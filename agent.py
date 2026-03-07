from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from board import Entity

CHANNELS = 16
ACTIONS = 5
DR = np.array([0, -1, 1, 0, 0], dtype=np.int32)
DC = np.array([0, 0, 0, -1, 1], dtype=np.int32)
SAVE_PATH = os.path.join(os.path.dirname(__file__), "ant_model.pt")


class WrapConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1):
        super().__init__()
        self.pad = kernel // 2
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel, stride=stride, padding=0, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.pad
        return self.conv(F.pad(x, (p, p, p, p), mode="circular"))


class Block(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv1 = WrapConv2d(ch, ch)
        self.bn2 = nn.BatchNorm2d(ch)
        self.conv2 = WrapConv2d(ch, ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.relu(self.bn1(x), inplace=True))
        return x + self.conv2(F.relu(self.bn2(h), inplace=True))


class Net(nn.Module):
    def __init__(self, filters: int = 32, n_blocks: int = 6, in_ch: int = CHANNELS):
        super().__init__()
        self.stem = nn.Sequential(
            WrapConv2d(in_ch, filters), nn.BatchNorm2d(filters), nn.ReLU(inplace=True)
        )
        self.body = nn.Sequential(*[Block(filters) for _ in range(n_blocks)])
        self.actor = nn.Sequential(
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            WrapConv2d(filters, filters // 2),
            nn.BatchNorm2d(filters // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters // 2, ACTIONS, 1),
        )
        self.critic = nn.Sequential(
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(filters, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        if isinstance(self.actor[-1], nn.Conv2d):
            nn.init.zeros_(self.actor[-1].weight)
            if self.actor[-1].bias is not None:
                nn.init.zeros_(self.actor[-1].bias)

    def forward(self, x: torch.Tensor):
        feat = self.body(self.stem(x))
        return self.actor(feat), self.critic(feat)

    def act(self, x: torch.Tensor):
        return self.actor(self.body(self.stem(x)))


class Player:
    name = "AntNet"

    def __init__(
        self,
        walls,
        harvest_r,
        vision_r,
        battle_r,
        max_t,
        turn_time,
        model_path=SAVE_PATH,
    ):
        self.H, self.W = walls.shape
        self.vision_r, self.max_t = vision_r, max_t
        self.walls_bool, self.walls_f32 = walls.astype(bool), walls.astype(np.float32)
        self.explore_map = np.zeros((self.H, self.W), dtype=np.float32)
        self.friendly_ant_map = np.zeros((self.H, self.W), dtype=np.float32)
        self.enemy_ant_map = np.zeros((self.H, self.W), dtype=np.float32)
        self.food_map = np.zeros((self.H, self.W), dtype=np.float32)
        self.friendly_hill_map = np.zeros((self.H, self.W), dtype=np.float32)
        self.enemy_hill_map = np.zeros((self.H, self.W), dtype=np.float32)
        self.turn = self.stored_food = 0
        self.device = torch.device("cpu")
        self.model = Net().to(self.device).eval()
        self.mean, self.std = torch.zeros(CHANNELS), torch.ones(CHANNELS)
        r = np.arange(self.H)
        c = np.arange(self.W)
        self.sin_r, self.cos_r = (
            np.sin(2 * np.pi * r / self.H),
            np.cos(2 * np.pi * r / self.H),
        )
        self.sin_c, self.cos_c = (
            np.sin(2 * np.pi * c / self.W),
            np.cos(2 * np.pi * c / self.W),
        )
        if model_path and os.path.isfile(model_path):
            state = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(
                state.get("model_state_dict", state.get("model", state)), strict=False
            )
            if "obs_norm" in state:
                self.mean = torch.from_numpy(
                    state["obs_norm"]["mean"].astype(np.float32)
                )
                self.std = torch.from_numpy(
                    np.sqrt(state["obs_norm"]["var"] + 1e-8).astype(np.float32)
                )

    def _update_memory(self, vision: set, stored_food: int):
        self.turn += 1
        self.stored_food = stored_food
        self.explore_map *= 0.98
        self.enemy_ant_map *= 0.90
        self.food_map *= 0.95
        self.friendly_ant_map[:] = 0.0
        if not vision:
            return
        n = len(vision)
        vr, vc, ve = (
            np.empty(n, dtype=np.int32),
            np.empty(n, dtype=np.int32),
            np.empty(n, dtype=np.int8),
        )
        for i, ((r, c), ent) in enumerate(vision):
            vr[i], vc[i], ve[i] = r, c, ent.value
        self.explore_map[vr, vc] = 1.0
        visible = np.zeros((self.H, self.W), dtype=bool)
        visible[vr, vc] = True
        self.food_map[visible] = self.enemy_ant_map[visible] = self.enemy_hill_map[
            visible
        ] = 0.0
        was_f_hill = self.friendly_hill_map == 1.0
        self.friendly_hill_map[visible & was_f_hill] = 0.0
        f_ants, e_ants, food, f_hills, e_hills = (
            ve == Entity.FRIENDLY_ANT.value,
            ve == Entity.ENEMY_ANT.value,
            ve == Entity.FOOD.value,
            ve == Entity.FRIENDLY_HILL.value,
            ve == Entity.ENEMY_HILL.value,
        )
        self.friendly_ant_map[vr[f_ants], vc[f_ants]] = 1.0
        self.food_map[vr[food], vc[food]] = 1.0
        self.enemy_ant_map[vr[e_ants], vc[e_ants]] = 1.0
        self.enemy_hill_map[vr[e_hills], vc[e_hills]] = 1.0
        hr, hc = vr[f_hills], vc[f_hills]
        self.friendly_hill_map[hr, hc] = 1.0
        new_h = ~was_f_hill[hr, hc]
        if new_h.any():
            self.enemy_hill_map[self.H - 1 - hr[new_h], self.W - 1 - hc[new_h]] = 1.0

    def _get_obs(self):
        fa, ea = self.friendly_ant_map.sum(), self.enemy_ant_map.sum()
        fh, eh = self.friendly_hill_map.sum(), self.enemy_hill_map.sum()
        obs = np.empty((CHANNELS, self.H, self.W), dtype=np.float32)
        obs[0:8] = [
            self.walls_f32,
            self.friendly_hill_map,
            self.enemy_hill_map,
            self.friendly_ant_map,
            self.enemy_ant_map,
            self.food_map,
            self.explore_map,
            1.0 - self.explore_map,
        ]
        obs[8], obs[9], obs[10] = (
            self.turn / self.max_t,
            fa / (fa + ea + 1.0),
            fh / (fh + eh + 1.0),
        )
        obs[11] = self.stored_food / (self.stored_food + 20.0)
        obs[12], obs[13] = (
            self.sin_r[:, None] @ np.ones((1, self.W)),
            self.cos_r[:, None] @ np.ones((1, self.W)),
        )
        obs[14], obs[15] = (
            np.ones((self.H, 1)) @ self.sin_c[None, :],
            np.ones((self.H, 1)) @ self.cos_c[None, :],
        )
        t = torch.from_numpy(obs)
        return (
            (t - self.mean[:, None, None])
            .div_(self.std[:, None, None])
            .clamp_(-10.0, 10.0)
            .unsqueeze(0)
        )

    def move_ants(self, vision: set, stored_food: int) -> set:
        self._update_memory(vision, stored_food)
        pos = np.argwhere(self.friendly_ant_map > 0.5)
        if not len(pos):
            return set()
        with torch.no_grad():
            logits = self.model.act(self._get_obs().to(self.device)).squeeze(0)
        r, c = pos[:, 0], pos[:, 1]
        ant_logits = logits[:, r, c].cpu().numpy()
        probs = np.exp(ant_logits / 0.8)
        probs /= probs.sum(axis=0, keepdims=True)
        out, claimed = set(), set()
        for i in range(len(r)):
            curr, p = (int(r[i]), int(c[i])), probs[:, i]
            d_indices = np.argsort(-p)
            moved = False
            for d in d_indices:
                nr, nc = (curr[0] + DR[d]) % self.H, (curr[1] + DC[d]) % self.W
                dest = (int(nr), int(nc))
                if not self.walls_bool[nr, nc] and dest not in claimed:
                    out.add((curr, dest))
                    claimed.add(dest)
                    moved = True
                    break
            if not moved:
                out.add((curr, curr))
                claimed.add(curr)
        return out
