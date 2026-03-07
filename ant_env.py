import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ant_game import combat, flatten_hills, harvest, move_ants, spawn_ants
from board import Board, generate_board

CHANNELS = 16
ACTIONS = 5
DR = np.array([0, -1, 1, 0, 0], dtype=np.int32)
DC = np.array([0, 0, 0, -1, 1], dtype=np.int32)


def get_vision_disk(radius: int) -> np.ndarray:
    g = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    return np.argwhere((g * g).sum(0) <= radius * radius) - radius


def random_moves(board: Board, player: int, H: int, W: int) -> set:
    pos = np.argwhere(board.ants == player)
    if not len(pos):
        return set()
    r, c = pos[:, 0], pos[:, 1]
    hills = set(zip(*np.where(board.hills == player)))
    occupied = set(zip(r.tolist(), c.tolist()))
    out = set()
    for i in range(len(r)):
        curr = (int(r[i]), int(c[i]))
        occupied.remove(curr)
        dirs = np.random.permutation(ACTIONS)
        for d in dirs:
            nr, nc = (curr[0] + DR[d]) % H, (curr[1] + DC[d]) % W
            dest = (int(nr), int(nc))
            if (
                not board.walls[nr, nc]
                and dest not in occupied
                and (dest == curr or dest not in hills)
            ):
                out.add((curr, dest))
                occupied.add(dest)
                break
    return out


class Env(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self, board_size=32, harvest_r=1, vision_r=8, battle_r=3, max_t=1000, pool=None
    ):
        super().__init__()
        self.H = self.W = board_size
        self.harvest_r, self.vision_r, self.battle_r, self.max_t = (
            harvest_r,
            vision_r,
            battle_r,
            max_t,
        )
        self.observation_space = spaces.Box(
            0.0, 1.0, (CHANNELS, self.H, self.W), np.float32
        )
        self.action_space = spaces.Box(
            -np.inf, np.inf, (ACTIONS, self.H, self.W), np.float32
        )
        self.pool = pool or [None]
        self.opponent = None
        self.vision_disk = get_vision_disk(vision_r)
        self.board: Board = None
        self.hills1, self.hills2, self.food = {}, {}, {}
        self.turn = 0
        self.explore_map = np.zeros((self.H, self.W), dtype=np.float32)
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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(
            self.np_random.integers(0, 2**32)
            if seed is not None
            else np.random.randint(0, 2**32)
        )
        self.board = generate_board(self.H, self.W, rng=rng, percent_food=0.05)
        self.hills1 = {h: 0 for h in zip(*np.where(self.board.hills == 1))}
        self.hills2 = {h: 0 for h in zip(*np.where(self.board.hills == 2))}
        self.food = {1: len(self.hills1), 2: len(self.hills2)}
        self.turn = 0
        self.explore_map[:] = 0.0
        self.opponent = self.pool[int(self.np_random.integers(0, len(self.pool)))]
        return self.get_obs(), {}

    def step(self, action: np.ndarray):
        b = self.board
        ac0, hc0 = (
            np.bincount(b.ants.ravel(), minlength=3),
            np.bincount(b.hills.ravel(), minlength=3),
        )
        p1a0, p1h0, p2h0, f1_0 = int(ac0[1]), int(hc0[1]), int(hc0[2]), self.food[1]
        spawn_ants(b, self.food, self.hills1, self.hills2)
        ac_mid = np.bincount(b.ants.ravel(), minlength=3)
        births = max(0, int(ac_mid[1]) - p1a0)
        p1_moves, p1_final_acts = self.action_to_moves(np.asarray(action, np.float32))
        move_ants(b, p1_moves, self.opponent_moves())
        combat(b, self.battle_r)
        ac_after_combat = np.bincount(b.ants.ravel(), minlength=3)
        deaths, kills = (
            max(0, int(ac_mid[1]) - int(ac_after_combat[1])),
            max(0, int(ac_mid[2]) - int(ac_after_combat[2])),
        )
        flatten_hills(b)
        harvest(b, self.harvest_r, self.food)
        b.spawn_food()
        self.turn += 1
        obs = self.get_obs()
        p1_locs = np.argwhere((b.ants == 1) | (b.hills == 1))
        visible = np.zeros((self.H, self.W), dtype=bool)
        if len(p1_locs):
            rows, cols = (
                (p1_locs[:, 0:1] + self.vision_disk[:, 0]) % self.H,
                (p1_locs[:, 1:2] + self.vision_disk[:, 1]) % self.W,
            )
            visible[rows.ravel(), cols.ravel()] = True
        newly_explored = ((self.explore_map < 1.0) & visible).sum()
        np.add(self.explore_map, 1.0, where=visible, out=self.explore_map)
        np.clip(self.explore_map, 0.0, 1.0, out=self.explore_map)
        n_ants = len(p1_moves)
        move_reward = 0
        if n_ants > 0:
            stays = sum(1 for s, e in p1_moves if s == e)
            move_reward = 0.1 * (n_ants - 2 * stays) / n_ants
        ac1, hc1 = (
            np.bincount(b.ants.ravel(), minlength=3),
            np.bincount(b.hills.ravel(), minlength=3),
        )
        p1a1, p2a1, p1h1, p2h1 = int(ac1[1]), int(ac1[2]), int(hc1[1]), int(hc1[2])
        reward = (
            5.0 * max(0, self.food[1] - f1_0)
            + 0.2 * births
            + 0.5 * kills
            - 0.5 * deaths
            - 10.0 * max(0, p1h0 - p1h1)
            + 10.0 * max(0, p2h0 - p2h1)
            + 0.05 * newly_explored
            + move_reward
            - 0.02
        )
        if p1a1 + p2a1:
            reward += 0.01 * (p1a1 - p2a1) / (p1a1 + p2a1)
        done = p1h1 == 0 or p2h1 == 0 or self.turn >= self.max_t
        winner = -1
        if done:
            if p1h1 > p2h1:
                winner, reward = 1, reward + 10.0
            elif p2h1 > p1h1:
                winner, reward = 2, reward - 10.0
            else:
                s1, s2 = p1a1 + self.food[1], p2a1 + self.food[2]
                if s1 > s2:
                    winner, reward = 1, reward + 5.0
                elif s2 > s1:
                    winner, reward = 2, reward - 5.0
                else:
                    winner = 0
        return (
            obs,
            reward,
            done,
            False,
            {
                "turn": self.turn,
                "p1_ants": p1a1,
                "p1_hills": p1h1,
                "p1_food": self.food[1],
                "winner": winner,
                "final_acts": p1_final_acts,
            },
        )

    def get_obs(self) -> np.ndarray:
        b = self.board
        obs = np.empty((CHANNELS, self.H, self.W), dtype=np.float32)
        obs[0], obs[1], obs[2], obs[3] = (
            b.walls,
            b.hills == 1,
            b.hills == 2,
            b.ants == 1,
        )
        p1_locs = np.argwhere((b.ants == 1) | (b.hills == 1))
        visible = np.zeros((self.H, self.W), dtype=bool)
        if len(p1_locs):
            rows, cols = (
                (p1_locs[:, 0:1] + self.vision_disk[:, 0]) % self.H,
                (p1_locs[:, 1:2] + self.vision_disk[:, 1]) % self.W,
            )
            visible[rows.ravel(), cols.ravel()] = True
        obs[4], obs[5] = (b.ants == 2) & visible, (b.food == 1) & visible
        np.add(self.explore_map, 1.0 / self.max_t, where=visible, out=self.explore_map)
        np.clip(self.explore_map, 0.0, 1.0, out=self.explore_map)
        obs[6], obs[7] = self.explore_map, 1.0 - self.explore_map
        ac, hc = (
            np.bincount(b.ants.ravel(), minlength=3),
            np.bincount(b.hills.ravel(), minlength=3),
        )
        obs[8], obs[9], obs[10] = (
            self.turn / self.max_t,
            ac[1] / (ac[1] + ac[2]) if ac[1] + ac[2] else 0.5,
            hc[1] / (hc[1] + hc[2]) if hc[1] + hc[2] else 0.5,
        )
        obs[11] = self.food[1] / (self.food[1] + 20.0)
        obs[12], obs[13] = (
            self.sin_r[:, None] @ np.ones((1, self.W)),
            self.cos_r[:, None] @ np.ones((1, self.W)),
        )
        obs[14], obs[15] = (
            np.ones((self.H, 1)) @ self.sin_c[None, :],
            np.ones((self.H, 1)) @ self.cos_c[None, :],
        )
        return obs

    def action_to_moves(self, action: np.ndarray) -> tuple[set, np.ndarray]:
        pos = np.argwhere(self.board.ants == 1)
        final_acts = np.zeros((ACTIONS, self.H, self.W), dtype=np.float32)
        if not len(pos):
            return set(), final_acts
        r, c = pos[:, 0], pos[:, 1]
        hills, occupied, out = (
            set(self.hills1.keys()),
            set(zip(r.tolist(), c.tolist())),
            set(),
        )
        for i in range(len(r)):
            curr = (int(r[i]), int(c[i]))
            occupied.remove(curr)
            logits = action[:, curr[0], curr[1]]
            d_indices = np.argsort(-logits)
            for d in d_indices:
                nr, nc = (curr[0] + DR[d]) % self.H, (curr[1] + DC[d]) % self.W
                dest = (int(nr), int(nc))
                if (
                    not self.board.walls[nr, nc]
                    and dest not in occupied
                    and (dest == curr or dest not in hills)
                ):
                    out.add((curr, dest))
                    occupied.add(dest)
                    final_acts[d, curr[0], curr[1]] = 1.0
                    break
            if not any(m[0] == curr for m in out):
                out.add((curr, curr))
                occupied.add(curr)
                final_acts[0, curr[0], curr[1]] = 1.0
        return out, final_acts

    def opponent_moves(self) -> set:
        if self.opponent is None:
            return random_moves(self.board, 2, self.H, self.W)
        try:
            return self.opponent(
                self.board,
                self.food,
                self.hills2,
                self.harvest_r,
                self.vision_r,
                self.battle_r,
            )
        except:
            return random_moves(self.board, 2, self.H, self.W)
