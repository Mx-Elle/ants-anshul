import multiprocessing as mp
import os
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from agent import SAVE_PATH, Net
from ant_env import ACTIONS, CHANNELS, DC, DR, Env


@dataclass
class Config:
    board_size: int = 32
    harvest_r: int = 1
    vision_r: int = 8
    battle_r: int = 3
    max_t: int = 1000
    filters: int = 32
    n_blocks: int = 6
    n_envs: int = 20
    n_steps: int = 128
    batch_size: int = 512
    n_epochs: int = 4
    gamma: float = 0.99
    gae_lam: float = 0.95
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.05
    grad_norm: float = 0.5
    lr: float = 5e-4
    lr_min: float = 5e-5
    total_steps: int = 2_500_000
    stage1: int = 100_000
    stage2: int = 1_000_000
    pool_size: int = 8
    pool_freq: int = 200_000
    obs_clip: float = 10.0
    save_dir: str = "./models"
    ckpt_freq: int = 10
    device: str = "auto"


class Normalizer:
    def __init__(self, shape, clip: float = 10.0):
        C, H, W = shape
        self.mean, self.var, self.count, self.clip, self.C = (
            np.zeros(C, dtype=np.float64),
            np.ones(C, dtype=np.float64),
            1e-8,
            clip,
            C,
        )
        self.mean_t = self.std_t = None

    def update(self, x: np.ndarray):
        flat = x.reshape(-1, self.C, x.shape[-2] * x.shape[-1]).mean(axis=-1)
        n = flat.shape[0]
        total = self.count + n
        delta = flat.mean(axis=0) - self.mean
        self.mean += delta * n / total
        self.var = (
            self.count * self.var
            + n * flat.var(axis=0)
            + delta**2 * self.count * n / total
        ) / total
        self.count, self.mean_t, self.std_t = total, None, None

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean_t is None or self.mean_t.device != x.device:
            self.mean_t = torch.from_numpy(self.mean.astype(np.float32)).to(x.device)
            self.std_t = torch.from_numpy(
                np.sqrt(self.var + 1e-8).astype(np.float32)
            ).to(x.device)
        return (
            (x - self.mean_t[None, :, None, None])
            .div_(self.std_t[None, :, None, None])
            .clamp_(-self.clip, self.clip)
        )

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d):
        self.mean, self.var, self.count = d["mean"], d["var"], d["count"]


@dataclass
class Buffer:
    obs: np.ndarray
    acts: np.ndarray
    logp: np.ndarray
    vals: np.ndarray
    rews: np.ndarray
    dones: np.ndarray
    advs: np.ndarray = field(default_factory=lambda: np.array([]))
    rets: np.ndarray = field(default_factory=lambda: np.array([]))


def worker(rank, oq, aq, cfg, ashm_n, oshm_n, ashp, oshp):
    ashm, oshm = SharedMemory(name=ashm_n), SharedMemory(name=oshm_n)
    abuf, obuf = (
        np.ndarray(ashp, dtype=np.float32, buffer=ashm.buf),
        np.ndarray(oshp, dtype=np.float32, buffer=oshm.buf),
    )
    env = Env(cfg.board_size, cfg.harvest_r, cfg.vision_r, cfg.battle_r, cfg.max_t)
    obs, _ = env.reset()
    obuf[rank] = obs
    oq.put((rank, False, 0.0, {}))
    while True:
        msg = aq.get()
        if msg == "STOP":
            break
        if msg is not None:
            env.pool = msg
        obs, rew, term, trunc, info = env.step(abuf[rank].copy())
        done = term or trunc
        if done:
            obs, _ = env.reset()
        obuf[rank] = obs
        oq.put((rank, done, rew, info))
    ashm.close()
    oshm.close()


def heuristic_opponent(board, food, hills, harvest_r, vision_r, battle_r):
    H, W = board.shape
    INF = np.iinfo(np.int32).max

    def bfs(targets):
        dist = np.full((H, W), INF, dtype=np.int32)
        if not len(targets):
            return dist
        vis = board.walls.astype(bool)
        dist[targets[:, 0], targets[:, 1]] = 0
        vis[targets[:, 0], targets[:, 1]] = True
        frontier, dirs = (
            targets.copy(),
            np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int32),
        )
        d = 1
        while len(frontier):
            nxt = (frontier[:, None, :] + dirs[None, :, :]).reshape(-1, 2)
            nxt[:, 0] %= H
            nxt[:, 1] %= W
            mask = ~vis[nxt[:, 0], nxt[:, 1]]
            nxt = nxt[mask]
            if not len(nxt):
                break
            vis[nxt[:, 0], nxt[:, 1]] = True
            dist[nxt[:, 0], nxt[:, 1]] = d
            frontier, d = nxt, d + 1
        return dist

    d_f, d_h = bfs(np.argwhere(board.food == 1)), bfs(np.argwhere(board.hills == 1))
    ants = np.argwhere(board.ants == 2)
    r, c = ants[:, 0], ants[:, 1]
    dirs = np.array([[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]], dtype=np.int32)
    enemy_hills, occupied, out = (
        set(zip(*np.where(board.hills == 2))),
        set(zip(r.tolist(), c.tolist())),
        set(),
    )
    for i in range(len(r)):
        curr = (int(r[i]), int(c[i]))
        occupied.remove(curr)
        nr_all, nc_all = (curr[0] + dirs[:4, 0]) % H, (curr[1] + dirs[:4, 1]) % W
        scores = np.where(
            d_f[nr_all, nc_all] < INF,
            d_f[nr_all, nc_all],
            np.where(d_h[nr_all, nc_all] < INF, d_h[nr_all, nc_all] + 1000, INF),
        ).astype(np.int64)
        scores = np.append(scores, 2000)
        best_d = np.argsort(scores)
        for d in best_d:
            nr, nc = (curr[0] + dirs[d, 0]) % H, (curr[1] + dirs[d, 1]) % W
            dest = (int(nr), int(nc))
            if (
                not board.walls[nr, nc]
                and dest not in occupied
                and (dest == curr or dest not in enemy_hills)
            ):
                out.add((curr, dest))
                occupied.add(dest)
                break
    return out


def selfplay_opponent(path, norm, cfg):
    model = Net(cfg.filters, cfg.n_blocks, CHANNELS)
    model.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=False).get("model_state_dict")
    )
    model.eval()

    def opponent(board, food, hills, hr, vr, br):
        H, W = board.shape
        obs = np.zeros((CHANNELS, H, W), dtype=np.float32)
        obs[0], obs[1], obs[2], obs[3], obs[4] = (
            board.walls,
            board.hills == 2,
            board.hills == 1,
            board.ants == 2,
            board.ants == 1,
        )
        obs[5], p2c, p1c, p2h, p1h = (
            board.food.astype(np.float32),
            np.count_nonzero(board.ants == 2),
            np.count_nonzero(board.ants == 1),
            np.count_nonzero(board.hills == 2),
            np.count_nonzero(board.hills == 1),
        )
        obs[9], obs[10], obs[11] = (
            p2c / (p2c + p1c) if p2c + p1c else 0.5,
            p2h / (p2h + p1h) if p2h + p1h else 0.5,
            food[2] / (food[2] + 20.0),
        )
        t = norm.apply(torch.from_numpy(obs).unsqueeze(0))
        with torch.no_grad():
            logits = model.act(t).squeeze(0).numpy()
        ants = np.argwhere(board.ants == 2)
        r, c = ants[:, 0], ants[:, 1]
        occupied = set(zip(r.tolist(), c.tolist()))
        hills_set = set(zip(*np.where(board.hills == 2)))
        out = set()
        for i in range(len(r)):
            curr = (int(r[i]), int(c[i]))
            occupied.remove(curr)
            l = logits[:, curr[0], curr[1]]
            d_indices = np.argsort(-l)
            for d in d_indices:
                nr, nc = (curr[0] + DR[d]) % H, (curr[1] + DC[d]) % W
                dest = (int(nr), int(nc))
                if (
                    not board.walls[nr, nc]
                    and dest not in occupied
                    and (dest == curr or dest not in hills_set)
                ):
                    out.add((curr, dest))
                    occupied.add(dest)
                    break
        return out

    return opponent


def train_step(model, opt, buf, cfg, device, norm, stats_ref):
    T, E = buf.rews.shape
    tot, is_mps = T * E, device.type == "mps"
    norm.update(buf.obs)
    og = norm.apply(
        torch.from_numpy(buf.obs.reshape(tot, *buf.obs.shape[2:])).to(device)
    )
    ag, lpg, advg, retg, v_old = (
        torch.from_numpy(buf.acts.reshape(tot, *buf.acts.shape[2:])).to(device),
        torch.from_numpy(buf.logp.ravel()).to(device),
        torch.from_numpy(
            (buf.advs.ravel() - buf.advs.mean()) / (buf.advs.std() + 1e-8)
        ).to(device),
        torch.from_numpy(buf.rets.ravel()).to(device),
        torch.from_numpy(buf.vals.ravel()).to(device),
    )
    if is_mps:
        torch.mps.synchronize()
    for _ in range(cfg.n_epochs):
        perm = torch.randperm(tot, device=device)
        for s in range(0, tot, cfg.batch_size):
            idx = perm[s : s + cfg.batch_size]
            logits, vals = model(og[idx])
            vals = vals.squeeze(-1)
            lp_all = F.log_softmax(logits, dim=1)
            chosen = ag[idx].argmax(dim=1, keepdim=True)
            lp_cell = lp_all.gather(1, chosen).squeeze(1)
            mask = (og[idx, 3, :, :] > 0.0).float()
            na = mask.sum(dim=[1, 2]).clamp(min=1)
            nlp = (lp_cell * mask).sum(dim=[1, 2]) / na
            probs = lp_all.exp()
            ent = (-(probs * lp_all).sum(dim=1) * mask).sum(dim=[1, 2]).div(na).mean()
            ratio = torch.exp(torch.clamp(nlp - lpg[idx], -10, 10))
            ploss = -torch.min(
                ratio * advg[idx],
                torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * advg[idx],
            ).mean()
            v_clip = v_old[idx] + torch.clamp(vals - v_old[idx], -cfg.clip, cfg.clip)
            vloss = torch.max(
                F.smooth_l1_loss(vals, retg[idx]), F.smooth_l1_loss(v_clip, retg[idx])
            ).mean()
            loss = ploss + cfg.vf_coef * vloss - cfg.ent_coef * ent
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_norm)
            opt.step()
            if is_mps:
                torch.mps.synchronize()
            stats_ref["pl"], stats_ref["vl"], stats_ref["ent"] = (
                float(ploss.item()),
                float(vloss.item()),
                float(ent.item()),
            )


def calc_advantages(rews, vals, dones, last, gamma, lam):
    T, E = rews.shape
    advs, lg, vpad = (
        np.zeros_like(rews),
        np.zeros(E, dtype=np.float32),
        np.vstack([vals, last]),
    )
    for t in reversed(range(T)):
        m = 1.0 - dones[t]
        delta = rews[t] + gamma * vpad[t + 1] * m - vals[t]
        lg = delta + gamma * lam * m * lg
        advs[t] = lg
    return advs.astype(np.float32), (advs + vals).astype(np.float32)


def train(cfg=Config()):
    os.makedirs(cfg.save_dir, exist_ok=True)
    E, H, W = cfg.n_envs, cfg.board_size, cfg.board_size
    ashm, oshm = (
        SharedMemory(create=True, size=E * ACTIONS * H * W * 4),
        SharedMemory(create=True, size=E * CHANNELS * H * W * 4),
    )
    stop_event = threading.Event()

    def handler(sig, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, handler)
    try:
        s_acts, s_obs = (
            np.ndarray((E, ACTIONS, H, W), dtype=np.float32, buffer=ashm.buf),
            np.ndarray((E, CHANNELS, H, W), dtype=np.float32, buffer=oshm.buf),
        )
        ctx = mp.get_context("spawn")
        oqs, aqs = [ctx.Queue() for _ in range(E)], [ctx.Queue() for _ in range(E)]
        workers = [
            ctx.Process(
                target=worker,
                args=(
                    i,
                    oqs[i],
                    aqs[i],
                    cfg,
                    ashm.name,
                    oshm.name,
                    (E, ACTIONS, H, W),
                    (E, CHANNELS, H, W),
                ),
                daemon=True,
            )
            for i in range(E)
        ]
        for p in workers:
            p.start()
        cur_obs = np.zeros((E, CHANNELS, H, W), dtype=np.float32)
        [oqs[i].get() for i in range(E)]
        [np.copyto(cur_obs[i], s_obs[i]) for i in range(E)]
        device = (
            torch.device(
                "mps"
                if torch.backends.mps.is_available()
                else "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
            if cfg.device == "auto"
            else torch.device(cfg.device)
        )
        model, res_p, gs, uc, norm = (
            Net(cfg.filters, cfg.n_blocks, CHANNELS),
            os.path.join(cfg.save_dir, "latest.pt"),
            0,
            0,
            Normalizer((CHANNELS, H, W), cfg.obs_clip),
        )
        if os.path.isfile(res_p):
            ckpt = torch.load(res_p, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            gs, uc = ckpt.get("global_step", 0), ckpt.get("update_count", 0)
            if "obs_norm" in ckpt:
                norm.load_state_dict(ckpt["obs_norm"])
            print(f"Resumed from step {gs:,}")
        model.to(device)
        opt = AdamW(model.parameters(), lr=cfg.lr, eps=1e-5, weight_decay=1e-4)
        sched = CosineAnnealingLR(
            opt, T_max=max(1, cfg.total_steps // (cfg.n_steps * E)), eta_min=cfg.lr_min
        )
        pool, last_pool, T = deque(maxlen=cfg.pool_size), gs, cfg.n_steps
        b_obs, b_acts, b_lp, b_vals, b_rews, b_dones = (
            np.zeros((T, E, CHANNELS, H, W), dtype=np.float32),
            np.zeros((T, E, ACTIONS, H, W), dtype=np.float32),
            np.zeros((T, E), dtype=np.float32),
            np.zeros((T, E), dtype=np.float32),
            np.zeros((T, E), dtype=np.float32),
            np.zeros((T, E), dtype=np.float32),
        )
        ep_rew, roll = np.zeros(E, dtype=np.float32), deque(maxlen=100)
        stage, fps, mrew, win, stats, phase = (
            1,
            0.0,
            0.0,
            0.0,
            {"pl": 0.0, "vl": 0.0, "ent": 0.0},
            "INIT",
        )
        prog = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]PPO"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TextColumn("{task.fields[info]}"),
            refresh_per_second=60,
        )
        task, hstop = (
            prog.add_task("PPO", total=cfg.total_steps, completed=gs, info=""),
            threading.Event(),
        )

        def hud():
            while not hstop.is_set():
                prog.update(
                    task,
                    completed=min(gs, cfg.total_steps),
                    info=f"fps={fps:.0f} [{phase}] s={stage} rew={mrew:.2f} w={win:.1%} pl={stats['pl']:.3f} vl={stats['vl']:.3f}",
                )
                hstop.wait(1 / 60)

        threading.Thread(target=hud, daemon=True).start()
        prog.start()
        while gs < cfg.total_steps and not stop_event.is_set():
            phase = "ROLLOUT"
            stage = 1 if gs < cfg.stage1 else (2 if gs < cfg.stage2 else 3)
            if stage == 1:
                op_p = [None]
            elif stage == 2:
                op_p = [heuristic_opponent]
            else:
                op_p = [heuristic_opponent] + [
                    selfplay_opponent(p, norm, cfg) for p in pool
                ]
            if stage >= 3 and (gs - last_pool) >= cfg.pool_freq:
                p_path = os.path.join(cfg.save_dir, f"pool_{gs}.pt")
                torch.save(
                    {
                        "model_state_dict": model.cpu().state_dict(),
                        "obs_norm": norm.state_dict(),
                        "global_step": gs,
                    },
                    p_path,
                )
                model.to(device)
                pool.append(p_path)
                last_pool = gs
            model.eval()

            def fwd(o_np):
                ot = norm.apply(torch.from_numpy(o_np).to(device))
                with torch.no_grad():
                    l, v = model(ot)
                lpa, B_, C5, H_, W_ = F.log_softmax(l, dim=1), *l.shape
                gum = -torch.log(
                    -torch.log(
                        torch.rand(B_, C5, H_, W_, device=device).clamp_(min=1e-20)
                    ).clamp_(min=1e-20)
                )
                ch = (l + gum).argmax(dim=1)
                oh = torch.zeros_like(l).scatter_(1, ch.unsqueeze(1), 1.0)
                mask = (ot[:, 3, :, :] > 0.5).float()
                na = mask.sum(dim=[1, 2]).clamp(min=1)
                lp = lpa.gather(1, ch.unsqueeze(1)).squeeze(1)
                logp = (lp * mask).sum(dim=[1, 2]) / na
                return oh, logp, v.squeeze(-1)

            t_start = time.time()
            oh_g, lp_g, v_g = fwd(cur_obs)
            np.copyto(s_acts, oh_g.cpu().numpy())
            next_o, ohn, lpn, vn = np.empty_like(cur_obs), None, None, None
            for t in range(T):
                if stop_event.is_set():
                    break
                inj = op_p if t == 0 else None
                for i in range(E):
                    aqs[i].put(inj)
                b_obs[t] = cur_obs
                b_lp[t] = lp_g.cpu().numpy()
                b_vals[t] = v_g.cpu().numpy()
                if t < T - 1:
                    ohn, lpn, vn = fwd(cur_obs)
                for i in range(E):
                    rank, d, r, info = oqs[i].get()
                    next_o[i] = s_obs[i]
                    b_rews[t, i] = r
                    b_dones[t, i] = float(d)
                    ep_rew[i] += r
                    b_acts[t, i] = info.get("final_acts", np.zeros((ACTIONS, H, W)))
                    if d:
                        roll.append(
                            {"reward": ep_rew[i], "winner": info.get("winner", -1)}
                        )
                        ep_rew[i] = 0
                        er, ew = (
                            [e["reward"] for e in roll],
                            [e["winner"] for e in roll],
                        )
                        mrew, win = sum(er) / len(er), ew.count(1) / len(ew)
                cur_obs, gs = next_o.copy(), gs + E
                fps = (t + 1) * E / max(time.time() - t_start, 1e-6)
                if t < T - 1:
                    np.copyto(s_acts, ohn.cpu().numpy())
                    lp_g, v_g = lpn, vn
            if stop_event.is_set():
                break
            phase = "TRAIN"
            with torch.no_grad():
                otn = norm.apply(torch.from_numpy(cur_obs).to(device))
                _, lv = model(otn)
                lv_np = lv.squeeze(-1).cpu().numpy()
            model.train()
            train_step(
                model,
                opt,
                Buffer(
                    b_obs,
                    b_acts,
                    b_lp,
                    b_vals,
                    b_rews,
                    b_dones,
                    calc_advantages(
                        b_rews, b_vals, b_dones, lv_np, cfg.gamma, cfg.gae_lam
                    )[0],
                    calc_advantages(
                        b_rews, b_vals, b_dones, lv_np, cfg.gamma, cfg.gae_lam
                    )[1],
                ),
                cfg,
                device,
                norm,
                stats,
            )
            sched.step()
            uc += 1

            sd = model.cpu().state_dict()
            full_ckpt = {
                "model_state_dict": sd,
                "obs_norm": norm.state_dict(),
                "global_step": gs,
                "update_count": uc,
                "config": cfg.__dict__,
            }
            torch.save(full_ckpt, res_p)
            torch.save(full_ckpt, SAVE_PATH)

            if uc == 1 or uc % cfg.ckpt_freq == 0:
                torch.save(full_ckpt, os.path.join(cfg.save_dir, f"ant_model_{gs}.pt"))

            model.to(device)
        hstop.set()
        prog.stop()
        print("\n[STOP] Saving state...")
        sd = model.cpu().state_dict()
        full_ckpt = {
            "model_state_dict": sd,
            "obs_norm": norm.state_dict(),
            "global_step": gs,
            "update_count": uc,
        }
        torch.save(full_ckpt, SAVE_PATH)
        torch.save(full_ckpt, res_p)
        for i in range(E):
            aqs[i].put("STOP")
        for p in workers:
            p.join(timeout=0.5)
            if p.is_alive():
                p.terminate()
    finally:
        ashm.close()
        ashm.unlink()
