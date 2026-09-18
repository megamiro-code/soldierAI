# ============================================================
# Standalone local Tag / Micro-skill Evolution
# Knockout tournament + production RTS physics/rewards + replay
# ============================================================
# Usage:
#   python tag_game.py
#   python tag_game.py --soldiers 10 --updates 100
#   python tag_game.py --replay --policy 12 --bout 7
#   python tag_game.py --replay                    # latest policy/latest bout
#
# Training is LOCAL ONLY.  Modal is used only when --sync / --upload is given.
# Production PPO remains: modal run main.py
#
# Design:
#   - 8 individuals: 1 seed + 7 mutations
#   - Single elimination: 8 -> 4 -> 2 -> 1
#   - Each matchup: 8 games (candidate Red 4, candidate Blue 4)
#   - Only production Soldier/Commander Encoder + Micro heads mutate
#   - Main production step_one() is used directly, so Tag uses the SAME physics
#     and reward coefficients as the real RTS.
#   - No Tag-specific reward is introduced.
#   - Every game can be replayed later from its .npz; HTML is generated only by
#     an explicit --replay command and uses the production RTS replay renderer.
#   - Walls (both interior 2x2 blocks and the top/bottom edge 2x1 blocks) start
#     at zero and are introduced one at a time, in random order, as commander
#     kills become reliable.
# ============================================================

import argparse
import copy
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax, random

ROOT = Path(__file__).resolve().parent
LOCAL_BASE_DIR = ROOT / "PPO_RTS"
os.environ["RTS_BASE_DIR"] = str(LOCAL_BASE_DIR)

try:
    import modal  # local SDK only; no Modal compute is started here
except ModuleNotFoundError:
    modal = None
import main as rts

REMOTE_VOLUME_NAME = "rts-storage"
REMOTE_ROOT = "PPO_RTS"

TAG_POPULATION = 8
TAG_GAMES_PER_MATCH = 8
TAG_GAMES_PER_SIDE = TAG_GAMES_PER_MATCH // 2
TAG_MUTATION_STRENGTH = 0.06
TAG_BIAS_MUTATION_STRENGTH = 0.35
TAG_ATTACK_EXPLORATION_FLOOR = 0.15
TAG_ATTACK_EXPLORATION_CEIL = 0.85
TAG_MAX_STEPS = rts.MAX_STEPS
TAG_RESULT_EPS = 1e-6

# ------------------------------------------------------------
# Wall curriculum for the standalone knockout Tag system.
#
# All walls -- both the 15 interior 2x2 blocks and the 6 top/bottom edge
# 2x1 blocks -- are pooled into a single set of candidate blocks. The
# curriculum starts with ZERO active blocks (a fully open field) and adds
# one randomly chosen block (interior or edge, no preference) each time
# commander kills have been reliable for several consecutive generations.
# ------------------------------------------------------------

TAG_KILL_RATE_THRESHOLD = 0.25
TAG_KILL_STABLE_GENERATIONS = 3
TAG_INACTIVE_WALL_SENTINEL = 1000.0


def _make_2x2_wall_cells(cx, cz):
    return (
        (cx - 0.5, cz - 0.5),
        (cx - 0.5, cz + 0.5),
        (cx + 0.5, cz - 0.5),
        (cx + 0.5, cz + 0.5),
    )


def _make_edge_wall_cells(cx, cz):
    # 2 cells wide (x direction) x 1 cell high, centered at (cx, cz).
    return (
        (cx - 0.5, cz),
        (cx + 0.5, cz),
    )


# Interior 2x2 blocks (15 total): same 5x3 grid used previously.
TAG_INTERIOR_WALL_BLOCK_CENTERS = tuple(
    (float(x), float(z))
    for z in (-4.0, 0.0, 4.0)
    for x in (-4.0, 0.0, 4.0)
)

# Top/bottom edge 2x1 blocks (6 total): same footprint as the old fixed
# edge walls, but now treated as ordinary curriculum blocks.
TAG_EDGE_WALL_BLOCK_CENTERS = tuple(
    (float(x), float(z))
    for z in (-7.5, 7.5)
    for x in (-4.0, 0.0, 4.0)
)


def _build_wall_block_definitions():
    blocks = []
    for cx, cz in TAG_INTERIOR_WALL_BLOCK_CENTERS:
        blocks.append({
            "kind": "interior",
            "center": (cx, cz),
            "cells": _make_2x2_wall_cells(cx, cz),
        })
    for cx, cz in TAG_EDGE_WALL_BLOCK_CENTERS:
        blocks.append({
            "kind": "edge",
            "center": (cx, cz),
            "cells": _make_edge_wall_cells(cx, cz),
        })
    return tuple(blocks)


# Unified candidate pool. Block id is simply the index into this tuple
# (0..TAG_WALL_BLOCK_COUNT-1), fixed by construction order above.
TAG_WALL_BLOCKS = _build_wall_block_definitions()
TAG_WALL_BLOCK_COUNT = len(TAG_WALL_BLOCKS)
TAG_MAX_PHYSICS_WALL_CELLS = sum(len(b["cells"]) for b in TAG_WALL_BLOCKS)


def _build_active_tag_walls(active_wall_ids):
    """Build fixed-shape physics walls plus compact actual walls for replay."""
    active = sorted({int(i) for i in active_wall_ids})
    if any(i < 0 or i >= TAG_WALL_BLOCK_COUNT for i in active):
        raise ValueError(f"Invalid Tag wall block id(s): {active}")

    actual = []
    for i in active:
        actual.extend(TAG_WALL_BLOCKS[i]["cells"])

    physics = np.full(
        (TAG_MAX_PHYSICS_WALL_CELLS, 2),
        TAG_INACTIVE_WALL_SENTINEL,
        dtype=np.float32,
    )
    if actual:
        physics[:len(actual)] = np.asarray(actual, dtype=np.float32)
    return jnp.asarray(physics), np.asarray(actual, dtype=np.float32).reshape((-1, 2))


def _tag_wall_label(wall_id):
    block = TAG_WALL_BLOCKS[int(wall_id)]
    x, z = block["center"]
    tag = "E" if block["kind"] == "edge" else "I"
    return f"{tag}{int(wall_id):02d}@({x:+.0f},{z:+.0f})"


TAG_REPLAY_DIR = LOCAL_BASE_DIR / "tag_replay"
TAG_BOUT_DIR = LOCAL_BASE_DIR / "tag_bouts"
TAG_ELITE_DIR = LOCAL_BASE_DIR / "tag_elite"

for _d in (TAG_REPLAY_DIR, TAG_BOUT_DIR, TAG_ELITE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

TAG_ENCODER_KEYS = tuple(rts.TAG_ENCODER_KEYS)


# ============================================================
# Basic utilities
# ============================================================


def _tag_policy_number(path: str | Path) -> int:
    m = re.search(r"tag_policy_(\d+)", Path(path).name)
    return int(m.group(1)) if m else -1


def _bout_number(path: str | Path) -> int:
    m = re.search(r"_bout_(\d+)\.npz$", Path(path).name)
    return int(m.group(1)) if m else -1


def _latest_local_tag_policy():
    files = list(TAG_ELITE_DIR.glob("tag_policy_*.npz"))
    return (
        max(files, key=lambda p: (_tag_policy_number(p), p.stat().st_mtime))
        if files else None
    )


def _latest_local_bout(policy_number=None, bout_number=None):
    files = list(TAG_BOUT_DIR.glob("tag_policy_*_bout_*.npz"))
    if policy_number is not None:
        files = [p for p in files if _tag_policy_number(p) == int(policy_number)]
    if bout_number is not None:
        files = [p for p in files if _bout_number(p) == int(bout_number)]
    if not files:
        return None
    return max(files, key=lambda p: (_tag_policy_number(p), _bout_number(p)))


def _read_policy_npz(path: Path):
    with np.load(path, allow_pickle=False) as d:
        raw = {
            k[len("param_"):]: jnp.asarray(d[k])
            for k in d.files
            if k.startswith("param_")
        }
        meta = {}
        if "metadata_json" in d.files:
            try:
                meta = json.loads(str(d["metadata_json"]))
            except Exception:
                meta = {}
    return raw, meta


def _save_params(path: Path, params, metadata=None):
    # Tag evolution only mutates the production encoder + Micro heads.
    # Saving the entire production parameter tree made every Tag policy
    # needlessly large.  Store only the skill-bearing Tag subset.
    data = {}
    for k in TAG_ENCODER_KEYS:
        if k not in params:
            raise KeyError(f"Missing Tag parameter key: {k}")
        data[f"param_{k}"] = np.asarray(params[k])
    meta = dict(metadata or {})
    meta.update({
        "format": "tag_sparse_v2",
        "saved_parameter_keys": list(TAG_ENCODER_KEYS),
    })
    data["metadata_json"] = np.array(json.dumps(meta, ensure_ascii=False))
    np.savez_compressed(path, **data)
    return path


def _load_params(path: Path):
    raw, _meta = _read_policy_npz(path)
    raw_keys = set(raw)
    tag_keys = set(TAG_ENCODER_KEYS)

    # Compact Tag policy: reconstruct the full production tree from the
    # latest compatible Production Elite and overlay the saved Tag weights.
    if tag_keys.issubset(raw_keys) and not set(rts.required_policy_shapes()).issubset(raw_keys):
        production = rts.find_latest_elite()
        if production is None:
            raise FileNotFoundError(
                f"Tag policy {path.name} is a sparse Tag policy, but no Production Elite "
                "is available to provide the untouched parameters."
            )
        base, _ = rts.load_params(production)
        for k in TAG_ENCODER_KEYS:
            base[k] = raw[k]
        return base

    # Backward compatibility: older Tag policies stored the full parameter tree.
    params, _ = rts.migrate_policy_params(raw)
    return params


def _compact_legacy_tag_policies(remove_legacy=True):
    files = sorted(TAG_ELITE_DIR.glob("tag_policy_*.npz"), key=_tag_policy_number)
    converted = 0
    skipped = 0
    saved_bytes = 0

    required = set(rts.required_policy_shapes().keys())
    for path in files:
        try:
            raw, meta = _read_policy_npz(path)
            raw_keys = set(raw)
            if not required.issubset(raw_keys):
                skipped += 1
                continue

            out = {}
            for k in TAG_ENCODER_KEYS:
                out[f"param_{k}"] = np.asarray(raw[k])
            new_meta = dict(meta or {})
            new_meta.update({
                "format": "tag_sparse_v2",
                "saved_parameter_keys": list(TAG_ENCODER_KEYS),
                "compacted_from_full_policy": True,
            })
            out["metadata_json"] = np.array(json.dumps(new_meta, ensure_ascii=False))

            tmp = path.with_name(path.stem + ".compact.tmp.npz")
            np.savez_compressed(tmp, **out)
            old_size = path.stat().st_size
            new_size = tmp.stat().st_size
            if new_size >= old_size:
                tmp.unlink(missing_ok=True)
                skipped += 1
                continue

            if remove_legacy:
                os.replace(tmp, path)
            else:
                compact_path = path.with_name(path.stem + ".compact.npz")
                os.replace(tmp, compact_path)
            converted += 1
            saved_bytes += old_size - new_size
        except Exception as e:
            print(f"  Compact skip {path.name}: {e}")

    print(f"Tag policy compaction : converted={converted}, skipped={skipped}, "
          f"saved={saved_bytes / (1024 * 1024):.2f} MiB")
    return converted, skipped, saved_bytes


def _clone_params(params):
    return {k: jnp.array(v) for k, v in params.items()}


def _mutate_params(parent, key, strength=TAG_MUTATION_STRENGTH):
    """Mutation-only evolution with stronger exploration on Micro action biases.

    Encoder mutations stay relatively small.  Micro head biases are mutated more
    strongly because b_micro_s[2] directly controls the attack probability; with
    the previous tiny perturbation it stayed almost exactly at 0.5 and the
    knockout tournament frequently produced all-timeout ties.
    """
    keys = random.split(key, len(TAG_ENCODER_KEYS))
    child = {k: jnp.array(v) for k, v in parent.items()}
    for k, kk in zip(TAG_ENCODER_KEYS, keys):
        p = parent[k]
        noise = random.normal(kk, p.shape, dtype=p.dtype)
        if k in ("b_micro_s", "b_micro_c"):
            scale = jnp.float32(TAG_BIAS_MUTATION_STRENGTH)
        else:
            rms = jnp.sqrt(jnp.mean(p * p) + jnp.float32(1e-8))
            scale = jnp.float32(strength) * (rms + jnp.float32(0.01))
        child[k] = p + scale * noise
    return child


def _parameter_delta(parent, child):
    total = 0.0
    for k in TAG_ENCODER_KEYS:
        d = np.asarray(child[k] - parent[k], dtype=np.float32)
        total += float(np.sum(d * d))
    return float(np.sqrt(total))


def _production_reward_labels():
    return (
        f"HIT {rts.REWARD_SOLDIER_HIT:g} | "
        f"KILL {rts.REWARD_SOLDIER_KILL:g} | "
        f"MISS {rts.REWARD_SOLDIER_MISS:g} | "
        f"WALL {rts.REWARD_SOLDIER_WALL:g} | "
        f"APPROACH {rts.REWARD_SOLDIER_APPROACH:g}*progress | "
        f"CMD survival {rts.REWARD_COMMANDER_SURVIVAL:g}/step | "
        f"CMD hit {rts.REWARD_COMMANDER_HIT_BY_ENEMY:g} | "
        f"WIN {rts.REWARD_COMMANDER_WIN:g} / LOSS {rts.REWARD_COMMANDER_LOSS:g}"
    )


# ============================================================
# Tag battle state: exact production environment, fewer active soldiers
# ============================================================


def make_tag_initial_state(key, n_soldiers):
    """Create a randomized Tag start inside each team's own painted 2-column zone.

    The production replay paints Red at x in [-8,-6] and Blue at x in [6,8].
    Active soldiers and each team commander are sampled from the
    corresponding 2x16 start-cell pool without replacement, with a small in-cell
    jitter. This guarantees that no Tag unit starts outside the painted zone.
    """
    state = rts.reset_one(key)
    n = int(n_soldiers)
    if n > 31:
        raise ValueError("Tag soldiers/team must be <= 31 so commander cells remain free")

    k_red, k_blue, k_jit_r, k_jit_b = random.split(key, 4)

    inactive = jnp.concatenate([
        jnp.arange(rts.RED_SOLDIER_START + n, rts.RED_SOLDIER_END),
        jnp.arange(rts.BLUE_SOLDIER_START + n, rts.BLUE_SOLDIER_END),
    ])
    alive = state["alive"].at[inactive].set(0.0)
    hp = state["hp"].at[inactive].set(0.0)
    vx = state["vx"].at[inactive].set(0.0)
    vz = state["vz"].at[inactive].set(0.0)
    attack_timer = state["attack_timer"].at[inactive].set(0.0)
    speed = state["speed"].at[inactive].set(0.0)

    # Exactly the two columns used by the painted production start zones.
    start_z = jnp.arange(16, dtype=jnp.float32) - 7.5
    red_x_cols = jnp.array([-7.5, -6.5], dtype=jnp.float32)
    blue_x_cols = jnp.array([6.5, 7.5], dtype=jnp.float32)
    red_x_grid, red_z_grid = jnp.meshgrid(red_x_cols, start_z, indexing="xy")
    blue_x_grid, blue_z_grid = jnp.meshgrid(blue_x_cols, start_z, indexing="xy")
    red_pool = jnp.stack([red_x_grid.reshape(-1), red_z_grid.reshape(-1)], axis=1)
    blue_pool = jnp.stack([blue_x_grid.reshape(-1), blue_z_grid.reshape(-1)], axis=1)

    def sample_team(pool, perm_key, jitter_key):
        perm = random.permutation(perm_key, pool.shape[0])
        selected = pool[perm[: n + 1]]
        soldiers = selected[:n]
        cmd_a = selected[n]
        soldier_jitter = random.uniform(
            jitter_key, (n, 2), minval=-0.16, maxval=0.16
        )
        cmd_a_jitter = random.uniform(
            random.fold_in(jitter_key, 1), (2,), minval=-0.12, maxval=0.12
        )
        return (
            soldiers + soldier_jitter,
            cmd_a + cmd_a_jitter,
        )

    red_soldier_pos, red_cmd_pos = sample_team(red_pool, k_red, k_jit_r)
    blue_soldier_pos, blue_cmd_pos = sample_team(blue_pool, k_blue, k_jit_b)

    x = state["x"]
    z = state["z"]
    x = x.at[rts.RED_SOLDIER_START:rts.RED_SOLDIER_START + n].set(red_soldier_pos[:, 0])
    z = z.at[rts.RED_SOLDIER_START:rts.RED_SOLDIER_START + n].set(red_soldier_pos[:, 1])
    x = x.at[rts.BLUE_SOLDIER_START:rts.BLUE_SOLDIER_START + n].set(blue_soldier_pos[:, 0])
    z = z.at[rts.BLUE_SOLDIER_START:rts.BLUE_SOLDIER_START + n].set(blue_soldier_pos[:, 1])

    x = x.at[rts.RED_COMMANDER_INDEX].set(red_cmd_pos[0])
    z = z.at[rts.RED_COMMANDER_INDEX].set(red_cmd_pos[1])
    x = x.at[rts.BLUE_COMMANDER_INDEX].set(blue_cmd_pos[0])
    z = z.at[rts.BLUE_COMMANDER_INDEX].set(blue_cmd_pos[1])

    speed = speed.at[rts.RED_SOLDIER_START:rts.RED_SOLDIER_START + n].set(rts.INITIAL_SOLDIER_SPEED)
    speed = speed.at[rts.BLUE_SOLDIER_START:rts.BLUE_SOLDIER_START + n].set(rts.INITIAL_SOLDIER_SPEED)

    return {
        **state,
        "x": x,
        "z": z,
        "alive": alive,
        "hp": hp,
        "vx": vx,
        "vz": vz,
        "attack_timer": attack_timer,
        "speed": speed,
        "done": jnp.array(False),
        "time": jnp.array(0.0, dtype=jnp.float32),
    }


# ============================================================
# Micro-only Tag policy action
# ============================================================


def _micro_world_action(params, state, team, n_soldiers, rng_key, terrain_override):
    """Micro-only action using the same stochastic action style as production PPO.

    The Macro/attention branches are intentionally excluded.  Direction uses the
    Micro mean plus the production logstd exploration, and attack is sampled from
    the Micro Bernoulli probability rather than thresholded deterministically.
    """
    obs = rts.make_observation(state, float(team), terrain_override=terrain_override)[None, :]

    soldier_start = rts.TERRAIN_SIZE
    soldier_end = soldier_start + rts.N_SOLDIERS_TOTAL * rts.SOLDIER_FEATURES
    soldier_part = obs[:, soldier_start:soldier_end].reshape(
        1, rts.N_SOLDIERS_TOTAL, rts.SOLDIER_FEATURES
    )
    commander_part = obs[:, soldier_end:].reshape(
        1, rts.N_COMMANDERS, rts.COMMANDER_FEATURES
    )

    soldier_h = jnp.tanh(soldier_part @ params["Ws1"] + params["bs1"])
    soldier_emb = jnp.tanh(soldier_h @ params["Ws2"] + params["bs2"])
    commander_emb = jnp.tanh(commander_part @ params["Wc"] + params["bc"])

    micro_s_all = soldier_emb @ params["W_micro_s"] + params["b_micro_s"]
    micro_c_all = commander_emb @ params["W_micro_c"] + params["b_micro_c"]

    if int(team) == 0:
        own_s = micro_s_all[:, :rts.N_SOLDIERS_PER_TEAM, :]
        own_c = micro_c_all[:, 0, :]
    else:
        own_s = micro_s_all[:, rts.N_SOLDIERS_PER_TEAM:, :]
        own_c = micro_c_all[:, 1, :]

    s_vec = jnp.tanh(own_s[0, :, :2])
    s_vec = s_vec / (jnp.linalg.norm(s_vec, axis=-1, keepdims=True) + 1e-8)
    base_angle = jnp.arctan2(s_vec[:, 1], s_vec[:, 0])

    k_angle, k_attack, k_cmd = random.split(rng_key, 3)
    logstd = jnp.clip(params["logstd_s"], rts.LOGSTD_MIN, rts.LOGSTD_MAX)
    angle = base_angle + jnp.exp(logstd) * random.normal(
        k_angle, (rts.N_SOLDIERS_PER_TEAM,)
    )
    dx = jnp.cos(angle)
    dz = jnp.sin(angle)

    attack_prob_raw = jax.nn.sigmoid(own_s[0, :, 2])
    # Keep exploration alive during evolution: even a silent attack head has
    # a chance to discover the production HIT/KILL rewards.
    attack_prob = (
        jnp.float32(TAG_ATTACK_EXPLORATION_FLOOR)
        + jnp.float32(TAG_ATTACK_EXPLORATION_CEIL - TAG_ATTACK_EXPLORATION_FLOOR)
        * attack_prob_raw
    )
    attack = random.bernoulli(k_attack, attack_prob).astype(jnp.float32)

    c_vec = jnp.tanh(own_c[0])
    c_vec = c_vec / (jnp.linalg.norm(c_vec) + 1e-8)
    c_angle = jnp.arctan2(c_vec[1], c_vec[0])
    c_logstd = jnp.clip(params["logstd_c"], rts.LOGSTD_MIN, rts.LOGSTD_MAX)
    c_angle = c_angle + jnp.exp(c_logstd) * random.normal(k_cmd, ())
    cdx, cdz = jnp.cos(c_angle), jnp.sin(c_angle)

    local_action = jnp.zeros((rts.ACTION_SIZE,), dtype=jnp.float32)
    local_action = local_action.at[0:3 * n_soldiers:3].set(dx[:n_soldiers])
    local_action = local_action.at[1:3 * n_soldiers:3].set(dz[:n_soldiers])
    local_action = local_action.at[2:3 * n_soldiers:3].set(attack[:n_soldiers])
    local_action = local_action.at[rts.SOLDIER_ACTION_SIZE].set(cdx)
    local_action = local_action.at[rts.SOLDIER_ACTION_SIZE + 1].set(cdz)

    return rts.local_to_world_action(local_action[None, :], float(team))[0]


# JIT a single game rollout. n_soldiers is static because it only controls
# compile-time slicing/masking of action slots.
@jax.jit(static_argnums=(3,))
def _rollout_game(params_red, params_blue, initial_state, n_soldiers, game_key, physics_tag_walls, tag_terrain):
    """Run one game and return the exact production reward decomposition.

    `red_return` / `blue_return` are the same scalar reward definition used by
    production PPO: terminal win/timeout reward + mean(local reward over the
    100 soldier slots + commander slot).  Component rewards below are already
    scaled by the same 101-unit mean.
    """
    LOCAL_REWARD_DENOM = jnp.float32(rts.N_SOLDIERS_PER_TEAM + 1)

    def body(carry, step_idx):
        (
            state, finished, end_step,
            cum_red, cum_blue,
            red_attack_acc, blue_attack_acc,
            red_hit_acc, blue_hit_acc,
            red_kill_acc, blue_kill_acc,
        ) = carry

        red_action = _micro_world_action(
            params_red, state, 0, n_soldiers, random.fold_in(game_key, step_idx * 2), tag_terrain
        )
        blue_action = _micro_world_action(
            params_blue, state, 1, n_soldiers, random.fold_in(game_key, step_idx * 2 + 1), tag_terrain
        )

        nxt, terminal_red, terminal_blue, red_soldier_reward, blue_soldier_reward, red_cmd_reward, blue_cmd_reward, done_step = rts.step_one(
            state, red_action, blue_action, physics_tag_walls
        )

        red_local = jnp.mean(
            jnp.concatenate([red_soldier_reward, jnp.asarray([red_cmd_reward])])
        )
        blue_local = jnp.mean(
            jnp.concatenate([blue_soldier_reward, jnp.asarray([blue_cmd_reward])])
        )
        red_step_reward = terminal_red + red_local
        blue_step_reward = terminal_blue + blue_local

        # --------------------------------------------------------
        # Exact event counters/components using the same production state.
        # --------------------------------------------------------
        # Reconstruct the exact production attack bookkeeping.
        # Production does NOT count every attack command as an attack attempt:
        # a soldier must be alive, its cooldown must have expired, and only then
        # is the command evaluated against the post-movement target set.
        #
        # This is important for Tag because attack commands are sampled every
        # step, while the production attack cooldown is much longer than DT.
        # Counting commands directly makes the MISS penalty look ~10x too large.
        all_s = rts.ALL_SOLDIER_INDICES
        attack_cmd_all = jnp.concatenate([
            red_action[2:rts.SOLDIER_ACTION_SIZE:3],
            blue_action[2:rts.SOLDIER_ACTION_SIZE:3],
        ]) > 0.5
        timer_after_decay_all = jnp.maximum(
            0.0, state["attack_timer"][all_s] - rts.DT
        )
        attack_attempt_all = (
            attack_cmd_all
            & (timer_after_decay_all <= 0)
            & (state["alive"][all_s] > 0)
        )

        # Target availability matches main.step_one(): target search is done
        # using the positions after movement/separation, but the pre-attack
        # alive mask is used for valid targets.
        attacker_team = rts.teams[all_s]
        ax = nxt["x"][all_s]
        az = nxt["z"][all_s]
        ddx = nxt["x"][None, :] - ax[:, None]
        ddz = nxt["z"][None, :] - az[:, None]
        dist2 = ddx * ddx + ddz * ddz
        valid_target = (
            (rts.teams[None, :] != attacker_team[:, None])
            & (state["alive"][None, :] > 0)
            & (dist2 <= rts.ATTACK_RANGE ** 2)
            & (~(rts.ALL_UNIT_INDICES[None, :] == all_s[:, None]))
        )
        has_target = jnp.any(valid_target, axis=1)
        can_attack_all = attack_attempt_all & has_target
        attack_miss_all = attack_attempt_all & (~has_target)

        red_attack_attempt = attack_attempt_all[:rts.N_SOLDIERS_PER_TEAM]
        blue_attack_attempt = attack_attempt_all[rts.N_SOLDIERS_PER_TEAM:]
        red_can_attack = can_attack_all[:rts.N_SOLDIERS_PER_TEAM]
        blue_can_attack = can_attack_all[rts.N_SOLDIERS_PER_TEAM:]
        red_attack_miss = attack_miss_all[:rts.N_SOLDIERS_PER_TEAM]
        blue_attack_miss = attack_miss_all[rts.N_SOLDIERS_PER_TEAM:]

        red_attack_now = jnp.sum(red_attack_attempt).astype(jnp.float32)
        blue_attack_now = jnp.sum(blue_attack_attempt).astype(jnp.float32)
        red_hit_now = jnp.sum(red_can_attack).astype(jnp.float32)
        blue_hit_now = jnp.sum(blue_can_attack).astype(jnp.float32)

        red_enemy_hp_before = jnp.concatenate([
            state["hp"][rts.BLUE_COMMANDER_INDEX:rts.BLUE_COMMANDER_INDEX + 1],
            state["hp"][rts.BLUE_SOLDIER_START:rts.BLUE_SOLDIER_END],
        ])
        red_enemy_hp_after = jnp.concatenate([
            nxt["hp"][rts.BLUE_COMMANDER_INDEX:rts.BLUE_COMMANDER_INDEX + 1],
            nxt["hp"][rts.BLUE_SOLDIER_START:rts.BLUE_SOLDIER_END],
        ])
        blue_enemy_hp_before = jnp.concatenate([
            state["hp"][rts.RED_COMMANDER_INDEX:rts.RED_COMMANDER_INDEX + 1],
            state["hp"][rts.RED_SOLDIER_START:rts.RED_SOLDIER_END],
        ])
        blue_enemy_hp_after = jnp.concatenate([
            nxt["hp"][rts.RED_COMMANDER_INDEX:rts.RED_COMMANDER_INDEX + 1],
            nxt["hp"][rts.RED_SOLDIER_START:rts.RED_SOLDIER_END],
        ])

        red_kill_now = jnp.sum(
            (red_enemy_hp_before > 0) & (red_enemy_hp_after <= 0)
        ).astype(jnp.float32)
        blue_kill_now = jnp.sum(
            (blue_enemy_hp_before > 0) & (blue_enemy_hp_after <= 0)
        ).astype(jnp.float32)

        # Wall collision counts use the same pre-separation desired position test
        # as production step_one().
        red_dx, red_dz, _, red_cmd_dx, red_cmd_dz = rts.decode_actions(red_action)
        blue_dx, blue_dz, _, blue_cmd_dx, blue_cmd_dz = rts.decode_actions(blue_action)
        move_dx = jnp.zeros(rts.N_UNITS, dtype=jnp.float32)
        move_dz = jnp.zeros(rts.N_UNITS, dtype=jnp.float32)
        move_dx = move_dx.at[rts.RED_COMMANDER_INDEX].set(red_cmd_dx)
        move_dz = move_dz.at[rts.RED_COMMANDER_INDEX].set(red_cmd_dz)
        move_dx = move_dx.at[rts.BLUE_COMMANDER_INDEX].set(blue_cmd_dx)
        move_dz = move_dz.at[rts.BLUE_COMMANDER_INDEX].set(blue_cmd_dz)
        move_dx = move_dx.at[rts.ALL_SOLDIER_INDICES].set(jnp.concatenate([red_dx, blue_dx]))
        move_dz = move_dz.at[rts.ALL_SOLDIER_INDICES].set(jnp.concatenate([red_dz, blue_dz]))
        timer_after_decay = jnp.maximum(0.0, state["attack_timer"] - rts.DT)
        can_move = timer_after_decay <= 0
        distance = state["speed"] * rts.DT
        desired_x = state["x"] + move_dx * distance * can_move
        desired_z = state["z"] + move_dz * distance * can_move
        radius = rts.commander_mask * rts.COMMANDER_RADIUS + rts.soldier_mask * rts.SOLDIER_RADIUS
        inside = (
            (desired_x >= -rts.HALF_FIELD + radius)
            & (desired_x <= rts.HALF_FIELD - radius)
            & (desired_z >= -rts.HALF_FIELD + radius)
            & (desired_z <= rts.HALF_FIELD - radius)
        )
        attempted = (
            (state["alive"] > 0)
            & can_move
            & ((jnp.abs(move_dx) + jnp.abs(move_dz)) > 1e-6)
        )
        desired_wall = rts.wall_blocked(desired_x, desired_z, radius, physics_tag_walls)
        boundary_collision = attempted & (~inside)
        wall_collision = attempted & (desired_wall | (~inside))
        red_wall_now = jnp.sum(wall_collision[rts.RED_SOLDIER_START:rts.RED_SOLDIER_END]).astype(jnp.float32)
        blue_wall_now = jnp.sum(wall_collision[rts.BLUE_SOLDIER_START:rts.BLUE_SOLDIER_END]).astype(jnp.float32)
        red_cmd_wall_now = wall_collision[rts.RED_COMMANDER_INDEX].astype(jnp.float32)
        blue_cmd_wall_now = wall_collision[rts.BLUE_COMMANDER_INDEX].astype(jnp.float32)

        # Approach shaping: exact old/new distance change toward enemy commander.
        all_s = rts.ALL_SOLDIER_INDICES
        a_team = rts.teams[all_s]
        enemy_cmd_idx = jnp.where(a_team < 0.5, rts.BLUE_COMMANDER_INDEX, rts.RED_COMMANDER_INDEX)
        old_enemy_x = state["x"][enemy_cmd_idx]
        old_enemy_z = state["z"][enemy_cmd_idx]
        new_enemy_x = nxt["x"][enemy_cmd_idx]
        new_enemy_z = nxt["z"][enemy_cmd_idx]
        old_ax = state["x"][all_s]
        old_az = state["z"][all_s]
        new_ax = nxt["x"][all_s]
        new_az = nxt["z"][all_s]
        old_d = jnp.sqrt((old_enemy_x - old_ax) ** 2 + (old_enemy_z - old_az) ** 2 + 1e-8)
        new_d = jnp.sqrt((new_enemy_x - new_ax) ** 2 + (new_enemy_z - new_az) ** 2 + 1e-8)
        progress = old_d - new_d
        enemy_cmd_alive = (state["alive"][enemy_cmd_idx] > 0).astype(jnp.float32)
        approach = (
            rts.REWARD_SOLDIER_APPROACH
            * progress
            * (state["alive"][all_s] > 0).astype(jnp.float32)
            * enemy_cmd_alive
        )
        red_approach_now = jnp.sum(approach[:rts.N_SOLDIERS_PER_TEAM])
        blue_approach_now = jnp.sum(approach[rts.N_SOLDIERS_PER_TEAM:])

        red_cmd_hits_now = jnp.maximum(
            0.0,
            (state["hp"][rts.BLUE_COMMANDER_INDEX] - nxt["hp"][rts.BLUE_COMMANDER_INDEX])
            / rts.ATTACK_DAMAGE,
        )
        blue_cmd_hits_now = jnp.maximum(
            0.0,
            (state["hp"][rts.RED_COMMANDER_INDEX] - nxt["hp"][rts.RED_COMMANDER_INDEX])
            / rts.ATTACK_DAMAGE,
        )

        # Local rewards are averaged over 100 soldier slots + 1 commander slot,
        # exactly as the production PPO scalar reward does.
        # The event counters above are also aligned to production attack_attempt,
        # can_attack, and attack_miss semantics rather than raw action commands.
        red_hit_reward = rts.REWARD_SOLDIER_HIT * red_hit_now / LOCAL_REWARD_DENOM
        blue_hit_reward = rts.REWARD_SOLDIER_HIT * blue_hit_now / LOCAL_REWARD_DENOM
        red_kill_reward = rts.REWARD_SOLDIER_KILL * red_kill_now / LOCAL_REWARD_DENOM
        blue_kill_reward = rts.REWARD_SOLDIER_KILL * blue_kill_now / LOCAL_REWARD_DENOM
        red_miss_reward = rts.REWARD_SOLDIER_MISS * jnp.sum(red_attack_miss) / LOCAL_REWARD_DENOM
        blue_miss_reward = rts.REWARD_SOLDIER_MISS * jnp.sum(blue_attack_miss) / LOCAL_REWARD_DENOM
        red_wall_reward = (
            rts.REWARD_SOLDIER_WALL * red_wall_now
            + rts.REWARD_COMMANDER_WALL * red_cmd_wall_now
        ) / LOCAL_REWARD_DENOM
        blue_wall_reward = (
            rts.REWARD_SOLDIER_WALL * blue_wall_now
            + rts.REWARD_COMMANDER_WALL * blue_cmd_wall_now
        ) / LOCAL_REWARD_DENOM
        red_approach_reward = red_approach_now / LOCAL_REWARD_DENOM
        blue_approach_reward = blue_approach_now / LOCAL_REWARD_DENOM
        red_survival_reward = rts.REWARD_COMMANDER_SURVIVAL * nxt["alive"][rts.RED_COMMANDER_INDEX] / LOCAL_REWARD_DENOM
        blue_survival_reward = rts.REWARD_COMMANDER_SURVIVAL * nxt["alive"][rts.BLUE_COMMANDER_INDEX] / LOCAL_REWARD_DENOM
        red_cmdhit_reward = rts.REWARD_COMMANDER_HIT_BY_ENEMY * red_cmd_hits_now / LOCAL_REWARD_DENOM
        blue_cmdhit_reward = rts.REWARD_COMMANDER_HIT_BY_ENEMY * blue_cmd_hits_now / LOCAL_REWARD_DENOM

        active = ~finished
        newly_done = active & done_step
        effective_red = jnp.where(active, red_step_reward, 0.0)
        effective_blue = jnp.where(active, blue_step_reward, 0.0)

        next_state = jax.tree_util.tree_map(lambda n, o: jnp.where(active, n, o), nxt, state)
        next_finished = finished | done_step
        next_end_step = jnp.where(newly_done, step_idx + 1, end_step)
        next_red_attacks = red_attack_acc + jnp.where(active, red_attack_now, 0.0)
        next_blue_attacks = blue_attack_acc + jnp.where(active, blue_attack_now, 0.0)
        next_red_hits = red_hit_acc + jnp.where(active, red_hit_now, 0.0)
        next_blue_hits = blue_hit_acc + jnp.where(active, blue_hit_now, 0.0)
        next_red_kills = red_kill_acc + jnp.where(active, red_kill_now, 0.0)
        next_blue_kills = blue_kill_acc + jnp.where(active, blue_kill_now, 0.0)

        out = (
            next_state["x"], next_state["z"], next_state["hp"], next_state["alive"],
            red_action, blue_action, effective_red, effective_blue,
            jnp.where(active, red_attack_now, 0.0), jnp.where(active, blue_attack_now, 0.0),
            jnp.where(active, red_hit_now, 0.0), jnp.where(active, blue_hit_now, 0.0),
            jnp.where(active, red_kill_now, 0.0), jnp.where(active, blue_kill_now, 0.0),
            jnp.where(active, red_survival_reward, 0.0), jnp.where(active, blue_survival_reward, 0.0),
            jnp.where(active, red_hit_reward, 0.0), jnp.where(active, blue_hit_reward, 0.0),
            jnp.where(active, red_kill_reward, 0.0), jnp.where(active, blue_kill_reward, 0.0),
            jnp.where(active, red_miss_reward, 0.0), jnp.where(active, blue_miss_reward, 0.0),
            jnp.where(active, red_wall_reward, 0.0), jnp.where(active, blue_wall_reward, 0.0),
            jnp.where(active, red_approach_reward, 0.0), jnp.where(active, blue_approach_reward, 0.0),
            jnp.where(active, red_cmdhit_reward, 0.0), jnp.where(active, blue_cmdhit_reward, 0.0),
            jnp.where(active, terminal_red, 0.0), jnp.where(active, terminal_blue, 0.0),
        )
        return (
            (next_state, next_finished, next_end_step, cum_red + effective_red, cum_blue + effective_blue,
             next_red_attacks, next_blue_attacks, next_red_hits, next_blue_hits, next_red_kills, next_blue_kills),
            out,
        )

    init = (
        initial_state, jnp.array(False), jnp.int32(TAG_MAX_STEPS),
        jnp.float32(0.0), jnp.float32(0.0),
        jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0),
        jnp.float32(0.0), jnp.float32(0.0),
    )
    (final_state, _, end_step, cum_red, cum_blue, red_attack_acc, blue_attack_acc, red_hit_acc, blue_hit_acc, red_kill_acc, blue_kill_acc), traj = lax.scan(
        body, init, jnp.arange(TAG_MAX_STEPS)
    )

    (
        xs, zs, hps, alives, red_actions, blue_actions, red_rewards, blue_rewards,
        red_attack_steps, blue_attack_steps, red_hit_steps, blue_hit_steps,
        red_kill_steps, blue_kill_steps,
        red_survival_steps, blue_survival_steps,
        red_hit_reward_steps, blue_hit_reward_steps,
        red_kill_reward_steps, blue_kill_reward_steps,
        red_miss_reward_steps, blue_miss_reward_steps,
        red_wall_reward_steps, blue_wall_reward_steps,
        red_approach_reward_steps, blue_approach_reward_steps,
        red_cmdhit_reward_steps, blue_cmdhit_reward_steps,
        red_terminal_steps, blue_terminal_steps,
    ) = traj

    xs = jnp.concatenate([initial_state["x"][None, :], xs], axis=0)
    zs = jnp.concatenate([initial_state["z"][None, :], zs], axis=0)
    hps = jnp.concatenate([initial_state["hp"][None, :], hps], axis=0)
    alives = jnp.concatenate([initial_state["alive"][None, :], alives], axis=0)

    red_breakdown = {
        "survival": jnp.sum(red_survival_steps),
        "hit": jnp.sum(red_hit_reward_steps),
        "kill": jnp.sum(red_kill_reward_steps),
        "miss": jnp.sum(red_miss_reward_steps),
        "wall": jnp.sum(red_wall_reward_steps),
        "approach": jnp.sum(red_approach_reward_steps),
        "cmdhit": jnp.sum(red_cmdhit_reward_steps),
        "terminal": jnp.sum(red_terminal_steps),
        "total": cum_red,
    }
    blue_breakdown = {
        "survival": jnp.sum(blue_survival_steps),
        "hit": jnp.sum(blue_hit_reward_steps),
        "kill": jnp.sum(blue_kill_reward_steps),
        "miss": jnp.sum(blue_miss_reward_steps),
        "wall": jnp.sum(blue_wall_reward_steps),
        "approach": jnp.sum(blue_approach_reward_steps),
        "cmdhit": jnp.sum(blue_cmdhit_reward_steps),
        "terminal": jnp.sum(blue_terminal_steps),
        "total": cum_blue,
    }

    result = jnp.where(
        hps[-1, rts.RED_COMMANDER_INDEX] > 0,
        jnp.where(hps[-1, rts.BLUE_COMMANDER_INDEX] > 0, 3, 1),
        jnp.where(hps[-1, rts.BLUE_COMMANDER_INDEX] > 0, 2, 4),
    ).astype(jnp.int32)

    red_commander_kill = jnp.any(
        (hps[:-1, rts.RED_COMMANDER_INDEX] > 0)
        & (hps[1:, rts.RED_COMMANDER_INDEX] <= 0)
    )
    blue_commander_kill = jnp.any(
        (hps[:-1, rts.BLUE_COMMANDER_INDEX] > 0)
        & (hps[1:, rts.BLUE_COMMANDER_INDEX] <= 0)
    )

    return {
        "x": xs, "z": zs, "hp": hps, "alive": alives,
        "red_actions": red_actions, "blue_actions": blue_actions,
        "red_rewards": red_rewards, "blue_rewards": blue_rewards,
        "result": result, "end_step": end_step,
        "red_return": cum_red, "blue_return": cum_blue,
        "red_attack_attempts": red_attack_acc, "blue_attack_attempts": blue_attack_acc,
        "red_hits": red_hit_acc, "blue_hits": blue_hit_acc,
        "red_kills": red_kill_acc, "blue_kills": blue_kill_acc,
        "red_commander_kill": red_commander_kill,
        "blue_commander_kill": blue_commander_kill,
        "commander_kill": red_commander_kill | blue_commander_kill,
        "red_breakdown": red_breakdown, "blue_breakdown": blue_breakdown,
        "final_state": final_state,
    }


# ============================================================
# Game / match evaluation
# ============================================================


def _resolve_game_result(raw_result, red_return, blue_return, final_hp):
    raw_result = int(raw_result)
    red_return = float(red_return)
    blue_return = float(blue_return)
    hp = np.asarray(final_hp, dtype=np.float32)

    # Actual production commander win/loss always wins over any shaping.
    if raw_result == 1:
        return 0  # Red
    if raw_result == 2:
        return 1  # Blue
    if raw_result == 4:
        # Extremely rare simultaneous commander death: actual result is a draw;
        # resolve only if rewards separate it.
        pass

    diff = red_return - blue_return
    if abs(diff) > TAG_RESULT_EPS:
        return 0 if diff > 0 else 1

    hp_diff = float(hp[rts.RED_COMMANDER_INDEX] - hp[rts.BLUE_COMMANDER_INDEX])
    if abs(hp_diff) > TAG_RESULT_EPS:
        return 0 if hp_diff > 0 else 1
    return -1


def _match_score(winner_side, red_return, blue_return, red_hp, blue_hp, candidate_is_red):
    if winner_side < 0:
        return None
    candidate_return = red_return if candidate_is_red else blue_return
    opponent_return = blue_return if candidate_is_red else red_return
    candidate_hp = red_hp if candidate_is_red else blue_hp
    opponent_hp = blue_hp if candidate_is_red else red_hp
    return {
        "candidate_return": float(candidate_return),
        "opponent_return": float(opponent_return),
        "return_margin": float(candidate_return - opponent_return),
        "candidate_cmd_hp": float(candidate_hp),
        "opponent_cmd_hp": float(opponent_hp),
        "hp_margin": float(candidate_hp - opponent_hp),
    }




def _format_reward_short(label, reward):
    return (
        f"{label} total={reward['total']:+.3f} "
        f"(survive {reward['survival']:+.3f}, hit {reward['hit']:+.3f}, "
        f"kill {reward['kill']:+.3f}, miss {reward['miss']:+.3f}, "
        f"wall {reward['wall']:+.3f}, approach {reward['approach']:+.3f}, "
        f"cmdhit {reward['cmdhit']:+.3f}, terminal {reward['terminal']:+.3f})"
    )



def _run_match(candidate, opponent, rng, n_soldiers, policy_numbers, bout_start, physics_tag_walls, actual_tag_walls):
    wins = {0: 0, 1: 0}
    total_return = {0: 0.0, 1: 0.0}
    total_hp_margin = 0.0
    total_attacks = {0: 0.0, 1: 0.0}
    total_hits = {0: 0.0, 1: 0.0}
    total_kills = {0: 0.0, 1: 0.0}
    draws = 0
    commander_kill_games = 0
    commander_kill_events = 0
    game_records = []
    reward_totals = {0: {k: 0.0 for k in ("survival", "hit", "kill", "miss", "wall", "approach", "cmdhit", "terminal", "total")},
                     1: {k: 0.0 for k in ("survival", "hit", "kill", "miss", "wall", "approach", "cmdhit", "terminal", "total")}}

    for game_in_match in range(TAG_GAMES_PER_MATCH):
        candidate_is_red = game_in_match < TAG_GAMES_PER_SIDE
        params_red = candidate if candidate_is_red else opponent
        params_blue = opponent if candidate_is_red else candidate
        game_key = random.fold_in(rng, game_in_match + 1)
        state = make_tag_initial_state(game_key, n_soldiers)

        tag_terrain = rts.make_terrain_from_walls(physics_tag_walls)
        result = _rollout_game(
            params_red, params_blue, state, n_soldiers, game_key,
            physics_tag_walls, tag_terrain
        )
        result_np = {k: v for k, v in result.items() if k not in ("final_state", "red_breakdown", "blue_breakdown")}
        raw_result = int(np.asarray(result_np["result"]))
        red_return = float(np.asarray(result_np["red_return"]))
        blue_return = float(np.asarray(result_np["blue_return"]))
        end_step = int(np.asarray(result_np["end_step"]))
        # Final HP comes from the last trajectory frame.
        final_hp = np.asarray(result_np["hp"])[end_step]
        red_attack_attempts = float(np.asarray(result["red_attack_attempts"]))
        blue_attack_attempts = float(np.asarray(result["blue_attack_attempts"]))
        red_hits = float(np.asarray(result["red_hits"]))
        blue_hits = float(np.asarray(result["blue_hits"]))
        red_kills = float(np.asarray(result["red_kills"]))
        blue_kills = float(np.asarray(result["blue_kills"]))
        red_commander_kill = bool(np.asarray(result["red_commander_kill"]))
        blue_commander_kill = bool(np.asarray(result["blue_commander_kill"]))
        commander_kill = red_commander_kill or blue_commander_kill
        commander_kill_games += int(commander_kill)
        commander_kill_events += int(red_commander_kill) + int(blue_commander_kill)
        red_breakdown = {k: float(np.asarray(v)) for k, v in result["red_breakdown"].items()}
        blue_breakdown = {k: float(np.asarray(v)) for k, v in result["blue_breakdown"].items()}

        winner_side = _resolve_game_result(
            raw_result, red_return, blue_return, final_hp
        )
        if winner_side < 0:
            draws += 1
        else:
            winner_is_candidate = (
                (winner_side == 0 and candidate_is_red)
                or (winner_side == 1 and not candidate_is_red)
            )
            if winner_is_candidate:
                wins[0] += 1
            else:
                wins[1] += 1

        total_return[0] += red_return if candidate_is_red else blue_return
        total_return[1] += blue_return if candidate_is_red else red_return
        total_hp_margin += (
            float(final_hp[rts.RED_COMMANDER_INDEX] - final_hp[rts.BLUE_COMMANDER_INDEX])
            if candidate_is_red
            else float(final_hp[rts.BLUE_COMMANDER_INDEX] - final_hp[rts.RED_COMMANDER_INDEX])
        )
        total_attacks[0] += red_attack_attempts if candidate_is_red else blue_attack_attempts
        total_attacks[1] += blue_attack_attempts if candidate_is_red else red_attack_attempts
        total_hits[0] += red_hits if candidate_is_red else blue_hits
        total_hits[1] += blue_hits if candidate_is_red else red_hits
        total_kills[0] += red_kills if candidate_is_red else blue_kills
        total_kills[1] += blue_kills if candidate_is_red else red_kills

        bout_number = bout_start + game_in_match
        bout = {
            "policy_numbers": np.array(policy_numbers, dtype=np.int32),
            "match_game": game_in_match,
            "candidate_is_red": int(candidate_is_red),
            "winner_side": int(winner_side),
            "raw_result": raw_result,
            "winner_code": raw_result,
            "red_return": red_return,
            "blue_return": blue_return,
            "candidate_return": red_return if candidate_is_red else blue_return,
            "opponent_return": blue_return if candidate_is_red else red_return,
            "end_step": end_step,
            "dt": float(rts.DT),
            "field_size": float(rts.FIELD_SIZE),
            "terrain_res": int(rts.TERRAIN_RES),
            "raw_obs_size": int(rts.RAW_OBS_SIZE),
            "global_input_size": int(rts.GLOBAL_INPUT_SIZE),
            "action_size": int(rts.ACTION_SIZE),
            "attack_cooldown": float(rts.ATTACK_COOLDOWN),
            "start_x": np.asarray(state["x"]),
            "start_z": np.asarray(state["z"]),
            "start_hp": np.asarray(state["hp"]),
            "start_alive": np.asarray(state["alive"]),
            "x": np.asarray(result_np["x"]),
            "z": np.asarray(result_np["z"]),
            "hp": np.asarray(result_np["hp"]),
            "alive": np.asarray(result_np["alive"]),
            "red_actions": np.asarray(result_np["red_actions"]),
            "blue_actions": np.asarray(result_np["blue_actions"]),
            "red_rewards": np.asarray(result_np["red_rewards"]),
            "blue_rewards": np.asarray(result_np["blue_rewards"]),
            "red_attack_attempts": red_attack_attempts,
            "blue_attack_attempts": blue_attack_attempts,
            "red_hits": red_hits,
            "blue_hits": blue_hits,
            "red_kills": red_kills,
            "blue_kills": blue_kills,
            "red_commander_kill": int(red_commander_kill),
            "blue_commander_kill": int(blue_commander_kill),
            "commander_kill": int(commander_kill),
            "red_breakdown": red_breakdown,
            "blue_breakdown": blue_breakdown,
            "winner_team": int(winner_side),
            "result_text": rts.RESULT_TEXT.get(raw_result, "UNKNOWN"),
            "field_walls": np.asarray(rts.WALL_LIST, dtype=np.float32).reshape((-1, 2)),
            "tag_walls": np.asarray(actual_tag_walls, dtype=np.float32).reshape((-1, 2)),
            "bout_number": bout_number,
        }
        bout["reward_breakdown"] = {0: red_breakdown, 1: blue_breakdown}
        for team, bd in ((0, red_breakdown), (1, blue_breakdown)):
            for k in reward_totals[team]:
                reward_totals[team][k] += bd[k]
        game_records.append(bout)

    return {
        "candidate_wins": wins[0],
        "opponent_wins": wins[1],
        "candidate_return": total_return[0],
        "opponent_return": total_return[1],
        "candidate_return_margin": total_return[0] - total_return[1],
        "candidate_hp_margin": total_hp_margin,
        "candidate_attacks": total_attacks[0],
        "opponent_attacks": total_attacks[1],
        "candidate_hits": total_hits[0],
        "opponent_hits": total_hits[1],
        "candidate_kills": total_kills[0],
        "opponent_kills": total_kills[1],
        "draws": draws,
        "commander_kill_games": commander_kill_games,
        "commander_kill_events": commander_kill_events,
        "commander_kill_rate": commander_kill_games / float(TAG_GAMES_PER_MATCH),
        "games": game_records,
        "reward_totals": reward_totals,
    }


def _match_winner(match, rng):
    if match["candidate_wins"] != match["opponent_wins"]:
        return 0 if match["candidate_wins"] > match["opponent_wins"] else 1
    if abs(match["candidate_return_margin"]) > TAG_RESULT_EPS:
        return 0 if match["candidate_return_margin"] > 0 else 1
    if abs(match["candidate_hp_margin"]) > TAG_RESULT_EPS:
        return 0 if match["candidate_hp_margin"] > 0 else 1
    return int(random.bernoulli(rng))


# ============================================================
# Replay storage: production-compatible format
# ============================================================


def _save_bout(bout, policy_number, round_number, match_number, game_number):
    path = TAG_BOUT_DIR / (
        f"tag_policy_{policy_number:06d}_bout_{bout['bout_number']:03d}.npz"
    )
    steps = int(bout["end_step"])
    frames = steps + 1
    winner_side = int(bout["winner_team"])
    raw_result = int(bout["winner_code"])
    if winner_side < 0:
        winner_label = "Draw"
    else:
        winner_label = (
            f"Red policy {int(bout['policy_numbers'][0])}"
            if winner_side == 0
            else f"Blue policy {int(bout['policy_numbers'][1])}"
        )

    # Exact input schema expected by main.py's production build_replay_html().
    np.savez_compressed(
        path,
        generation=np.array(policy_number, dtype=np.int32),
        winner_label=np.array(winner_label),
        winner_team=np.array(winner_side, dtype=np.int32),
        result_code=np.array(raw_result, dtype=np.int32),
        win_time=np.array(steps * rts.DT, dtype=np.float32),
        end_step=np.array(steps, dtype=np.int32),
        dt=np.array(rts.DT, dtype=np.float32),
        candidate_is_red=np.array(bool(bout["candidate_is_red"])),
        candidate_wins=np.array(0, dtype=np.int32),
        elite_wins=np.array(0, dtype=np.int32),
        best_return=np.array(bout["candidate_return"], dtype=np.float32),
        field_size=np.array(rts.FIELD_SIZE, dtype=np.float32),
        terrain_res=np.array(rts.TERRAIN_RES, dtype=np.int32),
        raw_obs_size=np.array(rts.RAW_OBS_SIZE, dtype=np.int32),
        global_input_size=np.array(rts.GLOBAL_INPUT_SIZE, dtype=np.int32),
        action_size=np.array(rts.ACTION_SIZE, dtype=np.int32),
        attack_cooldown=np.array(rts.ATTACK_COOLDOWN, dtype=np.float32),
        tag_walls=np.asarray(bout["tag_walls"], dtype=np.float32).reshape((-1, 2)),
        start_x=bout["start_x"],
        start_z=bout["start_z"],
        start_hp=bout["start_hp"],
        start_alive=bout["start_alive"],
        x=bout["x"][:frames],
        z=bout["z"][:frames],
        hp=bout["hp"][:frames],
        alive=bout["alive"][:frames],
        red_actions=bout["red_actions"][:steps],
        blue_actions=bout["blue_actions"][:steps],
        red_rewards=bout["red_rewards"][:steps],
        blue_rewards=bout["blue_rewards"][:steps],
        red_attack_attempts=np.array(bout["red_attack_attempts"], dtype=np.float32),
        blue_attack_attempts=np.array(bout["blue_attack_attempts"], dtype=np.float32),
        red_hits=np.array(bout["red_hits"], dtype=np.float32),
        blue_hits=np.array(bout["blue_hits"], dtype=np.float32),
        red_kills=np.array(bout["red_kills"], dtype=np.float32),
        blue_kills=np.array(bout["blue_kills"], dtype=np.float32),

        red_commander_kill=np.array(bout["red_commander_kill"], dtype=np.int32),
        blue_commander_kill=np.array(bout["blue_commander_kill"], dtype=np.int32),
        commander_kill=np.array(bout["commander_kill"], dtype=np.int32),

        verification_pass=np.array(1, dtype=np.int32),
        max_state_error=np.array(0.0, dtype=np.float32),
        max_hp_error=np.array(0.0, dtype=np.float32),
        tag_policy_red=np.array(int(bout["policy_numbers"][0]), dtype=np.int32),
        tag_policy_blue=np.array(int(bout["policy_numbers"][1]), dtype=np.int32),
        tag_round=np.array(round_number, dtype=np.int32),
        tag_match=np.array(match_number, dtype=np.int32),
        tag_game=np.array(game_number, dtype=np.int32),
        tag_soldiers_per_team=np.array(int(np.sum(bout["start_alive"][2:102] > 0)), dtype=np.int32),
    )
    return path


def _render_tag_replay(bout_path, policy_number=None, bout_number=None):
    if policy_number is None:
        policy_number = _tag_policy_number(bout_path)
    if bout_number is None:
        bout_number = _bout_number(bout_path)
    out_path = TAG_REPLAY_DIR / (
        f"tag_replay_policy_{int(policy_number):06d}_bout_{int(bout_number):03d}.html"
    )
    html_path, _ = rts.build_replay_html(str(bout_path), str(out_path))
    return html_path


# ============================================================
# Evolution loop
# ============================================================


def _initial_seed_policy():
    production = rts.find_latest_elite()
    tag = _latest_local_tag_policy()
    if tag is None:
        if production is None:
            key = random.key(int(time.time()) & 0x7FFFFFFF)
            key, init_key = random.split(key)
            return rts.init_policy(init_key), "fresh random"
        return _load_params(Path(production)), f"production Elite {Path(production).name}"

    if production is None:
        return _load_params(tag), f"Tag Elite {tag.name}"

    # A Tag policy is sparse and is always merged onto the CURRENT Production
    # Elite inside _load_params().  mtime decides whether that Tag snapshot is
    # newer than the Production Elite itself.
    try:
        if tag.stat().st_mtime > Path(production).stat().st_mtime:
            return _load_params(tag), f"Tag Elite {tag.name}"
    except OSError:
        pass
    return _load_params(Path(production)), f"production Elite {Path(production).name}"


def _make_population(seed_params, rng):
    keys = random.split(rng, TAG_POPULATION - 1)
    pop = [_clone_params(seed_params)]
    for k in keys:
        pop.append(_mutate_params(seed_params, k))
    return pop


def _generation_tournament(population, policy_numbers, rng, n_soldiers, policy_number, save_bouts, physics_tag_walls, actual_tag_walls):
    """Run one 8->4->2->1 knockout and return the champion plus bout count."""
    next_bout = 0
    round_number = 1
    current = list(zip(population, policy_numbers))

    while len(current) > 1:
        if len(current) % 2 != 0:
            raise RuntimeError(
                f"Knockout tournament requires an even survivor count; got {len(current)}"
            )
        new_survivors = []
        total_matches = len(current) // 2
        print(f"  ROUND {round_number}: {total_matches} match(es)")
        for match_number, ((candidate, candidate_id), (opponent, opponent_id)) in enumerate(
            zip(current[0::2], current[1::2]), start=1
        ):
            print(
                f"    Match {match_number}/{total_matches}: "
                f"P{candidate_id:02d} vs P{opponent_id:02d}",
                end="",
                flush=True,
            )
            match_key = random.fold_in(rng, round_number * 10000 + match_number)
            match = _run_match(
                candidate, opponent, match_key, n_soldiers,
                (candidate_id, opponent_id), next_bout,
                physics_tag_walls, actual_tag_walls
            )
            winner_key = random.fold_in(match_key, 999)
            winner = _match_winner(match, winner_key)
            winner_id = candidate_id if winner == 0 else opponent_id
            winner_params = candidate if winner == 0 else opponent

            print(
                f" -> P{winner_id:02d} | "
                f"score {match['candidate_wins']}-{match['opponent_wins']} | "
                f"return {match['candidate_return_margin']:+.3f} | "
                f"HP {match['candidate_hp_margin']:+.3f} | "
                f"atk {match['candidate_attacks']:.0f}-{match['opponent_attacks']:.0f} | "
                f"hit {match['candidate_hits']:.0f}-{match['opponent_hits']:.0f} | "
                f"kill {match['candidate_kills']:.0f}-{match['opponent_kills']:.0f} | "
                f"draw {match['draws']}",
                flush=True,
            )
            rr = match["reward_totals"][0]
            oo = match["reward_totals"][1]
            print("      " + _format_reward_short(f"P{candidate_id:02d}", rr))
            print("      " + _format_reward_short(f"P{opponent_id:02d}", oo))
            print(
                f"      Commander kills: {match['commander_kill_games']}/{TAG_GAMES_PER_MATCH} "
                f"games ({match['commander_kill_rate'] * 100:.1f}%), events={match['commander_kill_events']}"
            )

            if save_bouts:
                for game_in_match, bout in enumerate(match["games"]):
                    _save_bout(bout, policy_number, round_number, match_number, game_in_match)

            next_bout += len(match["games"])
            new_survivors.append((winner_params, winner_id))

        current = new_survivors
        round_number += 1

    return current[0], next_bout


def _prune_bout_files(current_policy):
    """Keep only the current Tag generation's bouts."""
    current_policy = int(current_policy)
    removed = 0
    for path in TAG_BOUT_DIR.glob("tag_policy_*_bout_*.npz"):
        number = _tag_policy_number(path)
        if number == current_policy:
            continue
        try:
            path.unlink()
            removed += 1
        except OSError:
            pass
    return removed


def run_training(updates: int, soldiers: int, sync_remote=False, upload_remote=False):
    if not 1 <= int(soldiers) <= rts.N_SOLDIERS_PER_TEAM:
        raise SystemExit(
            f"--soldiers must be between 1 and {rts.N_SOLDIERS_PER_TEAM}, got {soldiers}"
        )
    if updates < 1:
        raise SystemExit("--updates must be >= 1")

    if sync_remote:
        sync_remote_seed_files()

    seed_params, seed_source = _initial_seed_policy()
    master_key = random.key(int(time.time()) & 0x7FFFFFFF)

    latest = _latest_local_tag_policy()
    next_policy = (_tag_policy_number(latest) + 1) if latest else 1
    print("============================================")
    print("STANDALONE TAG EVOLUTION")
    print("============================================")
    print(f"Seed                 : {seed_source}")
    print(f"Evolution generations : {updates}")
    print(f"Soldiers / team       : {soldiers}")
    print(f"Population            : {TAG_POPULATION}")
    print(f"Games / matchup       : {TAG_GAMES_PER_MATCH} (Red {TAG_GAMES_PER_SIDE} / Blue {TAG_GAMES_PER_SIDE})")
    print(f"Mutation strength     : encoder {TAG_MUTATION_STRENGTH:.3f}, micro-bias {TAG_BIAS_MUTATION_STRENGTH:.3f}")
    print(f"Attack exploration    : {TAG_ATTACK_EXPLORATION_FLOOR:.2f} .. {TAG_ATTACK_EXPLORATION_CEIL:.2f}")
    print(f"Max steps / game      : {TAG_MAX_STEPS}")
    print(
        f"Wall curriculum       : {TAG_WALL_BLOCK_COUNT} blocks total "
        f"({len(TAG_INTERIOR_WALL_BLOCK_CENTERS)} interior 2x2 + "
        f"{len(TAG_EDGE_WALL_BLOCK_CENTERS)} top/bottom edge 2x1); "
        f"starts at 0 active blocks; "
        f"+1 random block after {TAG_KILL_STABLE_GENERATIONS} stable generations at "
        f"≥{TAG_KILL_RATE_THRESHOLD * 100:.0f}% commander-kill games"
    )
    print(f"Physics                : main.step_one()")
    print(f"Rewards                : {_production_reward_labels()}")
    print("HTML replay            : only via explicit --replay")
    print("Rendering              : production build_replay_html()")
    print()

    champion = seed_params

    # Resume the wall curriculum from the latest sparse Tag policy metadata.
    active_wall_ids = []
    kill_stable_streak = 0
    latest_policy_path = _latest_local_tag_policy()
    if latest_policy_path is not None:
        try:
            _, latest_meta = _read_policy_npz(latest_policy_path)
            active_wall_ids = sorted({
                int(i) for i in latest_meta.get("active_wall_indices", [])
                if 0 <= int(i) < TAG_WALL_BLOCK_COUNT
            })
            kill_stable_streak = int(latest_meta.get("kill_stable_streak", 0))
        except Exception:
            active_wall_ids = []
            kill_stable_streak = 0

    total_t0 = time.time()
    for generation in range(1, updates + 1):
        master_key, pop_key, tour_key = random.split(master_key, 3)
        population = _make_population(champion, pop_key)
        ids = list(range(TAG_POPULATION))
        parent_champion = champion
        mutation_deltas = [_parameter_delta(parent_champion, p) for p in population[1:]]
        save_bouts = True

        physics_tag_walls, actual_tag_walls = _build_active_tag_walls(active_wall_ids)
        active_labels = [_tag_wall_label(i) for i in active_wall_ids]

        gen_t0 = time.time()
        print(f"===== EVOLUTION GENERATION {generation}/{updates} | policy {next_policy:06d} =====")
        print(
            f"  Mutations: avg Δ={np.mean(mutation_deltas):.4e}, "
            f"min Δ={np.min(mutation_deltas):.4e}, max Δ={np.max(mutation_deltas):.4e}"
        )
        print(
            f"  Walls: {len(active_wall_ids)}/{TAG_WALL_BLOCK_COUNT} active blocks "
            f"| active positions={active_labels if active_labels else 'none'}"
        )
        print("  Bout files: SAVE current generation only")
        champion_item, generation_bouts = _generation_tournament(
            population,
            ids,
            tour_key,
            soldiers,
            next_policy,
            save_bouts,
            physics_tag_walls,
            actual_tag_walls,
        )
        champion, champion_local_id = champion_item
        champion_delta = _parameter_delta(parent_champion, champion)

        # Read the current generation's saved bouts and summarize actual combat.
        # This makes zero-attack / zero-hit generations immediately visible.
        gen_atk = gen_hit = gen_kill = 0.0
        gen_reward = {0: 0.0, 1: 0.0}
        gen_draws = 0
        for bout_path in TAG_BOUT_DIR.glob(f"tag_policy_{next_policy:06d}_bout_*.npz"):
            try:
                with np.load(bout_path, allow_pickle=False) as d:
                    gen_atk += float(d["red_attack_attempts"]) + float(d["blue_attack_attempts"])
                    gen_hit += float(d["red_hits"]) + float(d["blue_hits"])
                    gen_kill += float(d["red_kills"]) + float(d["blue_kills"])
                    gen_reward[0] += float(np.sum(d["red_rewards"]))
                    gen_reward[1] += float(np.sum(d["blue_rewards"]))
                    if int(d["winner_team"]) < 0:
                        gen_draws += 1
            except Exception:
                pass

        gen_commander_kill_games = 0
        gen_commander_kill_events = 0
        for bout_path in TAG_BOUT_DIR.glob(f"tag_policy_{next_policy:06d}_bout_*.npz"):
            try:
                with np.load(bout_path, allow_pickle=False) as d:
                    gen_commander_kill_games += int(np.asarray(d["commander_kill"]))
                    gen_commander_kill_events += int(np.asarray(d["red_commander_kill"])) + int(np.asarray(d["blue_commander_kill"]))
            except Exception:
                pass

        commander_kill_rate = gen_commander_kill_games / float(max(1, generation_bouts))
        stable_this_generation = commander_kill_rate >= TAG_KILL_RATE_THRESHOLD
        if stable_this_generation:
            kill_stable_streak += 1
        else:
            kill_stable_streak = 0

        wall_added = None
        if (
            kill_stable_streak >= TAG_KILL_STABLE_GENERATIONS
            and len(active_wall_ids) < TAG_WALL_BLOCK_COUNT
        ):
            remaining = [i for i in range(TAG_WALL_BLOCK_COUNT) if i not in active_wall_ids]
            add_key = random.fold_in(tour_key, 0xC0FFEE + generation)
            chosen_pos = int(np.asarray(random.randint(add_key, (), 0, len(remaining))))
            wall_added = remaining[chosen_pos]
            active_wall_ids = sorted(active_wall_ids + [wall_added])
            kill_stable_streak = 0
            print(
                f"  WALL CURRICULUM: added {_tag_wall_label(wall_added)} "
                f"center={TAG_WALL_BLOCKS[wall_added]['center']} "
                f"after {TAG_KILL_STABLE_GENERATIONS} stable generations"
            )
        elif len(active_wall_ids) < TAG_WALL_BLOCK_COUNT:
            print(
                f"  WALL CURRICULUM: kill rate={commander_kill_rate * 100:.1f}% "
                f"(threshold={TAG_KILL_RATE_THRESHOLD * 100:.1f}%) "
                f"stable={kill_stable_streak}/{TAG_KILL_STABLE_GENERATIONS}; no new wall"
            )

        elapsed = time.time() - gen_t0
        tag_policy_path = TAG_ELITE_DIR / f"tag_policy_{next_policy:06d}.npz"
        _save_params(
            tag_policy_path,
            champion,
            {
                "source": "knockout_tag_evolution",
                "generation": generation,
                "policy_number": next_policy,
                "soldiers_per_team": soldiers,
                "population": TAG_POPULATION,
                "games_per_match": TAG_GAMES_PER_MATCH,
                "mutation_strength": TAG_MUTATION_STRENGTH,
                "champion_local_id": champion_local_id,
                "reward_definition": _production_reward_labels(),
                "source_seed": seed_source,
                "active_wall_indices": list(active_wall_ids),
                "active_wall_positions": [list(TAG_WALL_BLOCKS[i]["center"]) for i in active_wall_ids],
                "commander_kill_rate": commander_kill_rate,
                "commander_kill_events": gen_commander_kill_events,
                "kill_stable_streak": kill_stable_streak,
                "wall_added_this_generation": None if wall_added is None else int(wall_added),
            },
        )

        # Generation summary is emitted on every generation and stays visible.
        update_state = "UPDATED" if champion_delta > 1e-10 else "UNCHANGED"
        if gen_atk <= 0.0:
            print("  WARNING: zero attack attempts this generation.")
        elif gen_hit <= 0.0:
            print("  WARNING: attacks occurred, but zero HITs this generation.")
        print(
            f"Generation {generation:4d}/{updates} DONE | "
            f"Champion=P{champion_local_id:02d} | "
            f"Policy={next_policy:06d} | "
            f"{update_state} Δ={champion_delta:.4e} | "
            f"Bouts={generation_bouts} | "
            f"atk={gen_atk:.0f} hit={gen_hit:.0f} kill={gen_kill:.0f} draws={gen_draws} | "
            f"cmdkill={gen_commander_kill_games}/{generation_bouts} ({commander_kill_rate * 100:.1f}%) "
            f"stable={kill_stable_streak}/{TAG_KILL_STABLE_GENERATIONS} | "
            f"reward Red={gen_reward[0]:+.3f} Blue={gen_reward[1]:+.3f} | "
            f"time={elapsed:.1f}s"
        )
        removed = _prune_bout_files(next_policy)
        print(f"Tag policy saved      : {tag_policy_path}")
        print(f"Bout retention        : current policy only | removed {removed} old bout file(s)")
        print()

        next_policy += 1

    total_elapsed = time.time() - total_t0
    print("============================================")
    print("TAG EVOLUTION FINISHED")
    print("============================================")
    print(f"Final policy          : {next_policy - 1:06d}")
    print(f"Total time            : {total_elapsed:.1f} s")
    print(f"Raw bouts             : {TAG_BOUT_DIR}")
    print("HTML                  : generate only with --replay")

    if upload_remote:
        upload_tag_results()


# ============================================================
# Modal sync helpers
# ============================================================


def sync_remote_seed_files():
    if modal is None:
        raise RuntimeError("--sync requires the Modal SDK in the local Python environment.")
    vol = modal.Volume.from_name(REMOTE_VOLUME_NAME, create_if_missing=True)
    downloaded = 0
    for remote_dir, local_dir, prefix in (
        (f"/{REMOTE_ROOT}/elite", Path(rts.ELITE_DIR), "generation_"),
        (f"/{REMOTE_ROOT}/tag_elite", TAG_ELITE_DIR, "tag_policy_"),
        (f"/{REMOTE_ROOT}/tag_bouts", TAG_BOUT_DIR, "tag_policy_"),
    ):
        try:
            entries = list(vol.listdir(remote_dir, recursive=False))
        except Exception:
            continue
        tag_bout_entries = []
        for entry in entries:
            remote_path = str(entry.path)
            name = Path(remote_path).name
            if not name.endswith(".npz") or not name.startswith(prefix):
                continue
            if remote_dir.endswith("/tag_bouts"):
                tag_bout_entries.append((remote_path, name))
                continue
            data = b"".join(vol.read_file(remote_path))
            (local_dir / name).write_bytes(data)
            downloaded += 1
        if remote_dir.endswith("/tag_bouts") and tag_bout_entries:
            latest_policy = max(tag_bout_entries, key=lambda x: _tag_policy_number(x[1]))[0]
            latest_num = _tag_policy_number(latest_policy)
            for remote_path, name in tag_bout_entries:
                if _tag_policy_number(name) != latest_num:
                    continue
                data = b"".join(vol.read_file(remote_path))
                (local_dir / name).write_bytes(data)
                downloaded += 1
    print(f"Modal seed sync        : {downloaded} file(s)")


def upload_tag_results():
    if modal is None:
        raise RuntimeError("--upload requires the Modal SDK in the local Python environment.")
    vol = modal.Volume.from_name(REMOTE_VOLUME_NAME, create_if_missing=True)
    files = []
    for base, remote_dir in (
        (TAG_ELITE_DIR, f"/{REMOTE_ROOT}/tag_elite"),
        (TAG_BOUT_DIR, f"/{REMOTE_ROOT}/tag_bouts"),
    ):
        for path in base.glob("*.npz"):
            files.append((path, remote_dir))
    if not files:
        print("Modal upload            : no Tag files")
        return
    # Remove obsolete remote Tag bouts so old every-100th / historical files
    # do not keep consuming Volume storage after the retention policy changed.
    local_bout_names = {path.name for path in TAG_BOUT_DIR.glob("*.npz")}
    try:
        remote_bout_dir = f"/{REMOTE_ROOT}/tag_bouts"
        for entry in list(vol.listdir(remote_bout_dir, recursive=False)):
            remote_path = str(entry.path)
            name = Path(remote_path).name
            if name.endswith(".npz") and name.startswith("tag_policy_") and name not in local_bout_names:
                try:
                    vol.remove_file(remote_path)
                except Exception as exc:
                    print(f"Modal remote bout prune skipped: {remote_path}: {exc}")
    except Exception as exc:
        print(f"Modal remote bout listing/prune skipped: {exc}")

    with vol.batch_upload(force=True) as batch:
        for path, remote_dir in files:
            batch.put_file(str(path), f"{remote_dir}/{path.name}")
    try:
        vol.commit()
    except Exception:
        pass
    print(f"Modal Tag upload        : {len(files)} file(s)")


# ============================================================
# Replay CLI
# ============================================================


def run_replay(policy=None, bout=None, sync_remote=False):
    if sync_remote:
        sync_remote_seed_files()
    path = _latest_local_bout(policy, bout)
    if path is None:
        raise FileNotFoundError(
            f"No Tag bout found for policy={policy!r}, bout={bout!r} in {TAG_BOUT_DIR}"
        )
    html = _render_tag_replay(path, policy, bout)
    print(f"Tag replay source      : {path}")
    print(f"Tag replay HTML        : {html}")
    print("Replay appearance      : production RTS (soldiers + commanders + HP + attacks)")


def main():
    parser = argparse.ArgumentParser(description="Local knockout evolution for RTS Micro skills")
    parser.add_argument(
        "--updates", "--tag-updates", dest="updates", type=int, default=100,
        help="Number of evolutionary generations (default: 100)",
    )
    parser.add_argument(
        "--soldiers", "--n-soldiers", dest="soldiers", type=int, default=10,
        help=f"Active soldiers per team (default: 10, max: {rts.N_SOLDIERS_PER_TEAM})",
    )
    parser.add_argument("--replay", action="store_true", help="Render an existing Tag bout as production-style HTML")
    parser.add_argument("--policy", type=int, default=None, help="Tag policy number for replay")
    parser.add_argument("--bout", type=int, default=None, help="Bout number within the Tag policy")
    parser.add_argument("--sync", action="store_true", help="Download Tag/production NPZ files from Modal Volume first")
    parser.add_argument("--upload", action="store_true", help="Upload local Tag policy/bouts to Modal Volume after training")
    parser.add_argument("--compact", action="store_true", help="Compact legacy full-size Tag policies into sparse compressed Tag-only files and exit")
    args = parser.parse_args()

    if args.compact:
        _compact_legacy_tag_policies(remove_legacy=True)
        return

    if args.replay:
        run_replay(args.policy, args.bout, sync_remote=args.sync)
        return

    run_training(
        updates=args.updates,
        soldiers=args.soldiers,
        sync_remote=args.sync,
        upload_remote=args.upload,
    )


if __name__ == "__main__":
    main()