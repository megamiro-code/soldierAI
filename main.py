# ============================================================
# JAX RTS : PPO + ELITE SELF-PLAY + 3D REPLAY
# Hierarchical local encoders + local combat rewards
# ============================================================

import os
import glob
import json
import time
import functools
import numpy as np

import jax
import jax.numpy as jnp
from jax import random, lax
import optax

# ========== ここから追加 ==========
import modal

# Modalの環境設定 (GPU対応のJAXとoptaxをインストール)
app_image = modal.Image.debian_slim().pip_install("jax[cuda12]", "optax")
# 学習途中データ(チェックポイント等)を保存・再開するための永続ボリューム
vol = modal.Volume.from_name("rts-storage", create_if_missing=True)
app = modal.App("rts-jax-ppo")
# ========== ここまで追加 ==========

print("JAX version :", jax.__version__)
print("Backend     :", jax.default_backend())
print("Devices     :", jax.devices())

# ============================================================
# PATHS
# ============================================================

# スクリプト（a.txtなど）が存在するディレクトリの絶対パスを取得
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# どこから実行しても、必ずスクリプトと同じ階層（soldierAIの中）にPPO_RTSを作る
BASE_DIR = os.environ.get("RTS_BASE_DIR", "/data/PPO_RTS")

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
ELITE_DIR = os.path.join(BASE_DIR, "elite")
BOUT_DIR = os.path.join(BASE_DIR, "elite_bouts")
REPLAY_DIR = os.path.join(BASE_DIR, "replay")

for d in (CHECKPOINT_DIR, ELITE_DIR, BOUT_DIR, REPLAY_DIR):
    os.makedirs(d, exist_ok=True)

# ============================================================
# PART 1 : ENVIRONMENT
# ============================================================

N_ENVS = 32
FIELD_SIZE = 16.0
HALF_FIELD = FIELD_SIZE / 2.0
FIELD_DIAG = float(np.sqrt(2.0) * HALF_FIELD)

N_SOLDIERS_PER_TEAM = 100
N_SOLDIERS_TOTAL = 200
N_COMMANDERS = 2
N_UNITS = N_SOLDIERS_TOTAL + N_COMMANDERS

DT = 0.20
MAX_TIME = 180.0
MAX_STEPS = int(MAX_TIME / DT)

ATTACK_RANGE = 0.42
ATTACK_COOLDOWN = 0.55
ATTACK_DAMAGE = 0.30

SOLDIER_RADIUS = 0.13
COMMANDER_RADIUS = 0.22
COMMANDER_EXTRA_MARGIN = 0.10
INITIAL_SOLDIER_SPEED = 0.50

TERRAIN_RES = 16
TERRAIN_SIZE = TERRAIN_RES * TERRAIN_RES

# ------------------------------------------------------------
# New observation structure
# ------------------------------------------------------------
# Soldier:
#   x, z, vx, vz, hp, own, commander_flag, alive       = 8
#   nearest enemy: distance, sin(theta), cos(theta)     = 3
#   own commander: distance, sin(theta), cos(theta)     = 3
#   enemy commander: distance, sin(theta), cos(theta)   = 3
#   total = 17
#
# Commander:
#   x, z, vx, vz, hp, own, commander_flag, alive       = 8
#   nearest enemy: distance, sin(theta), cos(theta)     = 3
#   surrounding 8 cells: wall / not wall                = 8
#   total = 19
#
# Raw observation = terrain + 200*17 + 2*19 = 3694
# Hierarchical encoder output = terrain + 200*64 + 2*64 = 13184
# Self-attention is applied independently to each team's 100 soldiers,
# preserving the 100-action symmetry while keeping both teams' embeddings.
# The policy observes all 200 soldiers (100 own + 100 enemy).
# ------------------------------------------------------------

SOLDIER_FEATURES = 17
COMMANDER_FEATURES = 19
LOCAL_EMBED_SIZE = 64
LOCAL_HIDDEN_SIZE = 64
ATTENTION_HEADS = 2
ATTENTION_HEAD_DIM = LOCAL_EMBED_SIZE // ATTENTION_HEADS
ATTENTION_TOKENS_PER_TEAM = N_SOLDIERS_PER_TEAM

RAW_OBS_SIZE = (
    TERRAIN_SIZE
    + N_SOLDIERS_TOTAL * SOLDIER_FEATURES
    + N_COMMANDERS * COMMANDER_FEATURES
)

GLOBAL_INPUT_SIZE = (
    TERRAIN_SIZE
    + N_SOLDIERS_TOTAL * LOCAL_EMBED_SIZE
    + N_COMMANDERS * LOCAL_EMBED_SIZE
)

OBS_SIZE = RAW_OBS_SIZE
OBS_DIM = OBS_SIZE

SOLDIER_ACTION_SIZE = N_SOLDIERS_PER_TEAM * 3
COMMANDER_ACTION_SIZE = 2
ACTION_SIZE = SOLDIER_ACTION_SIZE + COMMANDER_ACTION_SIZE
ACTION_DIM = ACTION_SIZE

# ------------------------------------------------------------
# Reward settings
# ------------------------------------------------------------

REWARD_SOLDIER_HIT = 0.050
REWARD_SOLDIER_MISS = -0.003
REWARD_SOLDIER_WALL = -0.050
REWARD_SOLDIER_KILL = 0.050
REWARD_SOLDIER_APPROACH = 0.005
REWARD_COMMANDER_WALL = 0.0
REWARD_COMMANDER_SURVIVAL = 0.0005
REWARD_COMMANDER_HIT_BY_ENEMY = -0.020
REWARD_COMMANDER_WIN = 4.0
REWARD_COMMANDER_LOSS = -4.0

# Original damage shaping is intentionally disabled.
SHAPING_COEF = 0.0

RED_COMMANDER_INDEX = 0
BLUE_COMMANDER_INDEX = 1
RED_SOLDIER_START = 2
RED_SOLDIER_END = 102
BLUE_SOLDIER_START = 102
BLUE_SOLDIER_END = 202

ALL_SOLDIER_INDICES = jnp.arange(RED_SOLDIER_START, BLUE_SOLDIER_END)
ALL_UNIT_INDICES = jnp.arange(N_UNITS)

WALL_LIST = [
]

walls = jnp.asarray(WALL_LIST, dtype=jnp.float32).reshape((-1, 2))

teams = jnp.concatenate([
    jnp.array([0.0, 1.0]),
    jnp.zeros(N_SOLDIERS_PER_TEAM),
    jnp.ones(N_SOLDIERS_PER_TEAM),
])

soldier_mask = jnp.concatenate([
    jnp.zeros(N_COMMANDERS),
    jnp.ones(N_SOLDIERS_TOTAL),
])
commander_mask = 1.0 - soldier_mask


def make_terrain():
    centers_1d = (
        jnp.arange(TERRAIN_RES, dtype=jnp.float32)
        - HALF_FIELD
        + 0.5
    )
    xx, zz = jnp.meshgrid(centers_1d, centers_1d)
    centers = jnp.stack([xx.reshape(-1), zz.reshape(-1)], axis=-1)

    def blocked(i):
        cell = centers[i]
        d = jnp.abs(cell[None, :] - walls)
        return jnp.any(jnp.all(d < 0.5, axis=1))

    t = jax.vmap(blocked)(jnp.arange(TERRAIN_SIZE))

    start_column = jnp.abs(centers[:, 0]) > (HALF_FIELD - 1.0)
    t = jnp.where(start_column, False, t)
    return t.astype(jnp.float32)


terrain = make_terrain()

# ============================================================
# RESET
# ============================================================


def reset_one(key):
    key_x, key_z = random.split(key)

    x = jnp.zeros(N_UNITS, dtype=jnp.float32)
    z = jnp.zeros(N_UNITS, dtype=jnp.float32)
    vx = jnp.zeros(N_UNITS, dtype=jnp.float32)
    vz = jnp.zeros(N_UNITS, dtype=jnp.float32)

    hp = jnp.ones(N_UNITS, dtype=jnp.float32)
    hp = hp.at[RED_COMMANDER_INDEX].set(0.30)
    hp = hp.at[BLUE_COMMANDER_INDEX].set(0.30)

    alive = jnp.ones(N_UNITS, dtype=jnp.float32)
    attack_timer = jnp.zeros(N_UNITS, dtype=jnp.float32)
    speed = jnp.zeros(N_UNITS, dtype=jnp.float32)

    x = x.at[RED_COMMANDER_INDEX].set(-5.80)
    z = z.at[RED_COMMANDER_INDEX].set(0.0)
    x = x.at[BLUE_COMMANDER_INDEX].set(5.80)
    z = z.at[BLUE_COMMANDER_INDEX].set(0.0)

    speed = speed.at[RED_COMMANDER_INDEX].set(INITIAL_SOLDIER_SPEED * 0.5)
    speed = speed.at[BLUE_COMMANDER_INDEX].set(INITIAL_SOLDIER_SPEED * 0.5)

    cell_x = jnp.array([-7.5, -6.5], dtype=jnp.float32)
    row_z = jnp.arange(10, dtype=jnp.float32) - 4.5

    angles = jnp.arange(5, dtype=jnp.float32) * (2.0 * jnp.pi / 5.0)
    offsets_x = 0.29 * jnp.cos(angles)
    offsets_z = 0.29 * jnp.sin(angles)

    cx, rz = jnp.meshgrid(cell_x, row_z)
    cx = cx.reshape(-1)
    rz = rz.reshape(-1)

    red_x = (cx[:, None] + offsets_x[None, :]).reshape(-1)
    red_z = (rz[:, None] + offsets_z[None, :]).reshape(-1)

    # Give each parallel environment a slightly different initial state while
    # preserving the left/right mirror symmetry between Red and Blue.
    jitter_x = random.uniform(
        key_x, shape=(N_SOLDIERS_PER_TEAM,), minval=-0.04, maxval=0.04
    )
    jitter_z = random.uniform(
        key_z, shape=(N_SOLDIERS_PER_TEAM,), minval=-0.04, maxval=0.04
    )
    red_x = red_x + jitter_x
    red_z = red_z + jitter_z
    blue_x = -red_x
    blue_z = red_z

    x = x.at[RED_SOLDIER_START:RED_SOLDIER_END].set(red_x)
    z = z.at[RED_SOLDIER_START:RED_SOLDIER_END].set(red_z)
    x = x.at[BLUE_SOLDIER_START:BLUE_SOLDIER_END].set(blue_x)
    z = z.at[BLUE_SOLDIER_START:BLUE_SOLDIER_END].set(blue_z)

    speed = speed.at[RED_SOLDIER_START:BLUE_SOLDIER_END].set(INITIAL_SOLDIER_SPEED)

    return {
        "x": x,
        "z": z,
        "vx": vx,
        "vz": vz,
        "hp": hp,
        "alive": alive,
        "attack_timer": attack_timer,
        "speed": speed,
        "time": jnp.array(0.0, dtype=jnp.float32),
        "done": jnp.array(False),
    }


reset_parallel = jax.jit(jax.vmap(reset_one))


def reset_env(key):
    return reset_one(key)


def reset(key):
    return reset_parallel(random.split(key, N_ENVS))


def reset_batch(key, n):
    return jax.vmap(reset_one)(random.split(key, n))


def get_alive_mask(state):
    return state["alive"]


def tree_index(tree, i):
    return jax.tree_util.tree_map(lambda a: a[i], tree)

# ============================================================
# PHYSICS / COMBAT
# ============================================================


def decode_actions(action):
    soldier_action = action[:SOLDIER_ACTION_SIZE].reshape(N_SOLDIERS_PER_TEAM, 3)
    raw_dx = soldier_action[:, 0]
    raw_dz = soldier_action[:, 1]
    raw_attack = soldier_action[:, 2]

    norm = jnp.sqrt(raw_dx * raw_dx + raw_dz * raw_dz + 1e-8)
    soldier_dx = raw_dx / norm
    soldier_dz = raw_dz / norm

    cmd_dx = action[SOLDIER_ACTION_SIZE]
    cmd_dz = action[SOLDIER_ACTION_SIZE + 1]
    cmd_norm = jnp.sqrt(cmd_dx * cmd_dx + cmd_dz * cmd_dz + 1e-8)

    return (
        soldier_dx,
        soldier_dz,
        (raw_attack > 0.5).astype(jnp.float32),
        cmd_dx / cmd_norm,
        cmd_dz / cmd_norm,
    )


def pairwise_separation(x, z, alive, movable=None):
    dx = x[:, None] - x[None, :]
    dz = z[:, None] - z[None, :]
    dist2 = dx * dx + dz * dz
    dist = jnp.sqrt(dist2 + 1e-8)

    radius = commander_mask * COMMANDER_RADIUS + soldier_mask * SOLDIER_RADIUS
    required = radius[:, None] + radius[None, :]

    cpair = (commander_mask[:, None] > 0) | (commander_mask[None, :] > 0)
    required = required + cpair.astype(jnp.float32) * COMMANDER_EXTRA_MARGIN

    overlap = jnp.maximum(required - dist, 0.0)
    valid = (
        (alive[:, None] > 0)
        & (alive[None, :] > 0)
        & (dist2 > 1e-10)
    )
    corr = overlap * valid.astype(jnp.float32) / (dist + 1e-8)

    push_x = jnp.sum(corr * dx, axis=1)
    push_z = jnp.sum(corr * dz, axis=1)

    push_x = jnp.where(commander_mask > 0, 0.0, push_x * 0.5)
    push_z = jnp.where(commander_mask > 0, 0.0, push_z * 0.5)

    if movable is not None:
        push_x = jnp.where(movable, push_x, 0.0)
        push_z = jnp.where(movable, push_z, 0.0)

    return push_x, push_z


def wall_blocked(x, z, radius):
    x2, z2, r2 = x[:, None], z[:, None], radius[:, None]
    wx, wz = walls[:, 0][None, :], walls[:, 1][None, :]
    cx = jnp.clip(x2, wx - 0.5, wx + 0.5)
    cz = jnp.clip(z2, wz - 0.5, wz + 0.5)
    dx, dz = x2 - cx, z2 - cz
    return jnp.any(dx * dx + dz * dz < r2 * r2, axis=1)


def step_one(state, red_action, blue_action):
    x, z = state["x"], state["z"]
    hp, alive = state["hp"], state["alive"]
    attack_timer, speed = state["attack_timer"], state["speed"]

    red_dx, red_dz, red_at, red_cmd_dx, red_cmd_dz = decode_actions(red_action)
    blue_dx, blue_dz, blue_at, blue_cmd_dx, blue_cmd_dz = decode_actions(blue_action)

    move_dx = jnp.zeros(N_UNITS, dtype=jnp.float32)
    move_dz = jnp.zeros(N_UNITS, dtype=jnp.float32)
    move_at = jnp.zeros(N_UNITS, dtype=jnp.float32)

    move_dx = move_dx.at[RED_COMMANDER_INDEX].set(red_cmd_dx)
    move_dz = move_dz.at[RED_COMMANDER_INDEX].set(red_cmd_dz)
    move_dx = move_dx.at[BLUE_COMMANDER_INDEX].set(blue_cmd_dx)
    move_dz = move_dz.at[BLUE_COMMANDER_INDEX].set(blue_cmd_dz)
    move_dx = move_dx.at[ALL_SOLDIER_INDICES].set(jnp.concatenate([red_dx, blue_dx]))
    move_dz = move_dz.at[ALL_SOLDIER_INDICES].set(jnp.concatenate([red_dz, blue_dz]))
    move_at = move_at.at[ALL_SOLDIER_INDICES].set(jnp.concatenate([red_at, blue_at]))

    attack_timer = jnp.maximum(0.0, attack_timer - DT)
    can_move = attack_timer <= 0
    distance = speed * DT

    nx = x + move_dx * distance * can_move
    nz = z + move_dz * distance * can_move

    radius = commander_mask * COMMANDER_RADIUS + soldier_mask * SOLDIER_RADIUS

    inside = (
        (nx >= -HALF_FIELD + radius)
        & (nx <= HALF_FIELD - radius)
        & (nz >= -HALF_FIELD + radius)
        & (nz <= HALF_FIELD - radius)
    )

    attempted_move = (
        (alive > 0)
        & can_move
        & ((jnp.abs(move_dx) + jnp.abs(move_dz)) > 1e-6)
    )
    desired_wall_hit = wall_blocked(nx, nz, radius)
    wall_collision = attempted_move & desired_wall_hit

    valid_move = inside & (~desired_wall_hit) & (alive > 0) & can_move

    nx = jnp.where(valid_move, nx, x)
    nz = jnp.where(valid_move, nz, z)
    vx = jnp.where(valid_move, move_dx, 0.0)
    vz = jnp.where(valid_move, move_dz, 0.0)

    movable = (
        (commander_mask > 0)
        | ((soldier_mask > 0) & can_move & (alive > 0))
    )
    px, pz = pairwise_separation(nx, nz, alive, movable)

    separated_x = nx + px
    separated_z = nz + pz
    separated_blocked = wall_blocked(separated_x, separated_z, radius)
    nx = jnp.where(separated_blocked, nx, separated_x)
    nz = jnp.where(separated_blocked, nz, separated_z)

    nx = jnp.clip(nx, -HALF_FIELD + radius, HALF_FIELD - radius)
    nz = jnp.clip(nz, -HALF_FIELD + radius, HALF_FIELD - radius)

    ax = nx[ALL_SOLDIER_INDICES]
    az = nz[ALL_SOLDIER_INDICES]

    ddx = nx[None, :] - ax[:, None]
    ddz = nz[None, :] - az[:, None]
    dist2 = ddx * ddx + ddz * ddz

    a_team = teams[ALL_SOLDIER_INDICES]
    enemy_mask = teams[None, :] != a_team[:, None]

    valid_target = (
        enemy_mask
        & (alive[None, :] > 0)
        & (dist2 <= ATTACK_RANGE ** 2)
        & (~(ALL_UNIT_INDICES[None, :] == ALL_SOLDIER_INDICES[:, None]))
    )

    target = jnp.argmin(jnp.where(valid_target, dist2, 1e9), axis=1)
    has_target = jnp.any(valid_target, axis=1)

    soldier_attack_timer = attack_timer[ALL_SOLDIER_INDICES]
    attack_attempt = (
        (move_at[ALL_SOLDIER_INDICES] > 0.5)
        & (soldier_attack_timer <= 0)
        & (alive[ALL_SOLDIER_INDICES] > 0)
    )

    can_attack = attack_attempt & has_target
    attack_miss = attack_attempt & (~has_target)

    damage_values = ATTACK_DAMAGE * can_attack.astype(jnp.float32)

    attack_damage = jnp.zeros(N_UNITS, dtype=jnp.float32)
    attack_damage = attack_damage.at[target].add(damage_values)

    hp_before = hp
    hp_after = hp - attack_damage
    target_died = (hp_before > 0) & (hp_after <= 0)

    attacker_kill = (
        can_attack
        & target_died[target]
    ).astype(jnp.float32)

    soldier_wall = wall_collision[ALL_SOLDIER_INDICES].astype(jnp.float32)

    # Per-soldier approach shaping: reward only actual progress toward the
    # enemy commander, not merely being near it. This is kept small so it does
    # not dominate hit/kill or the terminal win signal.
    enemy_cmd_idx = jnp.where(
        a_team < 0.5,
        BLUE_COMMANDER_INDEX,
        RED_COMMANDER_INDEX,
    )
    old_enemy_x = x[enemy_cmd_idx]
    old_enemy_z = z[enemy_cmd_idx]
    new_enemy_x = nx[enemy_cmd_idx]
    new_enemy_z = nz[enemy_cmd_idx]

    old_ax = x[ALL_SOLDIER_INDICES]
    old_az = z[ALL_SOLDIER_INDICES]
    old_cmd_dx = old_enemy_x - old_ax
    old_cmd_dz = old_enemy_z - old_az
    new_cmd_dx = new_enemy_x - ax
    new_cmd_dz = new_enemy_z - az
    old_cmd_dist = jnp.sqrt(old_cmd_dx * old_cmd_dx + old_cmd_dz * old_cmd_dz + 1e-8)
    new_cmd_dist = jnp.sqrt(new_cmd_dx * new_cmd_dx + new_cmd_dz * new_cmd_dz + 1e-8)
    distance_progress = old_cmd_dist - new_cmd_dist
    enemy_cmd_alive = (alive[enemy_cmd_idx] > 0).astype(jnp.float32)
    approach_reward = (
        REWARD_SOLDIER_APPROACH
        * distance_progress
        * (alive[ALL_SOLDIER_INDICES] > 0).astype(jnp.float32)
        * enemy_cmd_alive
    )

    soldier_rewards = (
        REWARD_SOLDIER_HIT * can_attack.astype(jnp.float32)
        + REWARD_SOLDIER_MISS * attack_miss.astype(jnp.float32)
        + REWARD_SOLDIER_WALL * soldier_wall
        + REWARD_SOLDIER_KILL * attacker_kill
        + approach_reward
    )

    local_reward = jnp.zeros(N_UNITS, dtype=jnp.float32)
    local_reward = local_reward.at[ALL_SOLDIER_INDICES].set(soldier_rewards)

    commander_alive_after = alive * commander_mask
    local_reward = (
        local_reward
        + REWARD_COMMANDER_WALL
        * wall_collision.astype(jnp.float32)
        * commander_mask
        + REWARD_COMMANDER_SURVIVAL * commander_alive_after
    )

    red_cmd_hits = jnp.sum(
        can_attack
        & (target == RED_COMMANDER_INDEX)
        & (a_team == 1.0)
    )
    blue_cmd_hits = jnp.sum(
        can_attack
        & (target == BLUE_COMMANDER_INDEX)
        & (a_team == 0.0)
    )
    local_reward = local_reward.at[RED_COMMANDER_INDEX].add(
        REWARD_COMMANDER_HIT_BY_ENEMY * red_cmd_hits
    )
    local_reward = local_reward.at[BLUE_COMMANDER_INDEX].add(
        REWARD_COMMANDER_HIT_BY_ENEMY * blue_cmd_hits
    )

    hp = hp_after
    alive = jnp.where(hp <= 0, 0.0, alive)

    old_t = attack_timer[ALL_SOLDIER_INDICES]
    attack_timer = attack_timer.at[ALL_SOLDIER_INDICES].set(
        jnp.where(attack_attempt, ATTACK_COOLDOWN, old_t)
    )

    new_time = state["time"] + DT

    red_cmd_alive = alive[RED_COMMANDER_INDEX] > 0
    blue_cmd_alive = alive[BLUE_COMMANDER_INDEX] > 0
    commander_done = (~red_cmd_alive) | (~blue_cmd_alive)
    timeout = new_time >= MAX_TIME
    done = commander_done | timeout

    red_n = jnp.sum(alive[RED_SOLDIER_START:RED_SOLDIER_END])
    blue_n = jnp.sum(alive[BLUE_SOLDIER_START:BLUE_SOLDIER_END])

    terminal_red = jnp.where(
        commander_done,
        jnp.where(
            red_cmd_alive & (~blue_cmd_alive),
            REWARD_COMMANDER_WIN,
            jnp.where(
                blue_cmd_alive & (~red_cmd_alive),
                REWARD_COMMANDER_LOSS,
                0.0,
            ),
        ),
        jnp.where(timeout, (red_n - blue_n) / N_SOLDIERS_PER_TEAM, 0.0),
    )
    terminal_blue = -terminal_red

    red_soldier_reward = local_reward[RED_SOLDIER_START:RED_SOLDIER_END]
    blue_soldier_reward = local_reward[BLUE_SOLDIER_START:BLUE_SOLDIER_END]
    red_commander_reward = local_reward[RED_COMMANDER_INDEX]
    blue_commander_reward = local_reward[BLUE_COMMANDER_INDEX]

    next_state = {
        "x": nx,
        "z": nz,
        "vx": vx,
        "vz": vz,
        "hp": hp,
        "alive": alive,
        "attack_timer": attack_timer,
        "speed": speed,
        "time": new_time,
        "done": done,
    }

    return (
        next_state,
        terminal_red,
        terminal_blue,
        red_soldier_reward,
        blue_soldier_reward,
        red_commander_reward,
        blue_commander_reward,
        done,
    )


def env_step(state, red_action, blue_action):
    return step_one(state, red_action, blue_action)



# ============================================================
# OBSERVATION HELPERS
# ============================================================


def relative_features(source_x, source_z, target_x, target_z):
    dx = target_x - source_x
    dz = target_z - source_z
    dist = jnp.sqrt(dx * dx + dz * dz + 1e-8)
    dist_norm = jnp.clip(dist / FIELD_DIAG, 0.0, 1.0)
    sin_theta = dz / dist
    cos_theta = dx / dist
    return jnp.stack([dist_norm, sin_theta, cos_theta], axis=-1)


def nearest_enemy_indices(state):
    x, z = state["x"], state["z"]
    alive = state["alive"]

    dx = x[None, :] - x[:, None]
    dz = z[None, :] - z[:, None]
    dist2 = dx * dx + dz * dz

    valid = (
        (teams[None, :] != teams[:, None])
        & (alive[None, :] > 0)
    )
    return jnp.argmin(jnp.where(valid, dist2, 1e9), axis=1)


def nearest_enemy_features(local_x, local_z, nearest_idx):
    nearest_x = local_x[nearest_idx]
    nearest_z = local_z[nearest_idx]
    return relative_features(local_x, local_z, nearest_x, nearest_z)


def commander_wall_features(cmd_x, cmd_z, terrain_local):
    # [NW, N, NE, W, E, SW, S, SE] in the local/mirrored frame.
    dc = jnp.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=jnp.int32)
    dr = jnp.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=jnp.int32)

    def one(cx, cz):
        col = jnp.floor(cx + HALF_FIELD).astype(jnp.int32)
        row = jnp.floor(cz + HALF_FIELD).astype(jnp.int32)
        cols = col + dc
        rows = row + dr
        inside = (
            (cols >= 0) & (cols < TERRAIN_RES)
            & (rows >= 0) & (rows < TERRAIN_RES)
        )
        safe_cols = jnp.clip(cols, 0, TERRAIN_RES - 1)
        safe_rows = jnp.clip(rows, 0, TERRAIN_RES - 1)
        idx = safe_rows * TERRAIN_RES + safe_cols
        vals = terrain_local[idx]
        # Outside the battlefield is treated as blocked for local navigation.
        return jnp.where(inside, vals, 1.0)

    return jax.vmap(one)(cmd_x, cmd_z)


def make_observation(state, perspective_team, nearest_idx=None):
    x, z = state["x"], state["z"]
    vx, vz = state["vx"], state["vz"]
    hp, alive = state["hp"], state["alive"]

    blue = (perspective_team == 1.0)

    local_x = jnp.where(blue, -x, x)
    local_z = z
    local_vx = jnp.where(blue, -vx, vx)
    local_vz = vz

    own = (teams == perspective_team).astype(jnp.float32)
    unit_type = commander_mask.astype(jnp.float32)

    base = jnp.stack([
        local_x / HALF_FIELD,
        local_z / HALF_FIELD,
        local_vx,
        local_vz,
        hp,
        own,
        unit_type,
        alive,
    ], axis=-1)

    if nearest_idx is None:
        nearest_idx = nearest_enemy_indices(state)
    nearest = nearest_enemy_features(local_x, local_z, nearest_idx)

    red_or_blue_cmd = jnp.where(
        perspective_team == 0.0,
        jnp.array(RED_COMMANDER_INDEX, dtype=jnp.int32),
        jnp.array(BLUE_COMMANDER_INDEX, dtype=jnp.int32),
    )
    enemy_cmd = jnp.where(
        perspective_team == 0.0,
        jnp.array(BLUE_COMMANDER_INDEX, dtype=jnp.int32),
        jnp.array(RED_COMMANDER_INDEX, dtype=jnp.int32),
    )

    own_cmd_x = local_x[red_or_blue_cmd]
    own_cmd_z = local_z[red_or_blue_cmd]
    enemy_cmd_x = local_x[enemy_cmd]
    enemy_cmd_z = local_z[enemy_cmd]

    soldier_idx = jnp.arange(RED_SOLDIER_START, BLUE_SOLDIER_END)
    soldier_x = local_x[soldier_idx]
    soldier_z = local_z[soldier_idx]

    own_cmd_features = relative_features(
        soldier_x, soldier_z, own_cmd_x, own_cmd_z
    )
    enemy_cmd_features = relative_features(
        soldier_x, soldier_z, enemy_cmd_x, enemy_cmd_z
    )

    soldier_base = base[soldier_idx]
    soldier_nearest = nearest[soldier_idx]
    soldier_features = jnp.concatenate([
        soldier_base,
        soldier_nearest,
        own_cmd_features,
        enemy_cmd_features,
    ], axis=-1)

    cmd_idx = jnp.array([RED_COMMANDER_INDEX, BLUE_COMMANDER_INDEX], dtype=jnp.int32)
    cmd_base = base[cmd_idx]
    cmd_nearest = nearest[cmd_idx]

    terrain_2d = terrain.reshape(TERRAIN_RES, TERRAIN_RES)
    terrain_local_2d = jnp.where(blue, terrain_2d[:, ::-1], terrain_2d)
    terrain_local = terrain_local_2d.reshape(-1)

    cmd_wall_features = commander_wall_features(
        local_x[cmd_idx], local_z[cmd_idx], terrain_local
    )

    commander_features = jnp.concatenate([
        cmd_base,
        cmd_nearest,
        cmd_wall_features,
    ], axis=-1)

    return jnp.concatenate([
        terrain_local,
        soldier_features.reshape(-1),
        commander_features.reshape(-1),
    ])


def make_observation_batch(state, perspective_team):
    nearest_idx = jax.vmap(nearest_enemy_indices)(state)
    return jax.vmap(
        make_observation, in_axes=(0, None, 0)
    )(state, perspective_team, nearest_idx)


def make_observation_pair_batch(state):
    """Build Red and Blue observations from one shared nearest-enemy calculation."""
    nearest_idx = jax.vmap(nearest_enemy_indices)(state)
    red_obs = jax.vmap(
        make_observation, in_axes=(0, None, 0)
    )(state, 0.0, nearest_idx)
    blue_obs = jax.vmap(
        make_observation, in_axes=(0, None, 0)
    )(state, 1.0, nearest_idx)
    return red_obs, blue_obs


def local_to_world_action(action, perspective_team):
    blue = (perspective_team == 1.0)
    soldier_action = action[..., :SOLDIER_ACTION_SIZE]
    ldx = soldier_action[..., 0::3]
    ldz = soldier_action[..., 1::3]
    at = soldier_action[..., 2::3]
    cmd_dx = action[..., SOLDIER_ACTION_SIZE]
    cmd_dz = action[..., SOLDIER_ACTION_SIZE + 1]

    wdx = jnp.where(blue, -ldx, ldx)
    wcmd_dx = jnp.where(blue, -cmd_dx, cmd_dx)

    out = jnp.zeros_like(action)
    out = out.at[..., 0:SOLDIER_ACTION_SIZE:3].set(wdx)
    out = out.at[..., 1:SOLDIER_ACTION_SIZE:3].set(ldz)
    out = out.at[..., 2:SOLDIER_ACTION_SIZE:3].set(at)
    out = out.at[..., SOLDIER_ACTION_SIZE].set(wcmd_dx)
    out = out.at[..., SOLDIER_ACTION_SIZE + 1].set(cmd_dz)
    return out

print()
print("Raw observation size :", RAW_OBS_SIZE)
print("Global network input :", GLOBAL_INPUT_SIZE)
print("Action size           :", ACTION_SIZE)
print("Simulation steps      :", MAX_STEPS)
print("Total units           :", N_UNITS)

# ============================================================
# PART 2 : POLICY NETWORK
# Hierarchical encoders + 2-layer team-wise self-attention
# + shared global encoder + split Actor/Critic branches
# ============================================================

HIDDEN1 = 512
ACTOR_HIDDEN = 512
CRITIC_HIDDEN = 512

GAMMA = 0.999
GAE_LAMBDA = 0.97
CLIP_EPS = 0.20
VALUE_COEF = 0.5
LEARNING_RATE = 5e-5

ENTROPY_START = 0.005
ENTROPY_END = 0.0005

LOGSTD_INIT = -1.0
LOGSTD_MIN = -3.0
LOGSTD_MAX = -0.3
LOGSTD_TARGET = -1.0
LOGSTD_REG_COEF = 0.01

PPO_EPOCHS = 4
MINIBATCHES = 8

# Stop PPO epoch/update early when the full-action KL grows too large.
# This KL is summed over the full RTS action, so it is intentionally
# looser than typical single-action PPO thresholds.
TARGET_KL = 0.30

PPO_ROLLOUT_STEPS = 512
ROLLOUT_STEPS = PPO_ROLLOUT_STEPS
BATCH_SIZE = ROLLOUT_STEPS * N_ENVS * 2
MINIBATCH_SIZE = BATCH_SIZE // MINIBATCHES

PPO_UPDATES_PER_GENERATION = 20
N_GENERATIONS = 10000
EVAL_GAMES_PER_SIDE = 32
EVAL_GAMES = EVAL_GAMES_PER_SIDE * 2

# Stagger generation starts so PPO 1 is not forced to begin with every
# environment at t=0. Warm-up data is never used as PPO training data.
WARMUP_MAX_STEPS = MAX_STEPS

EVAL_Z_OFFSETS = jnp.array(
    [-0.72, -0.48, -0.24, 0.00, 0.24, 0.48, 0.72, 0.00],
    dtype=jnp.float32,
)

ANGLE_MEAN_IDX = jnp.arange(0, SOLDIER_ACTION_SIZE, 3)
ANGLE_LOGSTD_IDX = jnp.arange(1, SOLDIER_ACTION_SIZE, 3)
ATTACK_IDX = jnp.arange(2, SOLDIER_ACTION_SIZE, 3)
COMMANDER_MEAN_IDX = SOLDIER_ACTION_SIZE
COMMANDER_LOGSTD_IDX = SOLDIER_ACTION_SIZE + 1


def glorot_uniform(key, shape):
    fan_in, fan_out = shape[0], shape[1]
    limit = jnp.sqrt(6.0 / float(fan_in + fan_out))
    return random.uniform(key, shape, minval=-limit, maxval=limit)


def layer_norm(x, gamma, beta, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps) * gamma + beta


def split_heads(x):
    b, n, _ = x.shape
    return x.reshape(b, n, ATTENTION_HEADS, ATTENTION_HEAD_DIM)


def merge_heads(x):
    b, n, _, _ = x.shape
    return x.reshape(b, n, LOCAL_EMBED_SIZE)


def self_attention_block(x, params, prefix):
    q = split_heads(x @ params[f"{prefix}_Wq"] + params[f"{prefix}_bq"])
    k = split_heads(x @ params[f"{prefix}_Wk"] + params[f"{prefix}_bk"])
    v = split_heads(x @ params[f"{prefix}_Wv"] + params[f"{prefix}_bv"])

    scores = jnp.einsum("bnhd,bmhd->bhnm", q, k)
    scores = scores / jnp.sqrt(float(ATTENTION_HEAD_DIM))
    weights = jax.nn.softmax(scores, axis=-1)
    attended = jnp.einsum("bhnm,bmhd->bnhd", weights, v)
    attended = merge_heads(attended)
    attended = attended @ params[f"{prefix}_Wo"] + params[f"{prefix}_bo"]

    x = layer_norm(
        x + attended,
        params[f"{prefix}_ln_gamma"],
        params[f"{prefix}_ln_beta"],
    )
    return x


def init_attention_block(key, prefix):
    keys = random.split(key, 8)
    p = {
        f"{prefix}_Wq": glorot_uniform(keys[0], (LOCAL_EMBED_SIZE, LOCAL_EMBED_SIZE)),
        f"{prefix}_bq": jnp.zeros((LOCAL_EMBED_SIZE,)),
        f"{prefix}_Wk": glorot_uniform(keys[1], (LOCAL_EMBED_SIZE, LOCAL_EMBED_SIZE)),
        f"{prefix}_bk": jnp.zeros((LOCAL_EMBED_SIZE,)),
        f"{prefix}_Wv": glorot_uniform(keys[2], (LOCAL_EMBED_SIZE, LOCAL_EMBED_SIZE)),
        f"{prefix}_bv": jnp.zeros((LOCAL_EMBED_SIZE,)),
        f"{prefix}_Wo": glorot_uniform(keys[3], (LOCAL_EMBED_SIZE, LOCAL_EMBED_SIZE)),
        f"{prefix}_bo": jnp.zeros((LOCAL_EMBED_SIZE,)),
        f"{prefix}_ln_gamma": jnp.ones((LOCAL_EMBED_SIZE,)),
        f"{prefix}_ln_beta": jnp.zeros((LOCAL_EMBED_SIZE,)),
    }
    return p


def init_policy(key):
    keys = random.split(key, 32)
    p = {
        "Ws1": glorot_uniform(keys[0], (SOLDIER_FEATURES, LOCAL_HIDDEN_SIZE)),
        "bs1": jnp.zeros((LOCAL_HIDDEN_SIZE,)),
        "Ws2": glorot_uniform(keys[1], (LOCAL_HIDDEN_SIZE, LOCAL_EMBED_SIZE)),
        "bs2": jnp.zeros((LOCAL_EMBED_SIZE,)),
        "Wc": glorot_uniform(keys[2], (COMMANDER_FEATURES, LOCAL_EMBED_SIZE)),
        "bc": jnp.zeros((LOCAL_EMBED_SIZE,)),
        "Wg1": glorot_uniform(keys[3], (GLOBAL_INPUT_SIZE, HIDDEN1)),
        "bg1": jnp.zeros((HIDDEN1,)),
        "W_actor_s": glorot_uniform(keys[4], (HIDDEN1, ACTOR_HIDDEN)),
        "b_actor_s": jnp.zeros((ACTOR_HIDDEN,)),
        "W_actor_c": glorot_uniform(keys[5], (HIDDEN1, ACTOR_HIDDEN)),
        "b_actor_c": jnp.zeros((ACTOR_HIDDEN,)),
        "W_critic": glorot_uniform(keys[6], (HIDDEN1, CRITIC_HIDDEN)),
        "b_critic": jnp.zeros((CRITIC_HIDDEN,)),
        "Wa_s": random.normal(keys[7], (ACTOR_HIDDEN, SOLDIER_ACTION_SIZE)) * 0.01,
        "ba_s": jnp.zeros((SOLDIER_ACTION_SIZE,)),
        "Wa_c": random.normal(keys[8], (ACTOR_HIDDEN, COMMANDER_ACTION_SIZE)) * 0.01,
        "ba_c": jnp.zeros((COMMANDER_ACTION_SIZE,)),
        "Wv": random.normal(keys[9], (CRITIC_HIDDEN, 1)) * 0.01,
        "bv": jnp.zeros((1,)),
    }
    p.update(init_attention_block(keys[10], "attn1"))
    p.update(init_attention_block(keys[11], "attn2"))

    # Soldier layout: [angle_mean, angle_logstd, attack] x 100.
    p["ba_s"] = p["ba_s"].at[ANGLE_LOGSTD_IDX].set(LOGSTD_INIT)
    # Commander layout: [angle_mean, angle_logstd].
    p["ba_c"] = p["ba_c"].at[COMMANDER_LOGSTD_IDX - SOLDIER_ACTION_SIZE].set(LOGSTD_INIT)
    return p


def policy_forward(params, obs, detach_encoders=False):
    terrain_part = obs[:, :TERRAIN_SIZE]
    soldier_start = TERRAIN_SIZE
    soldier_end = soldier_start + N_SOLDIERS_TOTAL * SOLDIER_FEATURES
    soldier_part = obs[:, soldier_start:soldier_end].reshape(
        obs.shape[0], N_SOLDIERS_TOTAL, SOLDIER_FEATURES
    )
    commander_part = obs[:, soldier_end:].reshape(
        obs.shape[0], N_COMMANDERS, COMMANDER_FEATURES
    )

    soldier_h = jnp.tanh(soldier_part @ params["Ws1"] + params["bs1"])
    soldier_emb = jnp.tanh(soldier_h @ params["Ws2"] + params["bs2"])
    commander_emb = jnp.tanh(commander_part @ params["Wc"] + params["bc"])

    red_soldier_emb = soldier_emb[:, :N_SOLDIERS_PER_TEAM, :]
    blue_soldier_emb = soldier_emb[:, N_SOLDIERS_PER_TEAM:, :]

    red_soldier_emb = self_attention_block(red_soldier_emb, params, "attn1")
    blue_soldier_emb = self_attention_block(blue_soldier_emb, params, "attn1")
    red_soldier_emb = self_attention_block(red_soldier_emb, params, "attn2")
    blue_soldier_emb = self_attention_block(blue_soldier_emb, params, "attn2")
    soldier_emb = jnp.concatenate([red_soldier_emb, blue_soldier_emb], axis=1)

    if detach_encoders:
        soldier_global = lax.stop_gradient(soldier_emb)
        commander_global = lax.stop_gradient(commander_emb)
    else:
        soldier_global = soldier_emb
        commander_global = commander_emb

    global_input = jnp.concatenate([
        terrain_part,
        soldier_global.reshape(obs.shape[0], -1),
        commander_global.reshape(obs.shape[0], -1),
    ], axis=-1)

    h_shared = jnp.tanh(global_input @ params["Wg1"] + params["bg1"])

    soldier_h_actor = jnp.tanh(
        h_shared @ params["W_actor_s"] + params["b_actor_s"]
    )
    commander_h_actor = jnp.tanh(
        h_shared @ params["W_actor_c"] + params["b_actor_c"]
    )
    critic_h = jnp.tanh(
        h_shared @ params["W_critic"] + params["b_critic"]
    )

    soldier_output = soldier_h_actor @ params["Wa_s"] + params["ba_s"]
    commander_output = commander_h_actor @ params["Wa_c"] + params["ba_c"]
    action_output = jnp.concatenate([soldier_output, commander_output], axis=-1)
    value = (critic_h @ params["Wv"] + params["bv"])[..., 0]
    return action_output, value


policy_forward_jit = jax.jit(policy_forward)


def wrap_angle(a):
    return (a + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def action_logprob_components(action_output, local_action):
    soldier_output = action_output[:, :SOLDIER_ACTION_SIZE]
    commander_output = action_output[:, SOLDIER_ACTION_SIZE:]

    angle_mean = soldier_output[:, ANGLE_MEAN_IDX]
    logstd = jnp.clip(soldier_output[:, ANGLE_LOGSTD_IDX], LOGSTD_MIN, LOGSTD_MAX)
    std = jnp.exp(logstd)

    soldier_action = local_action[:, :SOLDIER_ACTION_SIZE]
    dx = soldier_action[:, 0::3]
    dz = soldier_action[:, 1::3]
    a = jnp.arctan2(dz, dx)
    diff = wrap_angle(a - angle_mean)

    lp_angle = (
        -0.5 * (diff / std) ** 2
        - logstd
        - 0.5 * jnp.log(2.0 * jnp.pi)
    )

    logits = soldier_output[:, ATTACK_IDX]
    at = soldier_action[:, 2::3]
    lp_attack = (
        at * (-jnp.logaddexp(0.0, -logits))
        + (1.0 - at) * (-jnp.logaddexp(0.0, logits))
    )
    soldier_lp = lp_angle + lp_attack

    cmd_mean = commander_output[:, 0]
    cmd_logstd = jnp.clip(commander_output[:, 1], LOGSTD_MIN, LOGSTD_MAX)
    cmd_std = jnp.exp(cmd_logstd)
    cmd_dx = local_action[:, SOLDIER_ACTION_SIZE]
    cmd_dz = local_action[:, SOLDIER_ACTION_SIZE + 1]
    cmd_angle = jnp.arctan2(cmd_dz, cmd_dx)
    cmd_diff = wrap_angle(cmd_angle - cmd_mean)
    commander_lp = (
        -0.5 * (cmd_diff / cmd_std) ** 2
        - cmd_logstd
        - 0.5 * jnp.log(2.0 * jnp.pi)
    )

    team_lp = jnp.sum(soldier_lp, axis=1) + commander_lp
    return soldier_lp, commander_lp, team_lp


def action_logprob(action_output, local_action):
    _, _, team_lp = action_logprob_components(action_output, local_action)
    return team_lp


def policy_entropy(action_output):
    soldier_output = action_output[:, :SOLDIER_ACTION_SIZE]
    commander_output = action_output[:, SOLDIER_ACTION_SIZE:]

    logstd = jnp.clip(soldier_output[:, ANGLE_LOGSTD_IDX], LOGSTD_MIN, LOGSTD_MAX)
    e_angle = jnp.sum(
        logstd + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e), axis=1
    )

    p = jax.nn.sigmoid(soldier_output[:, ATTACK_IDX])
    e_attack = -(
        p * jnp.log(p + 1e-8)
        + (1.0 - p) * jnp.log(1.0 - p + 1e-8)
    )
    e_attack = jnp.sum(e_attack, axis=1)

    cmd_logstd = jnp.clip(commander_output[:, 1], LOGSTD_MIN, LOGSTD_MAX)
    e_cmd = cmd_logstd + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e)
    return e_angle + e_attack + e_cmd


def sample_action(params, obs, key):
    action_output, value = policy_forward(params, obs)
    B = obs.shape[0]
    k_noise_soldier, k_attack, k_noise_cmd = random.split(key, 3)

    soldier_output = action_output[:, :SOLDIER_ACTION_SIZE]
    commander_output = action_output[:, SOLDIER_ACTION_SIZE:]

    mean = soldier_output[:, ANGLE_MEAN_IDX]
    logstd = jnp.clip(soldier_output[:, ANGLE_LOGSTD_IDX], LOGSTD_MIN, LOGSTD_MAX)
    std = jnp.exp(logstd)

    angle = mean + std * random.normal(
        k_noise_soldier, (B, N_SOLDIERS_PER_TEAM)
    )
    dx, dz = jnp.cos(angle), jnp.sin(angle)

    prob = jax.nn.sigmoid(soldier_output[:, ATTACK_IDX])
    at = random.bernoulli(k_attack, prob).astype(jnp.float32)

    cmd_mean = commander_output[:, 0]
    cmd_logstd = jnp.clip(commander_output[:, 1], LOGSTD_MIN, LOGSTD_MAX)
    cmd_std = jnp.exp(cmd_logstd)
    cmd_angle = cmd_mean + cmd_std * random.normal(k_noise_cmd, (B,))
    cmd_dx, cmd_dz = jnp.cos(cmd_angle), jnp.sin(cmd_angle)

    la = jnp.zeros((B, ACTION_SIZE), dtype=jnp.float32)
    la = la.at[:, 0:SOLDIER_ACTION_SIZE:3].set(dx)
    la = la.at[:, 1:SOLDIER_ACTION_SIZE:3].set(dz)
    la = la.at[:, 2:SOLDIER_ACTION_SIZE:3].set(at)
    la = la.at[:, SOLDIER_ACTION_SIZE].set(cmd_dx)
    la = la.at[:, SOLDIER_ACTION_SIZE + 1].set(cmd_dz)

    soldier_lp, commander_lp, team_lp = action_logprob_components(action_output, la)
    return la, soldier_lp, commander_lp, team_lp, value


def deterministic_local_action(params, obs):
    action_output, value = policy_forward(params, obs)
    soldier_output = action_output[:, :SOLDIER_ACTION_SIZE]
    commander_output = action_output[:, SOLDIER_ACTION_SIZE:]

    mean = soldier_output[:, ANGLE_MEAN_IDX]
    dx, dz = jnp.cos(mean), jnp.sin(mean)
    at = (jax.nn.sigmoid(soldier_output[:, ATTACK_IDX]) >= 0.5).astype(jnp.float32)

    cmd_mean = commander_output[:, 0]
    cmd_dx = jnp.cos(cmd_mean)
    cmd_dz = jnp.sin(cmd_mean)

    la = jnp.zeros((obs.shape[0], ACTION_SIZE), dtype=jnp.float32)
    la = la.at[:, 0:SOLDIER_ACTION_SIZE:3].set(dx)
    la = la.at[:, 1:SOLDIER_ACTION_SIZE:3].set(dz)
    la = la.at[:, 2:SOLDIER_ACTION_SIZE:3].set(at)
    la = la.at[:, SOLDIER_ACTION_SIZE].set(cmd_dx)
    la = la.at[:, SOLDIER_ACTION_SIZE + 1].set(cmd_dz)
    return la, value


def deterministic_world_action_batch(params, state, team):
    obs = make_observation_batch(state, team)
    la, _ = deterministic_local_action(params, obs)
    return local_to_world_action(la, team)


def deterministic_world_action_single(params, state, team):
    obs = make_observation(state, team)[None, :]
    la, _ = deterministic_local_action(params, obs)
    return local_to_world_action(la, team)[0]

# ============================================================
# PART 3 : OPTIMIZER / PPO
# ============================================================

optimizer = optax.chain(
    optax.clip_by_global_norm(0.5),
    optax.adam(LEARNING_RATE),
)


def normalize_advantages(a):
    return (a - jnp.mean(a)) / (jnp.std(a) + 1e-8)


def compute_gae(rewards, values, dones, last_value):
    def rev(carry, xs):
        gae = carry
        r_t, v_t, d_t, nv = xs
        m = 1.0 - d_t.astype(jnp.float32)
        delta = r_t + GAMMA * m * nv - v_t
        gae = delta + GAMMA * GAE_LAMBDA * m * gae
        return gae, gae

    next_values = jnp.concatenate([values[1:], last_value[None, :]], axis=0)
    _, adv = lax.scan(
        rev,
        jnp.zeros_like(last_value),
        (rewards[::-1], values[::-1], dones[::-1], next_values[::-1]),
    )
    adv = adv[::-1]
    return adv, adv + values


def compute_discounted_returns(rewards, dones):
    """Monte-Carlo reward-to-go for local skill rewards.

    ``dones`` is environment-level with shape ``(batch,)`` while local
    Soldier rewards can be ``(batch, 100)``. Expand the done mask to match
    the reward tensor so each Soldier/Commander stream resets at episode end.
    """
    def rev(carry, xs):
        ret = carry
        r_t, d_t = xs
        m = 1.0 - d_t.astype(jnp.float32)
        while m.ndim < r_t.ndim:
            m = m[..., None]
        ret = r_t + GAMMA * m * ret
        return ret, ret

    _, returns = lax.scan(
        rev,
        jnp.zeros_like(rewards[0]),
        (rewards[::-1], dones[::-1]),
    )
    return returns[::-1]


# Gradient ownership for the three reward systems.
SOLDIER_LOCAL_KEYS = (
    "Ws1", "bs1", "Ws2", "bs2",
    "attn1_Wq", "attn1_bq", "attn1_Wk", "attn1_bk",
    "attn1_Wv", "attn1_bv", "attn1_Wo", "attn1_bo",
    "attn1_ln_gamma", "attn1_ln_beta",
    "attn2_Wq", "attn2_bq", "attn2_Wk", "attn2_bk",
    "attn2_Wv", "attn2_bv", "attn2_Wo", "attn2_bo",
    "attn2_ln_gamma", "attn2_ln_beta",
    "W_actor_s", "b_actor_s", "Wa_s", "ba_s",
)
COMMANDER_LOCAL_KEYS = (
    "Wc", "bc",
    "W_actor_c", "b_actor_c", "Wa_c", "ba_c",
)
TEAM_KEYS = (
    "Wg1", "bg1",
    "W_actor_s", "b_actor_s", "Wa_s", "ba_s",
    "W_actor_c", "b_actor_c", "Wa_c", "ba_c",
    "W_critic", "b_critic", "Wv", "bv",
)


def masked_tree(tree, allowed_keys):
    allowed = set(allowed_keys)
    return {
        k: (v if k in allowed else jnp.zeros_like(v))
        for k, v in tree.items()
    }


def ppo_loss_parts(
    params,
    obs,
    local_actions,
    old_soldier_log_prob,
    old_commander_log_prob,
    old_team_log_prob,
    soldier_advantages,
    commander_advantages,
    team_advantages,
    returns,
    ent_coef,
):
    """Build the three losses without mixing reward ownership."""
    # For the team loss, detach both local encoders. This makes the global
    # network/actor/critic learn from WIN/LOSS without sending that signal back
    # into Soldier/Commander encoders.
    team_action_output, team_values = policy_forward(
        params, obs, detach_encoders=True
    )
    team_soldier_lp, team_commander_lp, team_lp = action_logprob_components(
        team_action_output, local_actions
    )

    # Local skill losses use the full graph so their own encoder receives only
    # their own reward signal after gradient masking.
    local_action_output, _ = policy_forward(
        params, obs, detach_encoders=False
    )
    soldier_lp, commander_lp, _ = action_logprob_components(
        local_action_output, local_actions
    )

    # Soldier-specific PPO objective: one ratio per soldier.
    soldier_ratio = jnp.exp(soldier_lp - old_soldier_log_prob)
    soldier_unclipped = soldier_ratio * soldier_advantages
    soldier_clipped = (
        jnp.clip(soldier_ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS)
        * soldier_advantages
    )
    soldier_policy_loss = -jnp.mean(
        jnp.minimum(soldier_unclipped, soldier_clipped)
    )

    # Commander-specific PPO objective.
    commander_ratio = jnp.exp(commander_lp - old_commander_log_prob)
    commander_unclipped = commander_ratio * commander_advantages
    commander_clipped = (
        jnp.clip(commander_ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS)
        * commander_advantages
    )
    commander_policy_loss = -jnp.mean(
        jnp.minimum(commander_unclipped, commander_clipped)
    )

    # Team objective: the full action is judged by WIN/LOSS.
    team_ratio = jnp.exp(team_lp - old_team_log_prob)
    team_unclipped = team_ratio * team_advantages
    team_clipped = (
        jnp.clip(team_ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS)
        * team_advantages
    )
    team_policy_loss = -jnp.mean(
        jnp.minimum(team_unclipped, team_clipped)
    )

    entropy = jnp.mean(policy_entropy(team_action_output))
    soldier_logstd = team_action_output[:, :SOLDIER_ACTION_SIZE][:, ANGLE_LOGSTD_IDX]
    commander_logstd = team_action_output[:, SOLDIER_ACTION_SIZE + 1]
    soldier_logstd_reg = jnp.mean((soldier_logstd - LOGSTD_TARGET) ** 2)
    commander_logstd_reg = jnp.mean((commander_logstd - LOGSTD_TARGET) ** 2)
    logstd_reg = 0.5 * soldier_logstd_reg + 0.5 * commander_logstd_reg

    value_loss = 0.5 * jnp.mean((returns - team_values) ** 2)
    team_total_loss = (
        team_policy_loss
        + VALUE_COEF * value_loss
        - ent_coef * entropy
        + LOGSTD_REG_COEF * logstd_reg
    )

    metrics = {
        "soldier_policy_loss": soldier_policy_loss,
        "commander_policy_loss": commander_policy_loss,
        "team_policy_loss": team_policy_loss,
        "policy_loss": team_policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "entropy_per_soldier": entropy / float(N_SOLDIERS_PER_TEAM),
        "logstd_mean": jnp.mean(jnp.clip(soldier_logstd, LOGSTD_MIN, LOGSTD_MAX)),
        "commander_logstd_mean": jnp.mean(
            jnp.clip(commander_logstd, LOGSTD_MIN, LOGSTD_MAX)
        ),
        "logstd_reg": logstd_reg,
        "approx_kl": jnp.mean(old_team_log_prob - team_lp),
        "clip_fraction": jnp.mean(
            (jnp.abs(team_ratio - 1.0) > CLIP_EPS).astype(jnp.float32)
        ),
        "team_total_loss": team_total_loss,
    }
    return metrics


@jax.jit
def ppo_update_minibatch(
    params,
    opt_state,
    obs,
    local_actions,
    old_soldier_log_prob,
    old_commander_log_prob,
    old_team_log_prob,
    soldier_advantages,
    commander_advantages,
    team_advantages,
    returns,
    ent_coef,
):
    def soldier_loss_fn(p):
        m = ppo_loss_parts(
            p, obs, local_actions,
            old_soldier_log_prob,
            old_commander_log_prob,
            old_team_log_prob,
            soldier_advantages,
            commander_advantages,
            team_advantages,
            returns,
            ent_coef,
        )
        return m["soldier_policy_loss"]

    def commander_loss_fn(p):
        m = ppo_loss_parts(
            p, obs, local_actions,
            old_soldier_log_prob,
            old_commander_log_prob,
            old_team_log_prob,
            soldier_advantages,
            commander_advantages,
            team_advantages,
            returns,
            ent_coef,
        )
        return m["commander_policy_loss"]

    def team_loss_fn(p):
        m = ppo_loss_parts(
            p, obs, local_actions,
            old_soldier_log_prob,
            old_commander_log_prob,
            old_team_log_prob,
            soldier_advantages,
            commander_advantages,
            team_advantages,
            returns,
            ent_coef,
        )
        return m["team_total_loss"], m

    soldier_value, soldier_grads = jax.value_and_grad(
        soldier_loss_fn
    )(params)
    commander_value, commander_grads = jax.value_and_grad(
        commander_loss_fn
    )(params)
    (team_value, metrics), team_grads = jax.value_and_grad(
        team_loss_fn, has_aux=True
    )(params)
    del soldier_value, commander_value, team_value

    soldier_grads = masked_tree(soldier_grads, SOLDIER_LOCAL_KEYS)
    commander_grads = masked_tree(commander_grads, COMMANDER_LOCAL_KEYS)
    team_grads = masked_tree(team_grads, TEAM_KEYS)

    def grad_l2(tree):
        leaves = jax.tree_util.tree_leaves(tree)
        return jnp.sqrt(
            sum(jnp.sum(x * x) for x in leaves) + jnp.float32(1e-12)
        )

    metrics = dict(metrics)
    metrics["grad_norm_soldier_local"] = grad_l2(soldier_grads)
    metrics["grad_norm_commander_local"] = grad_l2(commander_grads)
    metrics["grad_norm_team"] = grad_l2(team_grads)

    grads = {
        k: soldier_grads[k] + commander_grads[k] + team_grads[k]
        for k in params
    }

    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, metrics

# ============================================================
# ROLLOUT
# ============================================================


def reset_finished_envs(state, done, key):
    fresh = reset_parallel(random.split(key, N_ENVS))

    def merge(old, new):
        mask = done if old.ndim == 1 else done.reshape(
            (N_ENVS,) + (1,) * (old.ndim - 1)
        )
        return jnp.where(mask, new, old)

    return jax.tree_util.tree_map(merge, state, fresh)


@jax.jit
def warmup_env_state(params, state, key):
    """Advance each environment by a random number of steps before PPO 1.

    The warm-up trajectory is discarded. Its only purpose is to stagger the
    episode phase across environments so the first PPO rollout contains a
    realistic mixture of early/mid/late-episode states.
    """
    warmup_steps = random.randint(
        key,
        (N_ENVS,),
        minval=0,
        maxval=WARMUP_MAX_STEPS + 1,
        dtype=jnp.int32,
    )

    def body(carry, step_idx):
        st, k, completed = carry
        k, kr, kb, kreset = random.split(k, 4)
        red_obs, blue_obs = make_observation_pair_batch(st)
        red_la, _, _, _, _ = sample_action(params, red_obs, kr)
        blue_la, _, _, _, _ = sample_action(params, blue_obs, kb)
        red_wa = local_to_world_action(red_la, 0.0)
        blue_wa = local_to_world_action(blue_la, 1.0)

        nxt, _, _, _, _, _, _, done = jax.vmap(
            step_one, in_axes=(0, 0, 0)
        )(st, red_wa, blue_wa)

        active = step_idx < warmup_steps
        effective_done = done & active
        fresh = reset_parallel(random.split(kreset, N_ENVS))

        def mask_for(old, mask):
            return mask.reshape((N_ENVS,) + (1,) * (old.ndim - 1)) if old.ndim > 1 else mask

        merged_nxt = jax.tree_util.tree_map(
            lambda old, new: jnp.where(mask_for(old, active), new, old),
            st,
            nxt,
        )
        merged = jax.tree_util.tree_map(
            lambda old, new: jnp.where(mask_for(old, effective_done), new, old),
            merged_nxt,
            fresh,
        )
        completed = completed + jnp.sum(effective_done.astype(jnp.int32))
        return (merged, k, completed), None

    (state, key, completed), _ = lax.scan(
        body,
        (state, key, jnp.array(0, dtype=jnp.int32)),
        jnp.arange(WARMUP_MAX_STEPS, dtype=jnp.int32),
    )
    return state, key, completed


@functools.partial(jax.jit, static_argnums=())
def collect_rollout(params, state, key):
    def body(carry, _):
        st, k = carry

        red_obs, blue_obs = make_observation_pair_batch(st)

        k, kr, kb, kreset = random.split(k, 4)

        (
            red_la,
            red_soldier_lp,
            red_commander_lp,
            red_team_lp,
            red_v,
        ) = sample_action(params, red_obs, kr)
        (
            blue_la,
            blue_soldier_lp,
            blue_commander_lp,
            blue_team_lp,
            blue_v,
        ) = sample_action(params, blue_obs, kb)

        red_wa = local_to_world_action(red_la, 0.0)
        blue_wa = local_to_world_action(blue_la, 1.0)

        (
            nxt,
            rr,
            br,
            red_sr,
            blue_sr,
            red_cr,
            blue_cr,
            done,
        ) = jax.vmap(step_one, in_axes=(0, 0, 0))(
            st, red_wa, blue_wa
        )

        red_cmd = nxt["alive"][:, RED_COMMANDER_INDEX] > 0
        blue_cmd = nxt["alive"][:, BLUE_COMMANDER_INDEX] > 0

        reason = jnp.where(
            done & red_cmd & (~blue_cmd),
            1,
            jnp.where(
                done & blue_cmd & (~red_cmd),
                2,
                jnp.where(
                    done & (~red_cmd) & (~blue_cmd),
                    4,
                    jnp.where(done, 3, 0),
                ),
            ),
        )

        new_state = reset_finished_envs(nxt, done, kreset)

        out = (
            red_obs,
            blue_obs,
            red_la,
            blue_la,
            red_soldier_lp,
            blue_soldier_lp,
            red_commander_lp,
            blue_commander_lp,
            red_team_lp,
            blue_team_lp,
            red_v,
            blue_v,
            rr,
            br,
            red_sr,
            blue_sr,
            red_cr,
            blue_cr,
            done,
            reason,
        )
        return (new_state, k), out

    (final_state, final_key), traj = lax.scan(
        body,
        (state, key),
        None,
        length=ROLLOUT_STEPS,
    )

    (
        red_obs,
        blue_obs,
        red_la,
        blue_la,
        red_soldier_lp,
        blue_soldier_lp,
        red_commander_lp,
        blue_commander_lp,
        red_team_lp,
        blue_team_lp,
        red_v,
        blue_v,
        red_team_r,
        blue_team_r,
        red_soldier_r,
        blue_soldier_r,
        red_commander_r,
        blue_commander_r,
        dones,
        reasons,
    ) = traj

    f_red_obs, f_blue_obs = make_observation_pair_batch(final_state)
    _, red_last_v = policy_forward(params, f_red_obs)
    _, blue_last_v = policy_forward(params, f_blue_obs)

    red_team_adv, red_ret = compute_gae(
        red_team_r, red_v, dones, red_last_v
    )
    blue_team_adv, blue_ret = compute_gae(
        blue_team_r, blue_v, dones, blue_last_v
    )

    red_soldier_adv = compute_discounted_returns(red_soldier_r, dones)
    blue_soldier_adv = compute_discounted_returns(blue_soldier_r, dones)
    red_commander_adv = compute_discounted_returns(red_commander_r, dones)
    blue_commander_adv = compute_discounted_returns(blue_commander_r, dones)

    obs = jnp.concatenate([red_obs, blue_obs], axis=1).reshape(-1, OBS_SIZE)
    acts = jnp.concatenate([red_la, blue_la], axis=1).reshape(-1, ACTION_SIZE)

    soldier_old_lp = jnp.concatenate(
        [red_soldier_lp, blue_soldier_lp], axis=1
    ).reshape(-1, N_SOLDIERS_PER_TEAM)
    commander_old_lp = jnp.concatenate(
        [red_commander_lp, blue_commander_lp], axis=1
    ).reshape(-1)
    team_old_lp = jnp.concatenate(
        [red_team_lp, blue_team_lp], axis=1
    ).reshape(-1)

    soldier_adv = jnp.concatenate(
        [red_soldier_adv, blue_soldier_adv], axis=1
    ).reshape(-1, N_SOLDIERS_PER_TEAM)
    commander_adv = jnp.concatenate(
        [red_commander_adv, blue_commander_adv], axis=1
    ).reshape(-1)
    team_adv = jnp.concatenate(
        [red_team_adv, blue_team_adv], axis=1
    ).reshape(-1)
    returns = jnp.concatenate([red_ret, blue_ret], axis=1).reshape(-1)

    soldier_adv = normalize_advantages(soldier_adv)
    commander_adv = normalize_advantages(commander_adv)
    team_adv = normalize_advantages(team_adv)

    stats = {
        "battles": jnp.sum(reasons != 0),
        "red_wins": jnp.sum(reasons == 1),
        "blue_wins": jnp.sum(reasons == 2),
        "timeouts": jnp.sum(reasons == 3),
        "draws": jnp.sum(reasons == 4),
        # These are the exact reward/advantage tensors consumed by each PPO
        # objective. They make reward-to-NN wiring visible in the log.
        "soldier_reward_abs_mean": 0.5 * (
            jnp.mean(jnp.abs(red_soldier_r))
            + jnp.mean(jnp.abs(blue_soldier_r))
        ),
        "commander_reward_abs_mean": 0.5 * (
            jnp.mean(jnp.abs(red_commander_r))
            + jnp.mean(jnp.abs(blue_commander_r))
        ),
        "team_reward_abs_mean": 0.5 * (
            jnp.mean(jnp.abs(red_team_r))
            + jnp.mean(jnp.abs(blue_team_r))
        ),
        "soldier_adv_abs_mean": 0.5 * (
            jnp.mean(jnp.abs(soldier_adv))
            + jnp.mean(jnp.abs(blue_soldier_adv))
        ),
        "commander_adv_abs_mean": 0.5 * (
            jnp.mean(jnp.abs(red_commander_adv))
            + jnp.mean(jnp.abs(blue_commander_adv))
        ),
        "team_adv_abs_mean": 0.5 * (
            jnp.mean(jnp.abs(red_team_adv))
            + jnp.mean(jnp.abs(blue_team_adv))
        ),
    }

    return (
        final_state,
        final_key,
        obs,
        acts,
        soldier_old_lp,
        commander_old_lp,
        team_old_lp,
        soldier_adv,
        commander_adv,
        team_adv,
        returns,
        stats,
    )


def tree_l2_norm(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    total = 0.0
    for leaf in leaves:
        arr = np.asarray(leaf)
        total += float(np.sum(arr.astype(np.float64) ** 2))
    return float(np.sqrt(total))


def tree_delta_l2_norm(before, after):
    before_leaves = jax.tree_util.tree_leaves(before)
    after_leaves = jax.tree_util.tree_leaves(after)
    total = 0.0
    for a, b in zip(before_leaves, after_leaves):
        da = np.asarray(b) - np.asarray(a)
        total += float(np.sum(da.astype(np.float64) ** 2))
    return float(np.sqrt(total))



def run_ppo_update(params, opt_state, state, key, global_update, total_updates):
    t0 = time.time()
    params_before = jax.tree_util.tree_map(lambda x: x.copy(), params)

    (
        state,
        key,
        obs,
        acts,
        old_soldier_lp,
        old_commander_lp,
        old_team_lp,
        soldier_adv,
        commander_adv,
        team_adv,
        returns,
        stats,
    ) = collect_rollout(params, state, key)

    alpha = global_update / max(1, total_updates - 1)
    ent_coef = jnp.array(
        ENTROPY_START + alpha * (ENTROPY_END - ENTROPY_START),
        dtype=jnp.float32,
    )

    n = obs.shape[0]
    usable = (n // MINIBATCH_SIZE) * MINIBATCH_SIZE
    n_mb = usable // MINIBATCH_SIZE

    acc = []
    stopped_early = False
    stop_epoch = 0
    stop_minibatch = 0
    max_kl = float("-inf")

    for epoch in range(1, PPO_EPOCHS + 1):
        key, sk = random.split(key)
        perm = random.permutation(sk, n)[:usable]
        for mb in range(n_mb):
            idx = perm[
                mb * MINIBATCH_SIZE:(mb + 1) * MINIBATCH_SIZE
            ]
            (
                params,
                opt_state,
                m,
            ) = ppo_update_minibatch(
                params,
                opt_state,
                obs[idx],
                acts[idx],
                old_soldier_lp[idx],
                old_commander_lp[idx],
                old_team_lp[idx],
                soldier_adv[idx],
                commander_adv[idx],
                team_adv[idx],
                returns[idx],
                ent_coef,
            )
            acc.append(m)

            mb_kl = float(m["approx_kl"])
            max_kl = max(max_kl, mb_kl)
            if mb_kl > TARGET_KL:
                stopped_early = True
                stop_epoch = epoch
                stop_minibatch = mb + 1
                break

        if stopped_early:
            break

    metrics = {
        k: float(np.mean([float(m[k]) for m in acc]))
        for k in acc[0]
        if k != "team_total_loss"
    }
    metrics["entropy_coef"] = float(ent_coef)
    metrics["max_kl"] = max_kl
    metrics["ppo_epochs_used"] = stop_epoch if stopped_early else PPO_EPOCHS
    metrics["early_stopped"] = 1.0 if stopped_early else 0.0
    metrics["parameter_l2"] = tree_l2_norm(params)
    metrics["parameter_delta_l2"] = tree_delta_l2_norm(params_before, params)
    metrics["soldier_policy_loss"] = float(
        np.mean([float(m["soldier_policy_loss"]) for m in acc])
    )
    metrics["commander_policy_loss"] = float(
        np.mean([float(m["commander_policy_loss"]) for m in acc])
    )
    metrics["team_policy_loss"] = float(
        np.mean([float(m["team_policy_loss"]) for m in acc])
    )
    metrics["stop_epoch"] = int(stop_epoch)
    metrics["stop_minibatch"] = int(stop_minibatch)

    integer_stats = {"battles", "red_wins", "blue_wins", "timeouts", "draws"}
    stats = {
        k: (int(v) if k in integer_stats else float(v))
        for k, v in stats.items()
    }
    return (
        params,
        opt_state,
        state,
        key,
        time.time() - t0,
        stats,
        metrics,
    )

# ============================================================
# PART 4 : EVALUATION
# ============================================================


def make_evaluation_states(base_states):
    n = base_states["x"].shape[0]
    offsets = EVAL_Z_OFFSETS[jnp.arange(n) % EVAL_Z_OFFSETS.shape[0]]

    def shift_axis(values, unit_mask):
        return values + offsets[:, None] * unit_mask[None, :]

    unit_mask = jnp.ones((N_UNITS,), dtype=jnp.float32)
    z = shift_axis(base_states["z"], unit_mask)
    return {
        **base_states,
        "z": z,
    }


@jax.jit
def evaluate_match(params_red, params_blue, init_states):
    E = init_states["x"].shape[0]

    def body(carry, step_idx):
        (
            st,
            finished,
            result,
            end_step,
            red_surv,
            blue_surv,
            red_return,
            blue_return,
        ) = carry

        red_a = deterministic_world_action_batch(params_red, st, 0.0)
        blue_a = deterministic_world_action_batch(params_blue, st, 1.0)

        (
            nxt,
            rr,
            br,
            _red_soldier_reward,
            _blue_soldier_reward,
            _red_commander_reward,
            _blue_commander_reward,
            done,
        ) = jax.vmap(step_one, in_axes=(0, 0, 0))(
            st, red_a, blue_a
        )

        active = ~finished
        red_return = red_return + jnp.where(active, rr, 0.0)
        blue_return = blue_return + jnp.where(active, br, 0.0)

        newly = done & active

        red_cmd = nxt["alive"][:, RED_COMMANDER_INDEX] > 0
        blue_cmd = nxt["alive"][:, BLUE_COMMANDER_INDEX] > 0

        res_now = jnp.where(
            red_cmd & (~blue_cmd),
            1,
            jnp.where(
                blue_cmd & (~red_cmd),
                2,
                jnp.where(
                    (~red_cmd) & (~blue_cmd),
                    4,
                    jnp.where(done, 3, 0),
                ),
            ),
        )

        rs = jnp.sum(
            nxt["alive"][:, RED_SOLDIER_START:RED_SOLDIER_END],
            axis=1,
        )
        bs = jnp.sum(
            nxt["alive"][:, BLUE_SOLDIER_START:BLUE_SOLDIER_END],
            axis=1,
        )

        result = jnp.where(newly, res_now, result)
        end_step = jnp.where(newly, step_idx + 1, end_step)
        red_surv = jnp.where(newly, rs, red_surv)
        blue_surv = jnp.where(newly, bs, blue_surv)
        finished = finished | done

        return (
            nxt,
            finished,
            result,
            end_step,
            red_surv,
            blue_surv,
            red_return,
            blue_return,
        ), None

    init = (
        init_states,
        jnp.zeros(E, dtype=bool),
        jnp.zeros(E, dtype=jnp.int32),
        jnp.full(E, MAX_STEPS, dtype=jnp.int32),
        jnp.zeros(E, dtype=jnp.float32),
        jnp.zeros(E, dtype=jnp.float32),
        jnp.zeros(E, dtype=jnp.float32),
        jnp.zeros(E, dtype=jnp.float32),
    )

    (
        final_state,
        _,
        result,
        end_step,
        red_surv,
        blue_surv,
        red_return,
        blue_return,
    ), _ = lax.scan(body, init, jnp.arange(MAX_STEPS))
    del final_state

    result = jnp.where(result == 0, 3, result)
    end_step = jnp.where(
        result == 3,
        jnp.minimum(end_step, MAX_STEPS),
        end_step,
    )

    return (
        result,
        end_step,
        red_surv,
        blue_surv,
        red_return,
        blue_return,
    )


def evaluate_elite_match(candidate_params, elite_params, base_states, verbose=True):
    E = base_states["x"].shape[0]

    if verbose:
        print(f"  Elite match: side A (candidate = Red)  {E} games ...")
    rA, sA, redA, blueA, retRedA, retBlueA = evaluate_match(
        candidate_params, elite_params, base_states
    )

    if verbose:
        print(f"  Elite match: side B (candidate = Blue) {E} games ...")
    rB, sB, redB, blueB, retRedB, retBlueB = evaluate_match(
        elite_params, candidate_params, base_states
    )

    result = np.concatenate([np.asarray(rA), np.asarray(rB)])
    end_step = np.concatenate([np.asarray(sA), np.asarray(sB)])
    red_surv = np.concatenate([np.asarray(redA), np.asarray(redB)])
    blue_surv = np.concatenate([np.asarray(blueA), np.asarray(blueB)])
    red_return = np.concatenate([np.asarray(retRedA), np.asarray(retRedB)])
    blue_return = np.concatenate([np.asarray(retBlueA), np.asarray(retBlueB)])

    candidate_is_red = np.concatenate([
        np.ones(E, dtype=bool),
        np.zeros(E, dtype=bool),
    ])

    candidate_won = np.where(candidate_is_red, result == 1, result == 2)
    elite_won = np.where(candidate_is_red, result == 2, result == 1)

    candidate_surv = np.where(candidate_is_red, red_surv, blue_surv)
    elite_surv = np.where(candidate_is_red, blue_surv, red_surv)
    candidate_return = np.where(candidate_is_red, red_return, blue_return)
    elite_return = np.where(candidate_is_red, blue_return, red_return)

    win_time = end_step.astype(np.float32) * DT

    candidate_wins = int(np.sum(candidate_won))
    elite_wins = int(np.sum(elite_won))
    timeouts = int(np.sum(result == 3))
    draws = int(np.sum(result == 4))
    unresolved = int(np.sum(result == 0))

    if unresolved:
        raise RuntimeError(
            f"Elite evaluation produced {unresolved} unresolved game(s)."
        )

    winner = 1 if candidate_wins > elite_wins else 2

    return {
        "result": result,
        "end_step": end_step,
        "win_time": win_time,
        "candidate_is_red": candidate_is_red,
        "candidate_won": candidate_won,
        "elite_won": elite_won,
        "candidate_surv": candidate_surv,
        "elite_surv": elite_surv,
        "candidate_return": candidate_return,
        "elite_return": elite_return,
        "candidate_wins": candidate_wins,
        "elite_wins": elite_wins,
        "timeouts": timeouts,
        "draws": draws,
        "unresolved": unresolved,
        "avg_candidate_time": (
            float(np.mean(win_time[candidate_won]))
            if candidate_wins else float("nan")
        ),
        "avg_elite_time": (
            float(np.mean(win_time[elite_won]))
            if elite_wins else float("nan")
        ),
        "winner": winner,
        "games_per_side": E,
    }

# ============================================================
# PART 5 : BEST BOUT RECORDING
# ============================================================


@jax.jit
def record_bout(params_red, params_blue, initial_state):
    state0 = jax.tree_util.tree_map(lambda a: a[None, ...], initial_state)
    E = 1

    def body(carry, step_idx):
        st, finished, result, end_step = carry

        red_a = deterministic_world_action_batch(params_red, st, 0.0)
        blue_a = deterministic_world_action_batch(params_blue, st, 1.0)

        (
            nxt,
            rr,
            br,
            _red_soldier_reward,
            _blue_soldier_reward,
            _red_commander_reward,
            _blue_commander_reward,
            done,
        ) = jax.vmap(step_one, in_axes=(0, 0, 0))(
            st, red_a, blue_a
        )
        del rr, br, _red_soldier_reward, _blue_soldier_reward, _red_commander_reward, _blue_commander_reward

        active = ~finished
        newly = done & active

        red_cmd = nxt["alive"][:, RED_COMMANDER_INDEX] > 0
        blue_cmd = nxt["alive"][:, BLUE_COMMANDER_INDEX] > 0
        res_now = jnp.where(
            red_cmd & (~blue_cmd),
            1,
            jnp.where(
                blue_cmd & (~red_cmd),
                2,
                jnp.where(
                    (~red_cmd) & (~blue_cmd),
                    4,
                    jnp.where(done, 3, 0),
                ),
            ),
        )

        result = jnp.where(newly, res_now, result)
        end_step = jnp.where(newly, step_idx + 1, end_step)
        finished = finished | done

        out = (
            nxt["x"],
            nxt["z"],
            nxt["hp"],
            nxt["alive"],
            red_a,
            blue_a,
        )
        return (nxt, finished, result, end_step), out

    init = (
        state0,
        jnp.zeros(E, dtype=bool),
        jnp.zeros(E, dtype=jnp.int32),
        jnp.full(E, MAX_STEPS, dtype=jnp.int32),
    )

    (final_state, _, result, end_step), traj = lax.scan(
        body, init, jnp.arange(MAX_STEPS)
    )

    result = jnp.where(result == 0, 3, result)

    xs, zs, hps, alives, red_actions, blue_actions = traj

    xs = xs[:, 0, :]
    zs = zs[:, 0, :]
    hps = hps[:, 0, :]
    alives = alives[:, 0, :]
    red_actions = red_actions[:, 0, :]
    blue_actions = blue_actions[:, 0, :]

    xs = jnp.concatenate([initial_state["x"][None, :], xs], axis=0)
    zs = jnp.concatenate([initial_state["z"][None, :], zs], axis=0)
    hps = jnp.concatenate([initial_state["hp"][None, :], hps], axis=0)
    alives = jnp.concatenate([initial_state["alive"][None, :], alives], axis=0)

    return {
        "x": xs,
        "z": zs,
        "hp": hps,
        "alive": alives,
        "red_actions": red_actions,
        "blue_actions": blue_actions,
        "result": result[0],
        "end_step": end_step[0],
        "final_state": jax.tree_util.tree_map(lambda a: a[0], final_state),
    }


@jax.jit
def verify_bout(initial_state, red_actions, blue_actions):
    def body(st, actions):
        ra, ba = actions
        nxt, _, _, _, _, _, _, _ = step_one(st, ra, ba)
        return nxt, (nxt["x"], nxt["z"], nxt["alive"])

    final_state, traj = lax.scan(
        body,
        initial_state,
        (red_actions, blue_actions),
    )
    del final_state
    xs, zs, alives = traj

    # Reconstruct HP independently as part of replay verification.
    def hp_body(st, actions):
        ra, ba = actions
        nxt, _, _, _, _, _, _, _ = step_one(st, ra, ba)
        return nxt, nxt["hp"]

    _, hps = lax.scan(
        hp_body,
        initial_state,
        (red_actions, blue_actions),
    )

    xs = jnp.concatenate([initial_state["x"][None, :], xs], axis=0)
    zs = jnp.concatenate([initial_state["z"][None, :], zs], axis=0)
    hps = jnp.concatenate([initial_state["hp"][None, :], hps], axis=0)
    alives = jnp.concatenate([initial_state["alive"][None, :], alives], axis=0)
    return xs, zs, hps, alives


def select_best_bout_index(match, key=None):
    """Choose a representative bout; fall back to a random game if no decisive win exists."""
    winner = int(match["winner"])
    if winner == 1:
        won = np.asarray(match["candidate_won"], dtype=bool)
        returns = np.asarray(match["candidate_return"], dtype=np.float32)
        surv = np.asarray(match["candidate_surv"], dtype=np.float32)
    elif winner == 2:
        won = np.asarray(match["elite_won"], dtype=bool)
        returns = np.asarray(match["elite_return"], dtype=np.float32)
        surv = np.asarray(match["elite_surv"], dtype=np.float32)
    else:
        won = np.zeros_like(np.asarray(match["result"], dtype=np.int32), dtype=bool)
        returns = np.maximum(
            np.asarray(match["candidate_return"], dtype=np.float32),
            np.asarray(match["elite_return"], dtype=np.float32),
        )
        surv = np.maximum(
            np.asarray(match["candidate_surv"], dtype=np.float32),
            np.asarray(match["elite_surv"], dtype=np.float32),
        )

    idx = np.where(won)[0]
    if len(idx) > 0:
        order = np.lexsort((-surv[idx], match["end_step"][idx], -returns[idx]))
        return int(idx[order[0]]), False

    n_games = len(match["result"])
    if n_games <= 0:
        return None, True

    if key is None:
        rng = np.random.default_rng()
        idx = int(rng.integers(0, n_games))
    else:
        idx = int(jax.random.randint(key, (), 0, n_games))
    return idx, True


def save_best_bout(
    path,
    generation,
    match,
    best_idx,
    bout,
    verification_pass,
    max_state_error,
    max_hp_error,
    winner_label,
    winner_team,
):
    steps = int(bout["end_step"])
    frames = steps + 1

    data = {
        "generation": np.array(generation, dtype=np.int32),
        "winner_code": np.array(int(bout["result"]), dtype=np.int32),
        "winner_label": np.array(winner_label),
        "winner_team": np.array(winner_team, dtype=np.int32),
        "overall_match_winner_code": np.array(match["winner"], dtype=np.int32),
        "result_code": np.array(int(bout["result"]), dtype=np.int32),
        "win_time": np.array(steps * DT, dtype=np.float32),
        "end_step": np.array(steps, dtype=np.int32),
        "dt": np.array(DT, dtype=np.float32),
        "candidate_is_red": np.array(bool(match["candidate_is_red"][best_idx])),
        "candidate_wins": np.array(match["candidate_wins"], dtype=np.int32),
        "elite_wins": np.array(match["elite_wins"], dtype=np.int32),
        "best_return": np.array(
            (
                match["candidate_return"]
                if match["winner"] == 1
                else match["elite_return"]
            )[best_idx],
            dtype=np.float32,
        ),
        "field_size": np.array(FIELD_SIZE, dtype=np.float32),
        "terrain_res": np.array(TERRAIN_RES, dtype=np.int32),
        "raw_obs_size": np.array(RAW_OBS_SIZE, dtype=np.int32),
        "global_input_size": np.array(GLOBAL_INPUT_SIZE, dtype=np.int32),
        "action_size": np.array(ACTION_SIZE, dtype=np.int32),
        "attack_cooldown": np.array(ATTACK_COOLDOWN, dtype=np.float32),
        "start_x": np.asarray(bout["x"][0]),
        "start_z": np.asarray(bout["z"][0]),
        "start_hp": np.asarray(bout["hp"][0]),
        "start_alive": np.asarray(bout["alive"][0]),
        "x": np.asarray(bout["x"][:frames]),
        "z": np.asarray(bout["z"][:frames]),
        "hp": np.asarray(bout["hp"][:frames]),
        "alive": np.asarray(bout["alive"][:frames]),
        "red_actions": np.asarray(bout["red_actions"][:steps]),
        "blue_actions": np.asarray(bout["blue_actions"][:steps]),
        "verification_pass": np.array(
            1 if verification_pass else 0,
            dtype=np.int32,
        ),
        "max_state_error": np.array(max_state_error, dtype=np.float32),
        "max_hp_error": np.array(max_hp_error, dtype=np.float32),
    }

    np.savez_compressed(path, **data)
    return path

# ============================================================
# PART 6 : CHECKPOINT / PARAM I-O
# ============================================================


def save_params(path, params, metadata=None):
    data = {f"param_{k}": np.asarray(v) for k, v in params.items()}
    if metadata is not None:
        data["metadata_json"] = np.array(json.dumps(metadata))
    np.savez(path, **data)
    return path



def save_checkpoint(
    path,
    params,
    opt_state,
    generation,
    ppo_index,
    elite_path,
    master_key,
    entropy_schedule_start_generation=None,
    entropy_schedule_total_updates=None,
):
    data = {f"param_{k}": np.asarray(v) for k, v in params.items()}

    leaves = jax.tree_util.tree_leaves(opt_state)
    for i, leaf in enumerate(leaves):
        data[f"opt_{i}"] = np.asarray(leaf)
    data["opt_n_leaves"] = np.array(len(leaves), dtype=np.int32)

    data["generation"] = np.array(generation, dtype=np.int32)
    data["ppo_index"] = np.array(ppo_index, dtype=np.int32)
    data["elite_path"] = np.array(elite_path if elite_path else "")
    data["master_key"] = np.asarray(random.key_data(master_key))
    if entropy_schedule_start_generation is not None:
        data["entropy_schedule_start_generation"] = np.array(
            entropy_schedule_start_generation, dtype=np.int32
        )
    if entropy_schedule_total_updates is not None:
        data["entropy_schedule_total_updates"] = np.array(
            entropy_schedule_total_updates, dtype=np.int32
        )

    np.savez(path, **data)
    return path


def required_new_policy_shapes():
    required_shapes = {
        "Ws1": (SOLDIER_FEATURES, LOCAL_HIDDEN_SIZE),
        "bs1": (LOCAL_HIDDEN_SIZE,),
        "Ws2": (LOCAL_HIDDEN_SIZE, LOCAL_EMBED_SIZE),
        "bs2": (LOCAL_EMBED_SIZE,),
        "Wc": (COMMANDER_FEATURES, LOCAL_EMBED_SIZE),
        "bc": (LOCAL_EMBED_SIZE,),
        "Wg1": (GLOBAL_INPUT_SIZE, HIDDEN1),
        "bg1": (HIDDEN1,),
        "W_actor_s": (HIDDEN1, ACTOR_HIDDEN),
        "b_actor_s": (ACTOR_HIDDEN,),
        "W_actor_c": (HIDDEN1, ACTOR_HIDDEN),
        "b_actor_c": (ACTOR_HIDDEN,),
        "W_critic": (HIDDEN1, CRITIC_HIDDEN),
        "b_critic": (CRITIC_HIDDEN,),
        "Wa_s": (ACTOR_HIDDEN, SOLDIER_ACTION_SIZE),
        "ba_s": (SOLDIER_ACTION_SIZE,),
        "Wa_c": (ACTOR_HIDDEN, COMMANDER_ACTION_SIZE),
        "ba_c": (COMMANDER_ACTION_SIZE,),
        "Wv": (CRITIC_HIDDEN, 1),
        "bv": (1,),
    }
    for prefix in ("attn1", "attn2"):
        required_shapes.update({
            f"{prefix}_Wq": (LOCAL_EMBED_SIZE, LOCAL_EMBED_SIZE),
            f"{prefix}_bq": (LOCAL_EMBED_SIZE,),
            f"{prefix}_Wk": (LOCAL_EMBED_SIZE, LOCAL_EMBED_SIZE),
            f"{prefix}_bk": (LOCAL_EMBED_SIZE,),
            f"{prefix}_Wv": (LOCAL_EMBED_SIZE, LOCAL_EMBED_SIZE),
            f"{prefix}_bv": (LOCAL_EMBED_SIZE,),
            f"{prefix}_Wo": (LOCAL_EMBED_SIZE, LOCAL_EMBED_SIZE),
            f"{prefix}_bo": (LOCAL_EMBED_SIZE,),
            f"{prefix}_ln_gamma": (LOCAL_EMBED_SIZE,),
            f"{prefix}_ln_beta": (LOCAL_EMBED_SIZE,),
        })
    return required_shapes


def required_old_policy_shapes():
    required_shapes = {
        "Ws1": (SOLDIER_FEATURES, LOCAL_HIDDEN_SIZE),
        "bs1": (LOCAL_HIDDEN_SIZE,),
        "Ws2": (LOCAL_HIDDEN_SIZE, LOCAL_EMBED_SIZE),
        "bs2": (LOCAL_EMBED_SIZE,),
        "Wc": (COMMANDER_FEATURES, LOCAL_EMBED_SIZE),
        "bc": (LOCAL_EMBED_SIZE,),
        "Wg1": (GLOBAL_INPUT_SIZE, HIDDEN1),
        "bg1": (HIDDEN1,),
        "W_actor": (HIDDEN1, ACTOR_HIDDEN),
        "b_actor": (ACTOR_HIDDEN,),
        "W_critic": (HIDDEN1, CRITIC_HIDDEN),
        "b_critic": (CRITIC_HIDDEN,),
        "Wa": (ACTOR_HIDDEN, ACTION_SIZE),
        "ba": (ACTION_SIZE,),
        "Wv": (CRITIC_HIDDEN, 1),
        "bv": (1,),
    }
    for prefix in ("attn1", "attn2"):
        required_shapes.update({
            f"{prefix}_Wq": (LOCAL_EMBED_SIZE, LOCAL_EMBED_SIZE),
            f"{prefix}_bq": (LOCAL_EMBED_SIZE,),
            f"{prefix}_Wk": (LOCAL_EMBED_SIZE, LOCAL_EMBED_SIZE),
            f"{prefix}_bk": (LOCAL_EMBED_SIZE,),
            f"{prefix}_Wv": (LOCAL_EMBED_SIZE, LOCAL_EMBED_SIZE),
            f"{prefix}_bv": (LOCAL_EMBED_SIZE,),
            f"{prefix}_Wo": (LOCAL_EMBED_SIZE, LOCAL_EMBED_SIZE),
            f"{prefix}_bo": (LOCAL_EMBED_SIZE,),
            f"{prefix}_ln_gamma": (LOCAL_EMBED_SIZE,),
            f"{prefix}_ln_beta": (LOCAL_EMBED_SIZE,),
        })
    return required_shapes


def _shapes_match(params, shapes):
    if set(params.keys()) != set(shapes.keys()):
        return False
    return all(tuple(np.asarray(params[k]).shape) == shape for k, shape in shapes.items())


def policy_params_compatible(params):
    return (
        _shapes_match(params, required_new_policy_shapes())
        or _shapes_match(params, required_old_policy_shapes())
    )


def migrate_policy_params(params):
    """Upgrade the previous shared-actor checkpoint layout to two actor branches."""
    if _shapes_match(params, required_new_policy_shapes()):
        return params, False
    if not _shapes_match(params, required_old_policy_shapes()):
        raise ValueError("Incompatible policy parameter shapes")

    new_params = {
        k: v for k, v in params.items()
        if k not in ("W_actor", "b_actor", "Wa", "ba")
    }
    new_params["W_actor_s"] = params["W_actor"].copy()
    new_params["b_actor_s"] = params["b_actor"].copy()
    new_params["W_actor_c"] = params["W_actor"].copy()
    new_params["b_actor_c"] = params["b_actor"].copy()
    new_params["Wa_s"] = params["Wa"][:, :SOLDIER_ACTION_SIZE].copy()
    new_params["ba_s"] = params["ba"][:SOLDIER_ACTION_SIZE].copy()
    new_params["Wa_c"] = params["Wa"][:, SOLDIER_ACTION_SIZE:].copy()
    new_params["ba_c"] = params["ba"][SOLDIER_ACTION_SIZE:].copy()
    return new_params, True


def saved_policy_file_compatible(path):
    try:
        with np.load(path, allow_pickle=False) as d:
            params = {
                k[len("param_"):]: d[k]
                for k in d.files
                if k.startswith("param_")
            }
        return policy_params_compatible(params)
    except Exception:
        return False


def load_params(path):
    d = np.load(path, allow_pickle=False)
    raw_params = {
        k[len("param_"):]: jnp.asarray(d[k])
        for k in d.files
        if k.startswith("param_")
    }
    params, migrated = migrate_policy_params(raw_params)
    if migrated:
        print(f"Migrated legacy shared Actor checkpoint: {os.path.basename(path)}")
    meta = {}
    if "metadata_json" in d.files:
        try:
            meta = json.loads(str(d["metadata_json"]))
        except Exception:
            meta = {}
    return params, meta


def load_checkpoint(path):
    d = np.load(path, allow_pickle=False)

    raw_params = {
        k[len("param_"):]: jnp.asarray(d[k])
        for k in d.files
        if k.startswith("param_")
    }
    params, migrated = migrate_policy_params(raw_params)

    if migrated:
        # Optimizer moments from the old parameter tree cannot be mapped safely
        # to the new split Actor tree. Reinitialize Adam for this one resume.
        print(
            "Legacy checkpoint architecture detected; migrating parameters "
            "and reinitializing optimizer state."
        )
        opt_state = optimizer.init(params)
    else:
        template = optimizer.init(params)
        template_leaves = jax.tree_util.tree_leaves(template)
        treedef = jax.tree_util.tree_structure(template)
        n = int(d["opt_n_leaves"])
        if n != len(template_leaves):
            raise ValueError(
                f"Incompatible optimizer state in checkpoint: {path}"
            )

        leaves = []
        for i, template_leaf in enumerate(template_leaves):
            leaf = jnp.asarray(d[f"opt_{i}"])
            if tuple(leaf.shape) != tuple(template_leaf.shape):
                raise ValueError(
                    f"Incompatible optimizer leaf {i} in checkpoint: {path}"
                )
            leaves.append(leaf)
        opt_state = jax.tree_util.tree_unflatten(treedef, leaves)

    generation = int(d["generation"])
    ppo_index = int(d["ppo_index"])
    elite_path = str(d["elite_path"]) if "elite_path" in d.files else ""
    master_key = random.wrap_key_data(jnp.asarray(d["master_key"]))
    schedule_start_generation = (
        int(d["entropy_schedule_start_generation"])
        if "entropy_schedule_start_generation" in d.files
        else generation
    )
    schedule_total_updates = (
        int(d["entropy_schedule_total_updates"])
        if "entropy_schedule_total_updates" in d.files
        else None
    )

    return (
        params, opt_state, generation, ppo_index, elite_path, master_key,
        schedule_start_generation, schedule_total_updates
    )

def generation_number(path):
    try:
        return int(
            os.path.basename(path).split("generation_")[1].split(".npz")[0]
        )
    except Exception:
        return -1


def checkpoint_number(path):
    try:
        name = os.path.basename(path)
        generation = int(name.split("checkpoint_g")[1].split("_p")[0])
        ppo = int(name.split("_p")[1].split(".npz")[0])
        return generation, ppo
    except Exception:
        return -1, -1


def find_latest_elite():
    files = glob.glob(os.path.join(ELITE_DIR, "generation_*.npz"))
    
    # 世代番号が新しい順（降順）に並び替える
    files_sorted = sorted(files, key=generation_number, reverse=True)
    
    # 最新のものから順番に1つずつチェックし、OKなら即座に返す
    for f in files_sorted:
        if saved_policy_file_compatible(f):
            return f
            
    # もし使えるファイルが1つも無ければ None
    if files:
        print(f"Ignoring incompatible Elite file(s) (current global input={GLOBAL_INPUT_SIZE}).")
    return None


def find_latest_checkpoint():
    files = glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_*.npz"))
    compatible = []
    incompatible = []

    for path in files:
        if saved_policy_file_compatible(path):
            compatible.append(path)
        else:
            incompatible.append(path)

    if incompatible:
        print(
            f"Ignoring {len(incompatible)} incompatible checkpoint(s) "
            f"(current global input={GLOBAL_INPUT_SIZE})."
        )

    if not compatible:
        return None

    return sorted(compatible, key=checkpoint_number)[-1]


def prune_checkpoints(keep_path=None):
    keep = os.path.abspath(keep_path) if keep_path is not None else None
    removed = 0
    for path in glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_*.npz")):
        if keep is not None and os.path.abspath(path) == keep:
            continue
        try:
            os.remove(path)
            removed += 1
        except OSError:
            pass
    return removed

# ============================================================
# PART 7 : TRAINING LOOP
# ============================================================

LAST_COMPLETED_GENERATION = None


def train(n_generations=N_GENERATIONS, resume=True):
    master_key = random.key(int(time.time()) & 0x7FFFFFFF)

    start_generation = 1
    start_ppo_index = 1
    resumed_params = None
    resumed_opt = None
    entropy_schedule_start_generation = None
    entropy_schedule_total_updates = None

    ckpt = find_latest_checkpoint() if resume else None
    if ckpt is not None:
        try:
            (
                resumed_params,
                resumed_opt,
                g,
                p,
                saved_elite,
                master_key,
                entropy_schedule_start_generation,
                entropy_schedule_total_updates,
            ) = load_checkpoint(ckpt)
            del saved_elite
            start_generation = g
            start_ppo_index = p + 1
            if entropy_schedule_total_updates is None:
                entropy_schedule_total_updates = n_generations * PPO_UPDATES_PER_GENERATION
            if start_ppo_index > PPO_UPDATES_PER_GENERATION:
                start_ppo_index = 1
                start_generation += 1
                resumed_params = None
                resumed_opt = None
            print(
                f"Resuming from checkpoint: generation {start_generation}, "
                f"PPO {start_ppo_index}/{PPO_UPDATES_PER_GENERATION}"
            )
        except (ValueError, KeyError, OSError, EOFError) as exc:
            print(f"Checkpoint resume skipped: {exc}")
            resumed_params = None
            resumed_opt = None
            start_generation = 1
            start_ppo_index = 1
            entropy_schedule_start_generation = None
            entropy_schedule_total_updates = None

    latest_elite = find_latest_elite()
    if latest_elite is None:
        master_key, ik = random.split(master_key)
        params0 = init_policy(ik)
        latest_elite = save_params(
            os.path.join(ELITE_DIR, "generation_0000.npz"),
            params0,
            {
                "generation": 0,
                "source": "random_init",
                "raw_obs_size": RAW_OBS_SIZE,
                "global_input_size": GLOBAL_INPUT_SIZE,
                "architecture": "hierarchical_64_attention2_role_separated_actor_rewards",
                "ppo_rollout_steps": PPO_ROLLOUT_STEPS,
            },
        )
        print("Created Generation 0 Elite from fresh random initialization.")
    else:
        print("Existing Elite found:", os.path.basename(latest_elite))

    if entropy_schedule_start_generation is None:
        entropy_schedule_start_generation = start_generation
    if entropy_schedule_total_updates is None:
        entropy_schedule_total_updates = n_generations * PPO_UPDATES_PER_GENERATION

    print()
    print("============================================")
    print("PPO + ELITE SELF-PLAY")
    print("============================================")
    print(f"Raw observation    : {RAW_OBS_SIZE}")
    print(f"Global network in  : {GLOBAL_INPUT_SIZE}")
    print(f"Action             : {ACTION_SIZE}")
    print(f"Environments       : {N_ENVS}")
    print(f"Episode max steps  : {MAX_STEPS}  ({MAX_TIME:.1f} sim sec)")
    print(f"PPO rollout steps  : {PPO_ROLLOUT_STEPS}  ({PPO_ROLLOUT_STEPS * DT:.1f} sim sec)")
    print(f"PPO batch          : {BATCH_SIZE}")
    print(f"Minibatch          : {MINIBATCH_SIZE}")
    print(f"PPO / generation   : {PPO_UPDATES_PER_GENERATION}")
    print(f"Elite eval games   : {EVAL_GAMES}")
    print(f"Soldier encoder    : {SOLDIER_FEATURES} -> {LOCAL_HIDDEN_SIZE} -> {LOCAL_EMBED_SIZE}")
    print(f"Commander encoder  : {COMMANDER_FEATURES} -> {LOCAL_EMBED_SIZE}")
    print(f"Self-attention     : {N_SOLDIERS_PER_TEAM} soldiers/team, {ATTENTION_HEADS} heads x 2 layers")
    print(f"Global network     : {GLOBAL_INPUT_SIZE} -> {HIDDEN1}")
    print(f"Soldier actor      : {HIDDEN1} -> {ACTOR_HIDDEN} -> {SOLDIER_ACTION_SIZE}")
    print(f"Commander actor    : {HIDDEN1} -> {ACTOR_HIDDEN} -> {COMMANDER_ACTION_SIZE}")
    print(f"Global critic      : {HIDDEN1} -> {CRITIC_HIDDEN} -> 1")
    print(f"Soldier rewards    : hit {REWARD_SOLDIER_HIT}, kill {REWARD_SOLDIER_KILL}, "
          f"miss {REWARD_SOLDIER_MISS}, wall {REWARD_SOLDIER_WALL}, "
          f"approach {REWARD_SOLDIER_APPROACH}")
    print(f"Commander rewards  : wall {REWARD_COMMANDER_WALL}, "
          f"survival {REWARD_COMMANDER_SURVIVAL} / step, "
          f"hit by enemy {REWARD_COMMANDER_HIT_BY_ENEMY}")
    print(f"Team rewards       : win {REWARD_COMMANDER_WIN}, loss {REWARD_COMMANDER_LOSS}")
    print("Reward ownership   : Soldier local -> Soldier Encoder + Soldier Actor")
    print("                     Commander local -> Commander Encoder + Commander Actor")
    print("                     Team win/loss -> Global + both Actors + Critic")
    print("                     Team win/loss -> encoders BLOCKED (stop-gradient)")
    print(f"Warm-up            : 0 -> {WARMUP_MAX_STEPS} steps (not used for PPO)")
    print(f"Damage shaping     : {SHAPING_COEF}")
    print(f"Entropy coef       : {ENTROPY_START} -> {ENTROPY_END}")
    print(
        f"Entropy schedule   : generation {entropy_schedule_start_generation}"
        f" + {int(entropy_schedule_total_updates)} updates"
    )
    print(f"Logstd             : init {LOGSTD_INIT:.1f}, target {LOGSTD_TARGET:.1f}, range [{LOGSTD_MIN:.1f}, {LOGSTD_MAX:.1f}]")
    print(f"Logstd regularizer : {LOGSTD_REG_COEF}")
    print(f"PPO target KL      : {TARGET_KL}")
    print(f"Output directory   : {os.path.abspath(BASE_DIR)}")
    print()

    # n_generations is the number of generations to run from the current
    # resume point. For example, resuming at 190 with 10000 means 190..10189.

    end_generation = start_generation + n_generations - 1
    total_updates = int(entropy_schedule_total_updates)

    for generation in range(start_generation, end_generation + 1):
        print()
        print("==================================================")
        print(f"GENERATION {generation}")
        print("==================================================")

        elite_params, _ = load_params(latest_elite)

        if resumed_params is not None:
            candidate_params = resumed_params
            candidate_opt = resumed_opt
            first_ppo = start_ppo_index
            resumed_params = None
            resumed_opt = None
        else:
            candidate_params = jax.tree_util.tree_map(
                lambda a: a.copy(), elite_params
            )
            candidate_opt = optimizer.init(candidate_params)
            first_ppo = 1

        master_key, rk, ek = random.split(master_key, 3)
        env_state = reset(rk)
        env_key = ek

        env_state, env_key, warmup_battles = warmup_env_state(
            candidate_params, env_state, env_key
        )
        print(f"Warm-up completed games : {int(warmup_battles)}")

        cum = {
            "battles": 0,
            "red_wins": 0,
            "blue_wins": 0,
            "timeouts": 0,
            "draws": 0,
            "reward_samples": 0,
            "soldier_reward_abs_sum": 0.0,
            "commander_reward_abs_sum": 0.0,
            "team_reward_abs_sum": 0.0,
            "soldier_adv_abs_sum": 0.0,
            "commander_adv_abs_sum": 0.0,
            "team_adv_abs_sum": 0.0,
        }

        for ppo_index in range(first_ppo, PPO_UPDATES_PER_GENERATION + 1):
            global_update = (
                (generation - entropy_schedule_start_generation)
                * PPO_UPDATES_PER_GENERATION
                + (ppo_index - 1)
            )
            global_update = max(0, int(global_update))

            (
                candidate_params,
                candidate_opt,
                env_state,
                env_key,
                elapsed,
                stats,
                metrics,
            ) = run_ppo_update(
                candidate_params,
                candidate_opt,
                env_state,
                env_key,
                global_update,
                total_updates,
            )

            for k in ("battles", "red_wins", "blue_wins", "timeouts", "draws"):
                cum[k] += stats[k]

            cum["reward_samples"] += 1
            cum["soldier_reward_abs_sum"] += stats["soldier_reward_abs_mean"]
            cum["commander_reward_abs_sum"] += stats["commander_reward_abs_mean"]
            cum["team_reward_abs_sum"] += stats["team_reward_abs_mean"]
            cum["soldier_adv_abs_sum"] += stats["soldier_adv_abs_mean"]
            cum["commander_adv_abs_sum"] += stats["commander_adv_abs_mean"]
            cum["team_adv_abs_sum"] += stats["team_adv_abs_mean"]

            checkpoint_path = os.path.join(
                CHECKPOINT_DIR,
                f"checkpoint_g{generation:04d}_p{ppo_index:03d}.npz",
            )
            save_checkpoint(
                checkpoint_path,
                candidate_params,
                candidate_opt,
                generation,
                ppo_index,
                latest_elite,
                master_key,
                entropy_schedule_start_generation=entropy_schedule_start_generation,
                entropy_schedule_total_updates=total_updates,
            )
            prune_checkpoints(checkpoint_path)

            decisive_games = cum["red_wins"] + cum["blue_wins"]
            early_tag = " EARLY-STOP" if metrics["early_stopped"] > 0.5 else ""
            print(
                f"Generation {generation} | "
                f"PPO {ppo_index:2d}/{PPO_UPDATES_PER_GENERATION} | "
                f"{elapsed:5.1f} s/update | "
                f"Battles {cum['battles']:4d} | "
                f"Decisive Games {decisive_games:3d} | "
                f"entropy {metrics['entropy']:7.2f} "
                f"({metrics['entropy_per_soldier']:.3f}/soldier) "
                f"logstd {metrics['logstd_mean']:.3f} "
                f"KL {metrics['approx_kl']:.5f} maxKL {metrics['max_kl']:.5f} "
                f"clip {metrics['clip_fraction']:.3f} "
                f"delta {metrics['parameter_delta_l2']:.3e}"
                f"{early_tag}"
            )
            print(
                f"  Reward→NN | "
                f"Soldier |R| {stats['soldier_reward_abs_mean']:.3e} "
                f"|A| {stats['soldier_adv_abs_mean']:.3e} "
                f"grad {metrics['grad_norm_soldier_local']:.3e}; "
                f"Commander |R| {stats['commander_reward_abs_mean']:.3e} "
                f"|A| {stats['commander_adv_abs_mean']:.3e} "
                f"grad {metrics['grad_norm_commander_local']:.3e}; "
                f"Team |R| {stats['team_reward_abs_mean']:.3e} "
                f"|A| {stats['team_adv_abs_mean']:.3e} "
                f"grad {metrics['grad_norm_team']:.3e}"
            )

        print()
        print("PPO summary")
        print("------------------------------------------")
        print(f"Battles completed : {cum['battles']}")
        print(f"Decisive games     : {cum['red_wins'] + cum['blue_wins']}")
        print(f"Timeouts          : {cum['timeouts']}")
        if cum["reward_samples"] > 0:
            d = float(cum["reward_samples"])
            print("Reward -> NN diagnostics (rollout averages)")
            print("------------------------------------------")
            print(
                f"Soldier   |R| {cum['soldier_reward_abs_sum']/d:.3e}  "
                f"|A| {cum['soldier_adv_abs_sum']/d:.3e}"
            )
            print(
                f"Commander |R| {cum['commander_reward_abs_sum']/d:.3e}  "
                f"|A| {cum['commander_adv_abs_sum']/d:.3e}"
            )
            print(
                f"Team      |R| {cum['team_reward_abs_sum']/d:.3e}  "
                f"|A| {cum['team_adv_abs_sum']/d:.3e}"
            )
        print()

        master_key, evk = random.split(master_key)
        base_states = reset_batch(evk, EVAL_GAMES_PER_SIDE)
        eval_states = make_evaluation_states(base_states)

        print("Elite match")
        print("------------------------------------------")
        t0 = time.time()
        match = evaluate_elite_match(
            candidate_params,
            elite_params,
            eval_states,
        )
        match_time = time.time() - t0

        print()
        print("Elite Match Result")
        print("------------------------------------------")
        print(f"Candidate wins : {match['candidate_wins']}")
        print(f"Elite wins     : {match['elite_wins']}")
        print(f"Timeouts       : {match['timeouts']}")
        print(f"Draws          : {match['draws']}")
        print(f"Unresolved     : {match['unresolved']}")
        if np.isfinite(match["avg_candidate_time"]):
            print(f"Candidate avg win time : {match['avg_candidate_time']:.2f} s")
        if np.isfinite(match["avg_elite_time"]):
            print(f"Elite avg win time     : {match['avg_elite_time']:.2f} s")
        print(f"Evaluation time        : {match_time:.1f} s")

        if match["candidate_wins"] > match["elite_wins"]:
            winner_label = "Candidate"
        elif match["elite_wins"] > match["candidate_wins"]:
            winner_label = "Elite"
        else:
            winner_label = "Tie / Elite retained"
        elite_changed = match["winner"] == 1
        print(f"Winner                 : {winner_label}")

        best_bout_saved = False
        master_key, best_bout_key = random.split(master_key)
        best_idx, is_random_pick = select_best_bout_index(match, best_bout_key)
        if is_random_pick:
            print("Best Bout selection   : random fallback (no decisive winner)")

        if best_idx is not None:
            cand_is_red = bool(match["candidate_is_red"][best_idx])
            base_idx = best_idx % match["games_per_side"]
            init_state = tree_index(eval_states, base_idx)

            params_red = candidate_params if cand_is_red else elite_params
            params_blue = elite_params if cand_is_red else candidate_params

            bout = record_bout(params_red, params_blue, init_state)
            steps = int(bout["end_step"])
            vx_, vz_, vhp_, va_ = verify_bout(
                init_state,
                bout["red_actions"][:steps],
                bout["blue_actions"][:steps],
            )

            err_x = float(jnp.max(jnp.abs(vx_ - bout["x"][:steps + 1])))
            err_z = float(jnp.max(jnp.abs(vz_ - bout["z"][:steps + 1])))
            err_hp = float(jnp.max(jnp.abs(vhp_ - bout["hp"][:steps + 1])))
            err_alive = float(jnp.max(jnp.abs(va_ - bout["alive"][:steps + 1])))
            max_err = max(err_x, err_z, err_hp, err_alive)
            verification_pass = (
                err_x < 1e-4
                and err_z < 1e-4
                and err_hp < 1e-4
                and err_alive < 1e-6
            )

            expected_result = int(match["result"][best_idx])
            replay_result = int(bout["result"])
            final_alive = np.asarray(bout["alive"][steps])

            if expected_result == 1:
                commander_outcome_pass = (
                    final_alive[RED_COMMANDER_INDEX] > 0
                    and final_alive[BLUE_COMMANDER_INDEX] <= 0
                )
                commander_outcome_status = "PASS" if commander_outcome_pass else "FAIL"
            elif expected_result == 2:
                commander_outcome_pass = (
                    final_alive[BLUE_COMMANDER_INDEX] > 0
                    and final_alive[RED_COMMANDER_INDEX] <= 0
                )
                commander_outcome_status = "PASS" if commander_outcome_pass else "FAIL"
            else:
                commander_outcome_pass = None
                commander_outcome_status = "SKIP (timeout/draw)"

            if expected_result == 1:
                bout_winner_label = "Candidate" if cand_is_red else "Elite"
                winner_team = 0
            elif expected_result == 2:
                bout_winner_label = "Elite" if cand_is_red else "Candidate"
                winner_team = 1
            else:
                bout_winner_label = "Timeout / Draw"
                winner_team = -1

            surv = max(
                float(match["candidate_surv"][best_idx]),
                float(match["elite_surv"][best_idx]),
            ) if expected_result >= 3 else (
                float(match["candidate_surv"][best_idx])
                if bout_winner_label == "Candidate"
                else float(match["elite_surv"][best_idx])
            )
            best_return = max(
                float(match["candidate_return"][best_idx]),
                float(match["elite_return"][best_idx]),
            ) if expected_result >= 3 else (
                float(match["candidate_return"][best_idx])
                if bout_winner_label == "Candidate"
                else float(match["elite_return"][best_idx])
            )

            print()
            print("Best Bout")
            print("------------------------------------------")
            side_text = "Tie" if winner_team < 0 else ("Red" if winner_team == 0 else "Blue")
            print(
                f"Winner              : {bout_winner_label} ({side_text})"
            )
            print(f"Return              : {float(best_return):.6f}")
            print(f"Win time            : {steps * DT:.2f} s  ({steps} steps)")
            print(f"Winner survivors    : {int(surv)}")
            print(f"Evaluation result   : {expected_result}")
            print(f"Replay result       : {replay_result}")
            print(
                f"Replay verification : "
                f"{'PASS' if verification_pass else f'FAIL (err {max_err:.2e})'}"
            )
            print(f"Commander outcome   : {commander_outcome_status}")
            print(
                f"HP verification    : {'PASS' if err_hp < 1e-4 else f'FAIL (err {err_hp:.2e})'}"
            )

            path = save_best_bout(
                os.path.join(
                    BOUT_DIR,
                    f"generation_{generation:04d}_best_bout.npz",
                ),
                generation,
                match,
                best_idx,
                bout,
                verification_pass,
                max_err,
                err_hp,
                bout_winner_label,
                winner_team,
            )
            best_bout_saved = True
            if replay_result != expected_result:
                print(
                    "WARNING            : replay result differs from evaluation; "
                    "evaluation remains authoritative"
                )
            print(f"Saved               : {path}")

        if elite_changed:
            latest_elite = save_params(
                os.path.join(ELITE_DIR, f"generation_{generation:04d}.npz"),
                candidate_params,
                {
                    "generation": generation,
                    "source": "candidate",
                    "raw_obs_size": RAW_OBS_SIZE,
                    "global_input_size": GLOBAL_INPUT_SIZE,
                },
            )

        print()
        print("Generation summary")
        print("------------------------------------------")
        print(f"Elite changed : {'YES' if elite_changed else 'NO'}")
        print(f"Current Elite : {os.path.basename(latest_elite)}")
        print(f"Best Bout     : {'SAVED' if best_bout_saved else 'NONE'}")
        print("Checkpoint    : latest PPO checkpoint retained")
        print()

        global LAST_COMPLETED_GENERATION
        LAST_COMPLETED_GENERATION = generation

    return latest_elite

# ============================================================
# PART 8 : 3D REPLAY
# ============================================================

RESULT_TEXT = {
    1: "BLUE COMMANDER KILLED",
    2: "RED COMMANDER KILLED",
    3: "TIMEOUT",
    4: "BOTH COMMANDERS KILLED",
    0: "UNRESOLVED",
}


def build_replay_html(bout_path, out_path=None):
    d = np.load(bout_path, allow_pickle=False)
    generation = int(d["generation"])
    winner_label = str(d["winner_label"])
    winner_team = int(d["winner_team"])
    result_code = int(d["result_code"])
    win_time = float(d["win_time"])
    end_step = int(d["end_step"])
    dt = float(d["dt"])
    saved_field_size = float(d["field_size"]) if "field_size" in d else FIELD_SIZE
    saved_terrain_res = int(d["terrain_res"]) if "terrain_res" in d else TERRAIN_RES
    saved_raw_obs = int(d["raw_obs_size"]) if "raw_obs_size" in d else RAW_OBS_SIZE
    saved_global_input = int(d["global_input_size"]) if "global_input_size" in d else GLOBAL_INPUT_SIZE

    if abs(saved_field_size - FIELD_SIZE) > 1e-6:
        raise ValueError(
            f"Best Bout field size mismatch: saved={saved_field_size}, current={FIELD_SIZE}"
        )
    if (
        saved_terrain_res != TERRAIN_RES
        or saved_raw_obs != RAW_OBS_SIZE
        or saved_global_input != GLOBAL_INPUT_SIZE
    ):
        raise ValueError("Best Bout is from an incompatible environment/network")

    verification_pass = bool(int(d["verification_pass"]))
    max_state_error = float(d["max_state_error"])
    max_hp_error = float(d["max_hp_error"]) if "max_hp_error" in d else max_state_error
    x = d["x"].astype(np.float32)
    z = d["z"].astype(np.float32)
    hp = d["hp"].astype(np.float32)
    alive = d["alive"].astype(np.float32)
    red_actions = d["red_actions"].astype(np.float32)
    blue_actions = d["blue_actions"].astype(np.float32)

    n_frames, n_units = x.shape
    if n_units != N_UNITS:
        raise ValueError(f"Unexpected unit count: {n_units}")
    if n_frames < 2:
        raise ValueError("Best Bout contains no replay frames.")
    if red_actions.shape[0] != end_step or blue_actions.shape[0] != end_step:
        raise ValueError("Action length does not match end_step.")

    payload = json.dumps(
        {
            "generation": generation,
            "winnerLabel": winner_label,
            "winnerTeam": winner_team,
            "winnerSide": ("Tie" if winner_team < 0 else ("Red" if winner_team == 0 else "Blue")),
            "resultText": RESULT_TEXT.get(result_code, "UNKNOWN"),
            "winTime": win_time,
            "endStep": end_step,
            "dt": dt,
            "steps": n_frames - 1,
            "verificationPass": verification_pass,
            "maxStateError": max_state_error,
            "maxHpError": max_hp_error,
            "fieldSize": FIELD_SIZE,
            "walls": WALL_LIST,
            "x": np.round(x, 4).tolist(),
            "z": np.round(z, 4).tolist(),
            "hp": np.round(hp, 3).tolist(),
            "alive": alive.astype(np.uint8).tolist(),
            "redActions": np.round(red_actions, 3).tolist(),
            "blueActions": np.round(blue_actions, 3).tolist(),
            "attackCooldown": float(ATTACK_COOLDOWN),
        },
        separators=(",", ":"),
    )

    html = r'''<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RTS Best Bout Replay</title>
<style>
html,body{margin:0;padding:0;width:100%;height:100%;overflow:hidden;background:#101010;font-family:Arial,Helvetica,sans-serif;}
#app{position:relative;width:100vw;height:100vh;}
#info{position:absolute;left:14px;top:14px;z-index:10;padding:12px 15px;background:rgba(0,0,0,.74);color:#fff;border-radius:8px;min-width:280px;line-height:1.5;font-size:14px;pointer-events:none;}
#controls{position:absolute;left:14px;bottom:14px;z-index:10;padding:10px 12px;background:rgba(0,0,0,.74);border-radius:8px;color:#fff;}
button{margin-right:5px;padding:5px 9px;border:0;border-radius:4px;cursor:pointer;}
#timeline{width:440px;max-width:45vw;vertical-align:middle;}#status{margin-top:7px;font-size:12px;opacity:.85;}
.pass{color:#5fd97a;font-weight:bold}.fail{color:#ff6a6a;font-weight:bold}.attack{color:#ffd95a;font-weight:bold}
hr{border:0;border-top:1px solid rgba(255,255,255,.25);margin:7px 0;}
#error{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);z-index:30;display:none;max-width:75vw;padding:18px 22px;background:rgba(40,0,0,.92);color:#fff;border:1px solid #f66;border-radius:10px;font-family:Consolas,monospace;white-space:pre-wrap;}
</style></head><body>
<div id="app"><div id="info"></div><div id="error"></div>
<div id="controls">
<button id="play">Play</button><button id="pause">Pause</button><button id="reset">Reset</button>
<button data-speed="0.25">0.25x</button><button data-speed="0.5">0.5x</button><button data-speed="1">1x</button><button data-speed="2">2x</button><br><br>
<input id="timeline" type="range" min="0" max="__MAXSTEP__" value="0" step="1"><div id="status"></div>
</div></div>
<script type="importmap">{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
const DATA=__PAYLOAD__;
const app=document.getElementById('app'),info=document.getElementById('info'),errorBox=document.getElementById('error'),timeline=document.getElementById('timeline'),statusEl=document.getElementById('status');
let replayTime=0,playing=false,speed=1,lastTs=null;
function showError(err){errorBox.style.display='block';errorBox.textContent=String(err&&err.stack?err.stack:err);}
try{
 const scene=new THREE.Scene();scene.background=new THREE.Color(0x101010);
 const camera=new THREE.PerspectiveCamera(45,innerWidth/innerHeight,.1,250);camera.position.set(0,18,17);
 const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio||1,2));renderer.setSize(innerWidth,innerHeight);app.appendChild(renderer.domElement);
 const controls=new OrbitControls(camera,renderer.domElement);controls.target.set(0,0,0);controls.enableDamping=true;controls.dampingFactor=.08;controls.update();
 scene.add(new THREE.HemisphereLight(0xffffff,0x444444,2.0));const dir=new THREE.DirectionalLight(0xffffff,1.3);dir.position.set(6,14,8);scene.add(dir);
 const fieldSize=DATA.fieldSize,half=fieldSize/2;
 const ground=new THREE.Mesh(new THREE.PlaneGeometry(fieldSize,fieldSize),new THREE.MeshStandardMaterial({color:0x303030}));ground.rotation.x=-Math.PI/2;ground.position.y=-.02;scene.add(ground);
 const grid=new THREE.GridHelper(fieldSize,fieldSize,0x777777,0x444444);grid.position.y=.01;scene.add(grid);
 const redStart=new THREE.Mesh(new THREE.PlaneGeometry(2,fieldSize),new THREE.MeshBasicMaterial({color:0xd94b4b,transparent:true,opacity:.16,side:THREE.DoubleSide}));redStart.rotation.x=-Math.PI/2;redStart.position.set(-half+1,.015,0);scene.add(redStart);
 const blueStart=new THREE.Mesh(new THREE.PlaneGeometry(2,fieldSize),new THREE.MeshBasicMaterial({color:0x4b7bd9,transparent:true,opacity:.16,side:THREE.DoubleSide}));blueStart.rotation.x=-Math.PI/2;blueStart.position.set(half-1,.016,0);scene.add(blueStart);
 const wallGeo=new THREE.BoxGeometry(1,.7,1),wallMat=new THREE.MeshStandardMaterial({color:0x777777});for(const p of DATA.walls){const w=new THREE.Mesh(wallGeo,wallMat);w.position.set(p[0],.35,p[1]);scene.add(w);}
 const soldierBodyGeo=new THREE.BoxGeometry(.22,.36,.22),soldierHeadGeo=new THREE.SphereGeometry(.105,12,12);
 const redBodyMat=new THREE.MeshStandardMaterial({color:0xd94b4b}),blueBodyMat=new THREE.MeshStandardMaterial({color:0x4b7bd9});
 const redFaceMat=new THREE.MeshStandardMaterial({color:0xd94b4b,emissive:0x000000}),blueFaceMat=new THREE.MeshStandardMaterial({color:0x4b7bd9,emissive:0x000000});
 const attackFaceMat=new THREE.MeshStandardMaterial({color:0xffe13b,emissive:0x4a3a00,emissiveIntensity:.35});
 const redEyeMat=new THREE.LineBasicMaterial({color:0xd94b4b}),blueEyeMat=new THREE.LineBasicMaterial({color:0x4b7bd9}),attackEyeMat=new THREE.LineBasicMaterial({color:0xffe13b});
 const redSoldiers=[],blueSoldiers=[];
 function makeSoldier(team){const g=new THREE.Group();const body=new THREE.Mesh(soldierBodyGeo,team===0?redBodyMat:blueBodyMat);body.position.y=.18;const face=new THREE.Mesh(soldierHeadGeo,team===0?redFaceMat:blueFaceMat);face.position.set(0,.43,0);const lineGeo=new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0,.43,.06),new THREE.Vector3(0,.43,.34)]);const eyeLine=new THREE.Line(lineGeo,team===0?redEyeMat:blueEyeMat);g.add(body,face,eyeLine);scene.add(g);return{team,group:g,body,face,eyeLine,baseFace:team===0?redFaceMat:blueFaceMat,baseEye:team===0?redEyeMat:blueEyeMat};}
 for(let i=0;i<100;i++)redSoldiers.push(makeSoldier(0));for(let i=0;i<100;i++)blueSoldiers.push(makeSoldier(1));
 const cmdGeo=new THREE.BoxGeometry(.44,.8,.44),redCommander=new THREE.Mesh(cmdGeo,redBodyMat),blueCommander=new THREE.Mesh(cmdGeo,blueBodyMat);scene.add(redCommander,blueCommander);
 const crownGeo=new THREE.ConeGeometry(.22,.22,5),crownMat=new THREE.MeshStandardMaterial({color:0xffd83d}),redCrown=new THREE.Mesh(crownGeo,crownMat),blueCrown=new THREE.Mesh(crownGeo,crownMat);redCommander.add(redCrown);blueCommander.add(blueCrown);redCrown.position.set(0,.51,0);blueCrown.position.set(0,.51,0);
 function setSoldierVisual(s,attackNow){s.face.material=attackNow?attackFaceMat:s.baseFace;s.eyeLine.material=attackNow?attackEyeMat:s.baseEye;}
 function frameInfo(){const t=Math.max(0,Math.min(DATA.steps*DATA.dt,replayTime)),raw=t/DATA.dt,i=Math.min(Math.floor(raw),DATA.steps),a=i>=DATA.steps?0:raw-i;return{t,i,a};}
 function updateReplay(){const f=frameInfo(),i=f.i,a=f.a;timeline.value=String(i);const X0=DATA.x[i],Z0=DATA.z[i],A=DATA.alive[i],X1=i<DATA.steps?DATA.x[i+1]:X0,Z1=i<DATA.steps?DATA.z[i+1]:Z0,px=k=>X0[k]+(X1[k]-X0[k])*a,pz=k=>Z0[k]+(Z1[k]-Z0[k])*a;const redActions=DATA.redActions[i]||null,blueActions=DATA.blueActions[i]||null;let redAttackCount=0,blueAttackCount=0;
  for(let k=0;k<100;k++){const ridx=2+k,bidx=102+k;const rAttack=!!redActions&&redActions[3*k+2]>.5&&A[ridx]>0,bAttack=!!blueActions&&blueActions[3*k+2]>.5&&A[bidx]>0;const rdx=!!redActions?redActions[3*k]:0,rdz=!!redActions?redActions[3*k+1]:0,bdx=!!blueActions?blueActions[3*k]:0,bdz=!!blueActions?blueActions[3*k+1]:0;redSoldiers[k].group.position.set(px(ridx),0,pz(ridx));redSoldiers[k].group.rotation.y=(Math.abs(rdx)+Math.abs(rdz)>1e-6)?Math.atan2(rdx,rdz):redSoldiers[k].group.rotation.y;redSoldiers[k].group.visible=A[ridx]>0;setSoldierVisual(redSoldiers[k],rAttack);if(rAttack)redAttackCount++;blueSoldiers[k].group.position.set(px(bidx),0,pz(bidx));blueSoldiers[k].group.rotation.y=(Math.abs(bdx)+Math.abs(bdz)>1e-6)?Math.atan2(bdx,bdz):blueSoldiers[k].group.rotation.y;blueSoldiers[k].group.visible=A[bidx]>0;setSoldierVisual(blueSoldiers[k],bAttack);if(bAttack)blueAttackCount++;}
  redCommander.position.set(px(0),0,pz(0));redCommander.visible=A[0]>0;redCrown.visible=A[0]>0;blueCommander.position.set(px(1),0,pz(1));blueCommander.visible=A[1]>0;blueCrown.visible=A[1]>0;
  const ra=Array.from(A.slice(2,102)).reduce((sum,v)=>sum+v,0),ba=Array.from(A.slice(102,202)).reduce((sum,v)=>sum+v,0);const verify=DATA.verificationPass?'<span class="pass">Replay verification: PASS</span>':'<span class="fail">Replay verification: FAIL</span><br>Max state error: '+DATA.maxStateError.toExponential(2);info.innerHTML='<b>Generation '+DATA.generation+'</b><br>Winner: <b>'+DATA.winnerLabel+'</b> ('+DATA.winnerSide+')<br>Battle time: '+DATA.winTime.toFixed(1)+' s<br>Replay time: '+f.t.toFixed(2)+' s<br>Step: '+i+' / '+DATA.steps+'<hr>Red soldiers: '+ra+'<br>Blue soldiers: '+ba+'<br>Red Commander HP: '+DATA.hp[i][0].toFixed(2)+'<br>Blue Commander HP: '+DATA.hp[i][1].toFixed(2)+'<hr><span class="attack">Red attacks: '+redAttackCount+'</span><br><span class="attack">Blue attacks: '+blueAttackCount+'</span><hr>Result: <b>'+DATA.resultText+'</b><br>'+verify;statusEl.textContent='Generation '+DATA.generation+' | Best Bout | '+f.t.toFixed(2)+' s';}
 document.getElementById('play').onclick=()=>playing=true;document.getElementById('pause').onclick=()=>playing=false;document.getElementById('reset').onclick=()=>{playing=false;replayTime=0;updateReplay();};document.querySelectorAll('button[data-speed]').forEach(b=>b.onclick=()=>speed=parseFloat(b.dataset.speed));timeline.oninput=()=>{playing=false;replayTime=parseInt(timeline.value,10)*DATA.dt;updateReplay();};
 addEventListener('resize',()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);});
 function animate(ts){requestAnimationFrame(animate);if(lastTs===null)lastTs=ts;const delta=Math.min(.05,(ts-lastTs)/1000);lastTs=ts;if(playing){replayTime+=delta*speed;if(replayTime>=DATA.steps*DATA.dt){replayTime=DATA.steps*DATA.dt;playing=false;}}updateReplay();controls.update();renderer.render(scene,camera);}updateReplay();requestAnimationFrame(animate);
}catch(err){showError(err);}
</script></body></html>'''

    html = html.replace("__MAXSTEP__", str(n_frames - 1)).replace("__PAYLOAD__", payload)
    if out_path is None:
        out_path = os.path.join(
            REPLAY_DIR,
            f"replay_generation_{generation:04d}.html",
        )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path, html


def _replay_file_compatible(path):
    try:
        with np.load(path, allow_pickle=False) as d:
            if "field_size" not in d:
                return False
            if "raw_obs_size" not in d and "obs_size" not in d:
                return False
            if "global_input_size" not in d:
                return False
            saved_raw = int(d["raw_obs_size"]) if "raw_obs_size" in d else int(d["obs_size"])
            return (
                abs(float(d["field_size"]) - FIELD_SIZE) < 1e-6
                and int(d["terrain_res"]) == TERRAIN_RES
                and saved_raw == RAW_OBS_SIZE
                and int(d["global_input_size"]) == GLOBAL_INPUT_SIZE
            )
    except Exception:
        return False


def show_replay(generation=None):
    files = sorted(
        glob.glob(os.path.join(BOUT_DIR, "generation_*_best_bout.npz")),
        key=generation_number,
    )
    compatible = [f for f in files if _replay_file_compatible(f)]
    if not compatible:
        raise FileNotFoundError(
            f"No compatible Best Bout found in {os.path.abspath(BOUT_DIR)}"
        )

    if generation is None:
        bout_path = compatible[-1]
    else:
        bout_path = os.path.join(
            BOUT_DIR,
            f"generation_{generation:04d}_best_bout.npz",
        )
        if not os.path.exists(bout_path):
            raise FileNotFoundError(bout_path)
        if not _replay_file_compatible(bout_path):
            raise ValueError(
                f"Generation {generation} Best Bout is incompatible "
                "with the current environment/network."
            )

    out_path, html = build_replay_html(bout_path)
    try:
        from IPython.display import display, HTML, IFrame
        try:
            display(
                IFrame(
                    src=os.path.relpath(out_path),
                    width="100%",
                    height=720,
                )
            )
        except Exception:
            display(HTML(html))
    except ImportError:
        pass
    return out_path

# ============================================================
# ENTRY POINT (Modal用に書き換え)
# ============================================================

# クラウドのGPUで実行するメイン処理
@app.function(image=app_image, gpu="A10G", timeout=86400, volumes={"/data": vol})
def run_training_remotely():
    # 既存のtrain関数をクラウド上で実行
    latest = train(n_generations=N_GENERATIONS, resume=True)
    
    try:
        replay_generation = LAST_COMPLETED_GENERATION
        path = (
            show_replay(replay_generation)
            if replay_generation is not None
            else show_replay()
        )
        print("Replay written to (Cloud):", os.path.abspath(path))
        
        # 学習完了後、生成されたHTMLファイルのテキストデータを返す（手元に持ってくるため）
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
            
    except FileNotFoundError as e:
        print("Replay skipped:", e)
        return None

# 手元のPC（ローカル）で動くエントリーポイント
@app.local_entrypoint()
def main():
    print("クラウドのGPUに接続して学習を開始します...")
    
    # クラウド側の関数を呼び出し、完了するまで待機（ログは手元に流れます）
    html_content = run_training_remotely.remote()
    
    # クラウドから受け取ったHTMLテキストを、手元のPCにファイルとして保存する
    if html_content:
        local_path = "latest_replay.html"
        with open(local_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"手元のPCにリプレイを保存しました: {local_path}")