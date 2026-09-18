# ============================================================
# JAX RTS : PPO + ELITE SELF-PLAY + 3D REPLAY
# Hierarchical local encoders + local combat rewards
# ============================================================

import os
import glob
import json
import time
import functools
import argparse
import re
import webbrowser
from pathlib import Path
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
TAG_ELITE_DIR = os.path.join(BASE_DIR, "tag_elite")
TAG_BOUT_DIR = os.path.join(BASE_DIR, "tag_bouts")
TAG_REPLAY_DIR = os.path.join(BASE_DIR, "tag_replay")

for d in (
    CHECKPOINT_DIR, ELITE_DIR, BOUT_DIR, REPLAY_DIR,
    TAG_ELITE_DIR, TAG_BOUT_DIR, TAG_REPLAY_DIR,
):
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
#   adjacent 8 terrain cells                            = 8
#   total = 25
#
# Commander:
#   x, z, vx, vz, hp, own, commander_flag, alive       = 8
#   nearest enemy: distance, sin(theta), cos(theta)     = 3
#   surrounding 8 cells: wall / not wall                = 8
#   total = 19
#
# Raw observation = terrain + 200*25 + 2*19 = 5294
# Hierarchical encoder output = terrain + 200*64 + 2*64 = 13184
# Self-attention is applied independently to each team's 100 soldiers,
# preserving the 100-action symmetry while keeping both teams' embeddings.
# The policy observes all 200 soldiers (100 own + 100 enemy).
# ------------------------------------------------------------

SOLDIER_FEATURES = 25
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
    # Production RTS walls: e.g. (-2.0, 0.0), (0.0, 2.0)
]

walls = jnp.asarray(WALL_LIST, dtype=jnp.float32).reshape((-1, 2))

# ------------------------------------------------------------
# Tag-game-only walls
# These walls are used only by the auxiliary "soldier vs enemy
# commander" chase task and its replay. They do not affect the
# main RTS environment.
# ------------------------------------------------------------
# Each entry is one 1x1 wall cell. Four entries make one 2x2 block.
# 16x16 field -> four 8x8 blocks, with a 2x2 wall at the center of each block.
TAG_WALL_LIST = [
]

tag_walls = jnp.asarray(TAG_WALL_LIST, dtype=jnp.float32).reshape((-1, 2))

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


def make_terrain_from_walls(wall_array):
    centers_1d = (
        jnp.arange(TERRAIN_RES, dtype=jnp.float32)
        - HALF_FIELD
        + 0.5
    )
    xx, zz = jnp.meshgrid(centers_1d, centers_1d)
    centers = jnp.stack([xx.reshape(-1), zz.reshape(-1)], axis=-1)

    def blocked(i):
        cell = centers[i]
        d = jnp.abs(cell[None, :] - wall_array)
        return jnp.any(jnp.all(d < 0.5, axis=1))

    t = jax.vmap(blocked)(jnp.arange(TERRAIN_SIZE))
    return t.astype(jnp.float32)


tag_terrain = make_terrain_from_walls(tag_walls)

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


def wall_blocked(x, z, radius, wall_array=None):
    """Vectorized wall collision test.

    Production uses ``walls`` by default. Tag passes ``tag_walls`` explicitly so
    its physics can use the Tag-specific wall layout without changing production.
    """
    wall_array = walls if wall_array is None else wall_array
    x2, z2, r2 = x[:, None], z[:, None], radius[:, None]
    wx, wz = wall_array[:, 0][None, :], wall_array[:, 1][None, :]
    cx = jnp.clip(x2, wx - 0.5, wx + 0.5)
    cz = jnp.clip(z2, wz - 0.5, wz + 0.5)
    dx, dz = x2 - cx, z2 - cz
    return jnp.any(dx * dx + dz * dz < r2 * r2, axis=1)


def step_one(state, red_action, blue_action, wall_array=None):
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
    desired_wall_hit = wall_blocked(nx, nz, radius, wall_array)
    boundary_collision = attempted_move & (~inside)
    wall_collision = attempted_move & (desired_wall_hit | (~inside))

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
    separated_blocked = wall_blocked(separated_x, separated_z, radius, wall_array)
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


def grid_8_features(local_x, local_z, terrain_local):
    """Return the eight neighboring terrain cells for each soldier.

    Order: [NW, N, NE, W, E, SW, S, SE]. Outside the field is treated as
    blocked, matching commander local-wall features.
    """
    dc = jnp.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=jnp.int32)
    dr = jnp.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=jnp.int32)

    def one(px, pz):
        col = jnp.floor(px + HALF_FIELD).astype(jnp.int32)
        row = jnp.floor(pz + HALF_FIELD).astype(jnp.int32)
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
        return jnp.where(inside, vals, 1.0)

    # This helper is used in two contexts:
    #   1) batched soldier features: (N,) -> (N, 8)
    #   2) single tag-game unit features: scalar -> (8,)
    # The previous implementation always vmapped, which crashed for the
    # scalar tag-game case with: "vmap ... rank should be at least 1".
    if local_x.ndim == 0:
        return one(local_x, local_z)
    return jax.vmap(one)(local_x, local_z)


def commander_wall_features(cmd_x, cmd_z, terrain_local):
    return grid_8_features(cmd_x, cmd_z, terrain_local)


def make_observation(state, perspective_team, nearest_idx=None, terrain_override=None):
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

    terrain_source = terrain if terrain_override is None else terrain_override
    terrain_2d = terrain_source.reshape(TERRAIN_RES, TERRAIN_RES)
    terrain_local_2d = jnp.where(blue, terrain_2d[:, ::-1], terrain_2d)
    terrain_local = terrain_local_2d.reshape(-1)
    soldier_local_terrain = grid_8_features(
        soldier_x, soldier_z, terrain_local
    )

    soldier_base = base[soldier_idx]
    soldier_nearest = nearest[soldier_idx]
    soldier_features = jnp.concatenate([
        soldier_base,
        soldier_nearest,
        own_cmd_features,
        enemy_cmd_features,
        soldier_local_terrain,
    ], axis=-1)

    cmd_idx = jnp.array([RED_COMMANDER_INDEX, BLUE_COMMANDER_INDEX], dtype=jnp.int32)
    cmd_base = base[cmd_idx]
    cmd_nearest = nearest[cmd_idx]
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


def make_observation_batch(state, perspective_team, terrain_override=None):
    nearest_idx = jax.vmap(nearest_enemy_indices)(state)
    return jax.vmap(
        make_observation,
        in_axes=(0, None, 0, None),
    )(state, perspective_team, nearest_idx, terrain_override)


def make_observation_pair_batch(state, terrain_override=None):
    nearest_idx = jax.vmap(nearest_enemy_indices)(state)
    red_obs = jax.vmap(
        make_observation, in_axes=(0, None, 0, None)
    )(state, 0.0, nearest_idx, terrain_override)
    blue_obs = jax.vmap(
        make_observation, in_axes=(0, None, 0, None)
    )(state, 1.0, nearest_idx, terrain_override)
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
# Micro branch (local skill) + Macro branch (global coordination)
# ============================================================

HIDDEN1 = 384
ACTOR_HIDDEN = 384
CRITIC_HIDDEN = 384

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
TARGET_KL = 0.30
PPO_ROLLOUT_STEPS = 512
ROLLOUT_STEPS = PPO_ROLLOUT_STEPS
BATCH_SIZE = ROLLOUT_STEPS * N_ENVS * 2
MINIBATCH_SIZE = BATCH_SIZE // MINIBATCHES
PPO_UPDATES_PER_GENERATION = 20
N_GENERATIONS = 10000
EVAL_GAMES_PER_SIDE = 32
EVAL_GAMES = EVAL_GAMES_PER_SIDE * 2
WARMUP_MAX_STEPS = MAX_STEPS

# Auxiliary tag-game training. This is intentionally lightweight so the A10G
# spends almost all time on the main PPO loop.
TAG_BATCH_SIZE = 512
# One standalone tag command performs this many optimizer updates.
# This is intentionally independent from PPO generations.
TAG_UPDATES_PER_RUN = 200
TAG_TRAIN_STEPS = 8  # retained as an internal/default diagnostic label
TAG_MAX_STEPS = 240
TAG_LEARNING_RATE = 2e-4
TAG_STEP_PENALTY = -0.002
TAG_CAPTURE_REWARD = 1.0
TAG_RUNNER_CAPTURE_REWARD = -1.0
TAG_CHASER_SPEED = INITIAL_SOLDIER_SPEED * 1.20
TAG_RUNNER_SPEED = INITIAL_SOLDIER_SPEED * 0.75
TAG_WALL_MARGIN = 0.02
TAG_REPLAY_SAMPLES_PER_RUN = 16

EVAL_Z_OFFSETS = jnp.array(
    [-0.72, -0.48, -0.24, 0.00, 0.24, 0.48, 0.72, 0.00],
    dtype=jnp.float32,
)

# Final action remains [dx, dz, attack] x 100 + [cmd_dx, cmd_dz].
# Macro soldier output is [x, z, attack, ratio] x 100.
# ratio is the macro-adoption ratio: 0 => all micro, 1 => all macro.
MACRO_SOLDIER_OUTPUT_SIZE = N_SOLDIERS_PER_TEAM * 4
MACRO_COMMANDER_OUTPUT_SIZE = 2
MICRO_SOLDIER_OUTPUT_SIZE = N_SOLDIERS_PER_TEAM * 3
MICRO_COMMANDER_OUTPUT_SIZE = 2


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
    return layer_norm(
        x + attended,
        params[f"{prefix}_ln_gamma"],
        params[f"{prefix}_ln_beta"],
    )


def init_attention_block(key, prefix):
    keys = random.split(key, 8)
    return {
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


def init_policy(key):
    keys = random.split(key, 40)
    p = {
        # Local encoders
        "Ws1": glorot_uniform(keys[0], (SOLDIER_FEATURES, LOCAL_HIDDEN_SIZE)),
        "bs1": jnp.zeros((LOCAL_HIDDEN_SIZE,)),
        "Ws2": glorot_uniform(keys[1], (LOCAL_HIDDEN_SIZE, LOCAL_EMBED_SIZE)),
        "bs2": jnp.zeros((LOCAL_EMBED_SIZE,)),
        "Wc": glorot_uniform(keys[2], (COMMANDER_FEATURES, LOCAL_EMBED_SIZE)),
        "bc": jnp.zeros((LOCAL_EMBED_SIZE,)),
        # Micro outputs: x,z,attack for Soldiers; x,z for Commander.
        "W_micro_s": random.normal(keys[3], (LOCAL_EMBED_SIZE, 3)) * 0.02,
        "b_micro_s": jnp.zeros((3,)),
        "W_micro_c": random.normal(keys[4], (LOCAL_EMBED_SIZE, 2)) * 0.02,
        "b_micro_c": jnp.zeros((2,)),
        # Smaller global network for macro decisions.
        "Wg1": glorot_uniform(keys[5], (GLOBAL_INPUT_SIZE, HIDDEN1)),
        "bg1": jnp.zeros((HIDDEN1,)),
        "W_actor_s": glorot_uniform(keys[6], (HIDDEN1 + LOCAL_EMBED_SIZE, ACTOR_HIDDEN)),
        "b_actor_s": jnp.zeros((ACTOR_HIDDEN,)),
        "W_actor_c": glorot_uniform(keys[7], (HIDDEN1 + LOCAL_EMBED_SIZE, ACTOR_HIDDEN)),
        "b_actor_c": jnp.zeros((ACTOR_HIDDEN,)),
        "W_critic": glorot_uniform(keys[8], (HIDDEN1, CRITIC_HIDDEN)),
        "b_critic": jnp.zeros((CRITIC_HIDDEN,)),
        # Macro soldier output: x,z,attack,ratio for each of 100 Soldiers.
        "Wa_s_macro": random.normal(keys[9], (ACTOR_HIDDEN, MACRO_SOLDIER_OUTPUT_SIZE)) * 0.01,
        "ba_s_macro": jnp.zeros((MACRO_SOLDIER_OUTPUT_SIZE,)),
        "Wa_c_macro": random.normal(keys[10], (ACTOR_HIDDEN, MACRO_COMMANDER_OUTPUT_SIZE)) * 0.01,
        "ba_c_macro": jnp.zeros((MACRO_COMMANDER_OUTPUT_SIZE,)),
        # Value branch
        "Wv": random.normal(keys[11], (CRITIC_HIDDEN, 1)) * 0.01,
        "bv": jnp.zeros((1,)),
        # Exploration is kept outside the macro/micro outputs.
        "logstd_s": jnp.full((N_SOLDIERS_PER_TEAM,), LOGSTD_INIT, dtype=jnp.float32),
        "logstd_c": jnp.array(LOGSTD_INIT, dtype=jnp.float32),
    }
    p.update(init_attention_block(keys[12], "attn1"))
    p.update(init_attention_block(keys[13], "attn2"))
    return p


def infer_blue_from_obs(obs, soldier_part):
    # In each local observation, the 100 own soldiers have own=1. Red
    # perspective stores them in the first 100 slots, Blue in the last 100.
    first_mean = jnp.mean(soldier_part[:, :N_SOLDIERS_PER_TEAM, 5], axis=1)
    return first_mean < 0.5


def raw_policy_outputs(params, obs, detach_encoders=False):
    terrain_part = obs[:, :TERRAIN_SIZE]
    soldier_start = TERRAIN_SIZE
    soldier_end = soldier_start + N_SOLDIERS_TOTAL * SOLDIER_FEATURES
    soldier_part = obs[:, soldier_start:soldier_end].reshape(
        obs.shape[0], N_SOLDIERS_TOTAL, SOLDIER_FEATURES
    )
    commander_part = obs[:, soldier_end:].reshape(
        obs.shape[0], N_COMMANDERS, COMMANDER_FEATURES
    )

    soldier_emb_pre = jnp.tanh(soldier_part @ params["Ws1"] + params["bs1"])
    soldier_emb_pre = jnp.tanh(soldier_emb_pre @ params["Ws2"] + params["bs2"])
    commander_emb = jnp.tanh(commander_part @ params["Wc"] + params["bc"])

    micro_s_all = soldier_emb_pre @ params["W_micro_s"] + params["b_micro_s"]
    micro_c_all = commander_emb @ params["W_micro_c"] + params["b_micro_c"]

    if detach_encoders:
        soldier_attention_in = lax.stop_gradient(soldier_emb_pre)
        commander_global = lax.stop_gradient(commander_emb)
    else:
        soldier_attention_in = soldier_emb_pre
        commander_global = commander_emb

    red_soldier_emb = soldier_attention_in[:, :N_SOLDIERS_PER_TEAM, :]
    blue_soldier_emb = soldier_attention_in[:, N_SOLDIERS_PER_TEAM:, :]

    red_soldier_emb = self_attention_block(red_soldier_emb, params, "attn1")
    blue_soldier_emb = self_attention_block(blue_soldier_emb, params, "attn1")
    red_soldier_emb = self_attention_block(red_soldier_emb, params, "attn2")
    blue_soldier_emb = self_attention_block(blue_soldier_emb, params, "attn2")
    soldier_global = jnp.concatenate([red_soldier_emb, blue_soldier_emb], axis=1)

    global_input = jnp.concatenate([
        terrain_part,
        soldier_global.reshape(obs.shape[0], -1),
        commander_global.reshape(obs.shape[0], -1),
    ], axis=-1)
    h_shared = jnp.tanh(global_input @ params["Wg1"] + params["bg1"])
    blue = infer_blue_from_obs(obs, soldier_part)

    # Macro branch is conditioned on both global context and the individual
    # unit's local embedding. This keeps each Soldier's macro action distinct
    # without restoring the much larger pre-redesign 512-wide head.
    own_soldier_global = jnp.where(
        blue[:, None, None],
        soldier_global[:, N_SOLDIERS_PER_TEAM:, :],
        soldier_global[:, :N_SOLDIERS_PER_TEAM, :],
    )
    soldier_macro_input = jnp.concatenate(
        [
            jnp.broadcast_to(
                h_shared[:, None, :],
                (obs.shape[0], N_SOLDIERS_PER_TEAM, HIDDEN1),
            ),
            own_soldier_global,
        ],
        axis=-1,
    )
    soldier_h_macro = jnp.tanh(
        soldier_macro_input @ params["W_actor_s"] + params["b_actor_s"]
    )

    own_commander_global = jnp.where(
        blue[:, None, None],
        commander_global[:, 1:2, :],
        commander_global[:, :1, :],
    )
    commander_macro_input = jnp.concatenate(
        [h_shared[:, None, :], own_commander_global], axis=-1
    )
    commander_h_macro = jnp.tanh(
        commander_macro_input @ params["W_actor_c"] + params["b_actor_c"]
    )
    critic_h = jnp.tanh(
        h_shared @ params["W_critic"] + params["b_critic"]
    )

    macro_s = (
        soldier_h_macro @ params["Wa_s_macro"] + params["ba_s_macro"]
    )
    macro_c_own = (
        commander_h_macro @ params["Wa_c_macro"] + params["ba_c_macro"]
    )[:, 0, :]

    own_micro_s = jnp.where(
        blue[:, None, None],
        micro_s_all[:, N_SOLDIERS_PER_TEAM:, :],
        micro_s_all[:, :N_SOLDIERS_PER_TEAM, :],
    )
    own_micro_c = jnp.where(
        blue[:, None],
        micro_c_all[:, 1, :],
        micro_c_all[:, 0, :],
    )

    value = (critic_h @ params["Wv"] + params["bv"])[..., 0]
    return own_micro_s, own_micro_c, macro_s, macro_c_own, value


def compose_final_policy(params, micro_s, micro_c, macro_s, macro_c):
    micro_vec = jnp.tanh(micro_s[..., :2])
    macro_vec = jnp.tanh(macro_s[..., :2])
    ratio = jax.nn.sigmoid(macro_s[..., 3])

    blended_vec = (1.0 - ratio[..., None]) * micro_vec + ratio[..., None] * macro_vec
    blended_norm = jnp.sqrt(jnp.sum(blended_vec * blended_vec, axis=-1, keepdims=True) + 1e-8)
    blended_vec = blended_vec / blended_norm
    soldier_mean = jnp.arctan2(blended_vec[..., 1], blended_vec[..., 0])

    micro_attack_p = jax.nn.sigmoid(micro_s[..., 2])
    macro_attack_p = jax.nn.sigmoid(macro_s[..., 2])
    attack_p = jnp.clip(
        (1.0 - ratio) * micro_attack_p + ratio * macro_attack_p,
        1e-5,
        1.0 - 1e-5,
    )
    soldier_attack_logit = jnp.log(attack_p) - jnp.log1p(-attack_p)

    # Commander has the same micro -> macro composition, but no ratio output
    # is exposed to the user; use a fixed 50/50 blend so the commander micro
    # branch is a real policy path during production as well.
    micro_c_vec = jnp.tanh(micro_c)
    macro_c_vec = jnp.tanh(macro_c)
    cmd_blend = 0.5 * micro_c_vec + 0.5 * macro_c_vec
    cmd_norm = jnp.sqrt(jnp.sum(cmd_blend * cmd_blend, axis=-1, keepdims=True) + 1e-8)
    cmd_blend = cmd_blend / cmd_norm
    commander_mean = jnp.arctan2(cmd_blend[..., 1], cmd_blend[..., 0])

    return {
        "soldier_mean": soldier_mean,
        "soldier_attack_logit": soldier_attack_logit,
        "soldier_ratio": ratio,
        "commander_mean": commander_mean,
        "logstd_s": jnp.clip(params["logstd_s"], LOGSTD_MIN, LOGSTD_MAX),
        "logstd_c": jnp.clip(params["logstd_c"], LOGSTD_MIN, LOGSTD_MAX),
        "micro_s": micro_s,
        "micro_c": micro_c,
        "macro_s": macro_s,
        "macro_c": macro_c,
    }


def policy_forward(params, obs, detach_encoders=False):
    micro_s, micro_c, macro_s, macro_c, value = raw_policy_outputs(
        params, obs, detach_encoders=detach_encoders
    )
    out = compose_final_policy(params, micro_s, micro_c, macro_s, macro_c)
    out["value"] = value
    return out


policy_forward_jit = jax.jit(policy_forward)


def wrap_angle(a):
    return (a + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def action_logprob_components(policy_out, local_action):
    angle_mean = policy_out["soldier_mean"]
    logstd = policy_out["logstd_s"][None, :]
    std = jnp.exp(logstd)

    soldier_action = local_action[:, :SOLDIER_ACTION_SIZE]
    dx = soldier_action[:, 0::3]
    dz = soldier_action[:, 1::3]
    at = soldier_action[:, 2::3]
    a = jnp.arctan2(dz, dx)
    diff = wrap_angle(a - angle_mean)
    lp_angle = (
        -0.5 * (diff / std) ** 2
        - logstd
        - 0.5 * jnp.log(2.0 * jnp.pi)
    )

    logits = policy_out["soldier_attack_logit"]
    lp_attack = (
        at * (-jnp.logaddexp(0.0, -logits))
        + (1.0 - at) * (-jnp.logaddexp(0.0, logits))
    )
    soldier_lp = lp_angle + lp_attack

    cmd_mean = policy_out["commander_mean"]
    cmd_logstd = policy_out["logstd_c"]
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


def action_logprob(policy_out, local_action):
    _, _, team_lp = action_logprob_components(policy_out, local_action)
    return team_lp


def policy_entropy(policy_out):
    logstd = policy_out["logstd_s"][None, :]
    e_angle = jnp.sum(
        logstd + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e), axis=1
    )
    p = jax.nn.sigmoid(policy_out["soldier_attack_logit"])
    e_attack = -(
        p * jnp.log(p + 1e-8)
        + (1.0 - p) * jnp.log(1.0 - p + 1e-8)
    )
    e_attack = jnp.sum(e_attack, axis=1)
    e_cmd = policy_out["logstd_c"] + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e)
    return e_angle + e_attack + e_cmd


def sample_action(params, obs, key):
    out = policy_forward(params, obs)
    B = obs.shape[0]
    k_noise_soldier, k_attack, k_noise_cmd = random.split(key, 3)

    mean = out["soldier_mean"]
    logstd = out["logstd_s"][None, :]
    std = jnp.exp(logstd)
    angle = mean + std * random.normal(
        k_noise_soldier, (B, N_SOLDIERS_PER_TEAM)
    )
    dx, dz = jnp.cos(angle), jnp.sin(angle)

    prob = jax.nn.sigmoid(out["soldier_attack_logit"])
    at = random.bernoulli(k_attack, prob).astype(jnp.float32)

    cmd_mean = out["commander_mean"]
    cmd_std = jnp.exp(out["logstd_c"])
    cmd_angle = cmd_mean + cmd_std * random.normal(k_noise_cmd, (B,))
    cmd_dx, cmd_dz = jnp.cos(cmd_angle), jnp.sin(cmd_angle)

    la = jnp.zeros((B, ACTION_SIZE), dtype=jnp.float32)
    la = la.at[:, 0:SOLDIER_ACTION_SIZE:3].set(dx)
    la = la.at[:, 1:SOLDIER_ACTION_SIZE:3].set(dz)
    la = la.at[:, 2:SOLDIER_ACTION_SIZE:3].set(at)
    la = la.at[:, SOLDIER_ACTION_SIZE].set(cmd_dx)
    la = la.at[:, SOLDIER_ACTION_SIZE + 1].set(cmd_dz)

    soldier_lp, commander_lp, team_lp = action_logprob_components(out, la)
    return la, soldier_lp, commander_lp, team_lp, out["value"]


def deterministic_local_action(params, obs):
    out = policy_forward(params, obs)
    mean = out["soldier_mean"]
    dx, dz = jnp.cos(mean), jnp.sin(mean)
    at = (jax.nn.sigmoid(out["soldier_attack_logit"]) >= 0.5).astype(jnp.float32)

    cmd_mean = out["commander_mean"]
    cmd_dx = jnp.cos(cmd_mean)
    cmd_dz = jnp.sin(cmd_mean)

    la = jnp.zeros((obs.shape[0], ACTION_SIZE), dtype=jnp.float32)
    la = la.at[:, 0:SOLDIER_ACTION_SIZE:3].set(dx)
    la = la.at[:, 1:SOLDIER_ACTION_SIZE:3].set(dz)
    la = la.at[:, 2:SOLDIER_ACTION_SIZE:3].set(at)
    la = la.at[:, SOLDIER_ACTION_SIZE].set(cmd_dx)
    la = la.at[:, SOLDIER_ACTION_SIZE + 1].set(cmd_dz)
    return la, out["value"]


def deterministic_world_action_batch(params, state, team):
    obs = make_observation_batch(state, team)
    la, _ = deterministic_local_action(params, obs)
    return local_to_world_action(la, team)


def deterministic_world_action_single(params, state, team):
    obs = make_observation(state, team)[None, :]
    la, _ = deterministic_local_action(params, obs)
    return local_to_world_action(la, team)[0]


# ============================================================
# PART 3 : OPTIMIZER / PPO + AUXILIARY TAG TRAINING
# ============================================================

optimizer = optax.chain(
    optax.clip_by_global_norm(0.5),
    optax.adam(LEARNING_RATE),
)
tag_optimizer = optax.chain(
    optax.clip_by_global_norm(0.5),
    optax.adam(TAG_LEARNING_RATE),
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


SOLDIER_LOCAL_KEYS = (
    "Ws1", "bs1", "Ws2", "bs2",
    "W_micro_s", "b_micro_s",
    "attn1_Wq", "attn1_bq", "attn1_Wk", "attn1_bk",
    "attn1_Wv", "attn1_bv", "attn1_Wo", "attn1_bo",
    "attn1_ln_gamma", "attn1_ln_beta",
    "attn2_Wq", "attn2_bq", "attn2_Wk", "attn2_bk",
    "attn2_Wv", "attn2_bv", "attn2_Wo", "attn2_bo",
    "attn2_ln_gamma", "attn2_ln_beta",
    "W_actor_s", "b_actor_s", "Wa_s_macro", "ba_s_macro",
)
COMMANDER_LOCAL_KEYS = (
    "Wc", "bc", "W_micro_c", "b_micro_c",
    "W_actor_c", "b_actor_c", "Wa_c_macro", "ba_c_macro",
)
TEAM_KEYS = (
    "Wg1", "bg1",
    "W_actor_s", "b_actor_s", "Wa_s_macro", "ba_s_macro",
    "W_actor_c", "b_actor_c", "Wa_c_macro", "ba_c_macro",
    "W_critic", "b_critic", "Wv", "bv",
    "logstd_s", "logstd_c",
)
TAG_ENCODER_KEYS = (
    "Ws1", "bs1", "Ws2", "bs2",
    "W_micro_s", "b_micro_s",
    "Wc", "bc", "W_micro_c", "b_micro_c",
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
    team_out = policy_forward(params, obs, detach_encoders=True)
    team_soldier_lp, team_commander_lp, team_lp = action_logprob_components(
        team_out, local_actions
    )

    local_out = policy_forward(params, obs, detach_encoders=False)
    soldier_lp, commander_lp, _ = action_logprob_components(
        local_out, local_actions
    )

    soldier_ratio = jnp.exp(soldier_lp - old_soldier_log_prob)
    soldier_unclipped = soldier_ratio * soldier_advantages
    soldier_clipped = (
        jnp.clip(soldier_ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS)
        * soldier_advantages
    )
    soldier_policy_loss = -jnp.mean(jnp.minimum(soldier_unclipped, soldier_clipped))

    commander_ratio = jnp.exp(commander_lp - old_commander_log_prob)
    commander_unclipped = commander_ratio * commander_advantages
    commander_clipped = (
        jnp.clip(commander_ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS)
        * commander_advantages
    )
    commander_policy_loss = -jnp.mean(jnp.minimum(commander_unclipped, commander_clipped))

    team_ratio = jnp.exp(team_lp - old_team_log_prob)
    team_unclipped = team_ratio * team_advantages
    team_clipped = (
        jnp.clip(team_ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS)
        * team_advantages
    )
    team_policy_loss = -jnp.mean(jnp.minimum(team_unclipped, team_clipped))

    entropy = jnp.mean(policy_entropy(team_out))
    soldier_logstd = team_out["logstd_s"]
    commander_logstd = team_out["logstd_c"]
    logstd_reg = (
        0.5 * jnp.mean((soldier_logstd - LOGSTD_TARGET) ** 2)
        + 0.5 * (commander_logstd - LOGSTD_TARGET) ** 2
    )

    value_loss = 0.5 * jnp.mean((returns - team_out["value"]) ** 2)
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
        "logstd_mean": jnp.mean(soldier_logstd),
        "commander_logstd_mean": commander_logstd,
        "logstd_reg": logstd_reg,
        "macro_ratio_mean": jnp.mean(local_out["soldier_ratio"]),
        "micro_attack_prob": jnp.mean(jax.nn.sigmoid(local_out["micro_s"][..., 2])),
        "macro_attack_prob": jnp.mean(jax.nn.sigmoid(local_out["macro_s"][..., 2])),
        "approx_kl": jnp.mean(old_team_log_prob - team_lp),
        "clip_fraction": jnp.mean((jnp.abs(team_ratio - 1.0) > CLIP_EPS).astype(jnp.float32)),
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
        return ppo_loss_parts(
            p, obs, local_actions,
            old_soldier_log_prob, old_commander_log_prob, old_team_log_prob,
            soldier_advantages, commander_advantages, team_advantages, returns, ent_coef,
        )["soldier_policy_loss"]

    def commander_loss_fn(p):
        return ppo_loss_parts(
            p, obs, local_actions,
            old_soldier_log_prob, old_commander_log_prob, old_team_log_prob,
            soldier_advantages, commander_advantages, team_advantages, returns, ent_coef,
        )["commander_policy_loss"]

    def team_loss_fn(p):
        m = ppo_loss_parts(
            p, obs, local_actions,
            old_soldier_log_prob, old_commander_log_prob, old_team_log_prob,
            soldier_advantages, commander_advantages, team_advantages, returns, ent_coef,
        )
        return m["team_total_loss"], m

    _, soldier_grads = jax.value_and_grad(soldier_loss_fn)(params)
    _, commander_grads = jax.value_and_grad(commander_loss_fn)(params)
    (_, metrics), team_grads = jax.value_and_grad(team_loss_fn, has_aux=True)(params)

    soldier_grads = masked_tree(soldier_grads, SOLDIER_LOCAL_KEYS)
    commander_grads = masked_tree(commander_grads, COMMANDER_LOCAL_KEYS)
    team_grads = masked_tree(team_grads, TEAM_KEYS)

    def grad_l2(tree):
        leaves = jax.tree_util.tree_leaves(tree)
        return jnp.sqrt(sum(jnp.sum(x * x) for x in leaves) + jnp.float32(1e-12))

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
# TAG GAME AUXILIARY TASK
# ============================================================


def tag_wall_blocked(x, z, radius):
    """Return a scalar bool for scalar tag-game coordinates."""
    x = jnp.asarray(x)
    z = jnp.asarray(z)
    radius = jnp.asarray(radius)
    wx = tag_walls[:, 0]
    wz = tag_walls[:, 1]
    cx = jnp.clip(x, wx - 0.5, wx + 0.5)
    cz = jnp.clip(z, wz - 0.5, wz + 0.5)
    dx, dz = x - cx, z - cz
    blocked = jnp.any(
        dx * dx + dz * dz < (radius + TAG_WALL_MARGIN) ** 2,
        axis=0,
    )
    return jnp.asarray(blocked, dtype=jnp.bool_)
















def tag_micro_outputs(params, soldier_features, commander_features):
    sh = jnp.tanh(soldier_features @ params["Ws1"] + params["bs1"])
    semb = jnp.tanh(sh @ params["Ws2"] + params["bs2"])
    ch = jnp.tanh(commander_features @ params["Wc"] + params["bc"])
    return (
        semb @ params["W_micro_s"] + params["b_micro_s"],
        ch @ params["W_micro_c"] + params["b_micro_c"],
    )






@jax.jit
def tag_update_minibatch(params, opt_state, soldier_features, commander_features, target_chase, target_run, attack_target):
    def loss_fn(p):
        return tag_update_loss(
            p, soldier_features, commander_features,
            target_chase, target_run, attack_target,
        )
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    del loss
    grads = masked_tree(grads, TAG_ENCODER_KEYS)
    updates, opt_state = tag_optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    metrics = dict(metrics)
    metrics["tag_grad_norm"] = jnp.sqrt(
        sum(jnp.sum(x * x) for x in jax.tree_util.tree_leaves(grads)) + 1e-12
    )
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
    red_last_v = policy_forward(params, f_red_obs)["value"]
    blue_last_v = policy_forward(params, f_blue_obs)["value"]

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
        data["entropy_schedule_start_generation"] = np.array(entropy_schedule_start_generation, dtype=np.int32)
    if entropy_schedule_total_updates is not None:
        data["entropy_schedule_total_updates"] = np.array(entropy_schedule_total_updates, dtype=np.int32)
    np.savez(path, **data)
    return path


def required_policy_shapes():
    required_shapes = {
        "Ws1": (SOLDIER_FEATURES, LOCAL_HIDDEN_SIZE),
        "bs1": (LOCAL_HIDDEN_SIZE,),
        "Ws2": (LOCAL_HIDDEN_SIZE, LOCAL_EMBED_SIZE),
        "bs2": (LOCAL_EMBED_SIZE,),
        "Wc": (COMMANDER_FEATURES, LOCAL_EMBED_SIZE),
        "bc": (LOCAL_EMBED_SIZE,),
        "W_micro_s": (LOCAL_EMBED_SIZE, 3),
        "b_micro_s": (3,),
        "W_micro_c": (LOCAL_EMBED_SIZE, 2),
        "b_micro_c": (2,),
        "Wg1": (GLOBAL_INPUT_SIZE, HIDDEN1),
        "bg1": (HIDDEN1,),
        "W_actor_s": (HIDDEN1 + LOCAL_EMBED_SIZE, ACTOR_HIDDEN),
        "b_actor_s": (ACTOR_HIDDEN,),
        "W_actor_c": (HIDDEN1 + LOCAL_EMBED_SIZE, ACTOR_HIDDEN),
        "b_actor_c": (ACTOR_HIDDEN,),
        "W_critic": (HIDDEN1, CRITIC_HIDDEN),
        "b_critic": (CRITIC_HIDDEN,),
        "Wa_s_macro": (ACTOR_HIDDEN, MACRO_SOLDIER_OUTPUT_SIZE),
        "ba_s_macro": (MACRO_SOLDIER_OUTPUT_SIZE,),
        "Wa_c_macro": (ACTOR_HIDDEN, MACRO_COMMANDER_OUTPUT_SIZE),
        "ba_c_macro": (MACRO_COMMANDER_OUTPUT_SIZE,),
        "Wv": (CRITIC_HIDDEN, 1),
        "bv": (1,),
        "logstd_s": (N_SOLDIERS_PER_TEAM,),
        "logstd_c": (),
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


def required_legacy_shapes():
    required_shapes = {
        "Ws1": (17, LOCAL_HIDDEN_SIZE),
        "bs1": (LOCAL_HIDDEN_SIZE,),
        "Ws2": (LOCAL_HIDDEN_SIZE, LOCAL_EMBED_SIZE),
        "bs2": (LOCAL_EMBED_SIZE,),
        "Wc": (COMMANDER_FEATURES, LOCAL_EMBED_SIZE),
        "bc": (LOCAL_EMBED_SIZE,),
        "Wg1": (13184, 512),
        "bg1": (512,),
        "W_actor_s": (512, 512),
        "b_actor_s": (512,),
        "W_actor_c": (512, 512),
        "b_actor_c": (512,),
        "W_critic": (512, 512),
        "b_critic": (512,),
        "Wa_s": (512, 300),
        "ba_s": (300,),
        "Wa_c": (512, 2),
        "ba_c": (2,),
        "Wv": (512, 1),
        "bv": (1,),
    }
    for prefix in ("attn1", "attn2"):
        required_shapes.update({
            f"{prefix}_Wq": (64, 64), f"{prefix}_bq": (64,),
            f"{prefix}_Wk": (64, 64), f"{prefix}_bk": (64,),
            f"{prefix}_Wv": (64, 64), f"{prefix}_bv": (64,),
            f"{prefix}_Wo": (64, 64), f"{prefix}_bo": (64,),
            f"{prefix}_ln_gamma": (64,), f"{prefix}_ln_beta": (64,),
        })
    return required_shapes


def _shapes_match(params, shapes):
    if set(params.keys()) != set(shapes.keys()):
        return False
    return all(tuple(np.asarray(params[k]).shape) == shape for k, shape in shapes.items())


def policy_params_compatible(params):
    return _shapes_match(params, required_policy_shapes()) or _shapes_match(params, required_legacy_shapes())


def migrate_policy_params(params):
    if _shapes_match(params, required_policy_shapes()):
        return params, False
    if not _shapes_match(params, required_legacy_shapes()):
        raise ValueError("Incompatible policy parameter shapes")

    # Start from fresh micro/macro architecture and retain the old learned
    # local encoder, attention, global trunk, critic, and shared actor where
    # dimensions overlap. The extra 8 soldier terrain inputs are freshly
    # initialized, while old 17-feature weights are copied intact.
    seed = random.PRNGKey(1234567)
    new_params = init_policy(seed)
    new_params["Ws1"] = new_params["Ws1"].at[:17, :].set(params["Ws1"])
    for k in ("bs1", "Ws2", "bs2", "Wc", "bc"):
        new_params[k] = params[k].copy()
    for prefix in ("attn1", "attn2"):
        for suffix in ("Wq","bq","Wk","bk","Wv","bv","Wo","bo","ln_gamma","ln_beta"):
            k = f"{prefix}_{suffix}"
            new_params[k] = params[k].copy()

    # Shrink old 512-wide trunks into the new 384-wide trunks by taking the
    # corresponding top-left blocks.
    h = HIDDEN1
    ah = ACTOR_HIDDEN
    ch = CRITIC_HIDDEN
    new_params["Wg1"] = new_params["Wg1"].at[:, :h].set(params["Wg1"][:, :h])
    new_params["bg1"] = new_params["bg1"].at[:h].set(params["bg1"][:h])
    for actor in ("W_actor_s", "W_actor_c"):
        new_params[actor] = new_params[actor].at[:h, :ah].set(params[actor][:h, :ah])
    new_params["W_critic"] = new_params["W_critic"].at[:h, :ch].set(params["W_critic"][:h, :ch])
    for actor_b in ("b_actor_s", "b_actor_c", "b_critic"):
        new_params[actor_b] = new_params[actor_b].at[:ah].set(params[actor_b][:ah])
    new_params["Wv"] = new_params["Wv"].at[:ch, :].set(params["Wv"][:ch, :])
    new_params["bv"] = params["bv"].copy()

    # Initialize the new macro heads. Preserve old angle/attack heads in the
    # corresponding macro x/z/attack positions, and set ratio to mild micro
    # preference so the inherited policy is not discarded immediately.
    old_ws = params["Wa_s"][:ah, :]
    old_bs = params["ba_s"][:]
    for i in range(N_SOLDIERS_PER_TEAM):
        old_base = 3 * i
        new_base = 4 * i
        # Old output is [angle_mean, angle_logstd, attack]. We map angle_mean
        # into a directional x component and keep attack. z starts at zero.
        new_params["Wa_s_macro"] = new_params["Wa_s_macro"].at[:, new_base].set(old_ws[:, old_base])
        new_params["Wa_s_macro"] = new_params["Wa_s_macro"].at[:, new_base + 2].set(old_ws[:, old_base + 2])
        new_params["ba_s_macro"] = new_params["ba_s_macro"].at[new_base].set(old_bs[old_base])
        new_params["ba_s_macro"] = new_params["ba_s_macro"].at[new_base + 2].set(old_bs[old_base + 2])
        new_params["ba_s_macro"] = new_params["ba_s_macro"].at[new_base + 3].set(-2.1972246)  # sigmoid ~= 0.10

    old_cmd_w = params["Wa_c"][:ah, :]
    old_cmd_b = params["ba_c"]
    new_params["Wa_c_macro"] = new_params["Wa_c_macro"].at[:,:].set(old_cmd_w)
    new_params["ba_c_macro"] = new_params["ba_c_macro"].at[:].set(old_cmd_b)

    old_s_logstd = params["ba_s"]
    new_params["logstd_s"] = new_params["logstd_s"].at[:].set(old_s_logstd[1::3])
    new_params["logstd_c"] = old_cmd_b[1]
    return new_params, True


def saved_policy_file_compatible(path):
    try:
        with np.load(path, allow_pickle=False) as d:
            params = {k[len("param_"):]: d[k] for k in d.files if k.startswith("param_")}
        return policy_params_compatible(params)
    except Exception:
        return False


def load_params(path):
    d = np.load(path, allow_pickle=False)
    raw_params = {k[len("param_"):]: jnp.asarray(d[k]) for k in d.files if k.startswith("param_")}
    params, migrated = migrate_policy_params(raw_params)
    if migrated:
        print(f"Migrated legacy checkpoint into micro/macro architecture: {os.path.basename(path)}")
    meta = {}
    if "metadata_json" in d.files:
        try:
            meta = json.loads(str(d["metadata_json"]))
        except Exception:
            meta = {}
    return params, meta


def load_checkpoint(path):
    d = np.load(path, allow_pickle=False)
    raw_params = {k[len("param_"):]: jnp.asarray(d[k]) for k in d.files if k.startswith("param_")}
    params, migrated = migrate_policy_params(raw_params)
    if migrated:
        print("Legacy checkpoint architecture detected; migrated parameters and reinitialized PPO optimizer state.")
        opt_state = optimizer.init(params)
    else:
        template = optimizer.init(params)
        template_leaves = jax.tree_util.tree_leaves(template)
        treedef = jax.tree_util.tree_structure(template)
        n = int(d["opt_n_leaves"])
        if n != len(template_leaves):
            raise ValueError(f"Incompatible optimizer state in checkpoint: {path}")
        leaves = []
        for i, template_leaf in enumerate(template_leaves):
            leaf = jnp.asarray(d[f"opt_{i}"])
            if tuple(leaf.shape) != tuple(template_leaf.shape):
                raise ValueError(f"Incompatible optimizer leaf {i} in checkpoint: {path}")
            leaves.append(leaf)
        opt_state = jax.tree_util.tree_unflatten(treedef, leaves)

    generation = int(d["generation"])
    ppo_index = int(d["ppo_index"])
    elite_path = str(d["elite_path"]) if "elite_path" in d.files else ""
    master_key = random.wrap_key_data(jnp.asarray(d["master_key"]))
    schedule_start_generation = int(d["entropy_schedule_start_generation"]) if "entropy_schedule_start_generation" in d.files else generation
    schedule_total_updates = int(d["entropy_schedule_total_updates"]) if "entropy_schedule_total_updates" in d.files else None
    return params, opt_state, generation, ppo_index, elite_path, master_key, schedule_start_generation, schedule_total_updates


def generation_number(path):
    m = re.search(r"generation_(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


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
    files_sorted = sorted(
        files,
        key=lambda f: (
            generation_number(f),
            1 if "_micro_macro_migrated" in os.path.basename(f) else 0,
        ),
        reverse=True,
    )
    for f in files_sorted:
        if saved_policy_file_compatible(f):
            return f
    if files:
        print(f"Ignoring incompatible Elite file(s); starting from a migrated/new architecture.")
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
        print(f"Ignoring {len(incompatible)} incompatible checkpoint(s).")
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


def tag_policy_number(path):
    try:
        name = os.path.basename(path)
        tail = name.split("tag_policy_", 1)[1]
        tail = tail.split("_bout_", 1)[0]
        tail = tail.split(".npz", 1)[0]
        return int(tail)
    except Exception:
        return -1


def find_latest_tag_policy():
    files = glob.glob(os.path.join(TAG_ELITE_DIR, "tag_policy_*.npz"))
    if not files:
        return None
    # Policy numbers are the primary ordering key.  mtime breaks ties so a
    # Production->Tag synchronization and a Tag-evolution result can safely
    # share the same numeric range.
    return max(files, key=lambda f: (tag_policy_number(f), os.path.getmtime(f)))


def _read_tag_policy_sparse(path):
    """Load a sparse Tag policy file without pretending it is a full Elite."""
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


def _merge_tag_policy_into_production(production_params, tag_path):
    """Overlay only TAG_ENCODER_KEYS from a sparse Tag policy onto a full Elite."""
    raw, meta = _read_tag_policy_sparse(tag_path)
    missing = [k for k in TAG_ENCODER_KEYS if k not in raw]
    if missing:
        raise ValueError(
            f"Tag policy {os.path.basename(tag_path)} is missing Tag parameter(s): {missing}"
        )
    merged = {k: jnp.array(v) for k, v in production_params.items()}
    for k in TAG_ENCODER_KEYS:
        merged[k] = raw[k]
    return merged, meta


def _next_tag_policy_number():
    files = glob.glob(os.path.join(TAG_ELITE_DIR, "tag_policy_*.npz"))
    if not files:
        return 1
    return max(tag_policy_number(f) for f in files) + 1


def save_tag_policy_snapshot(params, production_generation, production_elite_path, source="production_elite_sync"):
    """Persist only the Tag-mutated subset of a Production policy.

    This is deliberately independent of the large full Elite files.  The
    resulting sparse compressed file can later be merged back onto whatever
    Production Elite is current.
    """
    os.makedirs(TAG_ELITE_DIR, exist_ok=True)
    number = _next_tag_policy_number()
    path = os.path.join(TAG_ELITE_DIR, f"tag_policy_{number:06d}.npz")
    data = {}
    for k in TAG_ENCODER_KEYS:
        if k not in params:
            raise KeyError(f"Missing Tag parameter key while syncing Production Elite: {k}")
        data[f"param_{k}"] = np.asarray(params[k])
    metadata = {
        "format": "tag_sparse_v2",
        "source": source,
        "generation": int(production_generation),
        "production_generation": int(production_generation),
        "production_elite": os.path.basename(str(production_elite_path)),
        "saved_parameter_keys": list(TAG_ENCODER_KEYS),
    }
    data["metadata_json"] = np.array(json.dumps(metadata, ensure_ascii=False))
    np.savez_compressed(path, **data)
    return path


def choose_tag_seed_policy(production_elite):
    """Return the newer of the full Production Elite and sparse Tag Elite."""
    tag_path = find_latest_tag_policy()
    if tag_path is None:
        return production_elite, "production Elite"
    if production_elite is None:
        return tag_path, "Tag Elite"
    try:
        if os.path.getmtime(tag_path) > os.path.getmtime(production_elite):
            return tag_path, "Tag Elite"
    except OSError:
        pass
    return production_elite, "production Elite"


def load_tag_seed_params(seed_path, production_elite):
    """Load either a full Production Elite or a sparse Tag overlay."""
    if os.path.abspath(seed_path) == os.path.abspath(production_elite):
        return load_params(seed_path), {}

    raw, meta = _read_tag_policy_sparse(seed_path)
    raw_keys = set(raw)
    full_keys = set(required_policy_shapes().keys())

    # New Tag format: only TAG_ENCODER_KEYS are stored.
    if set(TAG_ENCODER_KEYS).issubset(raw_keys) and not full_keys.issubset(raw_keys):
        base_params, _ = load_params(production_elite)
        return _merge_tag_policy_into_production(base_params, seed_path)

    # Backward compatibility for old full-size Tag policy files.
    params, meta2 = load_params(seed_path)
    if meta2:
        meta.update(meta2)
    return params, meta




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
                "architecture": "micro_macro_25feat_64_attention2_tag_aux_role_separated",
                "ppo_rollout_steps": PPO_ROLLOUT_STEPS,
            },
        )
        print("Created Generation 0 Elite from fresh random initialization.")
    else:
        print("Existing Elite found:", os.path.basename(latest_elite))

        # If the latest available Elite is from the old architecture, migrate it
        # once and persist that migrated parameter tree. Otherwise every new
        # generation would repeatedly restart from random micro heads and the
        # tag-trained encoder progress would be discarded whenever the Candidate
        # failed to replace Elite.
        try:
            with np.load(latest_elite, allow_pickle=False) as d_probe:
                raw_probe = {
                    k[len("param_"):]: d_probe[k]
                    for k in d_probe.files
                    if k.startswith("param_")
                }
            if _shapes_match(raw_probe, required_legacy_shapes()):
                migrated_params, _ = migrate_policy_params(
                    {k: jnp.asarray(v) for k, v in raw_probe.items()}
                )
                migrated_path = os.path.join(
                    ELITE_DIR,
                    f"generation_{generation_number(latest_elite):04d}_micro_macro_migrated.npz",
                )
                latest_elite = save_params(
                    migrated_path,
                    migrated_params,
                    {
                        "source": "legacy_elite_migrated_once",
                        "base_elite": os.path.basename(latest_elite),
                        "architecture": "micro_macro_25feat_64_attention2_tag_aux_role_separated",
                    },
                )
                try:
                    synced_tag_path = save_tag_policy_snapshot(
                        migrated_params,
                        generation_number(latest_elite),
                        latest_elite,
                        source="production_elite_migration_sync",
                    )
                    print(f"Tag policy synced    : {os.path.basename(synced_tag_path)}")
                except Exception as exc:
                    print(f"Tag policy sync failed: {exc}")
                print("Persisted migrated Elite:", os.path.basename(latest_elite))
        except Exception as exc:
            print(f"Elite migration persistence skipped: {exc}")

    if entropy_schedule_start_generation is None:
        entropy_schedule_start_generation = start_generation

    # Standalone tag training is deliberately outside PPO.  If a newer Tag
    # Elite exists, consume it only as the initial Candidate for this run.
    tag_seed_path, tag_seed_source = choose_tag_seed_policy(latest_elite)
    use_tag_seed = (
        resumed_params is None
        and os.path.abspath(tag_seed_path) != os.path.abspath(latest_elite)
    )
    if use_tag_seed:
        print(f"Initial Candidate seed : {os.path.basename(tag_seed_path)} ({tag_seed_source})")

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
    print("Micro Soldier      : 64 -> [x,z,attack]")
    print("Micro Commander    : 64 -> [x,z]")
    print(f"Self-attention     : {N_SOLDIERS_PER_TEAM} soldiers/team, {ATTENTION_HEADS} heads x 2 layers")
    print(f"Global network     : {GLOBAL_INPUT_SIZE} -> {HIDDEN1}")
    print(f"Soldier macro       : [global {HIDDEN1} + local {LOCAL_EMBED_SIZE}] -> {ACTOR_HIDDEN} -> 4  ([x,z,attack,ratio] x {N_SOLDIERS_PER_TEAM})")
    print(f"Commander macro     : [global {HIDDEN1} + local {LOCAL_EMBED_SIZE}] -> {ACTOR_HIDDEN} -> {MACRO_COMMANDER_OUTPUT_SIZE}")
    print("Final soldier act   : normalized((1-ratio)*micro + ratio*macro)")
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
    print("                     Team win/loss -> encoders BLOCKED; micro heads are masked")
    print("Tag training        : external command only (not run inside PPO)")
    print(f"Tag soldiers/team   : {TAG_SOLDIERS_PER_TEAM}")
    print(f"Tag updates/run     : {TAG_UPDATES_PER_RUN}, batch {TAG_BATCH_SIZE}")
    print(f"Tag walls           : {len(TAG_WALL_LIST)} entries")
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
            seed_path = tag_seed_path if (use_tag_seed and generation == start_generation) else latest_elite
            seed_params, _ = load_tag_seed_params(seed_path, latest_elite)
            candidate_params = jax.tree_util.tree_map(
                lambda a: a.copy(), seed_params
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

            # Keep the Tag side synchronized with every NEW Production Elite.
            # Only Tag-mutated encoder/Micro parameters are copied, so the file
            # remains tiny and can later be merged onto the current full Elite.
            try:
                synced_tag_path = save_tag_policy_snapshot(
                    candidate_params,
                    generation,
                    latest_elite,
                    source="production_elite_sync",
                )
                print(f"Tag policy synced    : {os.path.basename(synced_tag_path)}")
            except Exception as exc:
                print(f"Tag policy sync failed: {exc}")

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

    is_tag_replay = ("tag_policy_red" in d) or ("tag_policy_blue" in d) or ("tag_soldiers_per_team" in d)
    if is_tag_replay:
        if "tag_walls" in d:
            saved_walls = d["tag_walls"].astype(np.float32).reshape((-1, 2)).tolist()
        else:
            saved_walls = [list(p) for p in TAG_WALL_LIST]
    else:
        saved_walls = [list(p) for p in WALL_LIST]

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
            "walls": saved_walls,
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
# TAG GAME REPLAY
# ============================================================










def tag_bout_filename(policy_number, bout_number):
    return os.path.join(
        TAG_BOUT_DIR,
        f"tag_policy_{int(policy_number):06d}_bout_{int(bout_number):03d}.npz",
    )


def find_latest_tag_bout(policy_number=None, bout_number=None):
    if policy_number is None:
        files = sorted(
            glob.glob(os.path.join(TAG_BOUT_DIR, "tag_policy_*_bout_*.npz")),
            key=lambda p: (tag_policy_number(p), os.path.basename(p)),
        )
        if files:
            return files[-1]
        legacy = sorted(
            glob.glob(os.path.join(TAG_BOUT_DIR, "tag_policy_*_best_bout.npz")),
            key=tag_policy_number,
        )
        return legacy[-1] if legacy else None

    if bout_number is None:
        bout_number = 0
    path = tag_bout_filename(policy_number, bout_number)
    if os.path.exists(path):
        return path

    if int(bout_number) == 0:
        legacy = os.path.join(
            TAG_BOUT_DIR, f"tag_policy_{int(policy_number):06d}_best_bout.npz"
        )
        if os.path.exists(legacy):
            return legacy
    return None


def show_tag_replay(policy_number=None, bout_number=None):
    path = find_latest_tag_bout(policy_number, bout_number)
    if path is None:
        if policy_number is None:
            raise FileNotFoundError(
                f"No tag replay found in {os.path.abspath(TAG_BOUT_DIR)}"
            )
        raise FileNotFoundError(
            f"No tag replay found for policy={policy_number}, bout={bout_number}"
        )
    out_path, html = build_tag_replay_html(path)
    try:
        from IPython.display import display, HTML, IFrame
        try:
            display(IFrame(src=os.path.relpath(out_path), width="100%", height=720))
        except Exception:
            display(HTML(html))
    except ImportError:
        pass
    return out_path



# ============================================================
# LOCAL PRODUCTION REPLAY
# ============================================================

LOCAL_REPLAY_BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), "PPO_RTS"))
LOCAL_REPLAY_BOUT_DIR = os.path.join(LOCAL_REPLAY_BASE_DIR, "elite_bouts")
LOCAL_REPLAY_HTML_DIR = os.path.join(LOCAL_REPLAY_BASE_DIR, "replay")
os.makedirs(LOCAL_REPLAY_BOUT_DIR, exist_ok=True)
os.makedirs(LOCAL_REPLAY_HTML_DIR, exist_ok=True)


def _production_replay_name(generation):
    return f"generation_{int(generation):04d}_best_bout.npz"


def _list_remote_production_replays():
    """List production Best Bout files stored in the Modal Volume."""
    remote_volume = modal.Volume.from_name("rts-storage")
    entries = remote_volume.iterdir("/PPO_RTS/elite_bouts", recursive=False)
    files = []
    pattern = re.compile(r"generation_(\d+)_best_bout\.npz$")
    for entry in entries:
        name = os.path.basename(str(entry.path))
        m = pattern.fullmatch(name)
        if m:
            files.append((int(m.group(1)), str(entry.path)))
    return sorted(files, key=lambda x: x[0])


def _download_volume_file(remote_path, local_path):
    """Download exactly one file from the production Modal Volume."""
    remote_volume = modal.Volume.from_name("rts-storage")
    os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
    with open(local_path, "wb") as f:
        for chunk in remote_volume.read_file(remote_path):
            f.write(chunk)
    return local_path


def replay_production_locally(generation=None):
    """Fetch one production Best Bout from Modal Volume and open a local HTML replay.

    generation=None means the newest generation currently stored in the remote Volume.
    Only the selected .npz file is downloaded; the Volume is never copied wholesale.
    """
    remote_files = _list_remote_production_replays()
    if not remote_files:
        raise FileNotFoundError(
            "No production Best Bout found in Modal Volume: /PPO_RTS/elite_bouts"
        )

    if generation is None:
        selected_generation, remote_path = remote_files[-1]
    else:
        selected_generation = int(generation)
        expected_name = _production_replay_name(selected_generation)
        matches = [
            (g, p) for g, p in remote_files
            if g == selected_generation
        ]
        if not matches:
            raise FileNotFoundError(
                f"Generation {selected_generation} Best Bout was not found in the Modal Volume. "
                f"Latest available generation is {remote_files[-1][0]}."
            )
        selected_generation, remote_path = matches[0]

    local_npz = os.path.join(
        LOCAL_REPLAY_BOUT_DIR,
        _production_replay_name(selected_generation),
    )
    local_html = os.path.join(
        LOCAL_REPLAY_HTML_DIR,
        f"replay_generation_{selected_generation:04d}.html",
    )

    print(f"Downloading production replay: generation {selected_generation}")
    print(f"  Remote : {remote_path}")
    print(f"  Local  : {local_npz}")
    _download_volume_file(remote_path, local_npz)

    out_path, _html = build_replay_html(local_npz, out_path=local_html)
    print(f"Local replay written: {os.path.abspath(out_path)}")

    try:
        webbrowser.open(Path(out_path).resolve().as_uri())
        print("Opened the replay in the default browser.")
    except Exception as exc:
        print(f"Browser auto-open skipped: {exc}")
        print(f"Open manually: {os.path.abspath(out_path)}")

    return out_path

# ============================================================
# ENTRY POINT : PRODUCTION RTS ONLY
# ============================================================

@app.function(image=app_image, gpu="A10G", timeout=86400, volumes={"/data": vol})
def run_training_remotely(n_generations=N_GENERATIONS):
    """Production RTS PPO only. Standalone tag training is external."""
    latest = train(n_generations=int(n_generations), resume=True)
    try:
        replay_generation = LAST_COMPLETED_GENERATION
        path = (
            show_replay(replay_generation)
            if replay_generation is not None
            else show_replay()
        )
        print("Main replay written to (Cloud):", os.path.abspath(path))
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError as e:
        print("Main replay skipped:", e)
        return None


@app.local_entrypoint()
def main(generations: int = N_GENERATIONS):
    print("クラウドのGPUで本番RTS PPOだけを開始します...")
    html_content = run_training_remotely.remote(int(generations))
    if html_content:
        local_path = "latest_replay.html"
        with open(local_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"手元のPCに本番リプレイを保存しました: {local_path}")


# ============================================================
# TAG CONFIG / OVERRIDES (standalone local tag only)
# ============================================================

TAG_SOLDIERS_PER_TEAM = 1


def configure_tag_soldiers_per_team(n):
    """Set the number of active chasers per team for standalone Tag training."""
    global TAG_SOLDIERS_PER_TEAM
    n = int(n)
    if not 1 <= n <= N_SOLDIERS_PER_TEAM:
        raise ValueError(
            f"TAG_SOLDIERS_PER_TEAM must be between 1 and {N_SOLDIERS_PER_TEAM}, got {n}"
        )
    TAG_SOLDIERS_PER_TEAM = n


def tag_role_indices(role):
    """Return the N active chasers and the two relevant commanders."""
    role = jnp.asarray(role, dtype=jnp.int32)
    chaser_team = role
    runner_team = 1 - chaser_team
    start = jnp.where(
        chaser_team == 0,
        RED_SOLDIER_START,
        BLUE_SOLDIER_START,
    )
    chaser_idx = start + jnp.arange(TAG_SOLDIERS_PER_TEAM, dtype=jnp.int32)
    runner_cmd_idx = jnp.where(
        runner_team == 0,
        RED_COMMANDER_INDEX,
        BLUE_COMMANDER_INDEX,
    )
    chaser_cmd_idx = jnp.where(
        chaser_team == 0,
        RED_COMMANDER_INDEX,
        BLUE_COMMANDER_INDEX,
    )
    return chaser_team, runner_team, chaser_idx, runner_cmd_idx, chaser_cmd_idx


def _tag_formation_offsets():
    n = TAG_SOLDIERS_PER_TEAM
    cols = max(1, int(np.ceil(np.sqrt(n))))
    rows = int(np.ceil(n / cols))
    idx = np.arange(n, dtype=np.float32)
    c = idx % cols
    r = np.floor(idx / cols)
    ox = (c - (cols - 1) * 0.5) * 0.34
    oz = (r - (rows - 1) * 0.5) * 0.34
    return jnp.asarray(ox, dtype=jnp.float32), jnp.asarray(oz, dtype=jnp.float32)


def reset_tag_one(key, role):
    k1, k2, k3 = random.split(key, 3)
    x = jnp.zeros(N_UNITS, dtype=jnp.float32)
    z = jnp.zeros(N_UNITS, dtype=jnp.float32)
    vx = jnp.zeros(N_UNITS, dtype=jnp.float32)
    vz = jnp.zeros(N_UNITS, dtype=jnp.float32)
    hp = jnp.zeros(N_UNITS, dtype=jnp.float32)
    alive = jnp.zeros(N_UNITS, dtype=jnp.float32)
    attack_timer = jnp.zeros(N_UNITS, dtype=jnp.float32)
    speed = jnp.zeros(N_UNITS, dtype=jnp.float32)

    (
        chaser_team,
        runner_team,
        chaser_idx,
        runner_cmd_idx,
        chaser_cmd_idx,
    ) = tag_role_indices(role)

    # Both commanders are present. Only the runner commander moves.
    hp = hp.at[RED_COMMANDER_INDEX].set(0.30)
    hp = hp.at[BLUE_COMMANDER_INDEX].set(0.30)
    alive = alive.at[RED_COMMANDER_INDEX].set(1.0)
    alive = alive.at[BLUE_COMMANDER_INDEX].set(1.0)

    # Exactly N soldiers are active on the chasing team. The opposite team's
    # soldier slots remain inactive, preserving the production observation shape.
    ox, oz = _tag_formation_offsets()
    chaser_x0 = jnp.where(chaser_team == 0, -6.0, 6.0)
    chaser_z0 = random.uniform(k1, (), minval=-3.0, maxval=3.0)
    runner_x = jnp.where(chaser_team == 0, 5.5, -5.5)
    runner_z = random.uniform(k2, (), minval=-4.0, maxval=4.0)
    own_cmd_x = jnp.where(chaser_team == 0, -5.8, 5.8)
    own_cmd_z = random.uniform(k3, (), minval=-1.0, maxval=1.0)

    # The formation is centered around the team's starting side and offset in z.
    chaser_x = chaser_x0 + ox
    chaser_z = chaser_z0 + oz

    x = x.at[chaser_idx].set(chaser_x)
    z = z.at[chaser_idx].set(chaser_z)
    alive = alive.at[chaser_idx].set(1.0)
    hp = hp.at[chaser_idx].set(1.0)
    speed = speed.at[chaser_idx].set(TAG_CHASER_SPEED)

    x = x.at[runner_cmd_idx].set(runner_x)
    z = z.at[runner_cmd_idx].set(runner_z)
    speed = speed.at[runner_cmd_idx].set(TAG_RUNNER_SPEED)

    x = x.at[chaser_cmd_idx].set(own_cmd_x)
    z = z.at[chaser_cmd_idx].set(own_cmd_z)
    speed = speed.at[chaser_cmd_idx].set(0.0)

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


reset_tag_parallel = jax.jit(jax.vmap(reset_tag_one, in_axes=(0, 0)))


def reset_tag_batch(key, n):
    keys = random.split(key, n)
    roles = jnp.arange(n, dtype=jnp.int32) % 2
    return reset_tag_parallel(keys, roles), roles


def tag_local_frame(state, team, indices):
    blue = jnp.asarray(team, dtype=jnp.float32) > 0.5
    x = state["x"][indices]
    z = state["z"][indices]
    vx = state["vx"][indices]
    vz = state["vz"][indices]
    return (
        jnp.where(blue, -x, x),
        z,
        jnp.where(blue, -vx, vx),
        vz,
        blue,
    )


def _tag_terrain_for_team(team):
    base = tag_terrain.reshape(TERRAIN_RES, TERRAIN_RES)
    blue = jnp.asarray(team, dtype=jnp.float32) > 0.5
    return jnp.where(blue, base[:, ::-1], base).reshape(-1)


def make_tag_soldier_features(state, role, terrain_local=None):
    _, _, chaser_idx, runner_cmd_idx, chaser_cmd_idx = tag_role_indices(role)
    lx, lz, lvx, lvz, blue = tag_local_frame(state, role, chaser_idx)

    own_cmd_x = jnp.where(
        blue,
        -state["x"][chaser_cmd_idx],
        state["x"][chaser_cmd_idx],
    )
    own_cmd_z = state["z"][chaser_cmd_idx]
    enemy_cmd_x = jnp.where(
        blue,
        -state["x"][runner_cmd_idx],
        state["x"][runner_cmd_idx],
    )
    enemy_cmd_z = state["z"][runner_cmd_idx]

    base = jnp.stack([
        lx / HALF_FIELD,
        lz / HALF_FIELD,
        lvx,
        lvz,
        state["hp"][chaser_idx],
        jnp.ones_like(lx),
        jnp.zeros_like(lx),
        state["alive"][chaser_idx],
    ], axis=-1)
    nearest = relative_features(lx, lz, enemy_cmd_x, enemy_cmd_z)
    own_cmd = relative_features(lx, lz, own_cmd_x, own_cmd_z)
    enemy_cmd = relative_features(lx, lz, enemy_cmd_x, enemy_cmd_z)

    if terrain_local is None:
        terrain_local = _tag_terrain_for_team(role)
    local_terrain = grid_8_features(lx, lz, terrain_local)
    return jnp.concatenate([
        base,
        nearest,
        own_cmd,
        enemy_cmd,
        local_terrain,
    ], axis=-1)


def make_tag_commander_features(state, role, terrain_local=None):
    _, _, chaser_idx, runner_cmd_idx, _ = tag_role_indices(role)
    lx, lz, lvx, lvz, blue = tag_local_frame(state, role, runner_cmd_idx)
    nearest = jnp.sqrt(
        jnp.min(
            (jnp.where(blue, -state["x"][chaser_idx], state["x"][chaser_idx]) - lx) ** 2
            + (state["z"][chaser_idx] - lz) ** 2
        ) + 1e-8
    )

    if terrain_local is None:
        terrain_local = _tag_terrain_for_team(role)
    wall = commander_wall_features(lx, lz, terrain_local)
    nearest_idx = jnp.argmin(
        (jnp.where(blue, -state["x"][chaser_idx], state["x"][chaser_idx]) - lx) ** 2
        + (state["z"][chaser_idx] - lz) ** 2
    )
    nearest_x = jnp.where(
        blue,
        -state["x"][chaser_idx[nearest_idx]],
        state["x"][chaser_idx[nearest_idx]],
    )
    nearest_z = state["z"][chaser_idx[nearest_idx]]
    nearest_features = relative_features(lx, lz, nearest_x, nearest_z)
    base = jnp.stack([
        lx / HALF_FIELD,
        lz / HALF_FIELD,
        lvx,
        lvz,
        state["hp"][runner_cmd_idx],
        jnp.ones_like(lx),
        jnp.ones_like(lx),
        state["alive"][runner_cmd_idx],
    ], axis=-1)
    del nearest
    return jnp.concatenate([base, nearest_features, wall], axis=-1)


def tag_target_vectors(state, role):
    chaser_team, runner_team, chaser_idx, runner_cmd_idx, _ = tag_role_indices(role)
    chx = state["x"][chaser_idx]
    chz = state["z"][chaser_idx]
    rx = state["x"][runner_cmd_idx]
    rz = state["z"][runner_cmd_idx]

    # Soldier target in the chaser's local mirrored frame.
    dx = rx - chx
    dz = rz - chz
    to_runner_x = jnp.where(chaser_team > 0, -dx, dx)
    to_runner_z = dz
    dist = jnp.sqrt(dx * dx + dz * dz + 1e-8)
    target_chase = jnp.stack([to_runner_x, to_runner_z], axis=-1)
    target_chase = target_chase / (jnp.linalg.norm(target_chase, axis=-1, keepdims=True) + 1e-8)

    # Runner moves away from the nearest active chaser. Express the target in
    # the runner commander's own local mirrored frame.
    d2 = (chx - rx) ** 2 + (chz - rz) ** 2
    nearest_i = jnp.argmin(d2)
    nearest_x = chx[nearest_i]
    nearest_z = chz[nearest_i]
    away_x_world = rx - nearest_x
    away_z_world = rz - nearest_z
    runner_is_blue = runner_team > 0
    away_x_local = jnp.where(runner_is_blue, -away_x_world, away_x_world)
    target_run = jnp.array([away_x_local, away_z_world], dtype=jnp.float32)
    target_run = target_run / (jnp.linalg.norm(target_run) + 1e-8)
    return target_chase, target_run, dist


def tag_update_loss(params, soldier_features, commander_features, target_chase, target_run, attack_target):
    micro_s, micro_c = tag_micro_outputs(params, soldier_features, commander_features)
    ps = jnp.tanh(micro_s[..., :2])
    ps = ps / (jnp.linalg.norm(ps, axis=-1, keepdims=True) + 1e-8)
    pc = jnp.tanh(micro_c)
    pc = pc / (jnp.linalg.norm(pc, axis=-1, keepdims=True) + 1e-8)

    chase_dir_loss = 1.0 - jnp.sum(ps * target_chase, axis=-1)
    run_dir_loss = 1.0 - jnp.sum(pc * target_run, axis=-1)
    attack_logit = micro_s[..., 2]
    attack_target = attack_target.astype(jnp.float32)
    attack_bce = (
        jnp.maximum(attack_logit, 0.0)
        - attack_logit * attack_target
        + jnp.log1p(jnp.exp(-jnp.abs(attack_logit)))
    )

    loss = (
        jnp.mean(chase_dir_loss)
        + jnp.mean(run_dir_loss)
        + 0.25 * jnp.mean(attack_bce)
    )
    return loss, {
        "tag_loss": loss,
        "tag_chase_loss": jnp.mean(chase_dir_loss),
        "tag_run_loss": jnp.mean(run_dir_loss),
        "tag_attack_loss": jnp.mean(attack_bce),
    }


@jax.jit
def tag_micro_local_action(params, soldier_features, commander_features):
    micro_s, micro_c = tag_micro_outputs(params, soldier_features, commander_features)
    s_vec = jnp.tanh(micro_s[..., :2])
    s_vec = s_vec / (jnp.linalg.norm(s_vec, axis=-1, keepdims=True) + 1e-8)
    c_vec = jnp.tanh(micro_c)
    c_vec = c_vec / (jnp.linalg.norm(c_vec, axis=-1, keepdims=True) + 1e-8)
    attack = (jax.nn.sigmoid(micro_s[..., 2]) >= 0.5).astype(jnp.float32)
    return s_vec[..., 0], s_vec[..., 1], attack, c_vec[..., 0], c_vec[..., 1]


def tag_step_one(state, role, chaser_dx, chaser_dz, chaser_attack, runner_dx, runner_dz):
    _, _, chaser_idx, runner_cmd_idx, _ = tag_role_indices(role)
    x, z = state["x"], state["z"]
    chx, chz = x[chaser_idx], z[chaser_idx]
    rx, rz = x[runner_cmd_idx], z[runner_cmd_idx]

    old_dist = jnp.sqrt((rx - chx) ** 2 + (rz - chz) ** 2 + 1e-8)
    ch_nx = chx + chaser_dx * TAG_CHASER_SPEED * DT
    ch_nz = chz + chaser_dz * TAG_CHASER_SPEED * DT
    r_nx = rx + runner_dx * TAG_RUNNER_SPEED * DT
    r_nz = rz + runner_dz * TAG_RUNNER_SPEED * DT

    ch_inside = (
        (ch_nx >= -HALF_FIELD + SOLDIER_RADIUS)
        & (ch_nx <= HALF_FIELD - SOLDIER_RADIUS)
        & (ch_nz >= -HALF_FIELD + SOLDIER_RADIUS)
        & (ch_nz <= HALF_FIELD - SOLDIER_RADIUS)
    )
    r_inside = (
        (r_nx >= -HALF_FIELD + COMMANDER_RADIUS)
        & (r_nx <= HALF_FIELD - COMMANDER_RADIUS)
        & (r_nz >= -HALF_FIELD + COMMANDER_RADIUS)
        & (r_nz <= HALF_FIELD - COMMANDER_RADIUS)
    )
    ch_blocked = jax.vmap(lambda px, pz: tag_wall_blocked(px, pz, SOLDIER_RADIUS))(ch_nx, ch_nz)
    r_blocked = tag_wall_blocked(r_nx, r_nz, COMMANDER_RADIUS)
    ch_valid = ch_inside & (~ch_blocked)
    r_valid = r_inside & (~r_blocked)

    ch_nx = jnp.where(ch_valid, ch_nx, chx)
    ch_nz = jnp.where(ch_valid, ch_nz, chz)
    r_nx = jnp.where(r_valid, r_nx, rx)
    r_nz = jnp.where(r_valid, r_nz, rz)

    new_dist = jnp.sqrt((r_nx - ch_nx) ** 2 + (r_nz - ch_nz) ** 2 + 1e-8)
    captured = jnp.any(new_dist <= ATTACK_RANGE)
    new_time = state["time"] + DT
    timeout = new_time >= TAG_MAX_STEPS * DT
    done = captured | timeout

    nx = x.at[chaser_idx].set(ch_nx)
    nz = z.at[chaser_idx].set(ch_nz)
    nx = nx.at[runner_cmd_idx].set(r_nx)
    nz = nz.at[runner_cmd_idx].set(r_nz)
    nvx = state["vx"].at[chaser_idx].set(jnp.where(ch_valid, chaser_dx, 0.0))
    nvz = state["vz"].at[chaser_idx].set(jnp.where(ch_valid, chaser_dz, 0.0))
    nvx = nvx.at[runner_cmd_idx].set(jnp.where(r_valid, runner_dx, 0.0))
    nvz = nvz.at[runner_cmd_idx].set(jnp.where(r_valid, runner_dz, 0.0))

    progress = jnp.mean(old_dist - new_dist)
    reward_chaser = TAG_STEP_PENALTY + TAG_CAPTURE_REWARD * captured.astype(jnp.float32) + 0.05 * progress
    reward_runner = TAG_STEP_PENALTY + TAG_RUNNER_CAPTURE_REWARD * captured.astype(jnp.float32) - 0.05 * progress

    next_state = {
        **state,
        "x": nx,
        "z": nz,
        "vx": nvx,
        "vz": nvz,
        "time": new_time,
        "done": done,
    }
    return next_state, reward_chaser, reward_runner, done, captured


@jax.jit
def tag_rollout_step(params, states, roles):
    soldier_features = jax.vmap(make_tag_soldier_features)(states, roles)
    commander_features = jax.vmap(make_tag_commander_features)(states, roles)
    target_chase, target_run, dist_now = jax.vmap(tag_target_vectors)(states, roles)
    attack_target = (dist_now <= ATTACK_RANGE).astype(jnp.float32)
    return soldier_features, commander_features, target_chase, target_run, attack_target


def train_tag_game(params, key, steps=TAG_UPDATES_PER_RUN):
    states, roles = reset_tag_batch(key, TAG_BATCH_SIZE)
    opt_state = tag_optimizer.init(params)
    metrics_list = []
    captures = []
    for _ in range(int(steps)):
        key, reset_key = random.split(key)
        soldier_features, commander_features, target_chase, target_run, attack_target = tag_rollout_step(
            params, states, roles
        )
        params, opt_state, metrics = tag_update_minibatch(
            params, opt_state,
            soldier_features, commander_features,
            target_chase, target_run, attack_target,
        )
        sdx, sdz, sat, cdx, cdz = tag_micro_local_action(
            params, soldier_features, commander_features
        )
        states, _, _, done, captured = jax.vmap(tag_step_one)(
            states, roles, sdx, sdz, sat, cdx, cdz
        )
        fresh = reset_tag_parallel(random.split(reset_key, TAG_BATCH_SIZE), roles)
        def merge(old, new):
            mask = done if old.ndim == 1 else done.reshape((TAG_BATCH_SIZE,) + (1,) * (old.ndim - 1))
            return jnp.where(mask, new, old)
        states = jax.tree_util.tree_map(merge, states, fresh)
        metrics_list.append({k: float(v) for k, v in metrics.items()})
        captures.append(int(jnp.sum(captured)))

    if not metrics_list:
        raise ValueError("TAG updates must be at least 1")
    metrics = {
        k: float(np.mean([m[k] for m in metrics_list]))
        for k in metrics_list[0]
    }
    metrics["tag_captures"] = int(sum(captures))
    metrics["tag_batch"] = TAG_BATCH_SIZE
    metrics["tag_updates"] = int(steps)
    return params, key, metrics


def record_tag_bout(params, initial_state=None, role=0, rng_key=None):
    if initial_state is None:
        if rng_key is None:
            rng_key = random.PRNGKey(20260917)
        initial_state = reset_tag_one(rng_key, jnp.array(role, dtype=jnp.int32))

    role_j = jnp.array(role, dtype=jnp.int32)
    _, _, chaser_idx, runner_cmd_idx, _ = tag_role_indices(role_j)
    chaser_idx_np = np.asarray(chaser_idx)
    state = initial_state
    chaser_x = [np.asarray(state["x"])[chaser_idx_np]]
    chaser_z = [np.asarray(state["z"])[chaser_idx_np]]
    runner_x = [float(np.asarray(state["x"])[int(runner_cmd_idx)])]
    runner_z = [float(np.asarray(state["z"])[int(runner_cmd_idx)])]
    chaser_actions = []
    runner_actions = []
    winner_code = 0

    for _ in range(TAG_MAX_STEPS):
        sf = make_tag_soldier_features(state, role_j)[None, ...]
        cf = make_tag_commander_features(state, role_j)[None, ...]
        sdx, sdz, sat, cdx, cdz = tag_micro_local_action(params, sf, cf)
        sdx_np = np.asarray(sdx[0])
        sdz_np = np.asarray(sdz[0])
        sat_np = np.asarray(sat[0])
        cdx_f = float(np.asarray(cdx[0]))
        cdz_f = float(np.asarray(cdz[0]))
        chaser_actions.append(np.stack([sdx_np, sdz_np, sat_np], axis=-1))
        runner_actions.append([cdx_f, cdz_f])
        state, _, _, done, captured = tag_step_one(
            state,
            role_j,
            jnp.asarray(sdx_np, dtype=jnp.float32),
            jnp.asarray(sdz_np, dtype=jnp.float32),
            jnp.asarray(sat_np, dtype=jnp.float32),
            jnp.array(cdx_f, dtype=jnp.float32),
            jnp.array(cdz_f, dtype=jnp.float32),
        )
        chaser_x.append(np.asarray(state["x"])[chaser_idx_np])
        chaser_z.append(np.asarray(state["z"])[chaser_idx_np])
        runner_x.append(float(np.asarray(state["x"])[int(runner_cmd_idx)]))
        runner_z.append(float(np.asarray(state["z"])[int(runner_cmd_idx)]))
        if bool(done):
            winner_code = 1 if bool(captured) else 3
            break

    return {
        "generation": -1,
        "role": int(role),
        "chaser_team": int(role),
        "runner_team": int(1 - role),
        "tag_soldiers_per_team": int(TAG_SOLDIERS_PER_TEAM),
        "winner_code": int(winner_code),
        "chaser_x": np.asarray(chaser_x, dtype=np.float32),
        "chaser_z": np.asarray(chaser_z, dtype=np.float32),
        "runner_x": np.asarray(runner_x, dtype=np.float32),
        "runner_z": np.asarray(runner_z, dtype=np.float32),
        "chaser_actions": np.asarray(chaser_actions, dtype=np.float32),
        "runner_actions": np.asarray(runner_actions, dtype=np.float32),
        "end_step": len(chaser_actions),
        "dt": DT,
        "field_size": FIELD_SIZE,
        "terrain_res": TERRAIN_RES,
        "tag_walls": TAG_WALL_LIST,
    }


def verify_tag_bout(initial_state, role, chaser_actions, runner_actions):
    role_j = jnp.array(role, dtype=jnp.int32)
    _, _, chaser_idx, runner_cmd_idx, _ = tag_role_indices(role_j)
    chaser_idx_np = np.asarray(chaser_idx)
    runner_idx = int(runner_cmd_idx)
    state = initial_state
    cx = [np.asarray(state["x"])[chaser_idx_np]]
    cz = [np.asarray(state["z"])[chaser_idx_np]]
    rx = [float(np.asarray(state["x"])[runner_idx])]
    rz = [float(np.asarray(state["z"])[runner_idx])]
    for i in range(len(chaser_actions)):
        ca = chaser_actions[i]
        ra = runner_actions[i]
        state, _, _, _, _ = tag_step_one(
            state,
            role_j,
            jnp.asarray(ca[:, 0], dtype=jnp.float32),
            jnp.asarray(ca[:, 1], dtype=jnp.float32),
            jnp.asarray(ca[:, 2], dtype=jnp.float32),
            jnp.array(ra[0], dtype=jnp.float32),
            jnp.array(ra[1], dtype=jnp.float32),
        )
        cx.append(np.asarray(state["x"])[chaser_idx_np])
        cz.append(np.asarray(state["z"])[chaser_idx_np])
        rx.append(float(np.asarray(state["x"])[runner_idx]))
        rz.append(float(np.asarray(state["z"])[runner_idx]))
    return np.asarray(cx, dtype=np.float32), np.asarray(cz, dtype=np.float32), np.asarray(rx, dtype=np.float32), np.asarray(rz, dtype=np.float32)


def save_tag_bout(path, generation, bout, verification_pass=True, max_state_error=0.0):
    np.savez_compressed(
        path,
        generation=np.array(generation, dtype=np.int32),
        sample_index=np.array(int(bout.get("sample_index", 0)), dtype=np.int32),
        role=np.array(bout["role"], dtype=np.int32),
        chaser_team=np.array(bout["chaser_team"], dtype=np.int32),
        runner_team=np.array(bout["runner_team"], dtype=np.int32),
        tag_soldiers_per_team=np.array(bout["tag_soldiers_per_team"], dtype=np.int32),
        winner_code=np.array(bout["winner_code"], dtype=np.int32),
        end_step=np.array(bout["end_step"], dtype=np.int32),
        dt=np.array(bout["dt"], dtype=np.float32),
        field_size=np.array(bout["field_size"], dtype=np.float32),
        terrain_res=np.array(bout["terrain_res"], dtype=np.int32),
        tag_walls=np.asarray(bout["tag_walls"], dtype=np.float32).reshape((-1, 2)),
        verification_pass=np.array(1 if verification_pass else 0, dtype=np.int32),
        max_state_error=np.array(max_state_error, dtype=np.float32),
        chaser_x=bout["chaser_x"],
        chaser_z=bout["chaser_z"],
        runner_x=bout["runner_x"],
        runner_z=bout["runner_z"],
        chaser_actions=bout["chaser_actions"],
        runner_actions=bout["runner_actions"],
    )
    return path


def build_tag_replay_html(path, out_path=None):
    d = np.load(path, allow_pickle=False)
    generation = int(d["generation"])
    sample_index = int(d["sample_index"]) if "sample_index" in d else 0
    chaser_team = int(d["chaser_team"]) if "chaser_team" in d else int(d["role"])
    runner_team = int(d["runner_team"]) if "runner_team" in d else 1 - chaser_team
    winner_code = int(d["winner_code"])
    end_step = int(d["end_step"])
    dt = float(d["dt"])
    field_size = float(d["field_size"])
    walls_data = d["tag_walls"].astype(np.float32).tolist()
    if "chaser_x" in d:
        n_soldiers = int(d["tag_soldiers_per_team"]) if "tag_soldiers_per_team" in d else 1
        chx = d["chaser_x"].astype(np.float32)
        chz = d["chaser_z"].astype(np.float32)
        rx = d["runner_x"].astype(np.float32)
        rz = d["runner_z"].astype(np.float32)
        ca = d["chaser_actions"].astype(np.float32)
    else:
        # Backward compatibility with the old single-chaser format.
        n_soldiers = 1
        role_legacy = int(d["role"])
        full_x = d["x"].astype(np.float32)
        full_z = d["z"].astype(np.float32)
        if role_legacy == 0:
            ch_index, runner_index = RED_SOLDIER_START, BLUE_COMMANDER_INDEX
        else:
            ch_index, runner_index = BLUE_SOLDIER_START, RED_COMMANDER_INDEX
        chx = full_x[:, [ch_index]]
        chz = full_z[:, [ch_index]]
        rx = full_x[:, runner_index]
        rz = full_z[:, runner_index]
        ca = d["chaser_actions"].astype(np.float32)[:, None, :]
    verification_pass = bool(int(d["verification_pass"])) if "verification_pass" in d else True
    max_state_error = float(d["max_state_error"]) if "max_state_error" in d else 0.0
    result_text = "CAUGHT" if winner_code == 1 else "TIMEOUT"
    payload = json.dumps({
        "generation": generation,
        "sampleIndex": sample_index,
        "chaserTeam": "Red" if chaser_team == 0 else "Blue",
        "runnerTeam": "Red" if runner_team == 0 else "Blue",
        "soldiers": n_soldiers,
        "winner": result_text,
        "steps": end_step,
        "dt": dt,
        "fieldSize": field_size,
        "walls": walls_data,
        "chaserX": np.round(chx, 4).tolist(),
        "chaserZ": np.round(chz, 4).tolist(),
        "runnerX": np.round(rx, 4).tolist(),
        "runnerZ": np.round(rz, 4).tolist(),
        "chaserActions": np.round(ca, 3).tolist(),
        "attackRange": float(ATTACK_RANGE),
        "verificationPass": verification_pass,
        "maxStateError": max_state_error,
    }, separators=(",", ":"))

    html = r'''<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RTS Tag Replay</title><style>
html,body{margin:0;padding:0;width:100%;height:100%;overflow:hidden;background:#101010;font-family:Arial,Helvetica,sans-serif}#info{position:absolute;left:14px;top:14px;z-index:10;padding:12px 15px;background:rgba(0,0,0,.74);color:#fff;border-radius:8px;line-height:1.5;font-size:14px}#controls{position:absolute;left:14px;bottom:14px;z-index:10;padding:10px 12px;background:rgba(0,0,0,.74);border-radius:8px;color:#fff}button{margin-right:5px;padding:5px 9px;border:0;border-radius:4px;cursor:pointer}#timeline{width:440px;max-width:45vw}.ok{color:#5fd97a;font-weight:bold}
</style></head><body><div id="info"></div><div id="controls"><button id="play">Play</button><button id="pause">Pause</button><button id="reset">Reset</button><button data-speed="0.5">0.5x</button><button data-speed="1">1x</button><button data-speed="2">2x</button><br><br><input id="timeline" type="range" min="0" max="__MAX__" value="0" step="1"><span id="status"></span></div><script type="importmap">{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}</script><script type="module">
import * as THREE from 'three';import {OrbitControls} from 'three/addons/controls/OrbitControls.js';const D=__DATA__;const info=document.getElementById('info'),timeline=document.getElementById('timeline'),status=document.getElementById('status');let t=0,playing=false,speed=1,last=null;const scene=new THREE.Scene();scene.background=new THREE.Color(0x101010);const cam=new THREE.PerspectiveCamera(45,innerWidth/innerHeight,.1,250);cam.position.set(0,18,17);const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio||1,2));renderer.setSize(innerWidth,innerHeight);document.body.appendChild(renderer.domElement);const ctl=new OrbitControls(cam,renderer.domElement);ctl.target.set(0,0,0);ctl.update();scene.add(new THREE.HemisphereLight(0xffffff,0x444444,2));const dl=new THREE.DirectionalLight(0xffffff,1.2);dl.position.set(6,14,8);scene.add(dl);const half=D.fieldSize/2;const ground=new THREE.Mesh(new THREE.PlaneGeometry(D.fieldSize,D.fieldSize),new THREE.MeshStandardMaterial({color:0x303030}));ground.rotation.x=-Math.PI/2;scene.add(ground);scene.add(new THREE.GridHelper(D.fieldSize,D.fieldSize,0x777777,0x444444));for(const p of D.walls){const w=new THREE.Mesh(new THREE.BoxGeometry(1,.7,1),new THREE.MeshStandardMaterial({color:0x777777}));w.position.set(p[0],.35,p[1]);scene.add(w)}const chMat=new THREE.MeshStandardMaterial({color:D.chaserTeam===0?0xd94b4b:0x4b7bd9}),runMat=new THREE.MeshStandardMaterial({color:D.runnerTeam===0?0xd94b4b:0x4b7bd9});const ch=[];for(let k=0;k<D.soldiers;k++){const m=new THREE.Mesh(new THREE.SphereGeometry(.17,16,16),chMat);m.position.y=.17;scene.add(m);ch.push(m)}const runner=new THREE.Mesh(new THREE.BoxGeometry(.44,.8,.44),runMat);runner.position.y=.4;scene.add(runner);function frame(){const raw=t/D.dt,i=Math.min(Math.floor(raw),D.steps),a=i>=D.steps?0:raw-i;return{i,a}}function update(){const f=frame(),i=f.i,a=f.a;for(let k=0;k<D.soldiers;k++){const x0=D.chaserX[i][k],z0=D.chaserZ[i][k],x1=i<D.steps?D.chaserX[i+1][k]:x0,z1=i<D.steps?D.chaserZ[i+1][k]:z0;ch[k].position.set(x0+(x1-x0)*a,.17,z0+(z1-z0)*a)}const rx0=D.runnerX[i],rz0=D.runnerZ[i],rx1=i<D.steps?D.runnerX[i+1]:rx0,rz1=i<D.steps?D.runnerZ[i+1]:rz0;runner.position.set(rx0+(rx1-rx0)*a,.4,rz0+(rz1-rz0)*a);const minD=Math.min(...ch.map((m)=>Math.hypot(m.position.x-runner.position.x,m.position.z-runner.position.z)));timeline.value=String(i);info.innerHTML='<b>Tag policy '+D.generation+'</b><br>Chaser: <b>'+D.chaserTeam+'</b> ('+D.soldiers+' soldiers)<br>Runner: <b>'+D.runnerTeam+' Commander</b><br>Time: '+(i*D.dt).toFixed(2)+' s<br>Nearest distance: '+minD.toFixed(3)+'<br>Result: <b>'+D.winner+'</b><br>'+((D.verificationPass)?'<span class="ok">Replay verification: PASS</span>':'<span style="color:#ff6a6a;font-weight:bold">Replay verification: FAIL ('+D.maxStateError.toExponential(2)+')</span>');status.textContent='Tag Replay | '+(i*D.dt).toFixed(2)+' s'}document.getElementById('play').onclick=()=>playing=true;document.getElementById('pause').onclick=()=>playing=false;document.getElementById('reset').onclick=()=>{playing=false;t=0;update()};document.querySelectorAll('button[data-speed]').forEach(b=>b.onclick=()=>speed=parseFloat(b.dataset.speed));timeline.oninput=()=>{playing=false;t=parseInt(timeline.value)*D.dt;update()};addEventListener('resize',()=>{cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight)});function anim(ts){requestAnimationFrame(anim);if(last===null)last=ts;const dt=Math.min(.05,(ts-last)/1000);last=ts;if(playing){t+=dt*speed;if(t>=D.steps*D.dt){t=D.steps*D.dt;playing=false}}update();ctl.update();renderer.render(scene,cam)}update();requestAnimationFrame(anim);
</script></body></html>'''
    html = html.replace("__MAX__", str(end_step)).replace("__DATA__", payload)
    if out_path is None:
        out_path = os.path.join(
            TAG_REPLAY_DIR,
            f"tag_replay_policy_{generation:06d}_bout_{sample_index:03d}.html",
        )
    Path(out_path).write_text(html, encoding="utf-8")
    return out_path, html


def run_tag_training(n_updates=TAG_UPDATES_PER_RUN, resume=True):
    """Standalone Tag training. No HTML replay is generated here."""
    if n_updates < 1:
        raise ValueError("n_updates must be >= 1")

    production_elite = find_latest_elite()
    if production_elite is None:
        key, ik = random.split(random.key(int(time.time()) & 0x7FFFFFFF))
        params0 = init_policy(ik)
        production_elite = save_params(
            os.path.join(ELITE_DIR, "generation_0000.npz"),
            params0,
            {
                "generation": 0,
                "source": "random_init_for_tag_training",
                "raw_obs_size": RAW_OBS_SIZE,
                "global_input_size": GLOBAL_INPUT_SIZE,
            },
        )

    start_path, start_source = choose_tag_seed_policy(production_elite)
    params, _ = load_params(start_path)
    master_key = random.key(int(time.time()) & 0x7FFFFFFF)
    master_key, tag_key = random.split(master_key)

    print()
    print("============================================")
    print("STANDALONE TAG TRAINING")
    print("============================================")
    print(f"Starting policy      : {os.path.basename(start_path)} ({start_source})")
    print(f"Soldiers per team    : {TAG_SOLDIERS_PER_TEAM}")
    print(f"Updates              : {n_updates}")
    print(f"Batch                : {TAG_BATCH_SIZE}")
    print(f"Field                : {FIELD_SIZE} x {FIELD_SIZE}")
    print(f"Walls                : {len(TAG_WALL_LIST)}")
    print(f"Chaser speed         : {TAG_CHASER_SPEED:.3f}")
    print(f"Runner speed         : {TAG_RUNNER_SPEED:.3f}")
    print(f"Max tag time         : {TAG_MAX_STEPS * DT:.1f} s")
    print("Updated parameters   : Soldier/Commander Encoders + Micro heads")
    print("Replay HTML          : only generated by an explicit --replay command")
    print()

    t0 = time.time()
    params, tag_key, metrics = train_tag_game(params, tag_key, steps=int(n_updates))
    elapsed = time.time() - t0

    existing = find_latest_tag_policy()
    next_number = tag_policy_number(existing) + 1 if existing else 1
    tag_policy_path = os.path.join(TAG_ELITE_DIR, f"tag_policy_{next_number:06d}.npz")
    save_params(
        tag_policy_path,
        params,
        {
            "source": "standalone_tag_training",
            "source_policy": os.path.abspath(start_path),
            "tag_updates": int(n_updates),
            "tag_batch": TAG_BATCH_SIZE,
            "tag_soldiers_per_team": int(TAG_SOLDIERS_PER_TEAM),
            "tag_walls": TAG_WALL_LIST,
        },
    )

    print(
        f"Tag training done     | loss {metrics['tag_loss']:.4f} "
        f"chase {metrics['tag_chase_loss']:.4f} "
        f"run {metrics['tag_run_loss']:.4f} "
        f"attack {metrics['tag_attack_loss']:.4f} "
        f"grad {metrics['tag_grad_norm']:.3e} "
        f"captures {metrics['tag_captures']} "
        f"time {elapsed:.1f} s"
    )
    print(f"Tag policy saved      : {tag_policy_path}")

    # Save raw episode data only. HTML is intentionally created later by --replay.
    for sample_index in range(TAG_REPLAY_SAMPLES_PER_RUN):
        master_key, episode_key = random.split(master_key)
        role = int(sample_index % 2)
        tag_init = reset_tag_one(episode_key, jnp.array(role, dtype=jnp.int32))
        tag_bout = record_tag_bout(params, tag_init, role=role)
        tag_bout["sample_index"] = sample_index
        cx, cz, rx, rz = verify_tag_bout(
            tag_init, role, tag_bout["chaser_actions"], tag_bout["runner_actions"]
        )
        max_err = float(max(
            np.max(np.abs(cx - tag_bout["chaser_x"])),
            np.max(np.abs(cz - tag_bout["chaser_z"])),
            np.max(np.abs(rx - tag_bout["runner_x"])),
            np.max(np.abs(rz - tag_bout["runner_z"])),
        ))
        verify_pass = max_err < 1e-6
        save_tag_bout(
            os.path.join(
                TAG_BOUT_DIR,
                f"tag_policy_{next_number:06d}_bout_{sample_index:03d}.npz",
            ),
            int(next_number),
            tag_bout,
            verify_pass,
            max_err,
        )

    print(f"Raw Tag bouts saved   : {TAG_REPLAY_SAMPLES_PER_RUN}")
    print("No Tag HTML replay generated during training.")
    return tag_policy_path, None


def _run_local_cli():
    """Local-only utilities. This path is used by `python main.py ...`, not Modal."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Local utilities for the production RTS replay."
    )
    parser.add_argument(
        "--replay",
        nargs="?",
        const="latest",
        default=None,
        help=(
            "Download and open a production Best Bout locally. "
            "With no value, fetch the latest generation; with a number, fetch that generation."
        ),
    )
    args = parser.parse_args()

    if args.replay is None:
        parser.print_help()
        return

    if args.replay == "latest":
        replay_production_locally(None)
    else:
        try:
            generation = int(args.replay)
        except ValueError as exc:
            raise SystemExit(
                f"--replay expects a generation number or no value; got {args.replay!r}"
            ) from exc
        replay_production_locally(generation)


if __name__ == "__main__":
    _run_local_cli()
