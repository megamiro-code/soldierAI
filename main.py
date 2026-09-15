# ============================================================
# JAX RTS : PPO + ELITE SELF-PLAY + 3D REPLAY
# Single-file version (Lightning AI)
#
# Fixes #1-#23
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

print("JAX version :", jax.__version__)
print("Backend     :", jax.default_backend())
print("Devices     :", jax.devices())


# ============================================================
# PATHS  (fix #15 : single source of truth for all directories)
# ============================================================

BASE_DIR = os.environ.get("RTS_BASE_DIR", "./PPO_RTS")

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

# 16 x 16 battlefield
FIELD_SIZE = 16.0
HALF_FIELD = FIELD_SIZE / 2.0

N_SOLDIERS_PER_TEAM = 100
N_SOLDIERS_TOTAL = N_SOLDIERS_PER_TEAM * 2
N_COMMANDERS = 2
N_UNITS = N_SOLDIERS_TOTAL + N_COMMANDERS

DT = 0.20
MAX_TIME = 180.0
MAX_STEPS = int(MAX_TIME / DT)

ATTACK_RANGE = 0.42
ATTACK_COOLDOWN = 0.55
ATTACK_DAMAGE = 0.20

SOLDIER_RADIUS = 0.13
COMMANDER_RADIUS = 0.22
COMMANDER_EXTRA_MARGIN = 0.10

# Fixed initial state:
# every soldier starts at the same deterministic speed.
INITIAL_SOLDIER_SPEED = 0.50

N_FEATURES_PER_UNIT = 10
TERRAIN_RES = 16

OBS_SIZE = TERRAIN_RES * TERRAIN_RES + N_UNITS * N_FEATURES_PER_UNIT
ACTION_SIZE = N_SOLDIERS_PER_TEAM * 3

SHAPING_COEF = 0.005

# Backward-compatible aliases
OBS_DIM = OBS_SIZE
ACTION_DIM = ACTION_SIZE

RED_COMMANDER_INDEX = 0
BLUE_COMMANDER_INDEX = 1
RED_SOLDIER_START = 2
RED_SOLDIER_END = 102
BLUE_SOLDIER_START = 102
BLUE_SOLDIER_END = 202

ALL_SOLDIER_INDICES = jnp.arange(
    RED_SOLDIER_START,
    BLUE_SOLDIER_END,
)
ALL_UNIT_INDICES = jnp.arange(
    N_UNITS
)

# ------------------------------------------------------------
# Fixed wall layout for the 16 x 16 battlefield.
#
# The two outermost starting columns are intentionally kept
# completely free of walls.
# ------------------------------------------------------------

WALL_LIST = [
    [-5.5, -6.5],
    [-5.5,  6.5],
    [ 5.5, -6.5],
    [ 5.5,  6.5],

    [-1.0, -1.0],
    [-1.0,  1.0],
    [ 1.0, -1.0],
    [ 1.0,  1.0],
]

walls = jnp.array(
    WALL_LIST,
    dtype=jnp.float32,
)


def make_terrain():
    # Cell centers for a 16 x 16 battlefield:
    # -7.5, -6.5, ..., 6.5, 7.5
    centers_1d = (
        jnp.arange(TERRAIN_RES, dtype=jnp.float32)
        - HALF_FIELD
        + 0.5
    )

    xx, zz = jnp.meshgrid(
        centers_1d,
        centers_1d,
    )

    centers = jnp.stack(
        [
            xx.reshape(-1),
            zz.reshape(-1),
        ],
        axis=-1,
    )

    def blocked(i):
        cell = centers[i]
        d = jnp.abs(
            cell[None, :]
            - walls
        )

        return jnp.any(
            jnp.all(
                d < 0.5,
                axis=1,
            )
        )

    t = jax.vmap(
        blocked
    )(
        jnp.arange(
            TERRAIN_RES * TERRAIN_RES
        )
    )

    # Safety rule:
    # the two outermost starting columns must remain free.
    start_column = jnp.abs(
        centers[:, 0]
    ) > (HALF_FIELD - 1.0)

    t = jnp.where(
        start_column,
        False,
        t,
    )

    return t.astype(
        jnp.float32
    )


terrain = make_terrain()

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


# ------------------------------------------------------------
# RESET
# ------------------------------------------------------------

def reset_one(key):
    """
    Deterministic initial state.

    Red:
        starts from the leftmost column, which is farthest from
        the Blue commander.

    Blue:
        starts from the rightmost column, which is farthest from
        the Red commander.

    The 100 soldiers occupy the two columns farthest from the enemy,
    across 10 rows, with 5 soldiers inside each 1 x 1 cell:
        2 columns x 10 rows x 5 soldiers = 100 soldiers.

    Each group of five forms a compact regular pentagon inside its cell.
    Both starting columns are reserved for the formation; no wall is placed there.
    """

    del key

    x = jnp.zeros(
        N_UNITS,
        dtype=jnp.float32,
    )

    z = jnp.zeros(
        N_UNITS,
        dtype=jnp.float32,
    )

    vx = jnp.zeros(
        N_UNITS,
        dtype=jnp.float32,
    )

    vz = jnp.zeros(
        N_UNITS,
        dtype=jnp.float32,
    )

    hp = jnp.ones(
        N_UNITS,
        dtype=jnp.float32,
    )

    alive = jnp.ones(
        N_UNITS,
        dtype=jnp.float32,
    )

    attack_timer = jnp.zeros(
        N_UNITS,
        dtype=jnp.float32,
    )

    speed = jnp.zeros(
        N_UNITS,
        dtype=jnp.float32,
    )

    # --------------------------------------------------------
    # Commanders
    # --------------------------------------------------------

    # Commanders sit just behind the two-column starting area.
    x = x.at[RED_COMMANDER_INDEX].set(-5.80)
    z = z.at[RED_COMMANDER_INDEX].set(0.0)
    x = x.at[BLUE_COMMANDER_INDEX].set(5.80)
    z = z.at[BLUE_COMMANDER_INDEX].set(0.0)

    speed = speed.at[RED_COMMANDER_INDEX].set(0.20)
    speed = speed.at[BLUE_COMMANDER_INDEX].set(0.20)

    # --------------------------------------------------------
    # Fixed 2 x 10 cell starting formation
    # --------------------------------------------------------
    # Each side occupies the two columns farthest from the enemy:
    #   Red  : x = -7.5, -6.5
    #   Blue : x = +7.5, +6.5
    # There are 10 rows (z = -4.5 ... +4.5), with 5 soldiers
    # packed inside each 1 x 1 cell. Total: 2 x 10 x 5 = 100.
    cell_x = jnp.array([-7.5, -6.5], dtype=jnp.float32)
    row_z = jnp.arange(10, dtype=jnp.float32) - 4.5

    # Regular pentagon, radius 0.29.
    # Nearest-neighbour spacing is about 0.34 > 2*SOLDIER_RADIUS.
    angles = jnp.arange(5, dtype=jnp.float32) * (2.0 * jnp.pi / 5.0)
    offsets_x = 0.29 * jnp.cos(angles)
    offsets_z = 0.29 * jnp.sin(angles)

    # 20 cells in row-major order, each repeated with 5 local offsets.
    cx, rz = jnp.meshgrid(cell_x, row_z)
    cx = cx.reshape(-1)
    rz = rz.reshape(-1)
    red_x = (cx[:, None] + offsets_x[None, :]).reshape(-1)
    red_z = (rz[:, None] + offsets_z[None, :]).reshape(-1)

    # Blue mirrors Red across x=0.
    blue_x = -red_x
    blue_z = red_z

    x = x.at[RED_SOLDIER_START:RED_SOLDIER_END].set(red_x)
    z = z.at[RED_SOLDIER_START:RED_SOLDIER_END].set(red_z)
    x = x.at[BLUE_SOLDIER_START:BLUE_SOLDIER_END].set(blue_x)
    z = z.at[BLUE_SOLDIER_START:BLUE_SOLDIER_END].set(blue_z)

    # Fixed speed: the entire reset state is deterministic.
    speed = speed.at[
        RED_SOLDIER_START:
        BLUE_SOLDIER_END
    ].set(
        INITIAL_SOLDIER_SPEED
    )

    return {
        "x": x,
        "z": z,
        "vx": vx,
        "vz": vz,
        "hp": hp,
        "alive": alive,
        "attack_timer": attack_timer,
        "speed": speed,
        "time": jnp.array(
            0.0,
            dtype=jnp.float32,
        ),
        "done": jnp.array(
            False
        ),
    }


reset_parallel = jax.jit(jax.vmap(reset_one))


def reset_env(key):
    """Single-environment reset (fix #4)."""
    return reset_one(key)


def reset(key):
    return reset_parallel(random.split(key, N_ENVS))


def reset_batch(key, n):
    return jax.vmap(reset_one)(random.split(key, n))


def get_alive_mask(state):
    """fix #5"""
    return state["alive"]


def tree_index(tree, i):
    """Take game i out of a batched state pytree (fix #6 helper)."""
    return jax.tree_util.tree_map(lambda a: a[i], tree)


# ------------------------------------------------------------
# PHYSICS / COMBAT
# ------------------------------------------------------------

def decode_actions(action):
    action = action.reshape(N_SOLDIERS_PER_TEAM, 3)
    raw_dx, raw_dz, raw_attack = action[:, 0], action[:, 1], action[:, 2]
    norm = jnp.sqrt(raw_dx * raw_dx + raw_dz * raw_dz + 1e-8)
    return raw_dx / norm, raw_dz / norm, (raw_attack > 0.5).astype(jnp.float32)


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
    valid = (alive[:, None] > 0) & (alive[None, :] > 0) & (dist2 > 1e-10)
    corr = overlap * valid.astype(jnp.float32) / (dist + 1e-8)

    push_x = jnp.sum(corr * dx, axis=1)
    push_z = jnp.sum(corr * dz, axis=1)

    push_x = jnp.where(commander_mask > 0, 0.0, push_x * 0.5)
    push_z = jnp.where(commander_mask > 0, 0.0, push_z * 0.5)

    # Soldiers in attack cooldown are physically frozen.  Do not let the
    # separation solver move them either; otherwise they can drift into a wall
    # even though their commanded movement is zero.
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

    red_dx, red_dz, red_at = decode_actions(red_action)
    blue_dx, blue_dz, blue_at = decode_actions(blue_action)

    move_dx = jnp.zeros(N_UNITS, dtype=jnp.float32)
    move_dz = jnp.zeros(N_UNITS, dtype=jnp.float32)
    move_at = jnp.zeros(N_UNITS, dtype=jnp.float32)

    move_dx = move_dx.at[ALL_SOLDIER_INDICES].set(jnp.concatenate([red_dx, blue_dx]))
    move_dz = move_dz.at[ALL_SOLDIER_INDICES].set(jnp.concatenate([red_dz, blue_dz]))
    move_at = move_at.at[ALL_SOLDIER_INDICES].set(jnp.concatenate([red_at, blue_at]))

    attack_timer = jnp.maximum(0.0, attack_timer - DT)
    can_move = attack_timer <= 0
    distance = speed * DT

    nx = x + move_dx * distance * can_move
    nz = z + move_dz * distance * can_move

    radius = commander_mask * COMMANDER_RADIUS + soldier_mask * SOLDIER_RADIUS

    inside = ((nx >= -HALF_FIELD + radius) & (nx <= HALF_FIELD - radius) &
              (nz >= -HALF_FIELD + radius) & (nz <= HALF_FIELD - radius))

    valid_move = inside & (~wall_blocked(nx, nz, radius)) & (alive > 0) & can_move

    nx = jnp.where(valid_move, nx, x)
    nz = jnp.where(valid_move, nz, z)
    vx = jnp.where(valid_move, move_dx, 0.0)
    vz = jnp.where(valid_move, move_dz, 0.0)

    movable = (commander_mask > 0) | ((soldier_mask > 0) & can_move & (alive > 0))
    px, pz = pairwise_separation(nx, nz, alive, movable)

    # Apply separation only if the corrected position remains outside walls.
    # If a separation push would move a unit into a wall, reject that push and
    # keep the already wall-safe position from the movement stage. This is
    # especially important for soldiers frozen by attack cooldown.
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

    valid_target = (enemy_mask & (alive[None, :] > 0) &
                    (dist2 <= ATTACK_RANGE ** 2) &
                    (~(ALL_UNIT_INDICES[None, :] == ALL_SOLDIER_INDICES[:, None])))

    target = jnp.argmin(jnp.where(valid_target, dist2, 1e9), axis=1)
    has_target = jnp.any(valid_target, axis=1)

    attack_attempt = ((move_at[ALL_SOLDIER_INDICES] > 0.5) &
                      (attack_timer[ALL_SOLDIER_INDICES] <= 0) &
                      (alive[ALL_SOLDIER_INDICES] > 0))
    can_attack = attack_attempt & has_target

    damage_values = ATTACK_DAMAGE * can_attack.astype(jnp.float32)

    attack_damage = jnp.zeros(N_UNITS, dtype=jnp.float32)
    attack_damage = attack_damage.at[target].add(damage_values)

    dmg_red = (jnp.sum(attack_damage[RED_SOLDIER_START:RED_SOLDIER_END])
               + attack_damage[RED_COMMANDER_INDEX])
    dmg_blue = (jnp.sum(attack_damage[BLUE_SOLDIER_START:BLUE_SOLDIER_END])
                + attack_damage[BLUE_COMMANDER_INDEX])

    shaping_red = (dmg_blue - dmg_red) / N_SOLDIERS_PER_TEAM * SHAPING_COEF

    hp = hp - attack_damage
    alive = jnp.where(hp <= 0, 0.0, alive)

    old_t = attack_timer[ALL_SOLDIER_INDICES]
    attack_timer = attack_timer.at[ALL_SOLDIER_INDICES].set(
        jnp.where(attack_attempt, ATTACK_COOLDOWN, old_t))

    new_time = state["time"] + DT

    red_cmd = alive[RED_COMMANDER_INDEX] > 0
    blue_cmd = alive[BLUE_COMMANDER_INDEX] > 0
    commander_done = (~red_cmd) | (~blue_cmd)
    timeout = new_time >= MAX_TIME
    done = commander_done | timeout

    red_n = jnp.sum(alive[RED_SOLDIER_START:RED_SOLDIER_END])
    blue_n = jnp.sum(alive[BLUE_SOLDIER_START:BLUE_SOLDIER_END])

    commander_reward = red_cmd.astype(jnp.float32) - blue_cmd.astype(jnp.float32)
    timeout_reward = (red_n - blue_n) / N_SOLDIERS_PER_TEAM

    terminal_red = jnp.where(commander_done, commander_reward,
                             jnp.where(timeout, timeout_reward, 0.0))

    reward_red = terminal_red + shaping_red

    next_state = {
        "x": nx, "z": nz, "vx": vx, "vz": vz,
        "hp": hp, "alive": alive,
        "attack_timer": attack_timer, "speed": speed,
        "time": new_time, "done": done,
    }
    return next_state, reward_red, -reward_red, done


def env_step(state, red_action, blue_action):
    """fix #2 : canonical single-environment step name."""
    return step_one(state, red_action, blue_action)


step_parallel = jax.jit(jax.vmap(step_one, in_axes=(0, 0, 0)))


# ------------------------------------------------------------
# OBSERVATION (fix #3)
# ------------------------------------------------------------

def make_observation(state, perspective_team):
    """
    Canonical (mirrored) observation for one environment.
    Red  : as-is.
    Blue : x -> -x, vx -> -vx, terrain mirrored in x.
    Layout: [terrain(100)] + [units(202 x 10)]
    """
    x, z = state["x"], state["z"]
    vx, vz = state["vx"], state["vz"]
    hp, alive = state["hp"], state["alive"]

    blue = (perspective_team == 1.0)

    tx = jnp.where(blue, -x, x)
    tvx = jnp.where(blue, -vx, vx)

    own = (teams == perspective_team)
    enemy = ~own

    unit_features = jnp.stack([
        tx / HALF_FIELD, z / HALF_FIELD, tvx, vz, hp,
        own.astype(jnp.float32), enemy.astype(jnp.float32),
        soldier_mask, commander_mask, alive,
    ], axis=-1).reshape(-1)

    t2d = terrain.reshape(TERRAIN_RES, TERRAIN_RES)
    tt = jnp.where(blue, t2d[:, ::-1], t2d).reshape(-1)

    return jnp.concatenate([tt, unit_features])


def make_observation_batch(state, perspective_team):
    return jax.vmap(make_observation, in_axes=(0, None))(state, perspective_team)


make_observation_batch_jit = jax.jit(make_observation_batch)


def local_to_world_action(action, perspective_team):
    blue = (perspective_team == 1.0)
    ldx = action[..., 0::3]
    ldz = action[..., 1::3]
    at = action[..., 2::3]

    wdx = jnp.where(blue, -ldx, ldx)

    out = jnp.zeros_like(action)
    out = out.at[..., 0::3].set(wdx)
    out = out.at[..., 1::3].set(ldz)
    out = out.at[..., 2::3].set(at)
    return out


print()
print("Observation size :", OBS_SIZE)
print("Action size      :", ACTION_SIZE)
print("Simulation steps :", MAX_STEPS)
print("Total units      :", N_UNITS)


# ============================================================
# PART 2 : POLICY NETWORK  (fix #7, #8)
# ============================================================

HIDDEN1 = 256
HIDDEN2 = 256

GAMMA = 0.999
GAE_LAMBDA = 0.97
CLIP_EPS = 0.20
VALUE_COEF = 0.5
LEARNING_RATE = 3e-4

ENTROPY_START = 0.005
ENTROPY_END = 0.0005

# Exploration control for the continuous movement policy.
# The policy starts with a moderate angular standard deviation and is prevented
# from becoming excessively random. A small regularizer also pulls logstd
# back toward the target instead of allowing entropy to drift to its maximum.
LOGSTD_INIT = -1.0
LOGSTD_MIN = -3.0
LOGSTD_MAX = -0.3
LOGSTD_TARGET = -1.0
LOGSTD_REG_COEF = 0.01

PPO_EPOCHS = 4
MINIBATCHES = 8

ROLLOUT_STEPS = MAX_STEPS
BATCH_SIZE = ROLLOUT_STEPS * N_ENVS * 2
MINIBATCH_SIZE = BATCH_SIZE // MINIBATCHES

PPO_UPDATES_PER_GENERATION = 20
N_GENERATIONS = 20
EVAL_GAMES_PER_SIDE = 32
EVAL_GAMES = EVAL_GAMES_PER_SIDE * 2

# Elite evaluation uses deterministic starting-state variants.
# Training/reset itself remains fully deterministic and unchanged.
EVAL_Z_OFFSETS = jnp.array(
    [-0.72, -0.48, -0.24, 0.00, 0.24, 0.48, 0.72, 0.00],
    dtype=jnp.float32,
)

ANGLE_MEAN_IDX = jnp.arange(0, ACTION_SIZE, 3)
ANGLE_LOGSTD_IDX = jnp.arange(1, ACTION_SIZE, 3)
ATTACK_IDX = jnp.arange(2, ACTION_SIZE, 3)


def init_policy(key):
    # fix #7 : four INDEPENDENT keys
    k1, k2, k3, k4 = random.split(key, 4)
    p = {
        "W1": random.normal(k1, (OBS_SIZE, HIDDEN1)) * jnp.sqrt(2.0 / OBS_SIZE),
        "b1": jnp.zeros((HIDDEN1,)),
        "W2": random.normal(k2, (HIDDEN1, HIDDEN2)) * jnp.sqrt(2.0 / HIDDEN1),
        "b2": jnp.zeros((HIDDEN2,)),
        "Wa": random.normal(k3, (HIDDEN2, ACTION_SIZE)) * 0.01,
        "ba": jnp.zeros((ACTION_SIZE,)),
        "Wv": random.normal(k4, (HIDDEN2, 1)) * 0.01,
        "bv": jnp.zeros((1,)),
    }
    p["ba"] = p["ba"].at[1::3].set(LOGSTD_INIT)
    return p


def policy_forward(params, obs):
    h1 = jnp.tanh(obs @ params["W1"] + params["b1"])
    h2 = jnp.tanh(h1 @ params["W2"] + params["b2"])
    action_output = h2 @ params["Wa"] + params["ba"]
    value = (h2 @ params["Wv"] + params["bv"])[..., 0]
    return action_output, value


policy_forward_jit = jax.jit(policy_forward)


def wrap_angle(a):
    return (a + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def action_logprob(action_output, local_action):
    angle_mean = action_output[:, ANGLE_MEAN_IDX]
    logstd = jnp.clip(action_output[:, ANGLE_LOGSTD_IDX], LOGSTD_MIN, LOGSTD_MAX)
    std = jnp.exp(logstd)

    dx = local_action[:, 0::3]
    dz = local_action[:, 1::3]
    a = jnp.arctan2(dz, dx)
    diff = wrap_angle(a - angle_mean)

    lp_angle = (-0.5 * (diff / std) ** 2 - logstd - 0.5 * jnp.log(2.0 * jnp.pi))
    lp_angle = jnp.sum(lp_angle, axis=1)

    logits = action_output[:, ATTACK_IDX]
    at = local_action[:, ATTACK_IDX]
    lp_attack = (at * (-jnp.logaddexp(0.0, -logits)) +
                 (1.0 - at) * (-jnp.logaddexp(0.0, logits)))
    lp_attack = jnp.sum(lp_attack, axis=1)

    return lp_angle + lp_attack


def policy_entropy(action_output):
    logstd = jnp.clip(action_output[:, ANGLE_LOGSTD_IDX], LOGSTD_MIN, LOGSTD_MAX)
    e_angle = jnp.sum(logstd + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e), axis=1)

    p = jax.nn.sigmoid(action_output[:, ATTACK_IDX])
    e_attack = -(p * jnp.log(p + 1e-8) + (1.0 - p) * jnp.log(1.0 - p + 1e-8))
    e_attack = jnp.sum(e_attack, axis=1)
    return e_angle + e_attack


def sample_action(params, obs, key):
    action_output, value = policy_forward(params, obs)
    B = obs.shape[0]
    k_noise, k_attack = random.split(key)

    mean = action_output[:, ANGLE_MEAN_IDX]
    logstd = jnp.clip(action_output[:, ANGLE_LOGSTD_IDX], LOGSTD_MIN, LOGSTD_MAX)
    std = jnp.exp(logstd)

    angle = mean + std * random.normal(k_noise, (B, N_SOLDIERS_PER_TEAM))
    dx, dz = jnp.cos(angle), jnp.sin(angle)

    prob = jax.nn.sigmoid(action_output[:, ATTACK_IDX])
    at = random.bernoulli(k_attack, prob).astype(jnp.float32)

    la = jnp.zeros((B, ACTION_SIZE), dtype=jnp.float32)
    la = la.at[:, 0::3].set(dx)
    la = la.at[:, 1::3].set(dz)
    la = la.at[:, 2::3].set(at)

    return la, action_logprob(action_output, la), value


def deterministic_local_action(params, obs):
    """obs : [B, OBS_SIZE]"""
    action_output, value = policy_forward(params, obs)
    mean = action_output[:, ANGLE_MEAN_IDX]
    dx, dz = jnp.cos(mean), jnp.sin(mean)
    at = (jax.nn.sigmoid(action_output[:, ATTACK_IDX]) >= 0.5).astype(jnp.float32)

    la = jnp.zeros((obs.shape[0], ACTION_SIZE), dtype=jnp.float32)
    la = la.at[:, 0::3].set(dx)
    la = la.at[:, 1::3].set(dz)
    la = la.at[:, 2::3].set(at)
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


def ppo_loss(params, obs, local_actions, old_log_prob, advantages, returns, ent_coef):
    action_output, values = policy_forward(params, obs)
    new_log_prob = action_logprob(action_output, local_actions)
    entropy = jnp.mean(policy_entropy(action_output))

    # Keep movement uncertainty centered near LOGSTD_TARGET instead of letting
    # the learned logstd drift toward the maximum-entropy boundary.
    raw_logstd = action_output[:, ANGLE_LOGSTD_IDX]
    logstd_reg = jnp.mean((raw_logstd - LOGSTD_TARGET) ** 2)

    ratio = jnp.exp(new_log_prob - old_log_prob)
    unclipped = ratio * advantages
    clipped = jnp.clip(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantages

    policy_loss = -jnp.mean(jnp.minimum(unclipped, clipped))
    value_loss = 0.5 * jnp.mean((returns - values) ** 2)

    total = (
        policy_loss
        + VALUE_COEF * value_loss
        - ent_coef * entropy
        + LOGSTD_REG_COEF * logstd_reg
    )

    metrics = {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "entropy_per_soldier": entropy / float(N_SOLDIERS_PER_TEAM),
        "logstd_mean": jnp.mean(jnp.clip(raw_logstd, LOGSTD_MIN, LOGSTD_MAX)),
        "logstd_reg": logstd_reg,
        "approx_kl": jnp.mean(old_log_prob - new_log_prob),
    }
    return total, metrics


@jax.jit
def ppo_update_minibatch(params, opt_state, obs, local_actions,
                         old_log_prob, advantages, returns, ent_coef):
    def loss_fn(p):
        return ppo_loss(p, obs, local_actions, old_log_prob,
                        advantages, returns, ent_coef)

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, metrics


# ------------------------------------------------------------
# ROLLOUT  (fix #9 : correct PRNG handling inside scan)
# ------------------------------------------------------------

def reset_finished_envs(state, done, key):
    fresh = reset_parallel(random.split(key, N_ENVS))

    def merge(old, new):
        mask = done if old.ndim == 1 else done.reshape((N_ENVS,) + (1,) * (old.ndim - 1))
        return jnp.where(mask, new, old)

    return jax.tree_util.tree_map(merge, state, fresh)


@functools.partial(jax.jit, static_argnums=())
def collect_rollout(params, state, key):
    def body(carry, _):
        st, k = carry

        red_obs = make_observation_batch(st, 0.0)
        blue_obs = make_observation_batch(st, 1.0)

        k, kr, kb, kreset = random.split(k, 4)

        red_la, red_lp, red_v = sample_action(params, red_obs, kr)
        blue_la, blue_lp, blue_v = sample_action(params, blue_obs, kb)

        red_wa = local_to_world_action(red_la, 0.0)
        blue_wa = local_to_world_action(blue_la, 1.0)

        nxt, rr, br, done = jax.vmap(step_one, in_axes=(0, 0, 0))(st, red_wa, blue_wa)

        red_cmd = nxt["alive"][:, RED_COMMANDER_INDEX] > 0
        blue_cmd = nxt["alive"][:, BLUE_COMMANDER_INDEX] > 0

        reason = jnp.where(done & red_cmd & (~blue_cmd), 1,
                   jnp.where(done & blue_cmd & (~red_cmd), 2,
                     jnp.where(done & (~red_cmd) & (~blue_cmd), 4,
                       jnp.where(done, 3, 0))))

        new_state = reset_finished_envs(nxt, done, kreset)

        out = (red_obs, blue_obs, red_la, blue_la, red_lp, blue_lp,
               red_v, blue_v, rr, br, done, reason)
        return (new_state, k), out

    (final_state, final_key), traj = lax.scan(body, (state, key), None,
                                              length=ROLLOUT_STEPS)

    (red_obs, blue_obs, red_la, blue_la, red_lp, blue_lp,
     red_v, blue_v, red_r, blue_r, dones, reasons) = traj

    f_red_obs = make_observation_batch(final_state, 0.0)
    f_blue_obs = make_observation_batch(final_state, 1.0)
    _, red_last_v = policy_forward(params, f_red_obs)
    _, blue_last_v = policy_forward(params, f_blue_obs)

    red_adv, red_ret = compute_gae(red_r, red_v, dones, red_last_v)
    blue_adv, blue_ret = compute_gae(blue_r, blue_v, dones, blue_last_v)

    obs = jnp.concatenate([red_obs, blue_obs], axis=1).reshape(-1, OBS_SIZE)
    acts = jnp.concatenate([red_la, blue_la], axis=1).reshape(-1, ACTION_SIZE)
    logp = jnp.concatenate([red_lp, blue_lp], axis=1).reshape(-1)
    adv = jnp.concatenate([red_adv, blue_adv], axis=1).reshape(-1)
    ret = jnp.concatenate([red_ret, blue_ret], axis=1).reshape(-1)

    adv = normalize_advantages(adv)

    stats = {
        "battles": jnp.sum(reasons != 0),
        "red_wins": jnp.sum(reasons == 1),
        "blue_wins": jnp.sum(reasons == 2),
        "timeouts": jnp.sum(reasons == 3),
        "draws": jnp.sum(reasons == 4),
    }

    return final_state, final_key, obs, acts, logp, adv, ret, stats


def run_ppo_update(params, opt_state, state, key, global_update, total_updates):
    t0 = time.time()

    state, key, obs, acts, logp, adv, ret, stats = collect_rollout(params, state, key)

    alpha = global_update / max(1, total_updates - 1)
    ent_coef = jnp.array(ENTROPY_START + alpha * (ENTROPY_END - ENTROPY_START),
                         dtype=jnp.float32)

    n = obs.shape[0]
    usable = (n // MINIBATCH_SIZE) * MINIBATCH_SIZE
    n_mb = usable // MINIBATCH_SIZE

    acc = []
    for _ in range(PPO_EPOCHS):
        key, sk = random.split(key)
        perm = random.permutation(sk, n)[:usable]
        for mb in range(n_mb):
            idx = perm[mb * MINIBATCH_SIZE:(mb + 1) * MINIBATCH_SIZE]
            params, opt_state, m = ppo_update_minibatch(
                params, opt_state,
                obs[idx], acts[idx], logp[idx], adv[idx], ret[idx], ent_coef)
            acc.append(m)

    metrics = {k: float(np.mean([float(m[k]) for m in acc])) for k in acc[0]}
    metrics["entropy_coef"] = float(ent_coef)

    stats = {k: int(v) for k, v in stats.items()}
    return params, opt_state, state, key, time.time() - t0, stats, metrics


# ============================================================
# PART 4 : EVALUATION  (fix #10, #13)
# ============================================================

def make_evaluation_states(base_states):
    """Create deterministic, symmetric starting-state variants for Elite evaluation.

    Every game is still fully reproducible. The entire battle (commanders and
    soldiers) is shifted together in z, so Red/Blue symmetry is preserved.
    Training environments keep the original fixed reset state.
    """
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
    """
    Deterministic head-to-head over a batch of games.
    Records the FIRST termination of each game exactly.

    result codes:
        1 = Red won   (blue commander killed)
        2 = Blue won  (red commander killed)
        3 = Timeout
        4 = Draw      (both commanders killed)
        0 = never terminated (should not happen)
    """
    E = init_states["x"].shape[0]

    def body(carry, step_idx):
        st, finished, result, end_step, red_surv, blue_surv = carry

        red_a = deterministic_world_action_batch(params_red, st, 0.0)
        blue_a = deterministic_world_action_batch(params_blue, st, 1.0)

        nxt, rr, br, done = jax.vmap(step_one, in_axes=(0, 0, 0))(st, red_a, blue_a)

        newly = done & (~finished)

        red_cmd = nxt["alive"][:, RED_COMMANDER_INDEX] > 0
        blue_cmd = nxt["alive"][:, BLUE_COMMANDER_INDEX] > 0

        res_now = jnp.where(
            red_cmd & (~blue_cmd), 1,
            jnp.where(
                blue_cmd & (~red_cmd), 2,
                jnp.where(
                    (~red_cmd) & (~blue_cmd), 4,
                    jnp.where(done, 3, 0),
                ),
            ),
        )

        rs = jnp.sum(nxt["alive"][:, RED_SOLDIER_START:RED_SOLDIER_END], axis=1)
        bs = jnp.sum(nxt["alive"][:, BLUE_SOLDIER_START:BLUE_SOLDIER_END], axis=1)

        result = jnp.where(newly, res_now, result)
        end_step = jnp.where(newly, step_idx + 1, end_step)
        red_surv = jnp.where(newly, rs, red_surv)
        blue_surv = jnp.where(newly, bs, blue_surv)
        finished = finished | done

        return (nxt, finished, result, end_step, red_surv, blue_surv), None

    init = (init_states,
            jnp.zeros(E, dtype=bool),
            jnp.zeros(E, dtype=jnp.int32),
            jnp.full(E, MAX_STEPS, dtype=jnp.int32),
            jnp.zeros(E, dtype=jnp.float32),
            jnp.zeros(E, dtype=jnp.float32))

    (final_state, _, result, end_step, red_surv, blue_surv), _ = lax.scan(
        body, init, jnp.arange(MAX_STEPS))

    # Every simulation reaches MAX_STEPS at the latest, because step_one
    # declares timeout when time >= MAX_TIME. The fallback prevents an
    # unresolved code 0 from silently entering the Elite comparison.
    result = jnp.where(result == 0, 3, result)
    end_step = jnp.where(result == 3, jnp.minimum(end_step, MAX_STEPS), end_step)

    return result, end_step, red_surv, blue_surv


def evaluate_elite_match(candidate_params, elite_params, base_states, verbose=True):
    """
    Side A : candidate = Red , elite = Blue   (games 0 .. E-1)
    Side B : elite = Red , candidate = Blue   (games E .. 2E-1)
    Deterministic evaluation variants are reused symmetrically for both sides.
    """
    E = base_states["x"].shape[0]

    if verbose:
        print(f"  Elite match: side A (candidate = Red)  {E} games ...")
    rA, sA, redA, blueA = evaluate_match(candidate_params, elite_params, base_states)

    if verbose:
        print(f"  Elite match: side B (candidate = Blue) {E} games ...")
    rB, sB, redB, blueB = evaluate_match(elite_params, candidate_params, base_states)

    result = np.concatenate([np.asarray(rA), np.asarray(rB)])
    end_step = np.concatenate([np.asarray(sA), np.asarray(sB)])
    red_surv = np.concatenate([np.asarray(redA), np.asarray(redB)])
    blue_surv = np.concatenate([np.asarray(blueA), np.asarray(blueB)])

    # fix #10 : side determines which colour the candidate actually played
    candidate_is_red = np.concatenate([np.ones(E, dtype=bool), np.zeros(E, dtype=bool)])

    candidate_won = np.where(candidate_is_red, result == 1, result == 2)
    elite_won = np.where(candidate_is_red, result == 2, result == 1)

    candidate_surv = np.where(candidate_is_red, red_surv, blue_surv)
    elite_surv = np.where(candidate_is_red, blue_surv, red_surv)

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
        "candidate_wins": candidate_wins,
        "elite_wins": elite_wins,
        "timeouts": timeouts,
        "draws": draws,
        "unresolved": unresolved,
        "avg_candidate_time": float(np.mean(win_time[candidate_won])) if candidate_wins else float("nan"),
        "avg_elite_time": float(np.mean(win_time[elite_won])) if elite_wins else float("nan"),
        "winner": winner,
        "games_per_side": E,
    }


# ============================================================
# PART 5 : BEST BOUT RECORDING  (fix #6, #13, #14, #23)
# ============================================================

@jax.jit
def record_bout(params_red, params_blue, initial_state):
    """
    Deterministic single-game replay recording.
    Frame 0 of the returned arrays IS the exact starting state (fix #23).
    """
    def body(carry, step_idx):
        st, finished, result, end_step = carry

        red_a = deterministic_world_action_single(params_red, st, 0.0)
        blue_a = deterministic_world_action_single(params_blue, st, 1.0)

        nxt, rr, br, done = step_one(st, red_a, blue_a)

        newly = done & (~finished)

        red_cmd = nxt["alive"][RED_COMMANDER_INDEX] > 0
        blue_cmd = nxt["alive"][BLUE_COMMANDER_INDEX] > 0

        res_now = jnp.where(
            red_cmd & (~blue_cmd), 1,
            jnp.where(
                blue_cmd & (~red_cmd), 2,
                jnp.where(
                    (~red_cmd) & (~blue_cmd), 4,
                    jnp.where(done, 3, 0),
                ),
            ),
        )

        result = jnp.where(newly, res_now, result)
        end_step = jnp.where(newly, step_idx + 1, end_step)
        finished = finished | done

        out = (nxt["x"], nxt["z"], nxt["hp"], nxt["alive"], red_a, blue_a)
        return (nxt, finished, result, end_step), out

    init = (initial_state,
            jnp.array(False),
            jnp.array(0, dtype=jnp.int32),
            jnp.array(MAX_STEPS, dtype=jnp.int32))

    (final_state, _, result, end_step), traj = lax.scan(
        body, init, jnp.arange(MAX_STEPS))

    result = jnp.where(result == 0, 3, result)

    xs, zs, hps, alives, red_actions, blue_actions = traj

    # prepend the true t=0 frame
    xs = jnp.concatenate([initial_state["x"][None, :], xs], axis=0)
    zs = jnp.concatenate([initial_state["z"][None, :], zs], axis=0)
    hps = jnp.concatenate([initial_state["hp"][None, :], hps], axis=0)
    alives = jnp.concatenate([initial_state["alive"][None, :], alives], axis=0)

    return {
        "x": xs, "z": zs, "hp": hps, "alive": alives,
        "red_actions": red_actions, "blue_actions": blue_actions,
        "result": result, "end_step": end_step,
        "final_state": final_state,
    }


@jax.jit
def verify_bout(initial_state, red_actions, blue_actions):
    """Replay the stored action sequence from the stored start state."""
    def body(st, actions):
        ra, ba = actions
        nxt, _, _, _ = step_one(st, ra, ba)
        return nxt, (nxt["x"], nxt["z"], nxt["alive"])

    final_state, traj = lax.scan(body, initial_state, (red_actions, blue_actions))
    xs, zs, alives = traj

    xs = jnp.concatenate([initial_state["x"][None, :], xs], axis=0)
    zs = jnp.concatenate([initial_state["z"][None, :], zs], axis=0)
    alives = jnp.concatenate([initial_state["alive"][None, :], alives], axis=0)
    return xs, zs, alives


def select_best_bout_index(match):
    """
    Winner-side victory, fastest first, then most survivors (tie-break).
    """
    winner = match["winner"]
    won = match["candidate_won"] if winner == 1 else match["elite_won"]
    idx = np.where(won)[0]
    if len(idx) == 0:
        return None

    surv = match["candidate_surv"] if winner == 1 else match["elite_surv"]
    order = np.lexsort((-surv[idx], match["end_step"][idx]))
    return int(idx[order[0]])


def save_best_bout(path, generation, match, best_idx, bout,
                   verification_pass, max_state_error,
                   winner_params, winner_label, winner_team):
    steps = int(bout["end_step"])
    frames = steps + 1

    data = {
        "generation": np.array(generation, dtype=np.int32),
        "winner_code": np.array(match["winner"], dtype=np.int32),
        "winner_label": np.array(winner_label),
        "winner_team": np.array(winner_team, dtype=np.int32),
        "result_code": np.array(int(bout["result"]), dtype=np.int32),
        "win_time": np.array(steps * DT, dtype=np.float32),
        "end_step": np.array(steps, dtype=np.int32),
        "dt": np.array(DT, dtype=np.float32),
        "candidate_is_red": np.array(bool(match["candidate_is_red"][best_idx])),
        "candidate_wins": np.array(match["candidate_wins"], dtype=np.int32),
        "elite_wins": np.array(match["elite_wins"], dtype=np.int32),
        "field_size": np.array(FIELD_SIZE, dtype=np.float32),
        "terrain_res": np.array(TERRAIN_RES, dtype=np.int32),
        "obs_size": np.array(OBS_SIZE, dtype=np.int32),
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

        # fix #14 : real verification outcome, not a hardcoded 1
        "verification_pass": np.array(1 if verification_pass else 0, dtype=np.int32),
        "max_state_error": np.array(max_state_error, dtype=np.float32),
    }

    for k, v in winner_params.items():
        data[f"winner_param_{k}"] = np.asarray(v)

    np.savez_compressed(path, **data)
    return path


# ============================================================
# PART 6 : CHECKPOINT / PARAM I-O  (fix #11, #12)
# ============================================================

def save_params(path, params, metadata=None):
    data = {f"param_{k}": np.asarray(v) for k, v in params.items()}
    if metadata is not None:
        data["metadata_json"] = np.array(json.dumps(metadata))
    np.savez(path, **data)
    return path


def load_params(path):
    d = np.load(path, allow_pickle=False)
    params = {k[len("param_"):]: jnp.asarray(d[k])
              for k in d.files if k.startswith("param_")}
    meta = {}
    if "metadata_json" in d.files:
        try:
            meta = json.loads(str(d["metadata_json"]))
        except Exception:
            meta = {}
    return params, meta


def save_checkpoint(path, params, opt_state, generation, ppo_index,
                    elite_path, master_key):
    """fix #11 : opt_state saved leaf-by-leaf so it can actually be restored."""
    data = {f"param_{k}": np.asarray(v) for k, v in params.items()}

    leaves = jax.tree_util.tree_leaves(opt_state)
    for i, leaf in enumerate(leaves):
        data[f"opt_{i}"] = np.asarray(leaf)
    data["opt_n_leaves"] = np.array(len(leaves), dtype=np.int32)

    data["generation"] = np.array(generation, dtype=np.int32)
    data["ppo_index"] = np.array(ppo_index, dtype=np.int32)
    data["elite_path"] = np.array(elite_path if elite_path else "")
    data["master_key"] = np.asarray(random.key_data(master_key))

    np.savez(path, **data)
    return path


def policy_params_compatible(params):
    """Return True only when a saved policy matches the current architecture."""
    required_shapes = {
        "W1": (OBS_SIZE, HIDDEN1),
        "b1": (HIDDEN1,),
        "W2": (HIDDEN1, HIDDEN2),
        "b2": (HIDDEN2,),
        "Wa": (HIDDEN2, ACTION_SIZE),
        "ba": (ACTION_SIZE,),
        "Wv": (HIDDEN2, 1),
        "bv": (1,),
    }
    if set(params.keys()) != set(required_shapes.keys()):
        return False
    return all(tuple(np.asarray(params[k]).shape) == shape
               for k, shape in required_shapes.items())


def saved_policy_file_compatible(path):
    """Check a saved Elite/checkpoint parameter block without loading Optax state."""
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


def load_checkpoint(path):
    d = np.load(path, allow_pickle=False)

    params = {k[len("param_"):]: jnp.asarray(d[k])
              for k in d.files if k.startswith("param_")}

    if not policy_params_compatible(params):
        raise ValueError(
            f"Incompatible checkpoint architecture: {path} "
            f"(current OBS_SIZE={OBS_SIZE})"
        )

    # Rebuild the optax state structure from a fresh init, then swap leaves in.
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

    return params, opt_state, generation, ppo_index, elite_path, master_key


def generation_number(path):
    try:
        return int(os.path.basename(path).split("generation_")[1].split(".npz")[0])
    except Exception:
        return -1


def find_latest_elite():
    files = glob.glob(os.path.join(ELITE_DIR, "generation_*.npz"))
    compatible = [f for f in files if saved_policy_file_compatible(f)]
    if not compatible:
        if files:
            print(
                f"Ignoring {len(files)} incompatible Elite file(s) "
                f"(current OBS_SIZE={OBS_SIZE})."
            )
        return None
    return sorted(compatible, key=generation_number)[-1]


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
            f"(current OBS_SIZE={OBS_SIZE})."
        )

    if not compatible:
        return None

    return sorted(compatible, key=os.path.getmtime)[-1]


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

    ckpt = find_latest_checkpoint() if resume else None
    if ckpt is not None:
        try:
            (resumed_params, resumed_opt, g, p, saved_elite, master_key) = load_checkpoint(ckpt)
            start_generation = g
            start_ppo_index = p + 1
            if start_ppo_index > PPO_UPDATES_PER_GENERATION:
                start_ppo_index = 1
                start_generation += 1
                resumed_params = None
                resumed_opt = None
            print(f"Resuming from checkpoint: generation {start_generation}, "
                  f"PPO {start_ppo_index}/{PPO_UPDATES_PER_GENERATION}")
        except (ValueError, KeyError, OSError, EOFError) as exc:
            print(f"Checkpoint resume skipped: {exc}")
            resumed_params = None
            resumed_opt = None
            start_generation = 1
            start_ppo_index = 1

    # ---- Generation 0 elite (fix #12 : random init fallback) ----
    latest_elite = find_latest_elite()
    if latest_elite is None:
        master_key, ik = random.split(master_key)
        params0 = init_policy(ik)
        latest_elite = save_params(
            os.path.join(ELITE_DIR, "generation_0000.npz"),
            params0,
            {"generation": 0, "source": "random_init"},
        )
        print("Created Generation 0 Elite from fresh random initialization.")
    else:
        print("Existing Elite found:", os.path.basename(latest_elite))

    print()
    print("============================================")
    print("PPO + ELITE SELF-PLAY")
    print("============================================")
    print(f"Observation       : {OBS_SIZE}")
    print(f"Action            : {ACTION_SIZE}")
    print(f"Environments      : {N_ENVS}")
    print(f"Rollout steps     : {ROLLOUT_STEPS}  ({ROLLOUT_STEPS * DT:.1f} sim sec)")
    print(f"PPO batch         : {BATCH_SIZE}")
    print(f"Minibatch         : {MINIBATCH_SIZE}")
    print(f"PPO / generation  : {PPO_UPDATES_PER_GENERATION}")
    print(f"Elite eval games  : {EVAL_GAMES}")
    print(f"Entropy coef      : {ENTROPY_START} -> {ENTROPY_END}")
    print(f"Logstd             : init {LOGSTD_INIT:.1f}, target {LOGSTD_TARGET:.1f}, range [{LOGSTD_MIN:.1f}, {LOGSTD_MAX:.1f}]")
    print(f"Logstd regularizer : {LOGSTD_REG_COEF}")
    print(f"Output directory  : {os.path.abspath(BASE_DIR)}")
    print()

    total_updates = n_generations * PPO_UPDATES_PER_GENERATION

    for generation in range(start_generation, n_generations + 1):
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
            candidate_params = jax.tree_util.tree_map(lambda a: a.copy(), elite_params)
            candidate_opt = optimizer.init(candidate_params)
            first_ppo = 1

        master_key, rk, ek = random.split(master_key, 3)
        env_state = reset(rk)
        env_key = ek

        cum = {"battles": 0, "red_wins": 0, "blue_wins": 0, "timeouts": 0, "draws": 0}

        for ppo_index in range(first_ppo, PPO_UPDATES_PER_GENERATION + 1):
            global_update = (generation - 1) * PPO_UPDATES_PER_GENERATION + (ppo_index - 1)

            (candidate_params, candidate_opt, env_state, env_key,
             elapsed, stats, metrics) = run_ppo_update(
                candidate_params, candidate_opt, env_state, env_key,
                global_update, total_updates)

            for k in cum:
                cum[k] += stats[k]

            save_checkpoint(
                os.path.join(CHECKPOINT_DIR,
                             f"checkpoint_g{generation:04d}_p{ppo_index:03d}.npz"),
                candidate_params, candidate_opt, generation, ppo_index,
                latest_elite, master_key)

            kills = cum["red_wins"] + cum["blue_wins"]
            print(f"Generation {generation} | "
                  f"PPO {ppo_index:2d}/{PPO_UPDATES_PER_GENERATION} | "
                  f"{elapsed:5.1f} s/update | "
                  f"Battles {cum['battles']:4d} | "
                  f"Commander Kills {kills:3d} | "
                  f"entropy {metrics['entropy']:7.2f} "
                  f"({metrics['entropy_per_soldier']:.3f}/soldier) "
                  f"logstd {metrics['logstd_mean']:.3f}")

        print()
        print("PPO summary")
        print("------------------------------------------")
        print(f"Battles completed : {cum['battles']}")
        print(f"Commander kills   : {cum['red_wins'] + cum['blue_wins']}")
        print(f"Timeouts          : {cum['timeouts']}")
        print()

        # ---- Elite match ----
        master_key, evk = random.split(master_key)
        base_states = reset_batch(evk, EVAL_GAMES_PER_SIDE)
        eval_states = make_evaluation_states(base_states)

        print("Elite match")
        print("------------------------------------------")
        t0 = time.time()
        match = evaluate_elite_match(candidate_params, elite_params, eval_states)
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

        winner_label = "Candidate" if match["winner"] == 1 else "Elite"
        elite_changed = (match["winner"] == 1)
        print(f"Winner                 : {winner_label}")

        # ---- Best bout ----
        best_bout_saved = False
        best_idx = select_best_bout_index(match)

        if best_idx is not None:
            cand_is_red = bool(match["candidate_is_red"][best_idx])
            base_idx = best_idx % match["games_per_side"]
            init_state = tree_index(eval_states, base_idx)

            params_red = candidate_params if cand_is_red else elite_params
            params_blue = elite_params if cand_is_red else candidate_params

            bout = record_bout(params_red, params_blue, init_state)

            steps = int(bout["end_step"])
            vx_, vz_, va_ = verify_bout(init_state,
                                        bout["red_actions"][:steps],
                                        bout["blue_actions"][:steps])

            err_x = float(jnp.max(jnp.abs(vx_ - bout["x"][:steps + 1])))
            err_z = float(jnp.max(jnp.abs(vz_ - bout["z"][:steps + 1])))
            err_a = float(jnp.max(jnp.abs(va_ - bout["alive"][:steps + 1])))
            max_err = max(err_x, err_z, err_a)
            verification_pass = max_err < 1e-4

            if match["winner"] == 1:
                winner_team = 0 if cand_is_red else 1
                winner_params = candidate_params
            else:
                winner_team = 1 if cand_is_red else 0
                winner_params = elite_params

            surv = (match["candidate_surv"] if match["winner"] == 1
                    else match["elite_surv"])[best_idx]

            path = save_best_bout(
                os.path.join(BOUT_DIR, f"generation_{generation:04d}_best_bout.npz"),
                generation, match, best_idx, bout,
                verification_pass, max_err,
                winner_params, winner_label, winner_team)

            best_bout_saved = True

            print()
            print("Best Bout")
            print("------------------------------------------")
            print(f"Winner              : {winner_label} "
                  f"({'Red' if winner_team == 0 else 'Blue'})")
            print(f"Win time            : {steps * DT:.2f} s  ({steps} steps)")
            print(f"Winner survivors    : {int(surv)}")
            print(f"Replay verification : "
                  f"{'PASS' if verification_pass else f'FAIL (err {max_err:.2e})'}")
            print(f"Saved               : {path}")

        # ---- Elite update ----
        if elite_changed:
            latest_elite = save_params(
                os.path.join(ELITE_DIR, f"generation_{generation:04d}.npz"),
                candidate_params,
                {"generation": generation, "source": "candidate"})

        print()
        print("Generation summary")
        print("------------------------------------------")
        print(f"Elite changed : {'YES' if elite_changed else 'NO'}")
        print(f"Current Elite : {os.path.basename(latest_elite)}")
        print(f"Best Bout     : {'SAVED' if best_bout_saved else 'NONE'}")
        print()

        global LAST_COMPLETED_GENERATION
        LAST_COMPLETED_GENERATION = generation

    return latest_elite


# ============================================================
# PART 8 : 3D REPLAY  (fix #15-#23)
# ============================================================

RESULT_TEXT = {
    1: "BLUE COMMANDER KILLED",
    2: "RED COMMANDER KILLED",
    3: "TIMEOUT",
    4: "BOTH COMMANDERS KILLED",
    0: "UNRESOLVED",
}


def build_replay_html(bout_path, out_path=None):
    # Three.js replay based directly on the original working replay architecture.
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
    saved_obs_size = int(d["obs_size"]) if "obs_size" in d else OBS_SIZE
    if abs(saved_field_size - FIELD_SIZE) > 1e-6:
        raise ValueError(f"Best Bout field size mismatch: saved={saved_field_size}, current={FIELD_SIZE}")
    if saved_terrain_res != TERRAIN_RES or saved_obs_size != OBS_SIZE:
        raise ValueError("Best Bout is from an incompatible environment")
    verification_pass = bool(int(d["verification_pass"]))
    max_state_error = float(d["max_state_error"])
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

    payload = json.dumps({
        "generation": generation,
        "winnerLabel": winner_label,
        "winnerTeam": winner_team,
        "winnerSide": "Red" if winner_team == 0 else "Blue",
        "resultText": RESULT_TEXT.get(result_code, "UNKNOWN"),
        "winTime": win_time,
        "endStep": end_step,
        "dt": dt,
        "steps": n_frames - 1,
        "verificationPass": verification_pass,
        "maxStateError": max_state_error,
        "fieldSize": FIELD_SIZE,
        "walls": WALL_LIST,
        "x": np.round(x, 4).tolist(),
        "z": np.round(z, 4).tolist(),
        "hp": np.round(hp, 3).tolist(),
        "alive": alive.astype(np.uint8).tolist(),
        "redActions": np.round(red_actions, 3).tolist(),
        "blueActions": np.round(blue_actions, 3).tolist(),
        "attackCooldown": float(ATTACK_COOLDOWN),
    }, separators=(",", ":"))

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
const app=document.getElementById('app'), info=document.getElementById('info'), errorBox=document.getElementById('error'), timeline=document.getElementById('timeline'), statusEl=document.getElementById('status');
let replayTime=0, playing=false, speed=1, lastTs=null;
function showError(err){errorBox.style.display='block';errorBox.textContent=String(err&&err.stack?err.stack:err);}
try{
 const scene=new THREE.Scene(); scene.background=new THREE.Color(0x101010);
 const camera=new THREE.PerspectiveCamera(45,innerWidth/innerHeight,.1,250); camera.position.set(0,18,17);
 const renderer=new THREE.WebGLRenderer({antialias:true}); renderer.setPixelRatio(Math.min(devicePixelRatio||1,2)); renderer.setSize(innerWidth,innerHeight); app.appendChild(renderer.domElement);
 const controls=new OrbitControls(camera,renderer.domElement); controls.target.set(0,0,0); controls.enableDamping=true; controls.dampingFactor=.08; controls.update();
 scene.add(new THREE.HemisphereLight(0xffffff,0x444444,2.0)); const dir=new THREE.DirectionalLight(0xffffff,1.3); dir.position.set(6,14,8); scene.add(dir);
 const fieldSize=DATA.fieldSize, half=fieldSize/2;
 const ground=new THREE.Mesh(new THREE.PlaneGeometry(fieldSize,fieldSize),new THREE.MeshStandardMaterial({color:0x303030})); ground.rotation.x=-Math.PI/2; ground.position.y=-.02; scene.add(ground);
 const grid=new THREE.GridHelper(fieldSize,fieldSize,0x777777,0x444444); grid.position.y=.01; scene.add(grid);
 const redStart=new THREE.Mesh(new THREE.PlaneGeometry(2,fieldSize),new THREE.MeshBasicMaterial({color:0xd94b4b,transparent:true,opacity:.16,side:THREE.DoubleSide})); redStart.rotation.x=-Math.PI/2; redStart.position.set(-half+1.0,.015,0); scene.add(redStart);
 const blueStart=new THREE.Mesh(new THREE.PlaneGeometry(2,fieldSize),new THREE.MeshBasicMaterial({color:0x4b7bd9,transparent:true,opacity:.16,side:THREE.DoubleSide})); blueStart.rotation.x=-Math.PI/2; blueStart.position.set(half-1.0,.016,0); scene.add(blueStart);
 const wallGeo=new THREE.BoxGeometry(1,.7,1), wallMat=new THREE.MeshStandardMaterial({color:0x777777}); for(const p of DATA.walls){const w=new THREE.Mesh(wallGeo,wallMat);w.position.set(p[0],.35,p[1]);scene.add(w);}
 const soldierBodyGeo=new THREE.BoxGeometry(.22,.36,.22);
 const soldierHeadGeo=new THREE.SphereGeometry(.105,12,12);
 const redBodyMat=new THREE.MeshStandardMaterial({color:0xd94b4b});
 const blueBodyMat=new THREE.MeshStandardMaterial({color:0x4b7bd9});
 const redFaceMat=new THREE.MeshStandardMaterial({color:0xd94b4b,emissive:0x000000});
 const blueFaceMat=new THREE.MeshStandardMaterial({color:0x4b7bd9,emissive:0x000000});
 const attackFaceMat=new THREE.MeshStandardMaterial({color:0xffe13b,emissive:0x4a3a00,emissiveIntensity:.35});
 const redEyeMat=new THREE.LineBasicMaterial({color:0xd94b4b});
 const blueEyeMat=new THREE.LineBasicMaterial({color:0x4b7bd9});
 const attackEyeMat=new THREE.LineBasicMaterial({color:0xffe13b});
 const redSoldiers=[],blueSoldiers=[];
 function makeSoldier(team){
   const g=new THREE.Group();
   const body=new THREE.Mesh(soldierBodyGeo,team===0?redBodyMat:blueBodyMat); body.position.y=.18;
   const face=new THREE.Mesh(soldierHeadGeo,team===0?redFaceMat:blueFaceMat); face.position.set(0,.43,0);
   const lineGeo=new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0,.43,.06),new THREE.Vector3(0,.43,.34)]);
   const eyeLine=new THREE.Line(lineGeo,team===0?redEyeMat:blueEyeMat);
   g.add(body,face,eyeLine); scene.add(g);
   return {team,group:g,body,face,eyeLine,baseFace:team===0?redFaceMat:blueFaceMat,baseEye:team===0?redEyeMat:blueEyeMat};
 }
 for(let i=0;i<100;i++) redSoldiers.push(makeSoldier(0));
 for(let i=0;i<100;i++) blueSoldiers.push(makeSoldier(1));
 const cmdGeo=new THREE.BoxGeometry(.44,.8,.44), redCommander=new THREE.Mesh(cmdGeo,redBodyMat), blueCommander=new THREE.Mesh(cmdGeo,blueBodyMat); scene.add(redCommander,blueCommander);
 const crownGeo=new THREE.ConeGeometry(.22,.22,5), crownMat=new THREE.MeshStandardMaterial({color:0xffd83d}), redCrown=new THREE.Mesh(crownGeo,crownMat), blueCrown=new THREE.Mesh(crownGeo,crownMat);
 redCommander.add(redCrown); blueCommander.add(blueCrown); redCrown.position.set(0,.51,0); blueCrown.position.set(0,.51,0);
 function setSoldierVisual(s,attackNow){s.face.material=attackNow?attackFaceMat:s.baseFace;s.eyeLine.material=attackNow?attackEyeMat:s.baseEye;}
 function frameInfo(){const t=Math.max(0,Math.min(DATA.steps*DATA.dt,replayTime)),raw=t/DATA.dt,i=Math.min(Math.floor(raw),DATA.steps),a=i>=DATA.steps?0:raw-i;return{t,i,a};}
 function updateReplay(){const f=frameInfo(),i=f.i,a=f.a;timeline.value=String(i);const X0=DATA.x[i],Z0=DATA.z[i],A=DATA.alive[i],X1=i<DATA.steps?DATA.x[i+1]:X0,Z1=i<DATA.steps?DATA.z[i+1]:Z0,px=k=>X0[k]+(X1[k]-X0[k])*a,pz=k=>Z0[k]+(Z1[k]-Z0[k])*a;
  const redActions=DATA.redActions[i]||null,blueActions=DATA.blueActions[i]||null;let redAttackCount=0,blueAttackCount=0;
  for(let k=0;k<100;k++){
    const ridx=2+k,bidx=102+k;
    const rAttack=!!redActions && redActions[3*k+2]>.5 && A[ridx]>0;
    const bAttack=!!blueActions && blueActions[3*k+2]>.5 && A[bidx]>0;
    redSoldiers[k].group.position.set(px(ridx),0,pz(ridx));redSoldiers[k].group.visible=A[ridx]>0;setSoldierVisual(redSoldiers[k],rAttack);if(rAttack)redAttackCount++;
    blueSoldiers[k].group.position.set(px(bidx),0,pz(bidx));blueSoldiers[k].group.visible=A[bidx]>0;setSoldierVisual(blueSoldiers[k],bAttack);if(bAttack)blueAttackCount++;
  }
  redCommander.position.set(px(0),0,pz(0));redCommander.visible=A[0]>0;redCrown.visible=A[0]>0;
  blueCommander.position.set(px(1),0,pz(1));blueCommander.visible=A[1]>0;blueCrown.visible=A[1]>0;
  const ra=Array.from(A.slice(2,102)).reduce((sum,v)=>sum+v,0),ba=Array.from(A.slice(102,202)).reduce((sum,v)=>sum+v,0);
  const verify=DATA.verificationPass?'<span class="pass">Replay verification: PASS</span>':'<span class="fail">Replay verification: FAIL</span><br>Max state error: '+DATA.maxStateError.toExponential(2);info.innerHTML='<b>Generation '+DATA.generation+'</b><br>Winner: <b>'+DATA.winnerLabel+'</b> ('+DATA.winnerSide+')<br>Battle time: '+DATA.winTime.toFixed(1)+' s<br>Replay time: '+f.t.toFixed(2)+' s<br>Step: '+i+' / '+DATA.steps+'<hr>Red soldiers: '+ra+'<br>Blue soldiers: '+ba+'<br>Red Commander HP: '+DATA.hp[i][0].toFixed(2)+'<br>Blue Commander HP: '+DATA.hp[i][1].toFixed(2)+'<hr><span class="attack">Red attacks: '+redAttackCount+'</span><br><span class="attack">Blue attacks: '+blueAttackCount+'</span><hr>Result: <b>'+DATA.resultText+'</b><br>'+verify;statusEl.textContent='Generation '+DATA.generation+' | Best Bout | '+f.t.toFixed(2)+' s';}
 document.getElementById('play').onclick=()=>playing=true; document.getElementById('pause').onclick=()=>playing=false; document.getElementById('reset').onclick=()=>{playing=false;replayTime=0;updateReplay();}; document.querySelectorAll('button[data-speed]').forEach(b=>b.onclick=()=>speed=parseFloat(b.dataset.speed)); timeline.oninput=()=>{playing=false;replayTime=parseInt(timeline.value,10)*DATA.dt;updateReplay();};
 addEventListener('resize',()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);});
 function animate(ts){requestAnimationFrame(animate);if(lastTs===null)lastTs=ts;const delta=Math.min(.05,(ts-lastTs)/1000);lastTs=ts;if(playing){replayTime+=delta*speed;if(replayTime>=DATA.steps*DATA.dt){replayTime=DATA.steps*DATA.dt;playing=false;}}updateReplay();controls.update();renderer.render(scene,camera);}
 updateReplay(); requestAnimationFrame(animate);
}catch(err){showError(err);}
</script></body></html>'''
    html=html.replace("__MAXSTEP__",str(n_frames-1)).replace("__PAYLOAD__",payload)
    if out_path is None:
        out_path=os.path.join(REPLAY_DIR,f"replay_generation_{generation:04d}.html")
    with open(out_path,"w",encoding="utf-8") as f:f.write(html)
    return out_path,html


def _replay_file_compatible(path):
    try:
        with np.load(path,allow_pickle=False) as d:
            if "field_size" not in d or "obs_size" not in d or "terrain_res" not in d:return False
            return abs(float(d["field_size"])-FIELD_SIZE)<1e-6 and int(d["obs_size"])==OBS_SIZE and int(d["terrain_res"])==TERRAIN_RES
    except Exception:return False


def show_replay(generation=None):
    files=sorted(glob.glob(os.path.join(BOUT_DIR,"generation_*_best_bout.npz")),key=generation_number)
    compatible=[f for f in files if _replay_file_compatible(f)]
    if not compatible:raise FileNotFoundError(f"No compatible Best Bout found in {os.path.abspath(BOUT_DIR)}")
    if generation is None:bout_path=compatible[-1]
    else:
        bout_path=os.path.join(BOUT_DIR,f"generation_{generation:04d}_best_bout.npz")
        if not os.path.exists(bout_path):raise FileNotFoundError(bout_path)
        if not _replay_file_compatible(bout_path):raise ValueError(f"Generation {generation} Best Bout is incompatible with the current environment.")
    out_path,html=build_replay_html(bout_path)
    try:
        from IPython.display import display,HTML,IFrame
        try:display(IFrame(src=os.path.relpath(out_path),width="100%",height=720))
        except Exception:display(HTML(html))
    except ImportError:pass
    return out_path


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    latest = train(n_generations=N_GENERATIONS, resume=True)
    try:
        replay_generation = LAST_COMPLETED_GENERATION
        path = show_replay(replay_generation) if replay_generation is not None else show_replay()
        print("Replay written to:", os.path.abspath(path))
    except FileNotFoundError as e:
        print("Replay skipped:", e)