import jax
import jax.numpy as jp


def actuator_map(action_type: str, actuation_type: str, action_val: jax.Array, qpos: jax.Array, cycle_time: float) -> jax.Array:
    if action_type == "joint_increment" and actuation_type == "velocity": # maps position increment to velocity
        return action_val / cycle_time
    elif action_type == "joint_increment" and actuation_type == "position": # maps position increment to position
        return qpos + action_val
    elif action_type == "joint" and actuation_type == "velocity": # maps velocity to velocity
        return action_val
    elif action_type == "joint" and actuation_type == "velocity-position": # maps velocity to position
        return action_val * cycle_time + qpos
    else:
        raise ValueError(f"Unsupported action_type {action_type} and actuation_type {actuation_type} combination.")