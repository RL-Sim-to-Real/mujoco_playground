# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Randomization functions."""
from typing import Tuple

import jax
import jax.numpy as jp
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src.manipulation.franka_emika_panda import push


def sample_light_position():
  position = np.zeros(3)
  while np.linalg.norm(position) < 1.0:
    position = np.random.uniform([1.5, -0.2, 0.8], [3, 0.2, 1.5])
  return position


def perturb_orientation(
    key: jax.Array, original: jax.Array, deg: float
) -> jax.Array:
  """Perturbs a 3D or 4D orientation by up to deg."""
  key_axis, key_theta, key = jax.random.split(key, 3)
  perturb_axis = jax.random.uniform(key_axis, (3,), minval=-1, maxval=1)
  # Only perturb upwards in the y axis.
  key_y, key = jax.random.split(key, 2)
  perturb_axis = perturb_axis.at[1].set(
      jax.random.uniform(key_y, (), minval=0, maxval=1)
  )
  perturb_axis = perturb_axis / jp.linalg.norm(perturb_axis)
  perturb_theta = jax.random.uniform(
      key_theta, shape=(1,), minval=0, maxval=np.deg2rad(deg)
  )
  rot_offset = math.axis_angle_to_quat(perturb_axis, perturb_theta)
  if original.shape == (4,):
    return math.quat_mul(rot_offset, original)
  elif original.shape == (3,):
    return math.rotate(original, rot_offset)
  else:
    raise ValueError('Invalid input shape:', original.shape)


def domain_randomize(
    mjx_model: mjx.Model, num_worlds: int
) -> Tuple[mjx.Model, mjx.Model]:
  """Tile the necessary axes for the Madrona BatchRenderer."""
  mj_model = push.PandaPushCuboid().mj_model
  floor_geom_id = mj_model.geom('floor').id
  box_geom_id = mj_model.geom('box').id
  custom_wood_material_id = mj_model.material('custom_wood').id
  strip_geom_id = mj_model.geom('white_strip').id
  box_body_id = mj_model.body('box').id

  in_axes = jax.tree_util.tree_map(lambda x: None, mjx_model)
  in_axes = in_axes.tree_replace({
      'geom_rgba': 0,
      'geom_matid': 0,
      'cam_pos': 0,
      'cam_quat': 0,
      'light_pos': 0,
      'light_dir': 0,
      'light_directional': 0,
      'light_castshadow': 0,
      'body_mass': 0,
      'body_inertia': 0
  })
  rng = jax.random.key(0)

  # Simpler logic implementing via Numpy.
  np.random.seed(0)
  light_positions = [sample_light_position() for _ in range(num_worlds)]
  light_positions = jp.array(light_positions)

  @jax.vmap
  def rand(rng: jax.Array, light_position: jax.Array):
    """Generate randomized model fields."""
    _, key = jax.random.split(rng, 2)

    #### Apearance ####
    
    # Sample a random color for the box
    key_box, key_strip, key_floor, key = jax.random.split(key, 4)
    rgba = jp.array(
        [jax.random.uniform(key_box, (), minval=0.7, maxval=1.0), 0.0, 0.0, 1.0]
    )
    geom_rgba = mjx_model.geom_rgba.at[box_geom_id].set(rgba)


    strip_white = jax.random.uniform(key_strip, (), minval=0.8, maxval=1.0)
    geom_rgba = geom_rgba.at[strip_geom_id].set(
        jp.array([strip_white, strip_white, strip_white, 1.0]) 
    )

    # # Sample a shade of gray -- I think this is for the floor

    # Randomize floor color
    gray_scale = jax.random.uniform(key_floor, (), minval=0.0, maxval=0.25)
    geom_rgba = geom_rgba.at[floor_geom_id].set(
        jp.array([0.21960785, 0.23529412, 0.14509804, 1.0])
    )

    mat_offset, num_geoms = 5, geom_rgba.shape[0]
    key_matid, key = jax.random.split(key)
    geom_matid = (
        jax.random.randint(key_matid, shape=(num_geoms,), minval=0, maxval=10)
        + mat_offset
    )
    geom_matid = geom_matid.at[box_geom_id].set(
        -2
    )  # Use the above randomized colors
    geom_matid = geom_matid.at[floor_geom_id].set(-2)
    geom_matid = geom_matid.at[strip_geom_id].set(-2)
    # Set floor material to "custom_wood"

    # geom_matid = geom_matid.at[floor_geom_id].set(custom_wood_material_id)

    #### Cameras ####
    key_pos, key_ori, key = jax.random.split(key, 3)
    cam_offset = jax.random.uniform(key_pos, (3,), minval=-0.01, maxval=0.01)
    assert (
        len(mjx_model.cam_pos) == 1
    ), f'Expected single camera, got {len(mjx_model.cam_pos)}'
    cam_pos = mjx_model.cam_pos.at[0].set(mjx_model.cam_pos[0] + cam_offset)
    cam_quat = mjx_model.cam_quat.at[0].set(
        perturb_orientation(key_ori, mjx_model.cam_quat[0], 0.2)
    )

    #### Lighting ####
    nlight = mjx_model.light_pos.shape[0]
    assert (
        nlight == 1
    ), f'Sim2Real was trained with a single light source, got {nlight}'
    key_lsha, key_ldir, key = jax.random.split(key, 3)

    # Direction
    shine_at = jp.array([0.661, -0.001, 0.179])  # Gripper starting position
    nom_dir = (shine_at - light_position) / jp.linalg.norm(
        shine_at - light_position
    )
    light_dir = mjx_model.light_dir.at[0].set(
        perturb_orientation(key_ldir, nom_dir, 20)
    )

    # Whether to cast shadows
    light_castshadow = jax.random.bernoulli(
        key_lsha, 0.75, shape=(nlight,)
    ).astype(jp.float32)

    # No need to randomize into specular lighting
    light_directional = jp.ones((nlight,))

    ### Randomize mass and intertia of the box ###

    key_mass, key = jax.random.split(key, 2)
    mass_scale = jax.random.uniform(key_mass, (), minval=1.0, maxval=1.5)  
    body_mass = mjx_model.body_mass.at[box_body_id].set(
        mjx_model.body_mass[box_body_id] * mass_scale
    )
    body_inertia = mjx_model.body_inertia.at[box_body_id].set(
        mjx_model.body_inertia[box_body_id] * mass_scale
    )

    return (
        geom_rgba,
        geom_matid,
        cam_pos,
        cam_quat,
        light_dir,
        light_directional,
        light_castshadow,
        body_mass,
        body_inertia,
    )

  (
      geom_rgba,
      geom_matid,
      cam_pos,
      cam_quat,
      light_dir,
      light_directional,
      light_castshadow,
      body_mass,
      body_inertia,
  ) = rand(jax.random.split(rng, num_worlds), light_positions)

  mjx_model = mjx_model.tree_replace({
      'geom_rgba': geom_rgba,
      'geom_matid': geom_matid,
      'cam_pos': cam_pos,
      'cam_quat': cam_quat,
      'light_pos': light_positions,
      'light_dir': light_dir,
      'light_directional': light_directional,
      'light_castshadow': light_castshadow,
      'body_mass': body_mass,
      'body_inertia': body_inertia,
  })

  return mjx_model, in_axes
