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
"""A simple task with demonstrating sim2real transfer for pixels observations.
Pick up a cube to a fixed location using a cartesian controller."""

from typing import Any, Dict, Optional, Union
import warnings

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground._src.manipulation.franka_emika_panda.actuator import actuator_map
from mujoco_playground._src import collision
from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.franka_emika_panda import panda
from mujoco_playground._src.manipulation.franka_emika_panda import panda_kinematics
from mujoco_playground._src.manipulation.franka_emika_panda import pick
import cv2


GEAR = np.array([150.0, 150.0, 150.0, 150.0, 20.0, 20.0, 20.0])

def default_vision_config() -> config_dict.ConfigDict:
  return config_dict.create(
      gpu_id=0,
      render_batch_size=1024,
      render_width=64,
      render_height=64,
      use_rasterizer=False,
      enabled_geom_groups=[0, 1, 2],
  )


def default_config():
  config = config_dict.create(
      ctrl_dt=0.04,
      sim_dt=0.004,
      episode_length=200,
      action_repeat=1,
      # Size of cartesian increment.
      action_scale=0.005,
      reward_config=config_dict.create(
          reward_scales=config_dict.create(
              # Gripper goes to the box.
              gripper_box=4.0,
              # Box goes to the target mocap.
              box_target=8.0,
              # Do not collide the gripper with the floor.
              no_floor_collision=0.25,
              # Do not collide cube with gripper
              no_box_collision=0.05,
              # Destabilizes training in cartesian action space.
              robot_target_qpos=0.0,
          ),
          action_rate=-0.0005,
          no_soln_reward=-0.01,
          lifted_reward=0.5,
          success_reward=2.0,
      ),
      vision=False,
      proprioception=False,
      vision_config=default_vision_config(),
      obs_noise=config_dict.create(brightness=[1.0, 1.0]),
      box_init_range=0.05,
      box_init_range_y=0.05,
      hide_white_strip=True,
      success_threshold=0.05,
      action_history_length=1,
      actuator='position',
      action='cartesian_increment',
  )
  return config

def rgb_to_hsv(img):
    # img: (..., 3), values in [0, 1]
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    maxc = jp.max(img, axis=-1)
    minc = jp.min(img, axis=-1)
    v = maxc
    deltac = maxc - minc
    s = jp.where(maxc == 0, 0, deltac / (maxc + 1e-8))
    # Hue calculation
    rc = (maxc - r) / (deltac + 1e-8)
    gc = (maxc - g) / (deltac + 1e-8)
    bc = (maxc - b) / (deltac + 1e-8)
    h = jp.where(
        deltac == 0,
        0.0,
        jp.where(
            maxc == r,
            (bc - gc) / 6.0,
            jp.where(
                maxc == g,
                (2.0 + rc - bc) / 6.0,
                (4.0 + gc - rc) / 6.0,
            ),
        ),
    )
    h = jp.mod(h, 1.0)
    return jp.stack([h, s, v], axis=-1)

def hsv_to_rgb(img):
    # img: (..., 3), values in [0, 1]
    h, s, v = img[..., 0], img[..., 1], img[..., 2]
    i = jp.floor(h * 6.0)
    f = h * 6.0 - i
    i = i.astype(jp.int32)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i_mod = i % 6
    conditions = [
        (i_mod == 0, jp.stack([v, t, p], axis=-1)),
        (i_mod == 1, jp.stack([q, v, p], axis=-1)),
        (i_mod == 2, jp.stack([p, v, t], axis=-1)),
        (i_mod == 3, jp.stack([p, q, v], axis=-1)),
        (i_mod == 4, jp.stack([t, p, v], axis=-1)),
        (i_mod == 5, jp.stack([v, p, q], axis=-1)),
    ]
    rgb = jp.zeros_like(img)
    for cond, val in conditions:
        rgb = jp.where(cond[..., None], val, rgb)
    return rgb


def augment_image(rng, 
                  img, 
                  contrast_range=(0.8, 1.2),
                  saturation_range=(0.8, 1.2),
                  hue_range=(-0.05, 0.05),
                  overlay: jax.Array=None,
                  random_overlay_rot90=True,   # random 0/90/180/270 rotation of overlay
                  overlay_flip_lr=True,        # random left-right flip of overlay
                  overlay_flip_ud=True ):
  # adjust contrast
  rng, rng_c = jax.random.split(rng)
  contrast = jax.random.uniform(rng_c, (), minval=contrast_range[0], maxval=contrast_range[1])
  mean = jp.mean(img, axis=(0, 1), keepdims=True)
  img = (img - mean) * contrast + mean


  # Random saturation (convert to grayscale and interpolate)
  rng, rng_s = jax.random.split(rng)
  saturation = jax.random.uniform(rng_s, (), minval=saturation_range[0], maxval=saturation_range[1])
  gray = jp.mean(img, axis=-1, keepdims=True)
  img = (img - gray) * saturation + gray

  # Random hue (shift in HSV space)
  # rng, rng_h = jax.random.split(rng)
  # hue = jax.random.uniform(rng_h, (), minval=hue_range[0], maxval=hue_range[1])
  # img_hsv = rgb_to_hsv(img)
  # img_hsv = img_hsv.at[..., 0].add(hue)
  # img_hsv = img_hsv.at[..., 0].set(jp.mod(img_hsv[..., 0], 1.0))  # wrap hue
  # img = hsv_to_rgb(img_hsv)

  # add image overlay with randomized orientation
  if overlay is not None:
    ovl = jp.asarray(overlay)  # expected shape (H, W, 3) in [0,1]
    if random_overlay_rot90 or overlay_flip_lr or overlay_flip_ud:
      rng, rng_r, rng_flr, rng_fud = jax.random.split(rng, 4)

      if random_overlay_rot90:
        k = jax.random.randint(rng_r, (), 0, 4)  # 0..3
        # Note: assumes square H==W for jit-safe static shapes.
        def _rot0(x): return x
        def _rot1(x): return jp.flip(jp.swapaxes(x, 0, 1), axis=0)  # 90°
        def _rot2(x): return jp.flip(jp.flip(x, axis=0), axis=1)    # 180°
        def _rot3(x): return jp.flip(jp.swapaxes(x, 0, 1), axis=1)  # 270°
        ovl = jax.lax.switch(k, (_rot0, _rot1, _rot2, _rot3), ovl)

      if overlay_flip_lr:
        do_flr = jax.random.bernoulli(rng_flr, 0.5)
        ovl = jax.lax.cond(do_flr, lambda x: jp.flip(x, axis=1), lambda x: x, ovl)

      if overlay_flip_ud:
        do_fud = jax.random.bernoulli(rng_fud, 0.5)
        ovl = jax.lax.cond(do_fud, lambda x: jp.flip(x, axis=0), lambda x: x, ovl)

    # pad/crop overlay to img size if needed
    H, W = img.shape[0], img.shape[1]
    h, w = min(int(ovl.shape[0]), int(H)), min(int(ovl.shape[1]), int(W))
    ovl_padded = jp.zeros_like(img)
    ovl_padded = ovl_padded.at[:h, :w, :].set(ovl[:h, :w, :])

    alpha = jp.asarray(0.3, dtype=img.dtype)  # 25% opacity
    img = (1.0 - alpha) * img + alpha * ovl_padded

  img = jp.clip(img, 0, 1)
  return img




def adjust_brightness(img, scale):
  """Adjusts the brightness of an image by scaling the pixel values."""
  return jp.clip(img * scale, 0, 1)


class PandaPickCuboid(pick.PandaPickCube):
  """Environment for training the Franka Panda robot to pick up a cube in
  Cartesian space."""

  def __init__(  # pylint: disable=non-parent-init-called,super-init-not-called
      self,
      config=default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):

    mjx_env.MjxEnv.__init__(self, config, config_overrides)
    self._vision = config.vision
    self._proprioception = config.proprioception

    xml_path = (
        mjx_env.ROOT_PATH
        / 'manipulation'
        / 'franka_emika_panda'
        / 'xmls'
        / 'mjx_single_cube_camera_modified.xml'
    )
    self._xml_path = xml_path.as_posix()

    mj_model = self.modify_model(
        mujoco.MjModel.from_xml_string(
            xml_path.read_text(), assets=panda.get_assets(actuator=config.actuator)
        )
    )

    texture_path = (
        mjx_env.ROOT_PATH
        / 'manipulation'
        / 'franka_emika_panda'
        / 'xmls'
        / 'texture-augment.jpeg'
    )
      

    self._overlay = None

    try:
      img_bgr = cv2.imread(str(texture_path))
      img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
      img_rgb = cv2.resize(img_rgb, \
                           (config.vision_config.render_width, config.vision_config.render_height))
      self._overlay = jp.asarray(img_rgb.astype(np.float32)) / 255.0 
    except Exception as e:
      print(f"Error loading overlay image: {e}")

    mj_model.opt.timestep = config.sim_dt

    self._mj_model = mj_model
    self._mjx_model = mjx.put_model(mj_model)

    # Set gripper in sight of camera
    self._post_init(obj_name='box', keyframe='low_home')
    self._box_geom = self._mj_model.geom('box').id
    self._max_torque = 8.0 # for torque control
    self._gear = GEAR

    if self._vision:
      try:
        # pylint: disable=import-outside-toplevel
        from madrona_mjx.renderer import BatchRenderer  # pytype: disable=import-error
      except ImportError:
        warnings.warn(
            'Madrona MJX not installed. Cannot use vision with'
            ' PandaPickCubeCartesian#D.'
        )
        return
      self.renderer = BatchRenderer(
          m=self._mjx_model,
          gpu_id=self._config.vision_config.gpu_id,
          num_worlds=self._config.vision_config.render_batch_size,
          batch_render_view_width=self._config.vision_config.render_width,
          batch_render_view_height=self._config.vision_config.render_height,
          enabled_geom_groups=np.asarray(
              self._config.vision_config.enabled_geom_groups
          ),
          enabled_cameras=None,  # Use all cameras.
          add_cam_debug_geo=False,
          use_rasterizer=self._config.vision_config.use_rasterizer,
          viz_gpu_hdls=None,
      )

  def _post_init(self, obj_name, keyframe):
    super()._post_init(obj_name, keyframe)
    self._guide_q = self._mj_model.keyframe('picked').qpos
    self._guide_ctrl = self._mj_model.keyframe('picked').ctrl
    # Use forward kinematics to init cartesian control
    self._start_tip_transform = panda_kinematics.compute_franka_fk(
        # self._init_ctrl[:7] use qpos instead
        self._init_q[:7]
    )

    self._sample_orientation = False

  def modify_model(self, mj_model: mujoco.MjModel):
    # Expand floor size to non-zero so Madrona can render it
    mj_model.geom_size[mj_model.geom('floor').id, :2] = [5.0, 5.0]

    # # Make the finger pads white for increased visibility
    # mesh_id = mj_model.mesh('finger_1').id
    # geoms = [
    #     idx
    #     for idx, data_id in enumerate(mj_model.geom_dataid)
    #     if data_id == mesh_id
    # ]
    # mj_model.geom_matid[geoms] = mj_model.mat('black').id
    return mj_model

  def _jnt_range(self):
    # TODO(siholt): Use joint limits from XML.
    return [
        [-2.8973, 2.8973],
        [-1.7628, 1.7628],
        [-2.8973, 2.8973],
        [-3.0718, -0.0698],
        [-2.8973, 2.8973],
        [-0.0175, 3.7525],
        [-2.8973, 2.8973],
    ]
    
  def _jnt_vel_range(self):
    return [
        [-2.1750, 2.1750],
        [-2.1750, 2.1750],
        [-2.1750, 2.1750],
        [-2.1750, 2.1750],
        [-2.6100, 2.6100],
        [-2.6100, 2.6100],
        [-2.6100, 2.6100],
    ]


  def reset(self, rng: jax.Array) -> mjx_env.State:
    """Resets the environment to an initial state."""
    # x_plane = self._start_tip_transform[0, 3] - 0.03  # Account for finite gain
    x_plane = 0.57 # this value is closer to the real robot setup
    # randomize end effector position
    rng, rng_plane = jax.random.split(rng)
    x_plane = x_plane + jax.random.uniform(rng_plane, (), minval=-0.02, maxval=0.02)
    target_tip_pose=jp.asarray([x_plane, 
                                  jax.random.uniform(rng_plane, (), minval=-0.01, maxval=0.01), 
                                  0.2 + jax.random.uniform(rng_plane, (), minval=-0.005, maxval=0.005)])

    # set initial pose to new plane
    reset_joint_pos, _, _ = self._move_tip_reset(
        target_tip_pose=target_tip_pose,
        current_tip_rot = self._start_tip_transform[:3, :3],
        current_jp=jp.asarray(self._init_q[:8]) # careful with this
    )
    init_q = (
        jp.array(self._init_q)
        .at[self._robot_arm_qposadr]
        .set(reset_joint_pos[:7])  # only move arm, not fingers
    )
    if self._config.actuator == "position":
      init_ctrl0 = reset_joint_pos
    else:
      init_ctrl0 = jp.asarray(self._init_ctrl)
    
    
    # intialize box position
    rng, rng_box = jax.random.split(rng)
    r_range = self._config.box_init_range
    box_pos = jp.array([
        x_plane + jax.random.uniform(rng_box, (), minval=-self._config.box_init_range_y, maxval=self._config.box_init_range_y), # randomize about white strip
        jax.random.uniform(rng_box, (), minval=-r_range, maxval=r_range),
        0.0,
    ])

    # Fixed target position to simplify pixels-only training.
    target_pos = jp.array([x_plane, target_tip_pose[1], 0.2])

    # initialize pipeline state
    init_q = (
        jp.array(init_q) # build on top of previous init_q
        .at[self._obj_qposadr : self._obj_qposadr + 3]
        .set(box_pos)
    )
    
    data = mjx_env.init(
        self._mjx_model,
        init_q,
        jp.zeros(self._mjx_model.nv, dtype=float),
        ctrl=init_ctrl0,
    )
    
    # reposition the white strip to be under the end effector
    # self._white_strip_geom = self._mj_model.geom("white_strip").id
    # self._mj_model.geom_rgba[self._white_strip_geom, 3] = 0.0 if self._config.hide_white_strip else 1.0  # Set alpha to 0 for invisibility
    # data = data.replace(
    #   xpos=data.xpos.at[self._white_strip_geom, 0].set(x_plane)
    # )

    target_quat = jp.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    data = data.replace(
        mocap_quat=data.mocap_quat.at[self._mocap_target, :].set(target_quat)
    )
    if not self._vision:
      # mocap target should not appear in the pixels observation.
      data = data.replace(
          mocap_pos=data.mocap_pos.at[self._mocap_target, :].set(target_pos)
      )

    # step simulator
    # data = mjx_env.step(self._mjx_model, data, self._init_ctrl, self.n_substeps)

    # initialize env state and info
    metrics = {
        'floor_collision': jp.array(0.0, dtype=float),
        'cube_collision': jp.array(0.0),
        'jerk': jp.array(0.0),
        'success': jp.array(0.0),
        'out_of_bounds': jp.array(0.0),
        **{
            f'reward/{k}': 0.0
            for k in self._config.reward_config.reward_scales.keys()
        },
        'reward/success': jp.array(0.0),
        'reward/lifted': jp.array(0.0),
    }

    info = {
        'rng': rng,
        'target_pos': target_pos,
        'reached_box': jp.array(0.0, dtype=float),
        'prev_reward': jp.array(0.0, dtype=float),
        'current_pos': target_tip_pose,
        'reset_pos': target_tip_pose,
        'newly_reset': jp.array(False, dtype=bool),
        'prev_action': jp.zeros(self.action_size),
        '_steps': jp.array(0, dtype=int),
        'action_history': jp.zeros((
            self._config.action_history_length,
        )),  # Gripper only
        'prev_qacc': jp.zeros(7),
    }

    reward, done = jp.zeros(2)

    obs = self._get_obs(data, info)
    obs = jp.concat([obs, jp.zeros(1), jp.zeros(3)], axis=0)

    

    if self._vision:
      rng_brightness, rng_img, rng = jax.random.split(rng, num=3)
      brightness = jax.random.uniform(
          rng_brightness,
          (1,),
          minval=self._config.obs_noise.brightness[0],
          maxval=self._config.obs_noise.brightness[1],
      )
      info.update({'brightness': brightness})
      info.update({'rng_img': rng_img}) 
      render_token, rgb, _ = self.renderer.init(data, self._mjx_model)
      info.update({'render_token': render_token})

      obs = jp.asarray(rgb[0][..., :3], dtype=jp.float32) / 255.0
      obs = adjust_brightness(obs, brightness)
      obs = augment_image(rng_img, img=obs, overlay=self._overlay)
      obs = {'pixels/view_0': obs}
      grasp = collision.geoms_colliding(data, self._box_geom, self._left_finger_geom) &\
          collision.geoms_colliding(data, self._box_geom, self._right_finger_geom)
      if self._proprioception:

        ee_height = data.xpos[self._left_finger_geom][2]
        joint_p = data.qpos[:7]  + jax.random.normal(rng, (7,)) * 0.1 # add noise
        normalized_jp = 2 * (joint_p - jp.array(self._jnt_range())[:, 0]) / (
          jp.array(self._jnt_range())[:, 1] - jp.array(self._jnt_range())[:, 0]
        ) - 1
        joint_v = data.qvel[:7]  + jax.random.normal(rng, (7,)) * 0.1 # add noise
        normalized_jv = 2 * (joint_v - jp.array(self._jnt_vel_range())[:, 0]) / (
          jp.array(self._jnt_vel_range())[:, 1] - jp.array(self._jnt_vel_range())[:, 0]
        ) - 1
        _prop = jp.concatenate([ 
                                normalized_jp,
                                normalized_jv, 
                                jp.zeros(self.action_size), jp.array([ee_height]), grasp.astype(float)[..., None]])


        obs["_prop"] = _prop

    return mjx_env.State(data, obs, reward, done, metrics, info)
  

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    """Runs one timestep of the environment's dynamics."""
    action_history = (
        jp.roll(state.info['action_history'], 1).at[0].set(action[-1])
    )
    state.info['action_history'] = action_history
    # Add action delay
    state.info['rng'], key = jax.random.split(state.info['rng'])
    action_idx = jax.random.randint(
        key, (), minval=0, maxval=self._config.action_history_length
    )
    action = action.at[-1].set(state.info['action_history'][action_idx])

    state.info['newly_reset'] = state.info['_steps'] == 0

    newly_reset = state.info['newly_reset']
    state.info['prev_reward'] = jp.where(
        newly_reset, 0.0, state.info['prev_reward']
    )
    state.info['current_pos'] = jp.where(
        newly_reset, state.info['reset_pos'], state.info['current_pos']
    )
    state.info['reached_box'] = jp.where(
        newly_reset, 0.0, state.info['reached_box']
    )
    state.info['prev_action'] = jp.where(
        newly_reset, jp.zeros(self.action_size), state.info['prev_action']
    )

    # Ocassionally aid exploration.
    state.info['rng'], key_swap = jax.random.split(state.info['rng'])
    to_sample = newly_reset * jax.random.bernoulli(key_swap, 0.05)
    swapped_data = state.data.replace(
        qpos=self._guide_q, ctrl=self._guide_ctrl
    )  # help hit the terminal sparse reward.
    data = jax.tree_util.tree_map(
        lambda x, y: (1 - to_sample) * x + to_sample * y,
        state.data,
        swapped_data,
    )

    # Cartesian control

    if self._config.action == 'cartesian_increment':
      increment = jp.zeros(4)
      increment = action  # directly set x, y, z and gripper commands.

      # this function was created with only position control in mind
      ctrl, new_tip_position, no_soln = self._move_tip(
          state.info['current_pos'],
          self._start_tip_transform[:3, :3],
          data.qpos[:8], # should be current joint positions including gripper
          increment,
      )

      if self._config.actuator == 'velocity': # careful with this
        delta_q = ctrl[:7] - data.qpos[:7] # calculate joint increments excluding gripper
        joint_cntrl = actuator_map("joint_increment", self._config.actuator, delta_q, data.qpos[:7], self._config.ctrl_dt) # ctrl is target joint pos
        ctrl = ctrl.at[:7].set(joint_cntrl)

      state.info.update({'current_pos': new_tip_position})
    elif self._config.action in {'joint_increment', 'joint'}:
      ctrl, no_soln = self._move_joints(data.qpos[:7], action)
    else:
      raise ValueError(f"Invalid action type: {self._config.action}")
      
    ctrl = jp.clip(ctrl, self._lowers, self._uppers)


    # Simulator step
    # keep claw open
    
    data = mjx_env.step(self._mjx_model, data, ctrl, self.n_substeps)

    # Dense rewards
    raw_rewards = self._get_reward(data, state.info)
    rewards = {
        k: v * self._config.reward_config.reward_scales[k]
        for k, v in raw_rewards.items()
    }

    # Penalize collision with box.
    hand_box = collision.geoms_colliding(data, self._box_geom, self._hand_geom)
    raw_rewards['no_box_collision'] = jp.where(hand_box, 0.0, 1.0)

    total_reward = jp.clip(sum(rewards.values()), -1e4, 1e4)

    if not self._vision:
      # Vision policy cannot access the required state-based observations.
      da = jp.linalg.norm(action - state.info['prev_action'])
      state.info['prev_action'] = action
      total_reward += self._config.reward_config.action_rate * da
      total_reward += no_soln * self._config.reward_config.no_soln_reward

    # Sparse rewards
    box_pos = data.xpos[self._obj_body]
    lifted = (box_pos[2] > 0.05) * self._config.reward_config.lifted_reward
    total_reward += lifted
    success = self._get_success(data, state.info)
    total_reward += success * self._config.reward_config.success_reward

    # Reward progress
    reward = jp.maximum(
        total_reward - state.info['prev_reward'], jp.zeros_like(total_reward)
    )
    state.info['prev_reward'] = jp.maximum(
        total_reward, state.info['prev_reward']
    )
    reward = jp.where(newly_reset, 0.0, reward)  # Prevent first-step artifact

    out_of_bounds = jp.any(jp.abs(box_pos) > 1.0)
    out_of_bounds |= box_pos[2] < 0.0
    state.metrics.update(out_of_bounds=out_of_bounds.astype(float))
    finger_collision: bool = collision.geoms_colliding(data, self._box_geom, self._left_finger_geom) ^\
        collision.geoms_colliding(data, self._box_geom, self._right_finger_geom) # if it's not grasping it's a collisiong
    state.metrics.update(cube_collision=(hand_box|finger_collision).astype(float)) # log collision only if lift wasn't successfull
    hand_floor_collision = [
        collision.geoms_colliding(data, self._floor_geom, g)
        for g in [
            self._left_finger_geom,
            self._right_finger_geom,
            self._hand_geom,
        ]
    ]
    current_qacc = data.qacc[:7]
    dt = self._config.ctrl_dt
    
    # Jerk is rate of change of acceleration
    jerk = jp.linalg.norm((current_qacc - state.info['prev_qacc']) / dt)
    jerk = jp.where(state.info['newly_reset'], 0.0, jerk)
    
    # Update acceleration history
    state.info['prev_qacc'] = current_qacc
    
    # Update metrics
    state.metrics.update(jerk=jerk.astype(float))
    
    floor_collision = sum(hand_floor_collision) > 0
    state.metrics.update(floor_collision=floor_collision.astype(float))
    state.metrics.update(success=jp.where(to_sample, 0.0, success).astype(float))
    state.metrics.update({f'reward/{k}': v for k, v in raw_rewards.items()})
    state.metrics.update({
        'reward/lifted': lifted.astype(float),
        'reward/success': (success * self._config.reward_config.success_reward).astype(float),
    })

    done = (
        out_of_bounds
        | jp.isnan(data.qpos).any()
        | jp.isnan(data.qvel).any()
        | success
    )


    # Ensure exact sync between newly_reset and the autoresetwrapper.
    state.info['_steps'] += self._config.action_repeat
    state.info['_steps'] = jp.where(
        done | (state.info['_steps'] >= self._config.episode_length),
        0,
        state.info['_steps'],
    )

    obs = self._get_obs(data, state.info)
    obs = jp.concat([obs, no_soln.reshape(1), action], axis=0)
  
    grasp = collision.geoms_colliding(data, self._box_geom, self._left_finger_geom) &\
        collision.geoms_colliding(data, self._box_geom, self._right_finger_geom)
    if self._vision:
      _, rgb, _ = self.renderer.render(state.info['render_token'], data)
      obs = jp.asarray(rgb[0][..., :3], dtype=jp.float32) / 255.0
      obs = adjust_brightness(obs, state.info['brightness'])
      # augment image
      rng_img = state.info['rng_img']  # Use the same RNG as in reset for consistency
      obs = augment_image(rng_img, img=obs, overlay=self._overlay)
      obs = {'pixels/view_0': obs }
      if self._proprioception:
        state.info['rng'], rng_prop = jax.random.split(state.info['rng'])

        ee_height = data.xpos[self._left_finger_geom][2]
        joint_p = data.qpos[:7] + jax.random.normal(rng_prop, (7,)) * 0.1
        joint_v = data.qvel[:7] + jax.random.normal(rng_prop, (7,)) * 0.1
        normalized_jv = 2 * (joint_v - jp.array(self._jnt_vel_range())[:, 0]) / (
          jp.array(self._jnt_vel_range())[:, 1] - jp.array(self._jnt_vel_range())[:, 0]
        ) - 1
        normalized_jp = 2 * (joint_p - jp.array(self._jnt_range())[:, 0]) / (
          jp.array(self._jnt_range())[:, 1] - jp.array(self._jnt_range())[:, 0]
        ) - 1
        _prop = jp.concatenate([

                                normalized_jp, 
                                normalized_jv,  # Include normalized joint velocity
                                action, jp.array([ee_height]), grasp.astype(float)[..., None]])

       
        obs["_prop"] = _prop

    return state.replace(
        data=data,
        obs=obs,
        reward=reward,
        done=done.astype(float),
        info=state.info,
    )

  def _get_success(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    box_pos = data.xpos[self._obj_body]
    target_pos = info['target_pos']
    if (
        self._vision
    ):  # Randomized camera positions cannot see location along y line.
      box_pos, target_pos = box_pos[2], target_pos[2]
    return jp.linalg.norm(box_pos - target_pos) < self._config.success_threshold # if the height difference is less than threshold
  
  def _move_tip_reset(self, 
                      target_tip_pose: jax.Array, 
                      current_tip_rot: jax.Array, 
                      current_jp: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
    new_tip_mat = jp.identity(4)
    new_tip_mat = new_tip_mat.at[:3, :3].set(current_tip_rot)
    new_tip_mat = new_tip_mat.at[:3, 3].set(target_tip_pose)

    target_tip_pose = target_tip_pose.at[0].set(jp.clip(target_tip_pose[0], 0.25, 0.77))
    target_tip_pose = target_tip_pose.at[1].set(jp.clip(target_tip_pose[1], -0.32, 0.32))
    target_tip_pose = target_tip_pose.at[2].set(jp.clip(target_tip_pose[2], 0.02, 0.5))

    out_jp = panda_kinematics.compute_franka_ik(
        new_tip_mat, current_jp[6], current_jp[:7]
    )
    no_soln = jp.any(jp.isnan(out_jp))
    out_jp = jp.where(no_soln, current_jp[:7], out_jp)
    no_soln = jp.logical_or(no_soln, jp.any(jp.isnan(out_jp)))
    
    new_jp = current_jp.at[:7].set(out_jp)
    return new_jp, target_tip_pose, no_soln


  def _move_tip(
      self,
      current_tip_pos: jax.Array,
      current_tip_rot: jax.Array,
      current_ctrl: jax.Array,
      action: jax.Array,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Calculate new tip position from cartesian increment."""
    # Discrete gripper action where a < 0 := closed
    close_gripper = jp.where(action[3] < 0, 1.0, 0.0)

    scaled_pos = action[:3] * self._config.action_scale
    new_tip_pos = current_tip_pos.at[:3].add(scaled_pos)

    new_ctrl = current_ctrl

    new_tip_pos = new_tip_pos.at[0].set(jp.clip(new_tip_pos[0], 0.25, 0.77))
    new_tip_pos = new_tip_pos.at[1].set(jp.clip(new_tip_pos[1], -0.32, 0.32))
    new_tip_pos = new_tip_pos.at[2].set(jp.clip(new_tip_pos[2], 0.02, 0.5))

    new_tip_mat = jp.identity(4)
    new_tip_mat = new_tip_mat.at[:3, :3].set(current_tip_rot)
    new_tip_mat = new_tip_mat.at[:3, 3].set(new_tip_pos)

    out_jp = panda_kinematics.compute_franka_ik(
        new_tip_mat, current_ctrl[6], current_ctrl[:7]
    )
    no_soln = jp.any(jp.isnan(out_jp))
    out_jp = jp.where(no_soln, current_ctrl[:7], out_jp)
    no_soln = jp.logical_or(no_soln, jp.any(jp.isnan(out_jp)))
    new_tip_pos = jp.where(
        jp.any(jp.isnan(out_jp)), current_tip_pos, new_tip_pos
    )

    new_ctrl = new_ctrl.at[:7].set(out_jp)
    jaw_action = jp.where(close_gripper, -1.0, 1.0)
    claw_delta = jaw_action * 0.02  # up to 2 cm movement per ctrl.
    new_ctrl = new_ctrl.at[7].set(new_ctrl[7] + claw_delta)

    return new_ctrl, new_tip_pos, no_soln

  def _move_joints(self, current_qpos: jax.Array, action: jax.Array):
    
    scaled_action = action[:-1] * self._config.action_scale # scake everything except gripper
    gripper_raw = action[-1]                                 # (batch,)
    gripper_raw = gripper_raw[..., None]  
    if self._config.action == 'joint_increment':
      # Incremental joint control.
      joints = actuator_map(self._config.action,
                            self._config.actuator,
                            scaled_action,
                            current_qpos,
                            self._config.ctrl_dt)            # (batch,7)
      new_ctrl = jp.concatenate([joints, gripper_raw], axis=-1)  # (batch,8)

    elif self._config.action == 'joint':
      # Absolute joint control.
      new_ctrl = jp.concatenate([scaled_action, gripper_raw], axis=-1)
      if self._config.actuator == 'position': # always assumes action as velocity
        joints = actuator_map(self._config.action,
                      'velocity-position',
                      scaled_action,
                      current_qpos,
                      self._config.ctrl_dt)            # (batch,7)
        new_ctrl = jp.concatenate([joints, gripper_raw], axis=-1)  # (batch,8)
      elif self._config.actuator == 'torque':
        new_ctrl = new_ctrl.at[:7].set(jp.clip(new_ctrl[:7],\
                                                -self._max_torque / self._gear, self._max_torque / self._gear))
    close_gripper = jp.where(action[-1] < 0, 1.0, 0.0)
    jaw_action = jp.where(close_gripper, -1.0, 1.0)
    claw_delta = jaw_action * 0.02  # up to 2 cm movement
    new_ctrl = new_ctrl.at[7].set(new_ctrl[7] + claw_delta)
    no_soln = jp.any(jp.isnan(new_ctrl))

    return new_ctrl, no_soln
    

  @property
  def action_size(self) -> int:
    if self._config.action == 'cartesian_increment':
      return 4
    elif self._config.action in {'joint_increment', 'joint'}:
      return 8 # for all 8 joints
    
    return -1

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model


if __name__ == '__main__':
  ## DEBUGGIN SECTION ##
  # For testing purposes, you can instantiate the environment like this:
  # xml_path = (
  #       mjx_env.ROOT_PATH
  #       / 'manipulation'
  #       / 'franka_emika_panda'
  #       / 'xmls'
  #       / 'mjx_single_cube_camera_modified.xml'
  #   )

  # # Load the model with assets
  # mj_model = mujoco.MjModel.from_xml_string(
  #     xml_path.read_text(), assets=panda.get_assets(actuator="torque")
  # )

  # mj_data = mujoco.MjData(mj_model)

  # import mujoco.viewer
  # # Launch the viewer
  # with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
  #     while viewer.is_running():
  #         # step the simulation
  #         mujoco.mj_step(mj_model, mj_data)
  #         viewer.sync()
  

  import cv2
  import time
  import jax
  import mujoco.viewer
  import pandas as pd


  key = jax.random.PRNGKey(1)
  env = PandaPickCuboid(config_overrides={'vision': False, 'action': "joint", "actuator":"position"})

  # IMPORTANT: use env.mj_model (mujoco.MjModel), not env.mjx_model (mjax model)
  mj_model_vis = env.mj_model
  mj_data_vis = mujoco.MjData(mj_model_vis)

  print("Action size:", env.action_size)

  # initial env state
  jit_reset = jax.jit(env.reset)
  jit_step = jax.jit(env.step)

  state = jit_reset(key)
  # print(state)
  # prefer offscreen if available, otherwise use the windowed viewer
  # mj_data_vis.qpos[: mj_data_vis.qpos.shape[0]] = np.asarray(state.data.qpos)[: mj_data_vis.qpos.shape[0]]
  # mj_data_vis.qvel[: mj_data_vis.qvel.shape[0]] = np.asarray(state.data.qvel)[: mj_data_vis.qvel.shape[0]]
  # windowed viewer (simple and reliable)
  # width, height = 640, 480  # Desired image dimensions
  # renderer = mujoco.Renderer(mj_model_vis, height, width)

  # # Render image from a specific camera
  # camera_name = "mounted"  # Replace with the actual camera name
  # camera_id = mujoco.mj_name2id(mj_model_vis, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
  # print(f"Using camera '{camera_name}' with ID {camera_id}")
  # if camera_id == -1:
  #     print(f"Camera '{camera_name}' not found in the model.")
  # else:
  #   while True:
  #       mujoco.mj_step(mj_model_vis, mj_data_vis)
  #       renderer.update_scene(mj_data_vis, camera=camera_id)
  #       image = renderer.render() # Render the image as a NumPy array

  #       # Convert the image to BGR format for OpenCV
  #       image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

  #       # Display the image using OpenCV
  #       cv2.imshow("MuJoCo Camera Feed", image_bgr)

  #       # Exit on 'q' press
  #       if cv2.waitKey(1) & 0xFF == ord('q'):
  #           break

    # cv2.destroyAllWindows()

  # action = state.data.ctrl.at[0].add(0.5)
  # joint1_positions = []
  # with mujoco.viewer.launch_passive(mj_model_vis, mj_data_vis) as viewer:
  #     reset_counter = 0
  #     while viewer.is_running():
  #         reset_counter += 1
  #                   # copy mjx state -> mujoco.MjData (slice safely to handle shape mismatches)
  #         mj_data_vis.qpos[: mj_data_vis.qpos.shape[0]] = np.asarray(state.data.qpos)[: mj_data_vis.qpos.shape[0]]
  #         mj_data_vis.qvel[: mj_data_vis.qvel.shape[0]] = np.asarray(state.data.qvel)[: mj_data_vis.qvel.shape[0]]
  #         ctrl_src = np.asarray(state.data.ctrl)
  #         mj_data_vis.ctrl[: min(mj_data_vis.ctrl.shape[0], ctrl_src.shape[0])] = ctrl_src[: mj_data_vis.ctrl.shape[0]]

  #         if reset_counter % 100 == 0:
  #             print("resetting")
  #             break
  #             state = jit_reset(jax.random.PRNGKey(int(time.time() * 1e6)))
  #             mj_data_vis.qpos[: mj_data_vis.qpos.shape[0]] = np.asarray(state.data.qpos)[: mj_data_vis.qpos.shape[0]]
  #             mj_data_vis.qvel[: mj_data_vis.qvel.shape[0]] = np.asarray(state.data.qvel)[: mj_data_vis.qvel.shape[0]]
  #             ctrl_src = np.asarray(state.data.ctrl)
  #             mj_data_vis.ctrl[: min(mj_data_vis.ctrl.shape[0], ctrl_src.shape[0])] = ctrl_src[: mj_data_vis.ctrl.shape[0]]
  #             mujoco.mj_step(mj_model_vis, mj_data_vis)
  #             viewer.sync()
  #             time.sleep(5)
              
  #         mujoco.mj_step(mj_model_vis, mj_data_vis)
  #         viewer.sync()


  #         # action = jax.random.uniform(
  #         #     jax.random.PRNGKey(int(time.time() * 1e6)),
  #         #     (env.action_size,),
  #         #     minval=-10,
  #         #     maxval=10,
  #         # )
  #         print("Action:", action)

  #         state = jit_step(state, action)
  #         joint1_positions.append(state.data.qpos[0])
  # print("Joint 1 positions over time:", joint1_positions)

  state = jit_reset(key)

  # Initialize MuJoCo data from JAX state once
  mj_data_vis.qpos[: mj_data_vis.qpos.shape[0]] = np.asarray(state.data.qpos)[: mj_data_vis.qpos.shape[0]]
  mj_data_vis.qvel[: mj_data_vis.qvel.shape[0]] = np.asarray(state.data.qvel)[: mj_data_vis.qvel.shape[0]]
  mujoco.mj_forward(mj_model_vis, mj_data_vis)

  # Add +0.5 rad to the first joint via ctrl, then step MuJoCo (no jit_step)
  ctrl = np.asarray(state.data.ctrl).copy()
  ctrl[0] += 0.5  # radians
  mj_data_vis.ctrl[: min(mj_data_vis.ctrl.shape[0], ctrl.shape[0])] = ctrl[: mj_data_vis.ctrl.shape[0]]

  joint1_positions = []
  with mujoco.viewer.launch_passive(mj_model_vis, mj_data_vis) as viewer:
      while viewer.is_running():
          mujoco.mj_step(mj_model_vis, mj_data_vis)  # advance physics using ctrl
          viewer.sync()
          joint1_positions.append(float(mj_data_vis.qpos[0]))
          # time.sleep(0.04)
          # Optional: exit after some frames
          if len(joint1_positions) >= 100:
              break
  df = pd.DataFrame(joint1_positions, columns=['joint1_pos'])
  df.to_csv('joint1_positions_sim.csv', index=False)
  print("Joint 1 positions over time:", joint1_positions)



