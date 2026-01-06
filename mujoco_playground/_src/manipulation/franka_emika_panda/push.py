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
from mujoco_playground._src.manipulation.franka_emika_panda.actuator import actuator_map
import numpy as np

from mujoco_playground._src import collision
from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.franka_emika_panda import panda
from mujoco_playground._src.manipulation.franka_emika_panda import panda_kinematics
from mujoco_playground._src.manipulation.franka_emika_panda import pick
import cv2
from mujoco.mjx._src import math

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
      frame_stack_size=1,
      action_repeat=1,
      # Size of cartesian increment.
      action_scale=0.005,
      reward_config=config_dict.create(
          reward_scales=config_dict.create(
              # Gripper goes to the box.
              gripper_box=4.0,

              # Do not collide the gripper with the floor.
              no_floor_collision=0.05,
              # Destabilizes training in cartesian action space.
              robot_target_qpos=0.0,
          ),
          action_rate=-0.0005,
          no_soln_reward=-0.01,
          contact_reward=0.5,
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

    alpha = jp.asarray(0.2, dtype=img.dtype)  # 25% opacity
    img = (1.0 - alpha) * img + alpha * ovl_padded

  img = jp.clip(img, 0, 1)
  return img




def adjust_brightness(img, scale):
  """Adjusts the brightness of an image by scaling the pixel values."""
  return jp.clip(img * scale, 0, 1)


class PandaPushCuboid(panda.PandaBase):
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
        / 'mjx_single_cube_camera_push.xml'
    )
    self._xml_path = xml_path.as_posix()

    mj_model = self.modify_model(
        mujoco.MjModel.from_xml_string(
            xml_path.read_text(), assets=panda.get_assets(actuator=config.actuator, task="push")
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
            ' PandaPushCuboid.'
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
    # x_plane = self._start_tip_transform[0, 3] - 0.03 # Account for finite gain and 0.05 for real position
    x_plane = 0.55 # position in real world
    # randomize end effector position
    rng, rng_plane = jax.random.split(rng)
    x_plane = x_plane + jax.random.uniform(rng_plane, (), minval=-0.02, maxval=0.02)
    target_tip_pose=jp.asarray([x_plane, 
                                  jax.random.uniform(rng_plane, (), minval=-0.02, maxval=0.02), 
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
    rng_box, rng_box_x, rng_box_y, rng_box_yaw = jax.random.split(rng_box, 4)
    box_pos = jp.array([
      jax.random.uniform(rng_box_x, minval=0.47, maxval=0.67), # + jax.random.uniform(rng_box, (), minval=0.05, maxval=0.05 + r_range), # randomize about white strip
      jax.random.uniform(rng_box_y, (), minval=-0.15, maxval=0.15),
        0.0,
    ])

    # randomize box orientation: yaw only (rotation about world z axis)
    yaw = jax.random.uniform(rng_box_yaw, (), minval=-0.8, maxval=0.8)
    half = 0.5 * yaw
    box_quat = jp.array([jp.cos(half), 0.0, 0.0, jp.sin(half)])  # [w, x, y, z]
    init_q = init_q.at[self._obj_qposadr + 3 : self._obj_qposadr + 7].set(box_quat)
    # Fixed target position to simplify pixels-only training.
    # determine white strip x position (fall back to x_plane if unavailable)

    ws_geom = self._mj_model.geom('white_strip')
    ws_id = ws_geom.id
    ws_pos = jp.asarray(self._mj_model.geom_pos[ws_id])
    white_strip_x = ws_pos[0]

    # target is white strip of line
    target_pos = jp.array([white_strip_x, box_pos[1], box_pos[2]])


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
    


    # initialize env state and info
    metrics = {
        'floor_collision': jp.array(0.0, dtype=float),
        'cube_collision': jp.array(0.0),
        'jerk_per_step': jp.array(0.0),
        'success': jp.array(0.0),
        'out_of_bounds': jp.array(0.0),
        **{
            f'reward/{k}': 0.0
            for k in self._config.reward_config.reward_scales.keys()
        },
        'reward/success': jp.array(0.0),
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
        'prev_box_pos': box_pos,
        'frame_stack': jp.zeros((self._config.vision_config.render_height, 
                                 self._config.vision_config.render_width, 3 * self._config.frame_stack_size), dtype=float)
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
      # frame stack along color channel
      if self._config.frame_stack_size > 1:
          obs = jp.concat([obs] * self._config.frame_stack_size, axis=-1)
          info.update({'frame_stack': obs})
      obs = {'pixels/view_0': obs}

      if self._proprioception:

        ee_height = data.xpos[self._left_finger_geom][2]
        joint_p = data.qpos[:7] + jax.random.normal(rng, 7) * 0.1
        normalized_jp = 2 * (joint_p - jp.array(self._jnt_range())[:, 0]) / (
          jp.array(self._jnt_range())[:, 1] - jp.array(self._jnt_range())[:, 0]
        ) - 1
        joint_v = data.qvel[:7] + jax.random.normal(rng, 7) * 0.1
        normalized_jv = 2 * (joint_v - jp.array(self._jnt_vel_range())[:, 0]) / (
          jp.array(self._jnt_vel_range())[:, 1] - jp.array(self._jnt_vel_range())[:, 0]
        ) - 1
        _prop = jp.concatenate([ 
                                normalized_jp,
                                normalized_jv, 
                                jp.zeros(self.action_size), jp.array([ee_height])])

        
        obs["_prop"] = _prop 
    return mjx_env.State(data, obs, reward, done, metrics, info)
  
  def _get_reward(self, data: mjx.Data, info: Dict[str, Any]) -> Dict[str, Any]:
    target_pos = info["target_pos"]
    box_pos = data.xpos[self._obj_body]
    box_pos = box_pos.at[0].add(-0.03) # reach for the back of the box
    gripper_pos = data.site_xpos[self._gripper_site]

    gripper_box = 1 - jp.tanh(5 * jp.linalg.norm(box_pos - gripper_pos))
    robot_target_qpos = 1 - jp.tanh(
        jp.linalg.norm(
            data.qpos[self._robot_arm_qposadr]
            - self._init_q[self._robot_arm_qposadr]
        )
    )

    # Check for collisions with the floor
    hand_floor_collision = [
        collision.geoms_colliding(data, self._floor_geom, g)
        for g in [
            self._left_finger_geom,
            self._right_finger_geom,
            self._hand_geom,
        ]
    ]
    floor_collision = sum(hand_floor_collision) > 0
    no_floor_collision = (1 - floor_collision).astype(float)

    info["reached_box"] = 1.0 * jp.maximum(
        info["reached_box"],
        (jp.linalg.norm(box_pos - gripper_pos) < 0.012),
    )

    rewards = {
        "gripper_box": gripper_box,
        "no_floor_collision": no_floor_collision,
        "robot_target_qpos": robot_target_qpos,
    }
    return rewards

  
  def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    gripper_pos = data.site_xpos[self._gripper_site]
    gripper_mat = data.site_xmat[self._gripper_site].ravel()
    target_mat = math.quat_to_mat(data.mocap_quat[self._mocap_target])
    obs = jp.concatenate([
        data.qpos,
        data.qvel,
        gripper_pos,
        gripper_mat[3:],
        data.xmat[self._obj_body].ravel()[3:],
        data.xpos[self._obj_body] - data.site_xpos[self._gripper_site],
        info["target_pos"] - data.xpos[self._obj_body],
        target_mat.ravel()[:6] - data.xmat[self._obj_body].ravel()[:6],
        data.ctrl - data.qpos[self._robot_qposadr[:-1]],
    ])
    return obs

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
    # action = action.at[-1].set(state.info['action_history'][action_idx]) # for the gripper not needed here

    state.info['newly_reset'] = state.info['_steps'] == 0

    newly_reset = state.info['newly_reset']
    state.info['prev_reward'] = jp.where(
        newly_reset, 0.0, state.info['prev_reward']
    )
    state.info['prev_box_pos'] = jp.where(
        newly_reset,
        state.data.xpos[self._obj_body],
        state.info['prev_box_pos'],
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

    data = state.data
    # Cartesian control

    if self._config.action == 'cartesian_increment':
      # increment = jp.zeros(3)
      # increment = action  # directly set x, y, z and gripper commands.
      # ctrl, new_tip_position, no_soln = self._move_tip(
      #     state.info['current_pos'],
      #     self._start_tip_transform[:3, :3],
      #     data.ctrl,
      #     increment,
      # )
      increment = jp.zeros(7)
      increment = action
      ctrl, new_tip_position, no_soln = self._move_tip_orient(
          state.info['current_pos'],
          panda_kinematics.compute_franka_fk(  # current orientation
              data.qpos[:7]
          )[:3, :3],
          data.qpos[:8], # expects joint positions
          increment,
      )
      state.info.update({'current_pos': new_tip_position})

      if self._config.actuator == 'velocity': # careful with this
        delta_q = ctrl[:7] - data.qpos[:7] # calculate joint increments excluding gripper
        joint_cntrl = actuator_map("joint_increment", self._config.actuator, delta_q, data.qpos[:7], self._config.ctrl_dt) # ctrl is target joint pos
        ctrl = ctrl.at[:7].set(joint_cntrl)
    elif self._config.action in {'joint_increment', 'joint'}:
      ctrl, no_soln = self._move_joints(data.ctrl, action)
    else:
      raise ValueError(f"Invalid action type: {self._config.action}")
      
    ctrl = jp.clip(ctrl, self._lowers, self._uppers)


    # Simulator step
    ctrl = ctrl.at[7].set(0.04) # ensure gripper remains open
    data = mjx_env.step(self._mjx_model, data, ctrl, self.n_substeps)

    # Dense rewards
    box_pos = data.xpos[self._obj_body]
    box_xy = data.xpos[self._obj_body][:2]
    prev_xy = state.info['prev_box_pos'][:2]
    delta = jp.linalg.norm(box_xy - prev_xy)

    # In-bounds and margin-to-edge (normalized)
    sid = self._mj_model.geom('white_strip').id
    strip_pos = jp.asarray(self._mj_model.geom_pos[sid])[:2]
    strip_half = jp.asarray(self._mj_model.geom_size[sid])[:2]   # [hx, hy]

    dx = strip_half[0] - jp.abs(box_xy[0] - strip_pos[0])
    dy = strip_half[1] - jp.abs(box_xy[1] - strip_pos[1])
    # margin = jp.minimum(dx, dy)  # how far from closest edge (in meters)
    # box_on_ground = collision.geoms_colliding(data, self._floor_geom, self._box_geom)
    box_on_ground = box_pos[2] < 0.025
    in_bounds = (dx >= 0) & (dy >= 0) & box_on_ground

    delta = jp.linalg.norm(box_xy - prev_xy)
    hand_floor_collision = [
        collision.geoms_colliding(data, self._floor_geom, g)
        for g in [
            self._left_finger_geom,
            self._right_finger_geom,
            self._hand_geom,
        ]
    ]
    floor_collision = sum(hand_floor_collision) > 0
    # Zero-baseline reward: only displacement inside bounds
    w_move = 5.0
    # no_move = (delta < 0.001).astype(jp.float32)

    # keep cube centered
    ee_pos = data.site_xpos[self._gripper_site]
    center_delta = jp.linalg.norm(box_pos - ee_pos)
    w_center = -0.5
    w_floor = -0.5
    reward = in_bounds.astype(jp.float32) * (w_move * delta) + (w_center * center_delta) + (w_floor * floor_collision)
    # jax.debug.print("reward {}", reward)
    # is_nan = jp.any(jp.isnan(reward))
    # jax.lax.cond(is_nan, lambda _: jax.debug.print("reward {}", reward), lambda _: None, operand=None)

    reward = jp.clip(reward, -1e3, 1e3)
    
    state.info['prev_box_pos'] = data.xpos[self._obj_body] 


    
    out_of_bounds = jp.any(jp.abs(box_pos) > 1.0)
    out_of_bounds |= box_pos[2] < 0.0

    state.metrics.update(out_of_bounds=out_of_bounds.astype(float))


    current_qacc = data.qacc[:7]
    dt = self._config.ctrl_dt
    
    # Jerk is rate of change of acceleration
    jerk = jp.linalg.norm((current_qacc - state.info['prev_qacc']) / dt)
    jerk = jp.where(state.info['newly_reset'], 0.0, jerk)
    
    # Update acceleration history
    state.info['prev_qacc'] = current_qacc
    
    # Update metrics
    state.metrics.update(jerk_per_step=jerk.astype(float))
    
    
    state.metrics.update(floor_collision=floor_collision.astype(float))
    # state.metrics.update(success=success.astype(float))
    # state.metrics.update({f'reward/{k}': v for k, v in raw_rewards.items()})
    # state.metrics.update({
    #     'reward/success': (success * self._config.reward_config.success_reward).astype(float),
    # })
    done = (
        out_of_bounds
        | jp.isnan(data.qpos).any()
        | jp.isnan(data.qvel).any()
    )
    # jax.debug.print("Done: {}", done)
    # jax.debug.print("out of bounds: {}", out_of_bounds)
    # jax.debug.print("success: {}", success)

    # Ensure exact sync between newly_reset and the autoresetwrapper.
    state.info['_steps'] += self._config.action_repeat
    state.info['_steps'] = jp.where(
        done | (state.info['_steps'] >= self._config.episode_length),
        0,
        state.info['_steps'],
    )

    obs = self._get_obs(data, state.info)
    obs = jp.concat([obs, no_soln.reshape(1), action], axis=0)
  
    if self._vision:
      _, rgb, _ = self.renderer.render(state.info['render_token'], data)
      obs = jp.asarray(rgb[0][..., :3], dtype=jp.float32) / 255.0
      obs = adjust_brightness(obs, state.info['brightness'])
      # augment image
      rng_img = state.info['rng_img']  # Use the same RNG as in reset for consistency
      obs = augment_image(rng_img, img=obs, overlay=self._overlay)
      # frame stack along color channel
      if self._config.frame_stack_size > 1:
        prev_frame_stack = state.info['frame_stack']
        new_frame_stack = jp.concatenate(
            [prev_frame_stack[..., 3:], obs], axis=-1
        )
        # jax.debug.print("Frame stack shape: {}", new_frame_stack.shape)
        state.info['frame_stack'] = new_frame_stack
        obs = new_frame_stack
      obs = {'pixels/view_0': obs }
      if self._proprioception:
        state.info['rng'], rng_prop = jax.random.split(state.info['rng'])

        ee_height = data.xpos[self._left_finger_geom][2]
        joint_p = data.qpos[:7] + jax.random.normal(rng_prop, 7) * 0.1
        joint_v = data.qvel[:7] + jax.random.normal(rng_prop, 7) * 0.1
        normalized_jv = 2 * (joint_v - jp.array(self._jnt_vel_range())[:, 0]) / (
          jp.array(self._jnt_vel_range())[:, 1] - jp.array(self._jnt_vel_range())[:, 0]
        ) - 1
        normalized_jp = 2 * (joint_p - jp.array(self._jnt_range())[:, 0]) / (
          jp.array(self._jnt_range())[:, 1] - jp.array(self._jnt_range())[:, 0]
        ) - 1
        _prop = jp.concatenate([
                                normalized_jp, 
                                normalized_jv,  # Include normalized joint velocity
                                action, jp.array([ee_height])])


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
      box_pos, target_pos = box_pos[0], target_pos[0] # target X positions
    return jp.linalg.norm(box_pos - target_pos) < self._config.success_threshold
  
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

    return new_ctrl, new_tip_pos, no_soln
  
  def _skew(self, w: jax.Array) -> jax.Array:
    """Skew-symmetric matrix from a 3-vector."""
    wx, wy, wz = w[0], w[1], w[2]
    return jp.array([
        [0.0, -wz,  wy],
        [wz,  0.0, -wx],
        [-wy, wx,  0.0],
    ])

  def _so3_exp(self, w: jax.Array) -> jax.Array:
    """SO(3) exponential map for a rotation vector w (axis * angle)."""
    theta2 = jp.dot(w, w)
    theta = jp.sqrt(theta2 + 1e-12)

    K = self._skew(w)

    # Stable sin(theta)/theta and (1-cos(theta))/theta^2
    small = theta < 1e-6
    A = jp.where(small, 1.0 - theta2 / 6.0 + (theta2 * theta2) / 120.0, jp.sin(theta) / theta)
    B = jp.where(small, 0.5 - theta2 / 24.0 + (theta2 * theta2) / 720.0, (1.0 - jp.cos(theta)) / theta2)

    I = jp.identity(3)
    return I + A * K + B * (K @ K)

  def _clip_facing_down(self, R: jax.Array, max_tilt_rad: float) -> jax.Array:
    """
    Clamp the tool's +Z axis (R[:,2]) to be within max_tilt of world 'down' (0,0,-1),
    while preserving twist/yaw as much as possible by keeping the x-axis close to original.
    """
    down = jp.array([0.0, 0.0, -1.0])

    z = R[:, 2]
    z = z / (jp.linalg.norm(z) + 1e-12)

    c = jp.clip(jp.dot(z, down), -1.0, 1.0)  # cos(angle to down)
    tilt = jp.arccos(c)

    def do_clip(_):
      # Direction of z away from down (tangent component)
      v = z - jp.dot(z, down) * down
      v_norm = jp.linalg.norm(v) + 1e-12
      v_hat = v / v_norm

      # Put z on the cone boundary around down
      z_clamped = jp.cos(max_tilt_rad) * down + jp.sin(max_tilt_rad) * v_hat
      z_clamped = z_clamped / (jp.linalg.norm(z_clamped) + 1e-12)

      # Preserve "yaw/twist" by keeping x axis as close as possible, re-orthonormalize
      x = R[:, 0]
      x = x - jp.dot(x, z_clamped) * z_clamped
      x_norm = jp.linalg.norm(x) + 1e-12
      x_hat = x / x_norm

      y_hat = jp.cross(z_clamped, x_hat)
      y_hat = y_hat / (jp.linalg.norm(y_hat) + 1e-12)

      # Recompute x to ensure perfect orthonormality
      x_hat = jp.cross(y_hat, z_clamped)
      return jp.stack([x_hat, y_hat, z_clamped], axis=1)

    return jp.where(tilt > max_tilt_rad, do_clip(None), R)

  def _move_tip_orient(
      self,
      current_tip_pos: jax.Array,
      current_tip_rot: jax.Array,
      current_ctrl: jax.Array,
      action: jax.Array,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Calculate new tip pose from cartesian increment + small rotation increment."""


    # --- Position update (same as before) ---
    scaled_pos = action[:3] * self._config.action_scale
    new_tip_pos = current_tip_pos.at[:3].add(scaled_pos)

    new_tip_pos = new_tip_pos.at[0].set(jp.clip(new_tip_pos[0], 0.25, 0.77))
    new_tip_pos = new_tip_pos.at[1].set(jp.clip(new_tip_pos[1], -0.32, 0.32))
    new_tip_pos = new_tip_pos.at[2].set(jp.clip(new_tip_pos[2], 0.02, 0.5))

    # --- Rotation update (NEW) ---
    # Keep rotation increments small via scaling + optional norm clip
    rot_scale = getattr(self._config, "rot_action_scale", 0.05)  # rad per step, ~3 degrees
    max_rot_step = getattr(self._config, "max_rot_step", 0.10)   # rad cap on ||drot||
    drot = action[3:6] * rot_scale
    drot_norm = jp.linalg.norm(drot) + 1e-12
    drot = jp.where(drot_norm > max_rot_step, drot * (max_rot_step / drot_norm), drot)

    dR = self._so3_exp(drot)
    new_tip_rot = current_tip_rot @ dR  # body-frame incremental rotation

    # Clip overall orientation so tool stays facing down (mostly)
    max_tilt = getattr(self._config, "max_tilt_rad", 0.35)  # ~20 degrees
    new_tip_rot = self._clip_facing_down(new_tip_rot, max_tilt)

    # --- Build 4x4 target pose for IK ---
    new_tip_mat = jp.identity(4)
    new_tip_mat = new_tip_mat.at[:3, :3].set(new_tip_rot)
    new_tip_mat = new_tip_mat.at[:3, 3].set(new_tip_pos)

    q7_scale = getattr(self._config, "q7_action_scale", 0.05)   # rad/step
    q7_min, q7_max = getattr(self._config, "q7_limits", (-2.9, 2.9))  # set from your model
    q7_des = jp.clip(current_ctrl[6] + action[7] * q7_scale, q7_min, q7_max)

    # --- IK + safety fallback (same pattern as before) ---
    out_jp = panda_kinematics.compute_franka_ik(
        new_tip_mat, q7_des, current_ctrl[:7]
    )
    no_soln = jp.any(jp.isnan(out_jp))
    out_jp = jp.where(no_soln, current_ctrl[:7], out_jp)
    no_soln = jp.logical_or(no_soln, jp.any(jp.isnan(out_jp)))

    # If still NaNs, revert the tip pos (and you may also want to revert rot logically)
    new_tip_pos = jp.where(jp.any(jp.isnan(out_jp)), current_tip_pos, new_tip_pos)
    new_tip_rot = jp.where(jp.any(jp.isnan(out_jp)), current_tip_rot, new_tip_rot)

    new_ctrl = current_ctrl
    new_ctrl = new_ctrl.at[:7].set(out_jp)

    return new_ctrl, new_tip_pos, no_soln

  def _move_joints(self, current_ctrl: jax.Array, action: jax.Array):
    new_ctrl = current_ctrl
    scaled_action = action * self._config.action_scale
    if self._config.action == 'joint_increment':
      # Incremental joint control.
      new_ctrl = new_ctrl.at[:7].add(scaled_action)
    elif self._config.action == 'joint':
      # Absolute joint control.
      new_ctrl = new_ctrl.at[:7].set(scaled_action)
      if self._config.actuator == 'torque':
        raise ValueError("Don't use torque")
        new_ctrl = new_ctrl.at[:7].set(jp.clip(new_ctrl[:7],\
                                                -self._max_torque / self._gear, self._max_torque / self._gear))

    no_soln = jp.any(jp.isnan(new_ctrl))

    return new_ctrl, no_soln
    

  @property
  def action_size(self) -> int:
    if self._config.action == 'cartesian_increment':
      return 7
    elif self._config.action in {'joint_increment', 'joint'}:
      return 7 # for all 7 joints
    
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
  #     xml_path.read_text(), assets=panda.get_assets(actuator="position", task="push")
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
  key = jax.random.PRNGKey(1)
  env = PandaPushCuboid(config_overrides={'vision': False, 'action': "cartesian_increment", "actuator":"position"})

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
  with mujoco.viewer.launch_passive(mj_model_vis, mj_data_vis) as viewer:
      reset_counter = 0
      while viewer.is_running():
          reset_counter += 1
                    # copy mjx state -> mujoco.MjData (slice safely to handle shape mismatches)
          mj_data_vis.qpos[: mj_data_vis.qpos.shape[0]] = np.asarray(state.data.qpos)[: mj_data_vis.qpos.shape[0]]
          mj_data_vis.qvel[: mj_data_vis.qvel.shape[0]] = np.asarray(state.data.qvel)[: mj_data_vis.qvel.shape[0]]
          ctrl_src = np.asarray(state.data.ctrl)
          mj_data_vis.ctrl[: min(mj_data_vis.ctrl.shape[0], ctrl_src.shape[0])] = ctrl_src[: mj_data_vis.ctrl.shape[0]]

          if reset_counter % 200 == 0:
              print("resetting")
              state = jit_reset(jax.random.PRNGKey(int(time.time() * 1e6)))
              mj_data_vis.qpos[: mj_data_vis.qpos.shape[0]] = np.asarray(state.data.qpos)[: mj_data_vis.qpos.shape[0]]
              mj_data_vis.qvel[: mj_data_vis.qvel.shape[0]] = np.asarray(state.data.qvel)[: mj_data_vis.qvel.shape[0]]
              ctrl_src = np.asarray(state.data.ctrl)
              mj_data_vis.ctrl[: min(mj_data_vis.ctrl.shape[0], ctrl_src.shape[0])] = ctrl_src[: mj_data_vis.ctrl.shape[0]]
              mujoco.mj_step(mj_model_vis, mj_data_vis)
              viewer.sync()
              time.sleep(5)
              
          mujoco.mj_step(mj_model_vis, mj_data_vis)
          viewer.sync()


          action = jax.random.uniform(
              jax.random.PRNGKey(int(time.time() * 1e6)),
              (env.action_size,),
              minval=-1,
              maxval=1,
          )
          print("Action:", action)
    
          state = jit_step(state, action)
          print("reward", state.reward)




