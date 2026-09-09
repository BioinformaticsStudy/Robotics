from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def ee_position(env, asset_cfg):
    robot = env.scene[asset_cfg.name]
    return robot.data.body_state_w[:, asset_cfg.body_ids[0], :3]

def cube_to_ee(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    cube = env.scene[object_cfg.name]
    robot = env.scene[asset_cfg.name]

    cube_pos = cube.data.root_pos_w[:, :3]
    ee_pos = robot.data.body_state_w[:, asset_cfg.body_ids[0], :3]
    
    return cube_pos - ee_pos

def cube_relative_velocity(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor: 
    cube = env.scene[object_cfg.name]
    robot = env.scene[asset_cfg.name]

    cube_vel = cube.data.root_lin_vel_w
    ee_vel = robot.data.body_state_w[:, asset_cfg.body_ids[0], 7:10]
    return cube_vel - ee_vel

def goal_to_cube(env: ManagerBasedRLEnv, goal_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg) -> torch.Tensor: 
    goal = env.scene[goal_cfg.name]
    cube = env.scene[object_cfg.name]

    goal_pos = goal.data.root_pos_w[:, :3]
    cube_pos = cube.data.root_pos_w[:, :3]

    return goal_pos - cube_pos

def ee_to_goal(env: ManagerBasedRLEnv, goal_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor: 
    goal = env.scene[goal_cfg.name]
    robot = env.scene[asset_cfg.name]

    goal_pos = goal.data.root_pos_w[:, :3]
    ee_pos = robot.data.body_state_w[:, asset_cfg.body_ids[0],: 3]

    return goal_pos - ee_pos
