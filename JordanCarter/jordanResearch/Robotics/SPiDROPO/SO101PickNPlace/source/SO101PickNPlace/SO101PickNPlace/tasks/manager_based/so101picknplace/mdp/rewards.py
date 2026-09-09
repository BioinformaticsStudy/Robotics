# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def reach_object_reward(env, asset_cfg, object_cfg):
    robot = env.scene[asset_cfg.name]
    obj = env.scene[object_cfg.name]

    ee_pos = env.scene["ee_frame"].data.target_pos_w[:, 0, :]
    obj_pos = obj.data.root_pos

    dist = torch.norm(ee_pos - obj_pos, dim=1)
    print(dist)
    return -dist

def grasp_bonus(env, contact_sensor):
    return contact_sensor.data.is_contact.float()

def lift_reward(env, object_cfg, contact_sensor):
    obj_z = env.scene[object_cfg.name].data.root_pos[:, 2]
    lift = torch.clamp(obj_z - 0.055, min=0.0)
    return lift * contact_sensor.data.is_contact.float()