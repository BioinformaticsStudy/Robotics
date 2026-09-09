# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore

    return torch.norm(curr_pos_w - des_pos_w, dim=1)

def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame) and maps it with a tanh kernel.
    """
    # extract the asset (to enable type hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)

    episode_step = env.episode_length_buf

    """if episode_step[0] in [0, 120, 300, 599]:

        print(f"\n===== step {episode_step[0].item()} =====")

        print("Robot 0")
        print("Desired:", des_pos_w[0])
        print("Current:", curr_pos_w[0])
        print("Distance:", distance[0])

        print("\nAverage over all environments")
        print("Desired:", torch.mean(des_pos_w, dim=0))
        print("Current:", torch.mean(curr_pos_w, dim=0))
        print("Distance:", torch.mean(distance, dim=0))
        print("Reward: ", 1 - torch.tanh(distance / std))"""

    return 1 - torch.tanh(distance / std)

def orientation_command_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize orientation error."""

    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    # Desired quaternion from command
    des_quat = command[:, 3:7]

    # Current EE quaternion
    curr_quat = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]

    # Angle error in radians
    angle_error = quat_error_magnitude(curr_quat, des_quat)

    return angle_error

def orientation_command_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward orientation tracking with tanh kernel."""

    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    des_quat = command[:, 3:7]
    curr_quat = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]

    angle_error = quat_error_magnitude(curr_quat, des_quat)

    return 1 - torch.tanh(angle_error / std)