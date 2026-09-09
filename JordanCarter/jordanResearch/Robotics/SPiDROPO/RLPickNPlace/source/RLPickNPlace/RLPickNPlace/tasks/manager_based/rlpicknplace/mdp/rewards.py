# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING 

import torch
import numpy as np

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi
from isaaclab.utils.math import quat_apply, quat_apply_inverse
from pxr import Usd, UsdGeom, Gf

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

usd_path = "/home/ubuntu/Robotics/JordanCarter/jordanResearch/Robotics/SPiDROPO/assets/robots/so101_follower.usd"
local_grasp_offset = torch.tensor([-0.0092, 0.0021, 0.0428], )

def reset_diagnostic(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
):
    # Initialize diagnostic tensors the first time this function is called.
    if not hasattr(env, "reach_success"):
        env.reach_success = torch.zeros(
            env.num_envs,
            dtype=torch.bool,
            device=env.device,
        )

        env.grasp_success = torch.zeros(
            env.num_envs,
            dtype=torch.bool,
            device=env.device,
        )

        env.lift_success = torch.zeros(
            env.num_envs,
            dtype=torch.bool,
            device=env.device,
        )

        env.min_grasp_distance = torch.full(
            (env.num_envs,),
            float("inf"),
            device=env.device,
        )

        env.gripper_open_at_min_distance = torch.zeros(
            env.num_envs,
            device=env.device,
        )

    # Diagnostics may not have been initialized yet.
    if not hasattr(env, "reach_success"):
        return

    env.reach_success[env_ids] = False
    env.grasp_success[env_ids] = False
    env.lift_success[env_ids] = False
    env.min_grasp_distance[env_ids] = float("inf")
    env.gripper_open_at_min_distance[env_ids] = 0.0

def diagnostic_print(env, asset_cfg, object_cfg):
    env_id = 0
    robot = env.scene["robot"]
    cube = env.scene["cube"]

    # --------------------------------------------------
    # Get jaw pose
    # --------------------------------------------------

    jaw_id = robot.find_bodies("jaw")[0][0]

    jaw_pos = robot.data.body_pos_w[:, jaw_id]
    jaw_quat = robot.data.body_quat_w[:, jaw_id]

    object_pos = cube.data.root_pos_w

    # --------------------------------------------------
    # SO-101 grasp reference point
    # Same offset used by reaching_object() and
    # grasping_object()
    # --------------------------------------------------

    grasp_offset = torch.tensor(
        [-0.0092, 0.0021, 0.0428],
        device=env.device,
        dtype=jaw_pos.dtype,
    )

    grasp_point = jaw_pos + quat_apply(
        jaw_quat,
        grasp_offset.expand(env.num_envs, -1),
    )

    # --------------------------------------------------
    # Distance to cube
    # --------------------------------------------------

    distance = torch.linalg.norm(
        grasp_point - object_pos,
        dim=1,
    )

    # --------------------------------------------------
    # Gripper state
    # --------------------------------------------------

    gripper_id = robot.find_joints("gripper")[0][0]

    gripper_pos = robot.data.joint_pos[:, gripper_id]

    gripper_lower = -0.1745
    gripper_upper = 1.7453

    gripper_open = torch.clamp(
        (gripper_pos - gripper_lower)
        / (gripper_upper - gripper_lower),
        0.0,
        1.0,
    )

    # --------------------------------------------------
    # Stage conditions
    # --------------------------------------------------

    near_object = distance < 0.02
    gripper_closed = gripper_open < 0.3

    grasped = near_object & gripper_closed

    object_height = object_pos[:, 2]
    lifted = object_height > 0.065

    # --------------------------------------------------
    # Persistent diagnostic flags
    # --------------------------------------------------

    if not hasattr(env, "reach_success"):
        env.reach_success = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        env.grasp_success = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        env.lift_success = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )

    env.reach_success |= near_object
    env.grasp_success |= grasped
    env.lift_success |= lifted

    # --------------------------------------------------
    # Print
    # --------------------------------------------------

    print("\n===== DIAGNOSTIC =====")

    print("Grasp point:", grasp_point[0].detach().cpu().numpy())
    print("Cube:", object_pos[0].detach().cpu().numpy())

    print("Distance:", distance[0].item())

    print("Gripper open:", gripper_open[0].item())

    print("Near object:", near_object[0].item())
    print("Grasped:", grasped[0].item())
    print("Object height:", object_height[0].item())
    print("Lifted:", lifted[0].item())

    print(
        "Reach success:",
        env.reach_success.float().mean().item()
    )

    print(
        "Grasp success:",
        env.grasp_success.float().mean().item()
    )

    print(
        "Lift success:",
        env.lift_success.float().mean().item()
    )

    gripper_id = robot.find_joints("gripper")[0][0]
    gripper_joint_id = robot.find_joints("gripper")[0][0]

    print(
        "Gripper action:",
        env.action_manager.action[:, -1].min().item(),
        env.action_manager.action[:, -1].max().item(),
        env.action_manager.action[:, -1].mean().item(),
    )

    print(
        "Gripper joint:",
        robot.data.joint_pos[:, gripper_joint_id].min().item(),
        robot.data.joint_pos[:, gripper_joint_id].max().item(),
        robot.data.joint_pos[:, gripper_joint_id].mean().item(),
    )

    gripper_term = env.action_manager.get_term("gripper_action")
    print(
        "Minimum grasp distance:",
        env.min_grasp_distance[env_id].item(),
    )

    print(
        "Gripper open at minimum:",
        env.gripper_open_at_min_distance[env_id].item(),
    )

    print(
        "Gripper joint at minimum:",
        (
            gripper_lower
            + env.gripper_open_at_min_distance[env_id]
            * (gripper_upper - gripper_lower)
        ).item(),
    )

def get_grasp_point(robot, env):
    jaw_id = robot.find_bodies("jaw")[0][0]

    jaw_pos = robot.data.body_pos_w[:, jaw_id]
    jaw_quat = robot.data.body_quat_w[:, jaw_id]

    # Fixed offset from jaw body frame to the desired
    # grasp reference point.
    local_offset = torch.tensor(
        [..., ..., ...],
        device=env.device,
        dtype=jaw_pos.dtype,
    ).expand(env.num_envs, -1)

    grasp_point = jaw_pos + quat_apply(jaw_quat, local_offset)

    return grasp_point

def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    return torch.sum(torch.square(joint_pos - target), dim=1)


def action_rate_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions, clamped to prevent reward explosion."""
    return torch.sum(
        torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
    ).clamp(-1000, 1000)


def joint_vel_l2_clamped(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint velocities, clamped to prevent reward explosion."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1).clamp(-1000, 1000)

def reaching_object(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    std: float,
) -> torch.Tensor:

    robot: Articulation = env.scene[asset_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    jaw_id = robot.find_bodies("jaw")[0][0]

    jaw_pos = robot.data.body_pos_w[:, jaw_id]
    jaw_quat = robot.data.body_quat_w[:, jaw_id]

    # SO-101 physical jaw reference point.
    #
    # Derived from the jaw mesh rather than using the jaw
    # rigid-body origin.
    local_grasp_offset = torch.tensor(
        [-0.0092, 0.0021, 0.0428],
        device=env.device,
        dtype=jaw_pos.dtype,
    )

    grasp_point = jaw_pos + quat_apply(
        jaw_quat,
        local_grasp_offset.expand(env.num_envs, -1),
    )

    object_pos = obj.data.root_pos_w

    distance = torch.linalg.norm(
        grasp_point - object_pos,
        dim=1,
    )

    gripper_joint_id = robot.find_joints("gripper")[0][0]

    gripper_pos = robot.data.joint_pos[:, gripper_joint_id]

    gripper_lower = -0.1745
    gripper_upper = 1.7453

    gripper_open = torch.clamp(
        (gripper_pos - gripper_lower)
        / (gripper_upper - gripper_lower),
        0.0,
        1.0,
    )

    # Determine which environments just reached a new minimum
    new_min = distance < env.min_grasp_distance

    # Save the gripper opening at that minimum
    env.gripper_open_at_min_distance = torch.where(
        new_min,
        gripper_open,
        env.gripper_open_at_min_distance,
    )

    # Update minimum distance
    env.min_grasp_distance = torch.minimum(
        env.min_grasp_distance,
        distance,
    )

    # Diagnostic only.
    reached = distance < 0.03
    env.reach_success |= reached

    if env.common_step_counter % 1000 == 0:
        diagnostic_print(env, asset_cfg, object_cfg)

    # OpenArm reward.
    return 1.0 - torch.tanh(distance / std)

"""def grasping_object(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    grasp_distance_threshold: float,
) -> torch.Tensor:

    robot: Articulation = env.scene[asset_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    # ---------------------------------------------------------
    # SO-101 physical grasp point
    # ---------------------------------------------------------

    jaw_id = robot.find_bodies("jaw")[0][0]

    jaw_pos = robot.data.body_pos_w[:, jaw_id]
    jaw_quat = robot.data.body_quat_w[:, jaw_id]

    local_grasp_offset = torch.tensor(
        [-0.0092, 0.0021, 0.0428],
        device=env.device,
        dtype=jaw_pos.dtype,
    )

    grasp_point = jaw_pos + quat_apply(
        jaw_quat,
        local_grasp_offset.expand(env.num_envs, -1),
    )

    object_pos = obj.data.root_pos_w

    distance = torch.linalg.norm(
        grasp_point - object_pos,
        dim=1,
    )

    # ---------------------------------------------------------
    # Distance gate
    # ---------------------------------------------------------

    near_object = distance < grasp_distance_threshold

    # ---------------------------------------------------------
    # SO-101 continuous gripper
    # ---------------------------------------------------------

    gripper_joint_id = robot.find_joints("gripper")[0][0]

    gripper_pos = robot.data.joint_pos[:, gripper_joint_id]

    gripper_lower = -0.1745
    gripper_upper = 1.7453

    gripper_open = torch.clamp(
        (gripper_pos - gripper_lower)
        / (gripper_upper - gripper_lower),
        0.0,
        1.0,
    )

    # 0 = open
    # 1 = closed
    closing_factor = 1.0 - gripper_open

    # Only encourage closing when the gripper is near
    # the object.
    reward = near_object.float() * closing_factor

    # ---------------------------------------------------------
    # Diagnostic grasp condition
    # ---------------------------------------------------------

    grasped = (
        near_object
        & (gripper_open < 0.3)
    )

    env.grasp_success |= grasped

    return reward
"""

def gripper_open_near_object(
        env: ManagerBasedRLEnv,
        object_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
        threshold: float,
    ) -> torch.Tensor:

        robot: Articulation = env.scene[asset_cfg.name]
        obj: RigidObject = env.scene[object_cfg.name]

        jaw_id = robot.find_bodies("jaw")[0][0]

        local_grasp_offset = torch.tensor(
            [-0.0092, 0.0021, 0.0428],
            device=env.device,
            dtype=robot.data.body_pos_w.dtype,
        )

        jaw_pos = robot.data.body_pos_w[:, jaw_id]
        jaw_quat = robot.data.body_quat_w[:, jaw_id]

        grasp_point = jaw_pos + quat_apply(
            jaw_quat,
            local_grasp_offset.expand(env.num_envs, -1),
        )

        distance = torch.linalg.norm(
            grasp_point - obj.data.root_pos_w,
            dim=1,
        )

        gripper_joint_id = robot.find_joints("gripper")[0][0]
        gripper_pos = robot.data.joint_pos[:, gripper_joint_id]

        gripper_open = torch.clamp(
            (gripper_pos - (-0.1745))
            / (1.7453 - (-0.1745)),
            0.0,
            1.0,
        )

        near = distance < threshold
        very_near = distance < (threshold / 2)

        return near.float() * (~very_near).float() * gripper_open


def object_lifted(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    minimal_height: float,
) -> torch.Tensor:

    obj: RigidObject = env.scene[object_cfg.name]

    object_height = obj.data.root_pos_w[:, 2]

    # Persistent diagnostic/state flag:
    # The object has been lifted after a successful grasp.
    lifted = object_height > minimal_height
    env.lift_success |= (lifted & env.grasp_success)

    # OpenArm-style binary lift reward.
    return lifted.float()


def object_goal_distance(
    env,
    object_cfg: SceneEntityCfg,
    goal_cfg: SceneEntityCfg,
    std: float,
    minimal_height: float,
) -> torch.Tensor:
    """Reward moving a lifted object toward the goal.

    Goal tracking is active only after the object has been lifted above
    the specified minimum height.
    """

    obj: RigidObject = env.scene[object_cfg.name]
    goal: RigidObject = env.scene[goal_cfg.name]

    object_pos = obj.data.root_pos_w
    goal_pos = goal.data.root_pos_w

    distance = torch.linalg.norm(
        object_pos[:, :2] - goal_pos[:, :2],
        dim=1,
    )

    distance_reward = 1.0 - torch.tanh(distance / std)

    lifted = object_pos[:, 2] > minimal_height

    return distance_reward * lifted.float()


def object_placed(
    env,
    object_cfg: SceneEntityCfg,
    goal_cfg: SceneEntityCfg,
    horizontal_threshold: float,
    velocity_threshold: float,
) -> torch.Tensor:
    """Determine whether the object has been successfully placed after lifting."""

    obj: RigidObject = env.scene[object_cfg.name]
    goal: RigidObject = env.scene[goal_cfg.name]

    object_pos = obj.data.root_pos_w
    goal_pos = goal.data.root_pos_w

    horizontal_distance = torch.linalg.norm(
        object_pos[:, :2] - goal_pos[:, :2],
        dim=1,
    )

    linear_velocity = torch.linalg.norm(
        obj.data.root_lin_vel_w,
        dim=1,
    )

    placed = (
        env.lift_success
        & (horizontal_distance < horizontal_threshold)
        & (linear_velocity < velocity_threshold)
    )

    return placed.float() 

def object_placed_success(
    env,
    object_cfg: SceneEntityCfg,
    goal_cfg: SceneEntityCfg,
    horizontal_threshold: float,
    velocity_threshold: float,
) -> torch.Tensor:

    obj: RigidObject = env.scene[object_cfg.name]
    goal: RigidObject = env.scene[goal_cfg.name]

    object_pos = obj.data.root_pos_w
    goal_pos = goal.data.root_pos_w

    horizontal_distance = torch.linalg.norm(
        object_pos[:, :2] - goal_pos[:, :2],
        dim=1,
    )

    linear_velocity = torch.linalg.norm(
        obj.data.root_lin_vel_w,
        dim=1,
    )

    return (
        env.lift_success
        & (horizontal_distance < horizontal_threshold)
        & (linear_velocity < velocity_threshold)
    )