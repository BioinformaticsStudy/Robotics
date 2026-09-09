# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.sim import CollisionPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR



from . import mdp

##
# Pre-defined configs
##

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG  # isort:skip
from .ur_gripper import UR_GRIPPER_CFG


##
# Scene definition
##


@configclass
class RlpicknplaceSceneCfg(InteractiveSceneCfg):
    """Configuration for a pick-and-place scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # robot
    robot: ArticulationCfg = UR_GRIPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    #block
    cube: RigidObjectCfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.14, 0.00, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.5, 0.5, 0.5),
                activate_contact_sensors=True,
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )
    
    #Goal 
    goal: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Goal",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.14, 0.10, 0.00),
            rot=(1,0,0,0),
        ),
        spawn=sim_utils.SphereCfg(
            radius=0.02,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0,0.0,0.0),
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            # remove collision_props entirely
        ),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    #Arm joints
    arm_action: ActionTerm = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=["shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll"], 
        scale=0.2, 
        use_default_offset=True, 
        debug_vis=True
    )

    #Gripper Joint
    gripper_action: mdp.BinaryJointPositionActionCfg = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["gripper"],
        open_command_expr={"gripper": 1.7453},
        close_command_expr={"gripper": -0.1745},
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Robot observation
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        ee_pos = ObsTerm(
            func=mdp.ee_position,
            params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["gripper"]
                )
            }
        )

        #Block observation
        cube_pos_rel = ObsTerm(
            func = mdp.cube_to_ee,
            params = {"object_cfg": SceneEntityCfg("cube"), "asset_cfg": SceneEntityCfg(
                "robot",
                body_names = ["gripper"])}
        )

        cube_vel = ObsTerm(
            func = mdp.cube_relative_velocity,
            params = {"object_cfg": SceneEntityCfg("cube"), "asset_cfg": SceneEntityCfg(
                "robot",
                body_names = ["gripper"])}
        )

        #Goal Observations
        goal_pos = ObsTerm(
            func = mdp.goal_to_cube, 
            params = {"goal_cfg": SceneEntityCfg("goal"), "object_cfg": SceneEntityCfg("cube")}
        )

        robot_to_goal_pos = ObsTerm(
            func = mdp.ee_to_goal,
            params = {"goal_cfg": SceneEntityCfg("goal"), "asset_cfg": SceneEntityCfg(
                "robot",
                body_names = ["gripper"])}
        )

        #Ending action
        last_action = ObsTerm(
            func=mdp.last_action
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    reset_diagnostic = EventTerm(
        func = mdp.reset_diagnostic,
        mode = "reset",
    )

    #Reset robot joints
    reset_robot_joints = EventTerm(
        func = mdp.reset_joints_by_scale, 
        mode = "reset",
        params = {
            "position_range": (0.95, 1.05),
            "velocity_range": (0.0, 0.0),
        },
    )

    #Reset Cube position in a randomized position 
    reset_cube = EventTerm(
        func = mdp.reset_root_state_uniform,
        mode = "reset",
        params = {
            "asset_cfg": SceneEntityCfg("cube"),
            "pose_range": { 
                "x": (0.14,0.14),
                "y": (0.00, 0.00),
                "z": (0.0, 0.0),
            },
            "velocity_range": {
                "linear": (0.0, 0.0),
                "angular": (0.0, 0.0), 
            },
        },
    )

    #Reset Goal position in a randomized position
    reset_goal = EventTerm(
        func = mdp.reset_root_state_uniform, 
        mode = "reset",
        params = {
            "asset_cfg": SceneEntityCfg("goal"),
            "pose_range": { 
                "x" : (0.14,0.14),
                "y" : (0.10,0.10),
                "z" : (0.00,0.00),
            },
            "velocity_range": {
                "linear": (0.0, 0.0),
                "angular": (0.0, 0.0), 
            },
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Reaching
    reaching_object = RewTerm(
        func=mdp.reaching_object,
        weight=1.0, 
        params={
            "object_cfg": SceneEntityCfg("cube"),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["jaw"],
            ),
            "std": 0.1,
        },
    )

    """ 
    # Grasping
    grasping_object = RewTerm(
        func=mdp.grasping_object,
        weight=1.0,
        params={
            "object_cfg": SceneEntityCfg("cube"),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["jaw"],
            ),
            "grasp_distance_threshold": 0.02,
        },
    )
    """

    gripper_open_near_object = RewTerm(
        func=mdp.gripper_open_near_object,
        weight=0.10,
        params={
            "object_cfg": SceneEntityCfg("cube"),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["jaw"],
            ),
            "threshold": 0.05,
        },
    )


    # Lifting
    object_lifted = RewTerm(
        func=mdp.object_lifted,
        weight=15.0,
        params={
            "object_cfg": SceneEntityCfg("cube"),
            "minimal_height": 0.025,
        },
    )

    # Moving the cube toward the goal
    object_goal_distance = RewTerm(
        func=mdp.object_goal_distance,
        weight=16.0,
        params={
            "object_cfg": SceneEntityCfg("cube"),
            "goal_cfg": SceneEntityCfg("goal"),
            "std": 0.05,
            "minimal_height": 0.065,
        },
    )

    # Placement
    object_placed_bonus = RewTerm(
        func=mdp.object_placed,
        weight=50.0,
        params={
            "object_cfg": SceneEntityCfg("cube"),
            "goal_cfg": SceneEntityCfg("goal"),
            "horizontal_threshold": 0.04,
            "velocity_threshold": 0.1,
        },
    )

    # Terminal success
    success = RewTerm(
        func=mdp.is_terminated_term,
        weight=200.0,
        params={
            "term_keys": "object_placed_success",
        },
    )

    # Smoothness
    action_rate = RewTerm(
        func=mdp.action_rate_l2_clamped,
        weight=-0.0001,
    )

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2_clamped,
        weight=-0.0001,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True,
    )

    object_placed_success = DoneTerm(
        func=mdp.object_placed_success,
        params={
            "object_cfg": SceneEntityCfg("cube"),
            "goal_cfg": SceneEntityCfg("goal"),
            "horizontal_threshold": 0.04,
            "velocity_threshold": 0.1,
        },
    )


##
# Environment configuration
##


@configclass
class RlpicknplaceEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: RlpicknplaceSceneCfg = RlpicknplaceSceneCfg(num_envs=256.0, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()



    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 10
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
