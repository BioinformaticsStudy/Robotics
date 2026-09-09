
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Universal Robots.
Reference: https://github.com/ros-industrial/universal_robot
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

UR_GRIPPER_CFG = ArticulationCfg(

# Where is the USD file for this robot?
spawn=sim_utils.UsdFileCfg(       
    usd_path=f"/home/ubuntu/Robotics/JordanCarter/jordanResearch/Robotics/SPiDROPO/SimEnvironment.usd", 
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=5.0,
            max_angular_velocity=20.0,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, 
            solver_position_iteration_count=15, 
            solver_velocity_iteration_count=4
        ),
    ),
# What is its initial position of the robot, and its joints?
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        },
    ),
# What parts of the robot move, and how stiff / damped are they?
    actuators={
        "arm": ImplicitActuatorCfg(
        joint_names_expr=[
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
        ],
        stiffness=100.0,
        damping=10.0,
        effort_limit=10.0,
    ),

    "gripper": ImplicitActuatorCfg(
        joint_names_expr=["gripper"],
        stiffness=50.0,
        damping=5.0,
        effort_limit=5.0,
    ),
    }
)
