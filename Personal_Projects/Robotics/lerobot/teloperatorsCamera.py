from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.teleoperators.so_leader import SO101LeaderConfig, SO101Leader
from lerobot.robots.so_follower import SO101FollowerConfig, SO101Follower

camera_config = {
    "front": OpenCVCameraConfig(index_or_path=1, width=1920, height=1080, fps=30)
}

robot_config = SO101FollowerConfig(
    port="/dev/tty.usbmodem5A7A0546471",
    id="my_awesome_follower_arm",
	cameras=camera_config
)

teleop_config = SO101LeaderConfig(
	port="/dev/tty.usbmodem5AB01789301",
    id="my_awesome_leader_arm",
)

robot = SO101Follower(robot_config)
teleop_device = SO101Leader(teleop_config)
robot.connect()
teleop_device.connect()

while True:
    observation = robot.get_observation()
    action = teleop_device.get_action()
    robot.send_action(action)
