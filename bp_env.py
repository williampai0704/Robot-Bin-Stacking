import pybullet as p
import pybullet_data
import numpy as np
import time

from assets.ycb_objects import getURDFPath
from utils import camera
from utils.control import get_movej_trajectory

class BinPickEnviornment:
    def __init__(self, gui=True):
        
        # 0 load environment
        if gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(1.5,45,-45,[0,0,0])
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._plane_id = p.loadURDF("plane.urdf")
        p.setGravity(0, 0, -9.8)

        # 1 load UR5 robot
        self.robot_body_id = p.loadURDF(
            "assets/ur5/ur5.urdf", [0, 0, 0.4], p.getQuaternionFromEuler([0, 0, 0]))
        self._mount_body_id = p.loadURDF(
            "assets/ur5/mount.urdf", [0, 0, 0.2], p.getQuaternionFromEuler([0, 0, 0]))

        # Get revolute joint indices of robot (skip fixed joints)
        robot_joint_info = [p.getJointInfo(self.robot_body_id, i) for i in range(
            p.getNumJoints(self.robot_body_id))]
        self._robot_joint_indices = [
            x[0] for x in robot_joint_info if x[2] == p.JOINT_REVOLUTE]

        # joint position threshold in radians (i.e. move until joint difference < epsilon)
        self._joint_epsilon = 1e-3

        # Robot home joint configuration 
        self.robot_home_joint_config = [
            -np.pi, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0]
        # Robot goal joint configuration 
        self.robot_goal_joint_config = [
            0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0]
    
        # 2. load gripper
        self.robot_end_effector_link_index = 9
        self._robot_tool_offset = [0, 0, -0.05]
        # Distance between tool tip and end-effector joint
        self._tool_tip_to_ee_joint = np.array([0, 0, 0.15])

        # Attach robotiq gripper to UR5 robot
        # - We use createConstraint to add a fixed constraint between the ur5 robot and gripper.
        self._gripper_body_id = p.loadURDF("assets/gripper/robotiq_2f_85.urdf")
        p.resetBasePositionAndOrientation(self._gripper_body_id, [
                                          0.5, 0.1, 0.2], p.getQuaternionFromEuler([np.pi, 0, 0]))

        p.createConstraint(self.robot_body_id, self.robot_end_effector_link_index, self._gripper_body_id, 0, jointType=p.JOINT_FIXED, jointAxis=[
                           0, 0, 0], parentFramePosition=[0, 0, 0], childFramePosition=self._robot_tool_offset, childFrameOrientation=p.getQuaternionFromEuler([0, 0, np.pi/2]))

        # Set friction coefficients for gripper fingers
        for i in range(p.getNumJoints(self._gripper_body_id)):
            p.changeDynamics(self._gripper_body_id, i, lateralFriction=1.0, spinningFriction=1.0,
                             rollingFriction=0.0001, frictionAnchor=True)
        
        self.set_joints(self.robot_home_joint_config)

        
    def load_box():
        
    def get_gripper_pose():
    
    def grasp_box():
    
    def release_box():
    
    def move_tool():
        
    def detect_collision():
    
    def pack_efficiency():
        
    def est_reward():