import pybullet as p
import pybullet_data
import numpy as np
import time
import math

# Add this import for GPU detection
import torch

from typing import Tuple, List, Dict


from utils.control import get_movej_trajectory


class StackingEfficiencyCalculator:
    @staticmethod
    def get_box_corners(position: List[float], orientation: List[float], 
                       half_extents: List[float]) -> np.ndarray:
        """
        Get the 8 corners of a box given its position, orientation, and dimensions.
        
        Args:
            position: [x, y, z] center position
            orientation: quaternion [x, y, z, w]
            half_extents: [x, y, z] half-lengths in each dimension
        """
        # Create corner points in local coordinates
        corners = np.array([
            [ 1,  1,  1], [ 1,  1, -1], [ 1, -1,  1], [ 1, -1, -1],
            [-1,  1,  1], [-1,  1, -1], [-1, -1,  1], [-1, -1, -1]
        ]) * np.array(half_extents)  # Now using different scales for each dimension
        
        # Convert quaternion to rotation matrix
        rot_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        
        # Rotate and translate corners
        transformed_corners = corners @ rot_matrix.T + position
        return transformed_corners

    @staticmethod
    def calculate_bounding_box_volume(points: np.ndarray) -> float:
        """Calculate volume of axis-aligned bounding box containing all points."""
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        dimensions = max_coords - min_coords
        return np.prod(dimensions)

    @staticmethod
    def calculate_total_box_volume(boxes: Dict[int, tuple]) -> float:
        """Calculate total volume of all boxes with different dimensions."""
        total_volume = 0
        for box_info in boxes.values():
            dimensions = box_info
            volume = dimensions[0] * dimensions[1] * dimensions[2]
            total_volume += volume
        return total_volume

class BinStackEnvironment:
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
        
        self.set_joints(self.robot_home_joint_config)

        self.robot_end_effector_link_index = 8 # refer to the last link of the UR5 robot (virtual ee)
        
        self.boxes = {}
        self.currently_grasped_box = None
        self.grasp_constraint = None

        # self.collision_threshold = 0.0001
        self.efficiency_weight = 0.6
        self.collision_weight = 0.2
        self.stack_weight = 0.2
        self.stack_efficiency_scale = 100.0
        # self.collision_scaling_factor = 10.0
        self.collision_history = []
        self.efficiency_calculator = StackingEfficiencyCalculator()
        
    
    def set_joints(self, target_joint_state, steps=1e2):
        assert len(self._robot_joint_indices) == len(target_joint_state)
        for joint, value in zip(self._robot_joint_indices, target_joint_state):
            p.resetJointState(self.robot_body_id, joint, value)
        if steps > 0:
            self.step_simulation(steps)
    
    def step_simulation(self, num_steps):
        """
        Step the simulation forward by num_steps timesteps.
        Each timestep is 1/240 seconds (PyBullet's default timestep).
        """
        for i in range(int(num_steps)*10):
            p.stepSimulation()

    def load_box(self):
        width = 0.100  # 5cm fixed width
        length_ = [0.1, 0.2]
        height_ = [0.1, 0.2]
        mass = 1.        # 100g mass
        length = self._to_3_decimals(np.random.uniform(length_[0], length_[1])) # truncate to 3 decimals
        height = self._to_3_decimals(np.random.uniform(height_[0], length_[1]))
        box_dimensions = [length,width,height]
        
        # Create collision and visual shapes with the random dimensions
        box_collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[dim/2 for dim in box_dimensions]
        )
        
        box_visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[dim/2 for dim in box_dimensions],
            rgbaColor=[1, 0, 0, 1]  # Red color
        )
        # Create the box body
        box_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=box_collision_shape,
            baseVisualShapeIndex=box_visual_shape,
            basePosition=[0.5, 0, 0.1]  # Default position
        )
        p.changeDynamics(box_id, -1,
            lateralFriction=1.0,
            spinningFriction=1.0,
            rollingFriction=0.0001)
        
        self.boxes[box_id] = box_dimensions
        return box_id, box_dimensions
    
    def robot_go_home(self, speed=3.0):
        self.move_joints(self.robot_home_joint_config, speed=speed)
    
    def move_joints(self, target_joint_state, acceleration=10, speed=3.0):
        """
            Move robot arm to specified joint configuration by appropriate motor control
        """
        assert len(self._robot_joint_indices) == len(target_joint_state)
        dt = 1./240
        q_current = np.array([x[0] for x in p.getJointStates(self.robot_body_id, self._robot_joint_indices)])
        q_target = np.array(target_joint_state)
        q_traj = get_movej_trajectory(q_current, q_target, 
            acceleration=acceleration, speed=speed)
        qdot_traj = np.gradient(q_traj, dt, axis=0)
        p_gain = 1 * np.ones(len(self._robot_joint_indices))
        d_gain = 1 * np.ones(len(self._robot_joint_indices))

        for i in range(len(q_traj)):
            p.setJointMotorControlArray(
                bodyUniqueId=self.robot_body_id, 
                jointIndices=self._robot_joint_indices,
                controlMode=p.POSITION_CONTROL, 
                targetPositions=q_traj[i],
                targetVelocities=qdot_traj[i],
                positionGains=p_gain,
                velocityGains=d_gain
            )
            self.step_simulation(1)
            
    def move_tool(self, position, orientation, acceleration=10, speed=3.0):
        """
            Move robot tool (end-effector) to a specified pose
            @param position: Target position of the end-effector link
            @param orientation: Target orientation of the end-effector link
        """
        joint_state = p.calculateInverseKinematics(bodyUniqueId = self.robot_body_id, 
                                                   endEffectorLinkIndex = self.robot_end_effector_link_index,
                                                   targetPosition = position, 
                                                   targetOrientation = orientation,
                                                   maxNumIterations = 80)
        self.move_joints(joint_state, acceleration=acceleration, speed=speed)
        
    
    def grasp_box(self, box_id: int):
        """
        Create a fixed constraint between the gripper and specified box that maintains
        the relative position between them at the time of grasping.
        """
        # Get current positions and orientations of both ee and box
        ee_state = p.getLinkState(self.robot_body_id, self.robot_end_effector_link_index)
        box_pos, box_orn = p.getBasePositionAndOrientation(box_id)
        
        # Calculate the relative transform between gripper and box
        inv_ee_pos, inv_ee_orn = p.invertTransform(ee_state[0], ee_state[1])
        relative_pos, relative_orn = p.multiplyTransforms(
            inv_ee_pos, inv_ee_orn,
            box_pos, box_orn
        )
        
        # Create constraint with the relative transform
        self.grasp_constraint = p.createConstraint(
            parentBodyUniqueId=self.robot_body_id,
            parentLinkIndex=self.robot_end_effector_link_index,
            childBodyUniqueId=box_id,
            childLinkIndex=-1,  # -1 for the base
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=relative_pos,
            childFramePosition=[0, 0, 0],
            parentFrameOrientation=relative_orn,
            childFrameOrientation=[0, 0, 0, 1]
        )

        
        # Set high maximum force to ensure rigid connection
        p.changeConstraint(self.grasp_constraint, maxForce=1000)
        self.currently_grasped_box = box_id
        
        # Optional: Disable collisions between gripper and box while grasped
    #     for joint_idx in range(p.getNumJoints(self._gripper_body_id)):
    #         p.setCollisionFilterPair(
    #             self._gripper_body_id, box_id,
    #             joint_idx, -1, 0
    #         )

    
    def release_box(self):
        """Release the currently grasped box by removing the constraint."""
        if self.grasp_constraint is not None:
            p.removeConstraint(self.grasp_constraint)
            self.grasp_constraint = None
            self.currently_grasped_box = None

    def execute_grasp(self, box_id, box_dimension):
        """
        Execute grasp sequence without gripper movements
        """
        grasp_position, _ = p.getBasePositionAndOrientation(box_id)
        grasp_position = np.array(grasp_position) + np.array([0, 0, box_dimension[2]/2 + 0.01])
        end_effector_orientation = p.getQuaternionFromEuler([np.pi, 0, 0])  # Adjust as needed
        
        pre_grasp_position = grasp_position + np.array([0, 0, 0.2])
        post_grasp_position = grasp_position + np.array([0, 0, 0.2])
        
        # Move to pre-grasp position
        self.move_tool(pre_grasp_position, end_effector_orientation)
        
        # Move to grasp position
        self.move_tool(grasp_position, end_effector_orientation)
        
        # Create fixed constraint
        self.grasp_box(box_id)
        
        # Move to post-grasp position
        self.move_tool(post_grasp_position, end_effector_orientation)
        
        # Move to home position
        self.robot_go_home()
        
        return True if self.grasp_constraint is not None else False
        
        
    def convert_pose_to_top_down():
        pass
    
    def move_joints(self, target_joint_state, acceleration=10, speed=3.0):
        """
            Move robot arm to specified joint configuration by appropriate motor control
        """
        assert len(self._robot_joint_indices) == len(target_joint_state)
        dt = 1./240
        q_current = np.array([x[0] for x in p.getJointStates(self.robot_body_id, self._robot_joint_indices)])
        q_target = np.array(target_joint_state)
        q_traj = get_movej_trajectory(q_current, q_target, 
            acceleration=acceleration, speed=speed)
        qdot_traj = np.gradient(q_traj, dt, axis=0)
        p_gain = 1 * np.ones(len(self._robot_joint_indices))
        d_gain = 1 * np.ones(len(self._robot_joint_indices))

        for i in range(len(q_traj)):
            p.setJointMotorControlArray(
                bodyUniqueId=self.robot_body_id, 
                jointIndices=self._robot_joint_indices,
                controlMode=p.POSITION_CONTROL, 
                targetPositions=q_traj[i],
                targetVelocities=qdot_traj[i],
                positionGains=p_gain,
                velocityGains=d_gain
            )
            # print(self.detect_collisions()["num_contacts"])
            collision_info = self.detect_collisions()
            self.step_simulation(1)

    def move_tool(self, position, orientation, acceleration=10, speed=3.0):
        """
            Move robot tool (end-effector) to a specified pose
            @param position: Target position of the end-effector link
            @param orientation: Target orientation of the end-effector link
        """
        joint_state = p.calculateInverseKinematics(bodyUniqueId = self.robot_body_id, 
                                                   endEffectorLinkIndex = self.robot_end_effector_link_index,
                                                   targetPosition = position, 
                                                   targetOrientation = orientation,
                                                   maxNumIterations = 80)
        
        self.move_joints(joint_state, acceleration=acceleration, speed=speed)
    
    def calculate_stacking_efficiency(self) -> float:
        """Calculate the ratio between occupied and bounding box volume."""
        if len(self.boxes) < 2:
            return 0.0
            
        # Collect all corner points from all boxes
        all_corners = []
        for box_id in self.boxes:
            pos, orn = p.getBasePositionAndOrientation(box_id)
            half_extents = [dim/2 for dim in self.boxes[box_id]]
            corners = self.efficiency_calculator.get_box_corners(
                list(pos), list(orn), half_extents)
            all_corners.extend(corners)
        
        all_corners = np.array(all_corners)
        
        # Calculate volumes
        bounding_volume = self.efficiency_calculator.calculate_bounding_box_volume(all_corners)
        occupied_volume = self.efficiency_calculator.calculate_total_box_volume(self.boxes)
        
        # Calculate efficiency ratio
        efficiency = occupied_volume / bounding_volume if bounding_volume > 0 else 0
        return efficiency
        
    def detect_collisions(self) -> Dict[str, float]:
        """
        Detect collisions between the grasped box and other boxes.
        Returns collision metrics including maximum impact force.
        """
        if self.currently_grasped_box is None:
            return {'collision_force': 0.0, 'max_impact': 0.0, 'num_contacts': 0}
        
        # total_collision_force = 0.0
        # max_impact = 0.0
        # num_contacts = 0
        
        for box_id in self.boxes:
            if box_id != self.currently_grasped_box:
                points = p.getContactPoints(self.currently_grasped_box, box_id)
                if points:
                    # Record the contact event with timestamp
                    collision_event = {
                        'timestamp': time.time(),
                        'box_ids': (self.currently_grasped_box, box_id)
                    }
                    self.collision_history.append(collision_event)
                    break  # Exit after finding first contact
        

        # for box_id in self.boxes:
        #     if box_id != self.currently_grasped_box:
        #         points = p.getContactPoints(self.currently_grasped_box, box_id)
        #         if points:
        #             for point in points:
        #                 normal_force = point[9]
        #                 lateral_friction_force = point[10]
        #                 total_force = np.sqrt(normal_force**2 + lateral_friction_force**2)
                        
        #                 if total_force > self.collision_threshold:
        #                     total_collision_force += total_force
        #                     max_impact = max(max_impact, total_force)
        #                     num_contacts += 1
        
        # return {
        #     'collision_force': total_collision_force,
        #     'max_impact': max_impact,
        #     'num_contacts': num_contacts
        # }
        
    def is_Stacked(self):
        threshold = 0.0001
        if len(self.boxes) < 2:
            return True
        box_items = list(self.boxes.items())
        current_box_id, curr_dim = box_items[-1]
        curr_pos, _ = p.getBasePositionAndOrientation(current_box_id)
        for prev_box_id, _ in box_items[:-1]:
            prev_pos, prev_dim = p.getBasePositionAndOrientation(prev_box_id)
            if((curr_pos[2] - curr_dim[2]/2. >= prev_pos[2] - prev_dim[2]/2. + threshold)):
                continue
            else:
                return False
        return True
    
    def get_total_reward(self) -> Dict[str, float]:
        """
        Calculate comprehensive reward based on:
        1. Stacking efficiency (higher is better)
        2. Collision penalties (lower is better)
        """
        # Get stacking efficiency
        efficiency = self.calculate_stacking_efficiency() * self.stack_efficiency_scale
        efficiency = self._to_3_decimals(efficiency)
        # # Get collision information
        # collision_info = self.detect_collisions()
        
        # # Calculate collision penalty (normalized between 0 and 1)
        # collision_penalty = min(1.0, collision_info['max_impact'] / self.collision_scaling_factor)
        
        # Calculate weighted reward components
        efficiency_reward = efficiency * self.efficiency_weight
        
        collision_penalty = -100.0 if self.collision_history else 0
        collision_reward = collision_penalty*self.collision_weight
        self.collision_history = []
        
        stack_penalty = -50.0 if not self.is_Stacked() else 0
        stack_reward = stack_penalty* self.stack_weight
        # collision_reward = (1.0 - collision_penalty) * self.collision_weight
        
        # Calculate total reward
        total_reward = self._to_3_decimals(efficiency_reward + collision_reward + stack_reward)
        
        return {
            'total_reward': total_reward,
            'efficiency_ratio': efficiency,
            'collision_penalty': collision_penalty,
            'stack_penalty': stack_penalty
        }
        
    def close(self):
        """Clean up resources."""
        p.disconnect()
        
    def _to_3_decimals(self,num):
        return math.trunc(num * 1000) / 1000