import pybullet as p
import pybullet_data
import numpy as np
from typing import Tuple, List, Dict, Set

class PerfectGraspSimulator:
    def __init__(self):
        # Initialize simulation
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Load UR5 with Robotiq gripper
        self.robot_id = p.loadURDF("ur5/ur5.urdf", [0, 0, 0], useFixedBase=True)
        self.gripper_id = p.loadURDF("robotiq_2f_85_gripper/robotiq_2f_85.urdf", 
                                    [0.5, 0, 0.5])
        
        # Track all boxes in the scene
        self.boxes: Dict[int, dict] = {}
        self.currently_grasped_box = None
        self.grasp_constraint = None
        
        # Collision detection parameters
        self.collision_threshold = 0.0001  # Minimum contact force to register
        
    def create_box(self, position: List[float], size: float = 0.05, 
                  mass: float = 0.1, color: List[float] = [1, 0, 0, 1]) -> int:
        """Create a new box and return its ID."""
        collision_shape = p.createCollisionShape(p.GEOM_BOX,
                                               halfExtents=[size/2]*3)
        visual_shape = p.createVisualShape(p.GEOM_BOX,
                                         halfExtents=[size/2]*3,
                                         rgbaColor=color)
        box_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position
        )
        
        # Store box information
        self.boxes[box_id] = {
            'size': size,
            'mass': mass,
            'color': color,
            'position': position
        }
        
        return box_id
        
    def get_gripper_center_pose(self) -> Tuple[List[float], List[float]]:
        """Get the center position and orientation of the gripper."""
        gripper_state = p.getLinkState(self.gripper_id, 0)
        return gripper_state[0], gripper_state[1]
        
    def create_perfect_grasp(self, box_id: int):
        """Create a fixed constraint between the gripper and specified box."""
        if self.grasp_constraint is None:
            gripper_pos, gripper_orn = self.get_gripper_center_pose()
            
            self.grasp_constraint = p.createConstraint(
                parentBodyUniqueId=self.gripper_id,
                parentLinkIndex=0,
                childBodyUniqueId=box_id,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=[0, 0, 0],
                parentFrameOrientation=gripper_orn,
                childFrameOrientation=p.getQuaternionFromEuler([0, 0, 0])
            )
            
            p.changeConstraint(self.grasp_constraint, maxForce=100)
            self.currently_grasped_box = box_id
    
    def release_grasp(self):
        """Remove the constraint to release the box."""
        if self.grasp_constraint is not None:
            p.removeConstraint(self.grasp_constraint)
            self.grasp_constraint = None
            self.currently_grasped_box = None
    
    def detect_collisions(self) -> Dict[str, float]:
        """
        Detect collisions between the grasped box and other boxes.
        Returns a dictionary with collision metrics for reward calculation.
        """
        if self.currently_grasped_box is None:
            return {'collision_force': 0.0, 'num_contacts': 0}
        
        total_collision_force = 0.0
        num_contacts = 0
        contact_points = []
        
        # Get all contact points for the grasped box
        for box_id in self.boxes:
            if box_id != self.currently_grasped_box:
                points = p.getContactPoints(self.currently_grasped_box, box_id)
                if points:
                    for point in points:
                        contact_force = point[9]  # Normal force
                        if contact_force > self.collision_threshold:
                            total_collision_force += contact_force
                            num_contacts += 1
                            contact_points.append({
                                'position': point[5],
                                'normal': point[7],
                                'force': contact_force
                            })
        
        return {
            'collision_force': total_collision_force,
            'num_contacts': num_contacts,
            'contact_points': contact_points
        }
    
    def get_stacking_reward(self) -> float:
        """
        Calculate a reward based on stacking success and collision penalties.
        You can modify this function to implement your specific reward logic.
        """
        collision_info = self.detect_collisions()
        
        # Example reward function:
        # - Positive reward for successful stacking (contact with small force)
        # - Negative reward for excessive collision forces
        base_reward = 0.0
        if collision_info['num_contacts'] > 0:
            if collision_info['collision_force'] < 1.0:  # Gentle contact
                base_reward = 1.0
            else:  # Penalty for rough contact
                base_reward = -collision_info['collision_force']
        
        return base_reward
    
    def step_simulation(self):
        """Step the simulation forward and return current reward."""
        p.stepSimulation()
        return self.get_stacking_reward()
    
    def cleanup(self):
        """Cleanup simulation resources."""
        p.disconnect(self.client)

# Example usage
if __name__ == "__main__":
    simulator = PerfectGraspSimulator()
    
    try:
        # Create some boxes for stacking
        base_box = simulator.create_box([0.5, 0, 0.1], color=[0, 1, 0, 1])
        grasp_box = simulator.create_box([0.5, 0.2, 0.1], color=[1, 0, 0, 1])
        
        # Simulate for a few seconds
        for i in range(1000):
            # Create perfect grasp after a few steps
            if i == 100:
                simulator.create_perfect_grasp(grasp_box)
            
            # Release grasp after more steps
            if i == 500:
                simulator.release_grasp()
            
            # Step simulation and get reward
            reward = simulator.step_simulation()
            
            # Print collision information every 100 steps
            if i % 100 == 0:
                collision_info = simulator.detect_collisions()
                print(f"Step {i}:")
                print(f"Collision force: {collision_info['collision_force']:.3f}")
                print(f"Number of contacts: {collision_info['num_contacts']}")
                print(f"Reward: {reward:.3f}")
            
            p.time.sleep(1./240.)
            
    finally:
        simulator.cleanup()
