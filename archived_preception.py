import os
import numpy as np
from pyquaternion import Quaternion
from utils import icp


def pose_est_state(obj_id, env):
    position, orientation = env.get_object_pose(obj_id)
    return position, orientation


def pose_est_segicp(obj_id, obj_name, depth_obs, mask, intrinsic_matrix, view_matrix):
    # TODO: Use the functions you implemented in previous hw to complete this part. 
    # Hints: functions provided in icp.py  
    # Input: 
    #   obj_id, int corresponding the index on mask, 
    #           used to find observed object pointcloud in depth. 
    #   obj_name, string corresponding to object name 
    #            used to load object model pointcloud. 
    # Goal find object pose [4x4] matrix using ICP 
    pts_depth = icp.obj_depth2pts(obj_id, depth_obs, mask, intrinsic_matrix, view_matrix)
    pts_mesh = icp.obj_mesh2pts(obj_name, point_num= len(pts_depth))
    obj_pose = icp.align_pts(pts_mesh,pts_depth)

    # Convert object pose to quaternion. 
    pos =  obj_pose[0:3,3].T
    orientation =  Quaternion(matrix = obj_pose)
    a = orientation.elements
    pybullet_quaternion = [a[1], a[2], a[3], a[0]] # [w,x,y,z] to [x,y,z,w]
    
    # Due to partial observation (the depth camera is pointed above the objects so only the top of the objects are seen), thus the estimated pose after running ICP is often shifted in z, so we offset with a constant. This is a hack to fix this.
    pos[2] = pos[2]-0.02
    
    if True: # debug visualization 
        if not os.path.exists('./debug'):
            os.mkdir('./debug')
        file_path_obs = './debug/' + obj_name + '_obs.ply'
        file_path_pred = './debug/' + obj_name + '_pred.ply'
        pts_pred = icp.transform_point3s(obj_pose, pts_mesh)

        icp.export_ply(pts_depth,file_path_obs,[0, 250, 0] )
        icp.export_ply(pts_pred,file_path_pred,[250, 0, 0] )
        

    return pos, pybullet_quaternion

