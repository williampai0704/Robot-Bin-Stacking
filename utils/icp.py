import numpy as np
import trimesh

def export_ply(pts, file_path, color):
    ptcloud = trimesh.points.PointCloud(vertices=pts, colors=color)   
    ptcloud.export(file_path)


def cam_view2pose(cam_view_matrix):
    """
    In:
        cam_view_matrix: 4x4 matrix, stored as a list of 16 floats
    Out:
        cam_pose_matrix: 4x4 matrix, stored as numpy array [4x4]
    Purpose:
        Convert camera view matrix to pose matrix
    """
    cam_pose_matrix = np.linalg.inv(np.array(cam_view_matrix).reshape(4, 4).T)
    
    # convert camera coordinate from (openGL y up x right z back) 
    # to align with image coordinate (openCV y down x right z forward)
    cam_pose_matrix[:, 1:3] = -cam_pose_matrix[:, 1:3]
    return cam_pose_matrix

def transform_point3s(t, ps):
    """Transfrom 3D points from one space to another.

    Args:
        t (numpy.array [4, 4]): SE3 transform.
        ps (numpy.array [n, 3]): Array of n 3D points (x, y, z).

    Raises:
        ValueError: If t is not a valid transform.
        ValueError: If ps does not have correct shape.

    Returns:
        numpy.array [n, 3]: Transformed 3D points.
    """


    # convert to homogeneous
    ps_homogeneous = np.hstack([ps, np.ones((len(ps), 1), dtype=np.float32)])
    ps_transformed = np.dot(t, ps_homogeneous.T).T

    return ps_transformed[:, :3]

def depth_to_point_cloud(intrinsics, depth_image):
    """Back project a depth image to a point cloud.
        Note: point clouds are unordered, so any permutation of points in the list is acceptable.
        Note: Only output those points whose depth > 0.

    Args:
        intrinsics (numpy.array [3, 3]): given as [[fu, 0, u0], [0, fv, v0], [0, 0, 1]]
        depth_image (numpy.array [h, w]): each entry is a z depth value.

    Returns:
        numpy.array [n, 3]: each row represents a different valid 3D point.
    """
    u0 = intrinsics[0, 2]
    v0 = intrinsics[1, 2]
    fu = intrinsics[0, 0]
    fv = intrinsics[1, 1]

    point_count = 0
    for v in range(depth_image.shape[0]):
        for u in range(depth_image.shape[1]):
            if depth_image[v, u] > 0:
                point_count += 1

    # back project for each u, v
    point_cloud = np.zeros((point_count, 3))
    point_count = 0
    for v in range(depth_image.shape[0]):
        for u in range(depth_image.shape[1]):
            if depth_image[v, u] > 0:
                point_cloud[point_count] = np.array([
                    (u - u0) * depth_image[v, u] / fu,
                    (v - v0) * depth_image[v, u] / fv,
                    depth_image[v, u]])
                point_count += 1

    return point_cloud

def obj_mesh2pts(obj_name, point_num, transform=None):
    """
    In:
        obj_id: int, indicating an object in LIST_OBJ_FOLDERNAME.
        point_num: int, number of points to sample.
        transform: Numpy array [4, 4] of float64.
    Out:
        pts: Numpy array [n, 3], sampled point cloud.
    Purpose:
         Sample a point cloud from the mesh of the specific object. If transform is not None, apply it.
    """
    mesh_path = './assets/ycb_objects/' + obj_name + '/textured_reoriented.obj'  # objects ID start from 1
    mesh = trimesh.load(mesh_path)
    pts, _ = trimesh.sample.sample_surface(mesh, count=point_num)
    return pts





def gen_obj_depth(obj_id, depth, mask):
    """
    In:
        obj_id: int, indicating an object in LIST_OBJ_FOLDERNAME.
        depth: Numpy array [height, width], where each value is z depth in meters.
        mask: Numpy array [height, width], where each value is an obj_id.
    Out:
        obj_depth: Numpy array [height, width] of float64, where depth value of all the pixels that don't belong to the object is 0.
    Purpose:
        Generate depth image for a specific object.
        Generate depth for all objects when obj_id == -1.
    Hint: 
        use 
        1. np.where() to get the right mask (1 for depth value to keep, 0 for others)
        2. apply mask on obj_depth 
    """
    
    if obj_id == -1:
        obj_mask = np.where(mask == 0, 0, 1).astype(np.float)
    else:
        obj_mask = np.where(mask == obj_id, 1, 0).astype(np.float)
    obj_depth = depth * obj_mask
    return obj_depth


def obj_depth2pts(obj_id, depth, mask, intrinsic_matrix, view_matrix):
    """
    In:
        obj_id: int, indicating an object in LIST_OBJ_FOLDERNAME.
        depth: Numpy array [height, width], where each value is z depth in meters.
        mask: Numpy array [height, width], where each value is an obj_id.
        camera: Camera instance.
        view_matrix: Numpy array [16,] of float64, representing a 4x4 matrix.
    Out:
        world_pts: Numpy array [n, 3], 3D points in the world frame of reference.
    Purpose:
        Generate point cloud projected from depth of the specific object(s) in the world frame of reference.
    Hint:
        Step 1: use gen_obj_depth() to generate the depth image for the specific object(s).
        Step 2: use depth_to_point_cloud() to project a depth image to a point cloud.
        Note that this method returns coordinates in the camera frame of reference,
        so don't forget to convert to the world frame of reference using the camera pose corresponding to this scene.
        The view matrices are provided in the /dataset/val/view_matrix or /dataset/test/view_matrix folder
        and the method cam_view2pose() in camera.py is provided to convert camera view matrix to pose matrix.
        Use transform_point3s to apply transformation.
        All the methods mentioned are imported.
    """
    obj_depth = gen_obj_depth(obj_id, depth, mask)
    cam_pts = depth_to_point_cloud(intrinsic_matrix, obj_depth)
    world_pts = transform_point3s(cam_view2pose(view_matrix), cam_pts)
    return world_pts


def align_pts(pts_a, pts_b, max_iterations=20, threshold=1e-05):
    """
    In:
        pts_a: Numpy array [n, 3].
        pts_b: Numpy array [n, 3].
        max_iterations: int, tunable parameter of trimesh.registration.icp().
        threshold: float，tunable parameter of trimesh.registration.icp().
    Out:
        matrix: Numpy array [4, 4], the transformation matrix sending pts_a to pts_b.
    Purpose:
        Apply the iterative closest point algorithm to estimate a transformation that aligns one point cloud with another.
    Hint:
        Use trimesh.registration.procrustes() to compute initial transformation and trimesh.registration.icp() to align.
    """
    
    # compute init_matrix
    init_matrix, transformed, cost = trimesh.registration.procrustes(pts_a,
                                                                     pts_b,
                                                                     reflection=False,
                                                                     translation=True,
                                                                     scale=False,
                                                                     return_cost=True)
    # use trimesh.registration.icp with init_matrix
    matrix, transformed, cost = trimesh.registration.icp(pts_a,
                                                         pts_b,
                                                         initial=init_matrix,
                                                         max_iterations=max_iterations,
                                                         scale=False,
                                                         threshold=threshold, )
    return matrix