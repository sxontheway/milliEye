import numpy as np
import pickle
import yaml
import os
import cv2
from .tracking import *


def load_data(file_stamp: str, file_point: str) -> (list, list):
    """Load camera and radar data.

    Load timestamp information from `file_stamp`, and pointcloud data from `file_point`

    Parameters
    ----------
    file_stamp : str
        The location of camera timestamp file
    file_point : str
        The location of radar pointcloud file

    Returns
    -------
    list, list
        A list of camera timestep information.
        A list of point cloud dicts corresponding to each frame.
    """
    video_stamps = []
    with open(file_stamp, "r") as f:
        for line in f:
            video_stamps.append(list(line.strip("\n").split()))
    with open(file_point, "rb") as f:
        point_data = pickle.load(f)
    return video_stamps, point_data



def match(video_stamps, point_data, neighbor_num: int) -> list:
    """To find the nearest `neighbor_num` radar frames to each camera frames.

    Parameters
    ----------
    neighbor_num : int
        The number of the nearest frames to find.

    Returns
    -------
    list
        A list of numpy array containing each camera frame's `neighbor_num` nearest radar frame indices.
    """
    all_closest = []
    for index, frame in enumerate(video_stamps):
        closest = np.argsort(
            list(map(lambda x: abs(x["Time"] - float(frame[0])), point_data))
        )
        all_closest.append(closest[:neighbor_num])
        if index > 1 and (closest[:neighbor_num] == all_closest[-2]).all():
            all_closest.pop()

    return all_closest



def load_calib(filename: str) -> np.array:
    """
    return a list [fx, cx, fy, cy, k1, k2, t1, t2, k3]
    where fx, fy, cx, cy are elements in projection_matrix
    and k1, k2, k3, t1, t2 are camera distortion_coefficients
    """
    trans_x, trans_y, trans_z = [-0.07, -0.05, 0]  
    with open(filename, "r") as f:
        y = yaml.load(f)
        camera_matrix = np.resize(y["camera_matrix"]["data"], (3, 3))
        distortion = np.array(y["distortion_coefficients"]["data"])
        calib_param = np.array([camera_matrix[0, 0], camera_matrix[0, 2], \
                                camera_matrix[1, 1], camera_matrix[1, 2], \
                                *distortion, *[trans_x, trans_y, trans_z]]) 
    return calib_param



def projection_xyr_to_uv(points: list, calib_param: np.array) -> (np.array, np.array):
    """
    from camera coordinate (x, y, r) to pixel coordinate (u, v)
    input: 
        points: [np.array([x1, x2, ...]), np.array([y1, y2, ...]), np.array([r1, r2, ...])]
        calib_param: a 1D np.array()
    """
    fx, cx, fy, cy, k1, k2, t1, t2, k3, trans_x, trans_y, trans_z = calib_param
    
    # translation between radar and camera 
    x, y = (points[0] + trans_x)/(points[2] + trans_z), (points[1] + trans_y)/(points[2] + trans_z)  
    
    # dirtortion with camera lens
    x_2, y_2 = x**2, y**2
    r_2 = x_2 + y_2 
    r_4 = r_2**2
    r_6 = r_2**3
    tmp = 1 + k1*r_2 + k2*r_4 + k3*r_6
    x_undistort, y_undistort = x*tmp + 2*t1*x*y + t2*(r_2 + 2*x_2), y*tmp + 2*t2*x*y + t1*(r_2 + 2*y_2)
    u, v = x_undistort * fx + cx, y_undistort * fy + cy
    return u, v



def from_3d_to_2d(points: np.array, calib_param: np.array) -> np.array:
    """
    3d to 2d projection
    input: 3d radar coordinate -> np.array with shape (4, n)
    output: image coordinate -> np.array: (n, 2) and (n, 4)
    camera coordinate: u, v, r = radar coordinate: x, -z, y
    """
    # radar coordinate -> image coordinate
    x, y, z, velocity = points[0], -points[2], points[1], points[3]
    u, v = projection_xyr_to_uv([x, y, z], calib_param)

    # outputs xyzV (in camera coordinate)
    trans_x, trans_y, trans_z = calib_param[-3:]    # actually, trans_z=0, so actually no translation applied
    uv = np.array([*zip(u, v)]).astype(np.int64)
    xyzV = np.array([*zip(x, y, z+trans_z, velocity)])
    return uv, xyzV



def draw_3d_boxes(center: np.array, size: np.array, frame: np.array,  calib_param: np.array):
    """
    Given center (shape(3, )),  size (shape(3, )) and calib_param, draw 3d boxes on 'frame'
    """
    multi = np.array([[1,1,1],[1,-1,1],[-1,-1,1],[-1,1,1],[1,1,-1],[1,-1,-1],[-1,-1,-1],[-1,1,-1]])
    corners = np.tile(center, (8,1)) + np.tile(size, (8,1))* multi/2    # shape (8, 3)
    connections = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
    for connection in connections:
        start_u, start_v = projection_xyr_to_uv(corners[connection[0], :], calib_param)
        end_u, end_v = projection_xyr_to_uv(corners[connection[1], :], calib_param)
        if max(start_u, start_v, end_u, end_v) > 1000 or min(start_u, start_v, end_u, end_v) < -1000:
            continue
        cv2.line(frame, (int(start_u), int(start_v)), (int(end_u), int(end_v)), (255,255,255), 2)



def draw_2d_boxes(center: np.array, size: np.array, z_multi: float, frame: np.array,  calib_param: np.array):
    """
    Args:
        center: shape(3, )
        size: shape(3, )
        z_multi: range from -1 to 1. -1 means the front side of the 3d bbox. 1 means the back side of the 3d bbox.
        frame: the image that will be drawn on
        calib_param: projection matrix from radar (3d) to camera (2d)
    Function: draw a 2d boxes on 'frame'
    """
    multi = np.array([[1,1,z_multi], [1,-1,z_multi], [-1,-1,z_multi], [-1,1,z_multi]])
    corners = np.tile(center,(4,1)) + np.tile(size,(4,1)) * multi/2    # shape (4, 3)
    connections = [[0,1], [1,2], [2,3], [3,0]]
    for connection in connections:
        start_u, start_v = projection_xyr_to_uv(corners[connection[0], :], calib_param)
        end_u, end_v = projection_xyr_to_uv(corners[connection[1], :], calib_param)
        if max(start_u, start_v, end_u, end_v) > 1000 or min(start_u, start_v, end_u, end_v) < -1000:
            continue
        cv2.line(frame, (int(start_u), int(start_v)), (int(end_u), int(end_v)), (255,255,255), 2)



def draw_solid_2d_boxes(center: np.array, size: np.array, z_multi: float, frame: np.array,  calib_param: np.array):
    """
    Args:
        center: shape(3, )
        size: shape(3, )
        z_multi: range from -1 to 1. -1 means the front side of the 3d bbox. 1 means the back side of the 3d bbox.
        frame: the image that will be drawn on
        calib_param: projection matrix from radar (3d) to camera (2d)
    Function: draw a solid 2d boxes on 'frame'
    """
    multi = np.array([[1,1,z_multi],[-1,-1,z_multi]])
    corners = np.tile(center, (2,1)) + np.tile(size, (2,1))* multi/2    # shape (2, 3)
    #connections = [[0,1],[1,2],[2,3],[3,0]]

    for corner in corners:
        start_u, start_v = projection_xyr_to_uv(corners[0, :], calib_param)
        end_u, end_v = projection_xyr_to_uv(corners[1, :], calib_param)
        if max(start_u, start_v, end_u, end_v) > 1000 or min(start_u, start_v, end_u, end_v) < -1000:
            continue
        cv2.rectangle(frame, (int(start_u), int(start_v)), (int(end_u), int(end_v)) , (255,255,255), -1)




if __name__ == "__main__":
    file_base = "./data/20200119-142610"
    stamp_data, point_data = load_data(
        os.path.join(file_base, "timestamps.txt"),
        os.path.join(file_base, "pointcloud.pkl"),
    )
    print(match(stamp_data, point_data, 5))

