# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================
import sys
from os import path as osp
current_dir = osp.abspath(osp.dirname(__file__))
sys.path.append(current_dir+"/../..")
carla_simulator_path = '/home/SENSETIME/maqiurui/reinforce/carla/carla_0.9.4/PythonAPI/carla-0.9.4-py3.5-linux-x86_64.egg'
try:
    sys.path.append(carla_simulator_path)
    sys.path.append(current_dir+'/../../../PythonAPI/carla-0.9.4-py3.5-linux-x86_64.egg')
except IndexError:
    pass

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================
import numpy as np
import numpy
from numpy.linalg import inv
import math

# util functions
def _pos(_object):
    type_obj = str(type(_object))
    if 'Actor' in type_obj or 'Vehicle' in type_obj or 'TrafficLight' in type_obj:
        return [_object.get_location().x, _object.get_location().y]
    elif 'BoundingBox' in type_obj or 'Transform' in type_obj:
        return [_object.location.x, _object.location.y]
    elif 'Vector3D' in type_obj or 'Location' in type_obj:
        return [_object.x, _object.y]
    elif 'Waypoint' in type_obj:
        return [_object.transform.location.x, _object.transform.location.y]
def _pos3d(_object):
    type_obj = str(type(_object))
    if 'Actor' in type_obj or 'Vehicle' in type_obj or 'TrafficLight' in type_obj:
        return [_object.get_location().x, _object.get_location().y, _object.get_location().z]
    elif 'BoundingBox' in type_obj or 'Transform' in type_obj:
        return [_object.location.x, _object.location.y, _object.location.z]
    elif 'Vector3D' in type_obj or 'Location' in type_obj:
        return [_object.x, _object.y, _object.z]
    elif 'Waypoint' in type_obj:
        return [_object.transform.location.x, _object.transform.location.y, _object.transform.location.z]
def _get_v(_object):
    return (_object.get_velocity().x, _object.get_velocity().y)
def _get_a(_object):
    return (_object.get_acceleration().x, _object.get_acceleration().y)
def _get_v_3d(_object):
    return (_object.get_velocity().x, _object.get_velocity().y, _object.get_velocity().z)
def _get_a_3d(_object):
    return (_object.get_acceleration().x, _object.get_acceleration().y, _object.get_acceleration().z)
def _get_r(_object):
    return (_object.get_transform().rotation.yaw, _object.get_transform().rotation.pitch, _object.get_transform().rotation.roll)

def _dis(a, b):
    return ((b[1]-a[1])**2 + (b[0]-a[0])**2)**0.5
def _dis3d(a, b):
    return ((b[1]-a[1])**2 + (b[0]-a[0])**2 + (b[2]-a[2])**2)**0.5

def _location(x):
    from carla import Location
    return Location(x[0], x[1], 0)
def _location3d(x):
    from carla import Location
    return Location(x[0], x[1], x[2])
def _rotation3d(x):
    from carla import Rotation
    return Rotation(pitch=x[0], yaw=x[1], roll=x[2])

def _cos(vector_a, vector_b):
    ab = vector_a[0]*vector_b[0]+vector_a[1]*vector_b[1]
    a_b = (vector_a[0]**2+vector_a[1]**2)**0.5 * (vector_b[0]**2+vector_b[1]**2)**0.5 + 1e-8
    return ab/a_b

def _cos3d(vector_a, vector_b):
    ab = vector_a[0]*vector_b[0]+vector_a[1]*vector_b[1]+vector_a[2]*vector_b[2]
    a_b = (vector_a[0]**2+vector_a[1]**2+vector_a[2]**2)**0.5 * (vector_b[0]**2+vector_b[1]**2+vector_b[2]**2)**0.5 +  1e-8
    return ab/a_b

def _norm2d(vec):
    return (vec[0]**2 + vec[1]**2)**0.5
def _norm3d(vec):
    return (vec[0]**2 + vec[1]**2 + vec[2]**2)**0.5

def _get_speed_3d(_object):
    v = _get_v_3d(_object)
    return _norm3d(v)

def _is_ahead(wp,target_pos):
    """
    Test if a target pos is ahead of the waypoint
    """
    wp_pos = _pos(wp)
    orientation = math.radians(wp.transform.rotation.yaw)
    target_vector = np.array([target_pos[0]-wp_pos[0],target_pos[1]-wp_pos[1]])
    forward_vector = np.array([np.cos(orientation),np.sin(orientation)])
    d_angle = math.degrees(math.acos(_cos(forward_vector,target_vector)))
    return d_angle<90

# corrdinate system class
class Coordinate(object):
    def __init__(self, vehicle):
        self.vehicle = vehicle
    def update_vehicle(self, vehicle):
        self.vehicle = vehicle
    def update_coordinate_systems(self):
        self.current_pos = _pos3d(self.vehicle)
        self.yaw = self.vehicle.get_transform().rotation.yaw
        self.pitch = self.vehicle.get_transform().rotation.pitch
        self.roll = self.vehicle.get_transform().rotation.roll
        def a2r(angle):
            return angle/180*np.pi
        self.yaw_radians = a2r(self.yaw) # fxxk
        self.pitch_radians = a2r(self.pitch)
        self.roll_radians = a2r(self.roll)
        R_X = np.array([[1,0,0],\
                        [0,np.cos(self.roll_radians),np.sin(self.roll_radians)],\
                        [0,-np.sin(self.roll_radians),np.cos(self.roll_radians)]])
        R_Y = np.array([[np.cos(self.pitch_radians),0,-np.sin(self.pitch_radians)],\
                        [0,1,0],\
                        [np.sin(self.pitch_radians),0,np.cos(self.pitch_radians)]])
        R_Z = np.array([[np.cos(self.yaw_radians),np.sin(self.yaw_radians),0],\
                        [-np.sin(self.yaw_radians),np.cos(self.yaw_radians),0],\
                        [0,0,1]])
        R = np.dot(np.dot(R_Z, R_Y),R_X)
        self.R = R
        self.R_inv = inv(R)
        self.R_2d = np.array([[np.cos(self.yaw_radians), np.sin(self.yaw_radians)],
                              [-np.sin(self.yaw_radians), np.cos(self.yaw_radians)]])
        self.R_2d_inv = inv(self.R_2d)
    
    def rotate_car2d(self, pos_world):
        self.update_coordinate_systems()
        pos_tran = (pos_world[0]*np.cos(self.yaw_radians)+pos_world[1]*np.sin(self.yaw_radians), -pos_world[0]*np.sin(self.yaw_radians)+pos_world[1]*np.cos(self.yaw_radians))
        return pos_tran

    def rotate_car3d(self, pos_world):
        self.update_coordinate_systems()
        pos_world = np.array(pos_world)
        pos_tran = np.dot(self.R,pos_world)
        return pos_tran

    def inverse_rotate_car3d(self, pos_world_car):
        self.update_coordinate_systems()
        pos_world_car = np.array(pos_world_car)
        pos_tran = np.dot(self.R_inv,pos_world_car)
        return pos_tran  

    def inverse_rotate_car2d(self, pos_world_car):
        self.update_coordinate_systems()
        pos_world_car = np.array(pos_world_car)
        pos_tran = np.dot(self.R_2d_inv,pos_world_car)
        return pos_tran                

    def transform_car2d(self, pos_actor):
        self.update_coordinate_systems()
        pos_ego = self.current_pos
        pos_ego_tran = self.rotate_car2d(pos_ego)
        pos_actor_tran = self.rotate_car2d(pos_actor)
        pos = [pos_actor_tran[0]-pos_ego_tran[0], pos_actor_tran[1]-pos_ego_tran[1]]
        return pos

    def transform_car3d(self, pos_actor):
        self.update_coordinate_systems()
        pos_ego = self.current_pos
        pos_ego_tran = self.rotate_car3d(pos_ego)
        pos_actor_tran = self.rotate_car3d(pos_actor)
        pos = [pos_actor_tran[0]-pos_ego_tran[0], pos_actor_tran[1]-pos_ego_tran[1], pos_actor_tran[2]-pos_ego_tran[2]]
        return pos
    
    def transform_world2d(self, pos_actor_tran):
        self.update_coordinate_systems()
        pos_ego = self.current_pos
        pos_ego_tran = self.rotate_car2d(pos_ego)
        pos_actor_tran = [pos_actor_tran[0]+pos_ego_tran[0], pos_actor_tran[1]+pos_ego_tran[1]]  
        pos_actor_world = self.inverse_rotate_car2d(pos_actor_tran)
        return pos_actor_world

    def transform_world3d(self, pos_actor_tran):
        self.update_coordinate_systems()
        pos_ego = self.current_pos
        pos_ego_tran = self.rotate_car3d(pos_ego)
        pos_actor_tran = [pos_actor_tran[0]+pos_ego_tran[0], pos_actor_tran[1]+pos_ego_tran[1], pos_actor_tran[2]+pos_ego_tran[2]]  
        pos_actor_world = self.inverse_rotate_car3d(pos_actor_tran)
        return pos_actor_world

    # obtain bounding box from a given vehicle
    def _bound(vehicle, fake=False): # transform the vehicle into 8 points bounding box
        ego_car = self.vehicle
        current_pos = _pos3d(ego_car)
        if not fake:
            vehicle_center = self._transform_car3d(current_pos, _pos3d(vehicle))
            extent_length = _pos3d(vehicle.bounding_box.extent)
        else:
            vehicle_center = [-70,0,0]
            extent_length = [0,0,0]

        return [(vehicle_center[0]-extent_length[0], vehicle_center[1]-extent_length[1], vehicle_center[2]-extent_length[2]),\
                (vehicle_center[0]+extent_length[0], vehicle_center[1]-extent_length[1], vehicle_center[2]-extent_length[2]),\
                (vehicle_center[0]-extent_length[0], vehicle_center[1]+extent_length[1], vehicle_center[2]-extent_length[2]),\
                (vehicle_center[0]+extent_length[0], vehicle_center[1]+extent_length[1], vehicle_center[2]-extent_length[2])]