#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module contains PID controllers to perform lateral and longitudinal control. """

from collections import deque
import math

import numpy as np

# ------------------------------------------------------------------------------
# import carla PythonApi
# ------------------------------------------------------------------------------
import sys
from os import path as osp
current_dir = osp.abspath(osp.dirname(__file__))
sys.path.append(current_dir+"/../..")

#carla_simulator_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
#carla_simulator_path += '/PythonAPI/carla-0.9.1-py3.5-linux-x86_64.egg'
carla_simulator_path = '/home/SENSETIME/maqiurui/reinforce/carla/carla_0.9.4/PythonAPI/carla-0.9.4-py3.5-linux-x86_64.egg'
try:
    sys.path.append(carla_simulator_path)
    sys.path.append(current_dir+'/../../../PythonAPI/carla-0.9.4-py3.5-linux-x86_64.egg')
except IndexError:
    pass
import carla
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
from baselines import logger

def get_speed(vehicle):
    """
    Compute speed of a vehicle in Kmh
    :param vehicle: the vehicle for which speed is calculated
    :return: speed as a float in Kmh
    """
    vel = vehicle.get_velocity()
    return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

THROTTLE_TABLE = {
0:0.2, 6: 0.3, 12: 0.4, 18: 0.5, 23: 0.6, 29: 0.65, 32: 0.65, 36: 0.656, 40: 0.66
}
SPEED = [0, 6, 12, 18, 23, 29, 32, 36, 40]
def _get_throttle(target_speed):
    for i in range(len(SPEED)-1):
        if target_speed >= SPEED[i] and target_speed <= SPEED[i+1]:
            speed_range = [SPEED[i], SPEED[i+1]] 
    throttle_range = [THROTTLE_TABLE[speed_range[0]], THROTTLE_TABLE[speed_range[1]]]
    speed_shift = (target_speed-speed_range[0])/(abs(speed_range[1]-speed_range[0]))
    throttle = throttle_range[0] + speed_shift * abs(throttle_range[1]-throttle_range[0])
    return throttle

class VehiclePIDController():
    """
    VehiclePIDController is the combination of two PID controllers (lateral and longitudinal) to perform the
    low level control a vehicle from client side
    """

    def __init__(self, vehicle, args_lateral=None, args_longitudinal=None):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param args_lateral: dictionary of arguments to set the lateral PID controller using the following semantics:
                             K_P -- Proportional term
                             K_D -- Differential term
                             K_I -- Integral term
        :param args_longitudinal: dictionary of arguments to set the longitudinal PID controller using the following
        semantics:
                             K_P -- Proportional term
                             K_D -- Differential term
                             K_I -- Integral term
        """
        if not args_lateral:
            args_lateral = {'K_P': 1.0, 'K_D': 0.0, 'K_I': 0.0, 'K_P_pos': logger.p_pos, 'K_D_pos': 0.0, 'K_I_pos': 0.0}
        if not args_longitudinal:
            args_longitudinal = {'K_P': logger.p_vel, 'K_D': 0.0, 'K_I': 0.0}
        print("============== PID parameters: CTE %s VEL %s =============" % (str(logger.p_pos), str(logger.p_vel)))

        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self._lon_controller = PIDLongitudinalController(self._vehicle, **args_longitudinal)
        self._lat_controller = PIDLateralController(self._vehicle, **args_lateral)

    def run_step(self, target_speed, waypoint, ref_point):
        """
        Execute one step of control invoking both lateral and longitudinal PID controllers to reach a target waypoint
        at a given target_speed.
        :param target_speed: desired vehicle speed
        :param waypoint: target angle
        :param refpoint: target location
        :return: distance (in meters) to the waypoint
        """
        throttle = self._lon_controller.run_step(target_speed)
        # steering = self._lat_controller.run_step(waypoint)

        steering_from_angle = self._lat_controller.run_step_orientation(waypoint)
        steering_from_pos = self._lat_controller.run_step_pos(ref_point)
        steering = steering_from_angle + steering_from_pos
        steering = np.clip(steering, -1, 1)
        logger.steering_from_angle = steering_from_angle
        logger.steering_from_pos = steering_from_pos

        control = carla.VehicleControl()
        control.steer = steering
        control.throttle = throttle
        control.brake = 0.0
        control.hand_brake = False
        control.manual_gear_shift = False
        return control

    def update_vehicle(self, vehicle):
        self._vehicle = vehicle
        self._lon_controller._vehicle = vehicle
        self._lat_controller._vehicle = vehicle

class PIDLongitudinalController():
    """
    PIDLongitudinalController implements longitudinal control using a PID.
    """

    def __init__(self, vehicle, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param K_P: Proportional term
        :param K_D: Differential term
        :param K_I: Integral term
        :param dt: time differential in seconds
        """
        self._vehicle = vehicle
        self._K_P = K_P
        self._K_D = K_D
        self._K_I = K_I
        self._dt = dt
        self._e_buffer = deque(maxlen=30)

    def run_step(self, target_speed, debug=False):
        """
        Execute one step of longitudinal control to reach a given target speed.
        :param target_speed: target speed in Km/h
        :return: throttle control in the range [0, 1]
        """
        current_speed = get_speed(self._vehicle)

        if debug:
            print('Current speed = {}'.format(current_speed))

        return self._pid_control(target_speed, current_speed)

    def _pid_control(self, target_speed, current_speed):
        """
        Estimate the throttle of the vehicle based on the PID equations
        :param target_speed:  target speed in Km/h
        :param current_speed: current speed of the vehicle in Km/h
        :return: throttle control in the range [0, 1]
        """
        _e = (target_speed - current_speed)
        self._e_buffer.append(_e)

        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2])# / self._dt
            _ie = sum(self._e_buffer)# * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        if target_speed <= 32. and not logger.scenario:
            return np.clip(_get_throttle(target_speed), 0.0, 1.0) 
        else:
            return np.clip((self._K_P * _e) + (self._K_D * _de / self._dt) + (self._K_I * _ie * self._dt), 0.0, 1.0)

class PIDLateralController():
    """
    PIDLateralController implements lateral control using a PID.
    """

    def __init__(self, vehicle, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03, K_P_pos=1.0, K_D_pos=0.0, K_I_pos=0.0):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param K_P: Proportional term
        :param K_D: Differential term
        :param K_I: Integral term
        :param dt: time differential in seconds
        """
        self._vehicle = vehicle
        self._K_P = K_P
        self._K_D = K_D
        self._K_I = K_I
        self._dt = dt
        self._K_P_pos = K_P_pos
        self._K_D_pos = K_D_pos
        self._K_I_pos = K_I_pos
        self._e_buffer = deque(maxlen=10)
        self._e_pos_buffer = deque(maxlen=10)

    def run_step(self, waypoint):
        """
        Execute one step of lateral control to steer the vehicle towards a certain waypoint.
        :param waypoint: target waypoint
        :return: steering control in the range [-1, 1] where:
            -1 represent maximum steering to left
            +1 maximum steering to right
        """
        return self._pid_control(waypoint, self._vehicle.get_transform())

    def run_step_orientation(self, waypoint):
        """
        Execute one step of lateral control to steer the vehicle towards a certain waypoint.
        :param waypoint: target waypoint
        :return: steering control in the range [-1, 1] where:
            -1 represent maximum steering to left
            +1 maximum steering to right
        """
        return self._pid_control(waypoint, self._vehicle.get_transform())

    def run_step_pos(self, ref_point):
        """
        Execute one step of lateral control to steer the vehicle towards a certain waypoin.t
        :param waypoint: target waypoint
        :return: steering control in the range [-1, 1] where:
            -1 represent maximum steering to left
            +1 maximum steering to right
        """
        ref_dir = np.array([ref_point.dir_x, ref_point.dir_y, 0.])
        car_dir = np.array([1, 0, 0.])
        ref_theta = math.acos(np.clip(np.dot(ref_dir, car_dir) / (np.linalg.norm(ref_dir) * np.linalg.norm(car_dir)), -1.0, 1.0))

        ref_pos = [ref_point.x, ref_point.y, 0]
        cte = ref_pos[1]*np.cos(ref_theta)-ref_pos[0]*np.sin(ref_theta)

        _cross = np.cross(car_dir, ref_dir)
        logger._ref_pos = ref_pos
        logger._ref_theta = ref_theta
        logger._cross = _cross

        if _cross[2] > 0:
            cte *= -1.0
        logger._ref_dir = ref_dir
        logger._car_dir = car_dir

        return self._pid_control_pos(cte)

    def _pid_control(self, waypoint, vehicle_transform):
        """
        Estimate the steering angle of the vehicle based on the PID equations
        :param waypoint: target waypoint
        :param vehicle_transform: current transform of the vehicle
        :return: steering control in the range [-1, 1]
        """
        v_begin = vehicle_transform.location
        v_end = v_begin + carla.Location(x=math.cos(math.radians(vehicle_transform.rotation.yaw)),
                                         y=math.sin(math.radians(vehicle_transform.rotation.yaw)))

        v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, 0.0])
        w_vec = np.array([waypoint.transform.location.x -
                          v_begin.x, waypoint.transform.location.y -
                          v_begin.y, 0.0])
        _dot = math.acos(np.clip(np.dot(w_vec, v_vec) /
                                 (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0))

        _cross = np.cross(v_vec, w_vec)
        if _cross[2] < 0:
            _dot *= -1.0

        logger._dot = _dot
        logger._theta = _dot * 180

        self._e_buffer.append(_dot)
        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) #/ self._dt
            _ie = sum(self._e_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0
        return np.clip((self._K_P * _dot) + (self._K_D * _de /
                                             self._dt) + (self._K_I * _ie * self._dt), -1.0, 1.0)

    def _pid_control_pos(self, cte):
        """
        Estimate the steering angle of the vehicle based on the PID equations
        :param waypoint: target waypoint
        :param vehicle_transform: current transform of the vehicle
        :return: steering control in the range [-1, 1]
        """
        self._e_pos_buffer.append(cte)
        if len(self._e_pos_buffer) >= 2:
            _de = (self._e_pos_buffer[-1] - self._e_pos_buffer[-2])# / self._dt
            _ie = sum(self._e_pos_buffer)# * self._dt
        else:
            _de = 0.0
            _ie = 0.0
        logger._cte = cte
        return np.clip((self._K_P_pos * cte) + (self._K_D_pos * _de /
                                             self._dt) + (self._K_I_pos * _ie * self._dt), -1.0, 1.0)
