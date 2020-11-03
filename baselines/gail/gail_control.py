#!/usr/bin/env python

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : toggle draw waypoints
    S            : brake
    AD           : steer
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot

    TAB          : change camera view
    `            : next sensor
    [1-9]        : change to sensor [1-9]
    C            : change weather (Shift+C reverse)
    Backspace    : change vehicle

    R            : toggle recording images to disk

    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
"""
from __future__ import print_function

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================
# # version 0.9.4
# import sys
# from os import path as osp
# current_dir = osp.abspath(osp.dirname(__file__))
# sys.path.append(current_dir+"/../..")
# carla_simulator_path = '/home/SENSETIME/maqiurui/reinforce/carla/carla_0.9.4/PythonAPI/carla-0.9.4-py3.5-linux-x86_64.egg'
# try:
#     sys.path.append(carla_simulator_path)
#     sys.path.append(current_dir+'/../../../CARLA/PythonAPI/carla-0.9.4-py3.5-linux-x86_64.egg')
# except IndexError:
#     pass

# version 0.9.5
import sys
from os import path as osp

current_dir = osp.abspath(osp.dirname(__file__))
sys.path.append(current_dir + "/../..")
carla_simulator_path = '/home/SENSETIME/maqiurui/reinforce/carla/carla_0.9.5/PythonAPI/carla-0.9.5-py3.5-linux-x86_64.egg'
try:
    sys.path.append(carla_simulator_path)
    sys.path.append(current_dir + '/../../../CARLA/PythonAPI/carla-0.9.5-py3.5-linux-x86_64.egg')
except IndexError:
    pass

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla
from carla import ColorConverter as cc

import argparse
import collections
import datetime
import time
import logging
import math
import random
import re
import weakref
from collections import deque
from enum import Enum
import copy
import heapq
import pickle
import os
from baselines.logger import TensorBoardOutputFormat
from baselines import logger
from scenarios import *

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

pygame.init()
pygame.font.init()
try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')


# -- World ---------------------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


DOTTED_SPAWN_POINTS = [2, 3, 4, 140, 6, 8, 10, 5, 13, 15, 16, 17, 18, 75, 99, 92, 98, 103, 101, 44, 45, 46, 72, 134,
                       135, 136, 137,
                       34, 35, 36, 37, 38, 88, 89, 125, 126, 24, 57, 60]
DOTTED_SPAWN_POINTS = sorted(DOTTED_SPAWN_POINTS)

SCENARIOS = [Cross_Join, Cross_Join, Ring_Join, Ring_Join, Straight_Follow_Single, Straight_Follow_Double, Cross_Turn_Left, Cross_Turn_Left]
SCENARIO_NAMES = ["Cross_Join", "Cross_Join", "Ring_Join", "Ring_Join", "Straight_Follow_Single", "Straight_Follow_Double", "Cross_Turn_Left", "Cross_Turn_Left"]

class RoadOption(Enum):
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4


class World(object):
    def __init__(self, carla_world, sync, sigmas, other_cars=0, autopilot=False, camera=True, A_skip=1, target_dis=10,
                 maxlen=100, mode='all', \
                 feature='lane_car', dim='2d', dis_max=1.35, spawn_mode='random', render=False, width=400, height=400,
                 obs_wp='exp', expected_velocity=7.0, \
                 draw_features=False, farther_features='rand', all_green_light=True, curriculumn_threshold=350,
                 save_ucb=True, max_lanes=3, scenario=True, \
                 scenario_name='OtherLeadingVehicle', client=None, host=None, port=None, checkkeys=None):

        # Carla Server world Settings
        print("EXPECTED V: ", expected_velocity)
        
        # inherent variables from logger
        self.actor_id=logger.actor_id
        self.checkkeys = checkkeys
        
        self.v_value = [0]
        self.fake_reward = [0]
        # camera = False
        self.world = carla_world
        self.sync = sync
        settings = self.world.get_settings()
        settings.synchronous_mode = True if self.sync else False
        self.world.apply_settings(settings)
        self.A_skip = A_skip
        self._mode = "Fake"
        # display and render
        self.render = render
        self.display = None
        self.render_width = width
        self.render_height = height
        if self.render:
            self.display = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.hud = HUD(self.render_width, self.render_height)
        self.clock = pygame.time.Clock()  # Client Clock
        self.scenario_name = scenario_name
        #self.keypointsinfos = {key:[] for key in logger.keypoints.keys()}
        #self.keypoints_saver = open("key_dict.pkl","wb")

        self.client = client
        self.host, self.port = host, port
        self._map = self.world.get_map()
        self._map_name = self._map.name
        self.fixed_spawn_points = self._map.get_spawn_points()
        bad_points = [12, 55, 42, 70, 71, 56, 63, 53, 54, 64, 68, 69, 86, 87, 94, 95, 105, 123, 124, 65, 67, 51, 121,
                      109]
        # selected_points = [3, 6, 9, 73, 176, 54, 53, 14, 15, 32, 169, 186, 67]
        # #selected_points = [186]
        # # Some waypoints cannot learn. Delete them: These are marked red in no_Rendering mode
        # #self.fixed_spawn_points = [ele for ele in self.fixed_spawn_points if
        # #                           self.fixed_spawn_points.index(ele) not in bad_points]
        # self.fixed_spawn_points = [ele for ele in self.fixed_spawn_points if
        #                            self.fixed_spawn_points.index(ele) not in bad_points and self.fixed_spawn_points.index(ele) in selected_points]

        
        # Some waypoints cannot learn. Delete them: These are marked red in no_Rendering mode
        self.fixed_spawn_points = [ele for ele in self.fixed_spawn_points if
                                   self.fixed_spawn_points.index(ele) not in bad_points]
        if other_cars > 0:
            self.fixed_spawn_points = [ele for ele in self.fixed_spawn_points if
                                       self.fixed_spawn_points.index(ele) in DOTTED_SPAWN_POINTS]

        self.save_ucb_path = current_dir + '/ucb_spawn_points_heap.pkl'
        self.save_ucb = save_ucb
        if self.save_ucb and os.path.exists(self.save_ucb_path):
            spawn_points_file = open(self.save_ucb_path, 'rb')
            self.spawn_points_heap = pickle.load(spawn_points_file)
            spawn_points_file.close()
            self._N = sum([ele[2] for ele in self.spawn_points_heap])
        else:
            self.spawn_points_heap = []  # Implement Upper Confident Bound
            self.update_start_heap()
        self.death_step = 0
        self.curriculumn_threshold = curriculumn_threshold
        self.sigmas = sigmas

        # setup start position
        self.spawn_idx = 0
        self.spawn_mode = "fixed"
        self.vehicle = None
        self.scenario = scenario
        if scenario:
            self.scenario_init(scenario_name, self._map, self.world)
        else:
            self.select_hero_actor()
        self.dim = dim

        # Hard Coding IDX index for expert data output
        self.idx_throttle = -1
        self.idx_first_wp_left_dis = 1
        self.idx_first_wp_right_dis = 4 if self.dim == '3d' else 3

        self.expected_velocity = expected_velocity

        # setup autopilot, for collecting data
        self.autopilot = autopilot
        if autopilot:
            pass
            # self.vehicle.set_autopilot()

        control = self.vehicle.get_control()
        control.steer, control.throttle, control.brake = 0., 0., 0.
        self.vehicle.apply_control(control)

        self.car_shape = [self.vehicle.bounding_box.extent.x, self.vehicle.bounding_box.extent.y,
                          self.vehicle.bounding_box.extent.z]
        self.car_length = self.car_shape[0]
        self.car_width = self.car_shape[1]
        self.car_height = self.car_shape[2]

        # setup sensors and collision detection
        #self.collision_sensor = CollisionSensor(self.vehicle, self.hud)
        # self.lane_invasion_sensor = LaneInvasionSensor(self.vehicle, self.hud)
        self._camera = camera
        if self._camera:
            self.camera_manager = CameraManager(self.vehicle, self.hud)
            self.camera_manager.set_sensor(0, notify=False)
        self._map = self.world.get_map()
        self._weather_presets = find_weather_presets()
        self._weather_index = 0

        self.target_dis = target_dis
        self.maxlen = maxlen
        self.feature = feature
        self.last_status = int(max_lanes / 2)

        # setup history for debugging
        self.history = deque()
        self.mode = mode

        # stats
        self.rew_now = 0.
        self.v_rew = 0.
        self.ang_rew = 0.
        self.track_rew = 0.
        self.dis_now = 100.
        self.ep_rew = 0.0
        self.ep_len = 0
        self.ep_rew_buffer = deque([], 100)
        self.ep_len_buffer = deque([], 100)
        self.ang = 0

        self.brake_control = carla.VehicleControl()
        self.brake_control.steer = 0
        self.brake_control.throttle = 0
        self.brake_control.brake = 1

        self.zero_control = carla.VehicleControl()
        self.zero_control.steer = 0
        self.zero_control.throttle = 1.0
        self.zero_control.brake = 0

        self.obs_wp = obs_wp  # equal and exp
        # 70 meters for visual
        self._maxlen = 701
        sys.setrecursionlimit(self._maxlen + 300)

        self.max_lanes = max_lanes
        self._waypoints_queue_l = [deque(maxlen=self._maxlen)]
        self._waypoints_pos_queue_l = [deque(maxlen=self._maxlen)]
        self._lanepoints_pos_queue_l = [deque(maxlen=self._maxlen)]
        # other lanes
        self._global_waypoints_queue_l = [None for i in range(self.max_lanes)]
        self._global_waypoints_pos_queue_l = [None for i in range(self.max_lanes)]
        self._global_lanepoints_pos_queue_l = [None for i in range(self.max_lanes)]
        self._global_current_waypoint_pos_car = [None for i in range(self.max_lanes)]
        self._global_reflines = [None for i in range(self.max_lanes)]
        self.shifts = []
        shift = int(self.max_lanes / 2) * -3.5
        for i in range(self.max_lanes):
            self._global_waypoints_queue_l[i] = [deque([], maxlen=self._maxlen)]
            self._global_waypoints_pos_queue_l[i] = [deque([], maxlen=self._maxlen)]
            self._global_lanepoints_pos_queue_l[i] = [deque([], maxlen=self._maxlen)]
            self._global_reflines[i] = [deque([], maxlen=self._maxlen)]
            self.shifts.append(shift)
            shift = shift + 3.5

        self._visual = 70
        self.all_green_light = all_green_light

        if self.autopilot:
            self.right_branch = 0
            self.just_merged = False
            self.past_intersect = True
            self.farther_features = farther_features
            # a debugging flag to test validity
            # save and draw features_world instead of features_car
            self.draw_features = draw_features

        self.zombie_cars, self.visible_zombie_cars = [], []
        # reading from world have to happen after world tick. Otherwise it gets nothing
        # self.zombie_cars = self.world.get_actors().filter('vehicle.*')
        self.other_cars = other_cars
        if self.scenario:
            self.zombie_cars = self.scenario_now.zombie_cars
            self.last_waypoint = self._map.get_waypoint(self.vehicle.get_location())
            self.speed = self.scenario_now.speed
            self.speed_freq_statistic = dict()
        else:
            self.zombie_cars = [actor for actor in self.world.get_actors() if
                                'vehicle' in actor.type_id and actor != self.vehicle]
        self.zombie_car_ref_pos = deque([], maxlen=self.other_cars)


        self.world.on_tick(self.on_world_tick)
        self.next_frame()

        try:  # autopilot initialized at crossroads does not know shape
            print("----------------World shape------------------", self.shape)
        except:
            print("----------------World shape------------------")

    def select_hero_actor(self):
        self.blueprint = blueprint = random.choice(self.world.get_blueprint_library().filter('vehicle.bmw.grandtourer'))
        blueprint.set_attribute('role_name', 'agent')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        while self.vehicle is None:
            self.spawn_point_now = self.next_start_point()
            self.vehicle = self.world.try_spawn_actor(blueprint, self.spawn_point_now)
        self.vehicle.set_simulate_physics(True)

    def scenario_init(self, name, map, world):
        if name == 'OtherLeadingVehicle':
            scenario = OtherLeadingVehicle
        elif name == 'CarFollowing':
            scenario = CarFollowing
        elif name == 'OtherLeadingVehicle_FullMap':
            scenario = OtherLeadingVehicle_FullMap
        elif name == 'Cross_Join':
            scenario = Cross_Join
            logger.select_scenario_id = self.actor_id % 2
        elif name == 'Ring_Join':
            scenario = Ring_Join
            logger.select_scenario_id = self.actor_id % 2
        elif name == 'Straight_Follow_Single':
            scenario = Straight_Follow_Single
            logger.select_scenario_id = self.actor_id % 2
        elif name == 'Straight_Follow_Double':
            scenario = Straight_Follow_Double
            logger.select_scenario_id = self.actor_id % 2
        elif name == 'Cross_Follow':
            scenario = Cross_Follow
            logger.select_scenario_id = self.actor_id % 2
        elif name == 'Cross_Turn_Left':
            scenario = Cross_Turn_Left
            logger.select_scenario_id = self.actor_id % 2
        elif name == 'Cross_Turn_Right':
            scenario = Cross_Turn_Right
            logger.select_scenario_id = self.actor_id % 2
        elif name == 'OverTake':
            scenario = OverTake
            logger.select_scenario_id = self.actor_id % 2
        elif name == 'Merge_Env':
            scenario = SCENARIOS[self.actor_id % (len(SCENARIOS))]
            self.scenario_name = SCENARIO_NAMES[self.actor_id % (len(SCENARIOS))]
            if self.scenario_name == 'Cross_Join':
                logger.select_scenario_id = self.actor_id % 2
            elif self.scenario_name == 'Ring_Join':
                logger.select_scenario_id = self.actor_id % 2
            elif self.scenario_name == 'Straight_Follow_Single':
                logger.select_scenario_id = self.actor_id % 2
            elif self.scenario_name == 'Straight_Follow_Double':
                logger.select_scenario_id = self.actor_id % 2
            elif self.scenario_name == 'Cross_Follow':
                logger.select_scenario_id = self.actor_id % 2
            elif self.scenario_name == 'Cross_Turn_Left':
                logger.select_scenario_id = self.actor_id % 2
            elif self.scenario_name == 'Cross_Turn_Right':
                logger.select_scenario_id = self.actor_id % 2
            elif self.scenario_name == 'OverTake':
                logger.select_scenario_id = self.actor_id % 2
        else:
            raise NotImplementedError('Scenario does not exist!')
        if name == 'OverTake':
            self.scenario_now = scenario(name, map, world, self.checkkeys)
        else:
            self.scenario_now = scenario(name, map, world)
        self.vehicle = self.scenario_now.hero_car

    def scenario_restart(self):
        self.scenario_now.restart()
        self.zombie_cars = self.scenario_now.zombie_cars
        self.vehicle = self.scenario_now.hero_car

    def update_start_heap(self, death_step=800):
        # The heap is a min heap
        if len(self.spawn_points_heap) == 0:
            self._N = len(self.fixed_spawn_points)
            # initial weighting: assume all waypoints are successful once.
            # Note that we can obtain a better prior. But not necessary
            # [UCB, death_time, overall_time, wp]
            init_val = 1 / 1 + np.sqrt(2 * np.log(self._N) / 1)
            for i in range(len(self.fixed_spawn_points)):
                heapq.heappush(self.spawn_points_heap, [-init_val, 0, 1, i])
            return

        self._N += 1  # Sum of frequencies
        self.spawn_points_heap[0][2] += 1  # Frequencies
        self.spawn_points_heap[0][1] += 1 if death_step < 300 else 0
        last_step = np.array(self.spawn_points_heap)  # update UCB
        last_step[:, 0] = -(last_step[:, 1] / last_step[:, 2] + np.power(2 * np.log(self._N) / last_step[:, 2], 0.5))
        self.spawn_points_heap = last_step.tolist()  # heapify stuff
        heapq.heapify(self.spawn_points_heap)

        if self.save_ucb and (self._N % 50 == 0):
            spawn_points_file = open(self.save_ucb_path, 'wb')
            pickle.dump(self.spawn_points_heap, spawn_points_file)
            spawn_points_file.close()

    def next_start_point(self):
        self.spawn_idx += 1
        if self.spawn_idx < self.curriculumn_threshold and self.spawn_mode == 'fixed':
            idx = self.spawn_idx % len(self.fixed_spawn_points)
            spawn_point = self.fixed_spawn_points[idx]
        elif self.spawn_idx < self.curriculumn_threshold and self.spawn_mode == 'random':
            spawn_point = random.choice(self.fixed_spawn_points) if self.fixed_spawn_points else carla.Transform()
        elif self.spawn_idx >= self.curriculumn_threshold:
            spawn_point = self.fixed_spawn_points[int(self.spawn_points_heap[0][-1])]
        else:
            raise NotImplementedError
        return spawn_point

    def _mode(self, spawn_point):
        raise NotImplementedError

    def restart(self):
        if self.scenario:
            if not self.speed in self.speed_freq_statistic:
                self.speed_freq_statistic[self.speed] = self.ep_len
                self.speed_freq_statistic[str(self.speed)] = 1
            else:
                # print(2, self.speed_freq_statistic)
                # self.speed_freq_statistic[self.speed] = (self.speed_freq_statistic[self.speed] * self.speed_freq_statistic[
                #     str(self.speed)] + self.ep_len) / (self.speed_freq_statistic[str(self.speed)] + 1)
                self.speed_freq_statistic[self.speed] = self.ep_len
                # print(3, self.speed_freq_statistic[self.speed])
                self.speed_freq_statistic[str(self.speed)] += 1
                # print(4, self.speed_freq_statistic[str(self.speed)])
            # print(self.speed_freq_statistic)
            self.speed = self.scenario_now.speed
        self.ep_rew_buffer.append(self.ep_rew)
        self.ep_len_buffer.append(self.ep_len)
        self.ep_rew = 0.0
        self.ep_len = 0

        if self.spawn_idx >= self.curriculumn_threshold:
            self.update_start_heap(self.death_step)
        self.spawn_point_now = self.next_start_point()
        self.death_step = 0

        self.destroy()

        self._waypoints_queue_l = [deque([], maxlen=self._maxlen)]
        self._waypoints_pos_queue_l = [deque([], maxlen=self._maxlen)]
        self._lanepoints_pos_queue_l = [deque([], maxlen=self._maxlen)]
        # other lanes
        self._global_waypoints_queue_l = [None for i in range(self.max_lanes)]
        self._global_waypoints_pos_queue_l = [None for i in range(self.max_lanes)]
        self._global_lanepoints_pos_queue_l = [None for i in range(self.max_lanes)]
        self._global_current_waypoint_pos_car = [None for i in range(self.max_lanes)]
        self._global_reflines = [None for i in range(self.max_lanes)]
        for i in range(self.max_lanes):
            self._global_waypoints_queue_l[i] = [deque([], maxlen=self._maxlen)]
            self._global_waypoints_pos_queue_l[i] = [deque([], maxlen=self._maxlen)]
            self._global_lanepoints_pos_queue_l[i] = [deque([], maxlen=self._maxlen)]
            self._global_reflines[i] = [deque([], maxlen=self._maxlen)]
        if self.autopilot:
            self.right_branch = 0
            self.just_merged = False
            self.past_intersect = False

        if self.scenario:
            self.scenario_restart()
        else:
            blueprint = self.blueprint
            self.vehicle = self.world.try_spawn_actor(blueprint, self.spawn_point_now)
            while self.vehicle is None:
                self.spawn_point_now = self.next_start_point()
                self.vehicle = self.world.try_spawn_actor(blueprint, self.spawn_point_now)

        self.vehicle.set_simulate_physics(True)

        control = self.vehicle.get_control()
        control.steer, control.throttle, control.brake = 0., 0., 0.
        self.vehicle.apply_control(control)

        self.last_status = int(self.max_lanes / 2)
        
        self.visible_zombie_cars.clear()
        self.steer = 0
        self.throttle_brake = 0

        if self.autopilot:
            pass
            # self.vehicle.set_autopilot()
        #self.collision_sensor = CollisionSensor(self.vehicle, self.hud)
        # self.lane_invasion_sensor = LaneInvasionSensor(self.vehicle, self.hud)
        if self._camera:
            self.camera_manager = CameraManager(self.vehicle, self.hud)
            self.camera_manager.set_sensor(0, notify=False)

        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self.vehicle_trajectory = deque([()], 10)
        actor_type = get_actor_display_name(self.vehicle)
        if self._camera:
            self.hud.notification(actor_type)

        self.next_frame()

    def _move_zombie_car_ahead(self):
        vehicle_location = self.vehicle.get_location()
        wp_baseline = self._map.get_waypoint(vehicle_location)
        wps_ahead_agent = [wp_baseline.next(30.)[0], wp_baseline.next(20.)[0], wp_baseline.next(10.)[0]]
        no_cars_on_wp = [self._no_cars_nearby(wp) for wp in wps_ahead_agent]
        print(no_cars_on_wp, "NO CARS ON WP")

        def _all_true(lst):
            for item in lst:
                if item != True:
                    return False
            return True

        all_true = _all_true(no_cars_on_wp)
        if all_true:
            self._place_zombie_car_onwp(wps_ahead_agent[1])
            print("Vehicle Ahead !!!!!!!!!!!!")
        else:
            print("Can't move new car ahead of the agent")

    def _no_cars_nearby(self, wp):
        pos_target = utils._pos3d(wp)
        print(len(self.zombie_cars), "ZOMBIE CARS")
        for car in self.zombie_cars:
            car_wp = self._map.get_waypoint(car.get_location())
            pos_zombie_car = utils._pos3d(car)
            if utils._dis3d(pos_zombie_car, pos_target) < 10. and car_wp.lane_id == wp.lane_id:
                return False
        return True

    def _place_zombie_car_onwp(self, wp):
        zombie_car = random.choice(self.zombie_cars)
        zombie_car.set_autopilot(False)
        zombie_car.set_velocity(carla.Vector3D(x=0, y=0, z=0))
        zombie_car.set_simulate_physics(False)
        _pos_baseline = [0, 0, 0]
        _rotation_baseline = [0, 0, 0]
        _location_baseline = utils._location3d(_pos_baseline)
        _rotation_baseline = utils._rotation3d(_rotation_baseline)
        fake_transform = carla.Transform(location=_location_baseline, rotation=_rotation_baseline)
        zombie_car.set_transform(fake_transform)
        zombie_car.set_transform(wp.transform)
        zombie_car.set_simulate_physics(True)
        zombie_car.set_autopilot(True)

    def _draw_waypoints(self, args=None):
        if args is not None:
            for p in args:
                self.world.debug.draw_point(carla.Location(x=p[0], y=p[1], z=p[2] + 1), life_time=1, size=0.05,
                                            color=carla.Color())
            return

        for wp_forward in self.waypoints_forward_l:
            for w in wp_forward[:, 0]:
                t = w.transform
                begin = t.location + carla.Location(z=1.0)
                angle = math.radians(t.rotation.yaw)
                end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
                color = carla.Color()
                #color.r, color.g, color.b, color.a = 255, 255, 51, 1
                color.r, color.g, color.b, color.a = 255, 255, 132, 1
                self.world.debug.draw_arrow(begin, end, arrow_size=0.5, life_time=0.5,
                                            color=color)
                #self.world.debug.draw_arrow(begin, end, arrow_size=0.3, life_time=0.5)
        # also draw lane position
        if (not self.autopilot) or (not self.draw_features):
            for lanepoints_pos_forward_world in self.lanepoints_pos_forward_world_l:
                t = 0
                for p in lanepoints_pos_forward_world:
                    if t < 2:
                        color = carla.Color()
                        #color.r, color.g, color.b, color.a = 255, 60, 0, 1
                        color.r, color.g, color.b, color.a = 255, 128, 0, 1 #255,142,0,1#255, 255, 255, 1
                    elif t >= 2: 
                        color = carla.Color()
                        color.r, color.g, color.b, color.a = 204, 0, 0, 1#166,91,47,1 #
                    t = (t + 1) % 4
                    self.world.debug.draw_point(carla.Location(x=p[0], y=p[1], z=p[2] + 1), life_time=0.1, size=0.08,
                                                color=color)

        # also draw bounding box for filted cars
        if len(self.visible_zombie_cars) > 0:
            infos = [(self.zombie_cars[index[1]].get_transform(), self.zombie_cars[index[1]].bounding_box) for index in
                     self.visible_zombie_cars]
            i = 0
            for transform, box in infos:
                i += 1
                box.location += transform.location
                self.world.debug.draw_box(box, transform.rotation, thickness=0.2, color=carla.Color(), life_time=0.1,
                                          persistent_lines=True)
                if i > 5: break
        
        # also draw goal positions and trajectory generated by planner
        if hasattr(self, "goal_wp_now"):
            goal_wp = self.goal_wp_now
            location = goal_wp.transform.location
            color_goal_wp = carla.Color()
            #color_goal_wp.r, color_goal_wp.g, color_goal_wp.b, color_goal_wp.a = 51, 51, 255, 1
            color_goal_wp.r, color_goal_wp.g, color_goal_wp.b, color_goal_wp.a = 74, 59, 255, 1 #178, 102, 255, 1 #153, 51, 255, 1
            self.world.debug.draw_point(carla.Location(x=location.x, y=location.y, z=location.z + 1.), life_time=0.05,
                                        size=0.2, color=color_goal_wp)
        
            color_traj = carla.Color()
            color_traj.r, color_traj.g, color_traj.b, color_traj.a = 22, 176, 71, 1
            #color_traj.r, color_traj.g, color_traj.b, color_traj.a = 153, 51, 255, 1
            for point in self.traj_ret:
                location = carla.Location(x=point[0], y=point[1], z=point[2] + 2)
                self.world.debug.draw_point(location, life_time=0.05, size=0.06, color=color_traj)
        
            location = self.ref_location
            color = carla.Color()
            color.r, color.g, color.b, color.a = 17, 45, 229, 1
            self.world.debug.draw_point(carla.Location(x=location.x, y=location.y, z=location.z + 1.), life_time=0.05,
                                        size=0.2, color=color)
        
            backward_wp = self.target_wp_pid
            location = backward_wp.transform.location
            color = carla.Color()
            color.r, color.g, color.b, color.a = 232, 232, 14, 1
            self.world.debug.draw_point(carla.Location(x=location.x, y=location.y, z=location.z + 1.), life_time=0.05,
                                        size=0.1, color=color)
        
        if hasattr(self, "wp_candidates"):
            for wp in self.wp_candidates:
                location = wp.transform.location
                color = carla.Color()
                color.r, color.g, color.b, color.a = 71, 175, 205, 1 #251,243,213 #51, 153, 255, 1
                self.world.debug.draw_point(carla.Location(x=location.x, y=location.y, z=location.z + 1.), life_time=0.05,
                            size=0.2, color=color)

    def _destroy_other_cars(self):
        for vehicle in self.world.get_actors():
            if vehicle is not None and vehicle is not self.vehicle and vehicle is carla.Vehicle:
                vehicle.destroy()

    def _add_cars(self, num=5):
        spawn_points = self.fixed_spawn_points
        random.shuffle(spawn_points)
        blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
        count = num
        for spawn_point in spawn_points:
            if count <= 0:
                break
            else:
                blueprint = random.choice(blueprints)
                vehicle = self.world.try_spawn_actor(blueprint, spawn_point)
                if vehicle is not None:
                    vehicle.set_simulate_physics(True)
                    vehicle.set_autopilot()
                    self.zombie_cars.append(vehicle)
                    count -= 1
        blueprints = self.world.get_blueprint_library().filter('vehicle.*')

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.vehicle.get_world().set_weather(preset[0])

    def apply_control(self, control):
        assert self.vehicle is not None
        for i in range(self.A_skip):
            self.vehicle.apply_control(control)
            self.next_frame()
            self.death_step += 1

    def late_init(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = True if self.sync else False
        self.world.apply_settings(settings)
        self._map = self.world.get_map()
        self._map_name = self._map.name
        self.fixed_spawn_points = self._map.get_spawn_points()
        bad_points = [12, 55, 42, 70, 71, 56, 63, 53, 54, 64, 68, 69, 86, 87, 94, 95, 105, 123, 124, 65, 67, 51, 121,
                      109]
        # Some waypoints cannot learn. Delete them: These are marked red in no_Rendering mode
        self.fixed_spawn_points = [ele for ele in self.fixed_spawn_points if
                                   self.fixed_spawn_points.index(ele) not in bad_points]
        if self.other_cars > 0:
            self.fixed_spawn_points = [ele for ele in self.fixed_spawn_points if
                                       self.fixed_spawn_points.index(ele) in DOTTED_SPAWN_POINTS]

        # setup start position
        self.spawn_idx = 0
        self.spawn_mode = "fixed"
        self.vehicle = None
        if self.scenario:
            self.scenario_init(self.scenario_name, self._map, self.world)
        else:
            self.select_hero_actor()

        control = self.vehicle.get_control()
        control.steer, control.throttle, control.brake = 0., 0., 0.
        self.vehicle.apply_control(control)

        self.car_shape = [self.vehicle.bounding_box.extent.x, self.vehicle.bounding_box.extent.y,
                          self.vehicle.bounding_box.extent.z]
        self.car_length = self.car_shape[0]
        self.car_width = self.car_shape[1]
        self.car_height = self.car_shape[2]

        # setup sensors and collision detection
        #self.collision_sensor = CollisionSensor(self.vehicle, self.hud)
        # self.lane_invasion_sensor = LaneInvasionSensor(self.vehicle, self.hud)
        if self._camera:
            self.camera_manager = CameraManager(self.vehicle, self.hud)
            self.camera_manager.set_sensor(0, notify=False)
        self._map = self.world.get_map()
        self._weather_presets = find_weather_presets()
        self._weather_index = 0

        self.last_status = int(self.max_lanes / 2)

        # setup history for debugging
        self.history = deque()

        # stats
        self.rew_now = 0.
        self.v_rew = 0.
        self.ang_rew = 0.
        self.track_rew = 0.
        self.dis_now = 100.
        self.ep_rew = 0.0
        self.ep_len = 0
        self.ep_rew_buffer = deque([], 100)
        self.ep_len_buffer = deque([], 100)
        self.ang = 0

        # 70 meters for visual
        self._maxlen = 701
        sys.setrecursionlimit(self._maxlen + 300)

        self._waypoints_queue_l = [deque(maxlen=self._maxlen)]
        self._waypoints_pos_queue_l = [deque(maxlen=self._maxlen)]
        self._lanepoints_pos_queue_l = [deque(maxlen=self._maxlen)]
        # other lanes
        self._global_waypoints_queue_l = [None for i in range(self.max_lanes)]
        self._global_waypoints_pos_queue_l = [None for i in range(self.max_lanes)]
        self._global_lanepoints_pos_queue_l = [None for i in range(self.max_lanes)]
        self._global_current_waypoint_pos_car = [None for i in range(self.max_lanes)]
        self._global_reflines = [None for i in range(self.max_lanes)]
        self.shifts = []
        shift = int(self.max_lanes / 2) * -3.5
        for i in range(self.max_lanes):
            self._global_waypoints_queue_l[i] = [deque([], maxlen=self._maxlen)]
            self._global_waypoints_pos_queue_l[i] = [deque([], maxlen=self._maxlen)]
            self._global_lanepoints_pos_queue_l[i] = [deque([], maxlen=self._maxlen)]
            self._global_reflines[i] = [deque([], maxlen=self._maxlen)]
            self.shifts.append(shift)
            shift = shift + 3.5

        self.zombie_cars, self.visible_zombie_cars = [], []
        # reading from world have to happen after world tick. Otherwise it gets nothing
        # self.zombie_cars = self.world.get_actors().filter('vehicle.*')
        if self.scenario:
            self.zombie_cars = self.scenario_now.zombie_cars
            pass
            # for actor in self.zombie_cars:
            #    actor.set_autopilot()
        else:
            self.zombie_cars = [actor for actor in self.world.get_actors() if
                                'vehicle' in actor.type_id and actor != self.vehicle]
        self.zombie_car_ref_pos = deque([], maxlen=self.other_cars)

        self.world.on_tick(self.on_world_tick)
        try:
            self.next_frame()
        except:
            self.force_restart()

    def force_restart(self):
        try:
            self.destroy()
            print("FORCE RESTART DESTROY")
        except:
            print("FORCE RESTART DESTROY FAILED")
        if self.scenario:
            self.scenario_now._remove_zombie_cars()
        client = carla.Client(self.host, self.port)
        client.set_timeout(600.0)
        self.world = client.get_world()
        self._map = self.world.get_map()
        self.late_init()

    def next_frame(self):
        self.clock.tick_busy_loop(60)
        if self.sync:
            self.world.tick()
            self.world.wait_for_tick(seconds=60.0)
        if self.scenario:
            self.scenario_now._update()
        obs = self._observation()
        # autopilot does not update reward
        if self.autopilot:
            # self._reward()
            pass
        if self._camera:
            self.hud.tick(self, self.timestamp)
            self._render()
        return obs

    def clear_and_next_frame(self):
        self._waypoints_queue_l = [deque([], maxlen=self._maxlen)]
        self._waypoints_pos_queue_l = [deque([], maxlen=self._maxlen)]
        self._lanepoints_pos_queue_l = [deque([], maxlen=self._maxlen)]
        if self.autopilot:
            self.right_branch = 0
            self.just_merged = False
            self.past_intersect = False
        self.next_frame()

    def on_world_tick(self, timestamp):
        """
        Call back function for Carla World tick:
        """
        self.delta_time = timestamp.delta_seconds
        self.timestamp = timestamp

    def _render(self):
        if self.render and self._camera:
            if self.camera_manager._draw_waypoints:
                self._draw_waypoints()
            self.camera_manager.render(self.display)
            self.hud.render(self.display)
            pygame.display.flip()

    def destroy(self):
        actors = [
            #self.collision_sensor.sensor,
            # self.lane_invasion_sensor.sensor,
            self.vehicle]
        if self._camera:
            actors.append(self.camera_manager.sensor)  # failed to destroy sometimes
        for actor in actors:
            if actor.is_alive:
                actor.destroy()

        # destroy zombie cars
        if len(self.zombie_cars) > 0 and not self.scenario:
            self._destroy_other_cars()

        # Remaining Content in history indicates that cannot tell where car is going
        # So naturally just flush them.
        self.history.clear()

    def destroy_agent(self):
        actors = [
            #self.collision_sensor.sensor,
            # self.lane_invasion_sensor.sensor,
            self.vehicle]
        if self._camera:
            actors.append(self.camera_manager.sensor)
        for actor in actors:
            if actor.is_alive:
                actor.destroy()

        # Remaining Content in history indicates that cannot tell where car is going
        # So naturally just flush them.
        self.history.clear()

    def collision(self, extension_factor=1, margin=1.15):
        """
        This function identifies if an obstacle is present in front of the reference actor
        """
        # world = CarlaDataProvider.get_world()
        actor = self.vehicle
        world_actors = self.zombie_cars
        actor_bbox = actor.bounding_box
        actor_transform = actor.get_transform()
        actor_location = actor_transform.location
        actor_vector = actor_transform.rotation.get_forward_vector()
        actor_vector = np.array([actor_vector.x, actor_vector.y])
        actor_vector = actor_vector / np.linalg.norm(actor_vector)
        actor_vector = actor_vector * (extension_factor - 1) * actor_bbox.extent.x
        actor_location = actor_location + carla.Location(actor_vector[0], actor_vector[1])
        actor_yaw = actor_transform.rotation.yaw
    
        is_hazard = False
        for adversary in world_actors:
            if adversary.id != actor.id and \
                    actor_transform.location.distance(adversary.get_location()) < 50:
                adversary_bbox = adversary.bounding_box
                adversary_transform = adversary.get_transform()
                adversary_loc = adversary_transform.location
                adversary_yaw = adversary_transform.rotation.yaw
                overlap_adversary = RotatedRectangle(
                    adversary_loc.x, adversary_loc.y,
                    2 * margin * adversary_bbox.extent.x, 2 * margin * adversary_bbox.extent.y, adversary_yaw)
                overlap_actor = RotatedRectangle(
                    actor_location.x, actor_location.y,
                    2 * margin * actor_bbox.extent.x * extension_factor, 2 * margin * actor_bbox.extent.y, actor_yaw)
                overlap_area = overlap_adversary.intersection(overlap_actor).area
                if overlap_area > 0:
                    is_hazard = True
                    break
        return is_hazard

    def _observation(self):
        self.yaw = self.vehicle.get_transform().rotation.yaw
        self.pitch = self.vehicle.get_transform().rotation.pitch
        self.roll = self.vehicle.get_transform().rotation.roll

        def a2r(angle):
            return angle / 180 * np.pi

        self.yaw_radians = a2r(self.yaw)  # fxxk
        self.pitch_radians = a2r(self.pitch)
        self.roll_radians = a2r(self.roll)
        R_X = np.array([[1, 0, 0], \
                        [0, np.cos(self.roll_radians), np.sin(self.roll_radians)], \
                        [0, -np.sin(self.roll_radians), np.cos(self.roll_radians)]])
        R_Y = np.array([[np.cos(self.pitch_radians), 0, -np.sin(self.pitch_radians)], \
                        [0, 1, 0], \
                        [np.sin(self.pitch_radians), 0, np.cos(self.pitch_radians)]])
        R_Z = np.array([[np.cos(self.yaw_radians), np.sin(self.yaw_radians), 0], \
                        [-np.sin(self.yaw_radians), np.cos(self.yaw_radians), 0], \
                        [0, 0, 1]])
        R = np.dot(np.dot(R_Z, R_Y), R_X)

        def _rotate_car(pos_world):
            pos_tran = (pos_world[0] * np.cos(self.yaw_radians) + pos_world[1] * np.sin(self.yaw_radians),
                        -pos_world[0] * np.sin(self.yaw_radians) + pos_world[1] * np.cos(self.yaw_radians))
            return pos_tran

        def _rotate_car3d(pos_world):
            pos_world = np.array(pos_world)
            pos_tran = np.dot(R, pos_world)
            return pos_tran

        def _transform_car(pos_ego, pos_actor):
            pos_ego_tran = _rotate_car(pos_ego)
            pos_actor_tran = _rotate_car(pos_actor)
            pos = [pos_actor_tran[0] - pos_ego_tran[0], pos_actor_tran[1] - pos_ego_tran[1]]
            return pos

        def _transform_car3d(pos_ego, pos_actor):
            pos_ego_tran = _rotate_car3d(pos_ego)
            pos_actor_tran = _rotate_car3d(pos_actor)
            pos = [pos_actor_tran[0] - pos_ego_tran[0], pos_actor_tran[1] - pos_ego_tran[1],
                   pos_actor_tran[2] - pos_ego_tran[2]]
            return pos

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

        def _dis(a, b):
            return ((b[1] - a[1]) ** 2 + (b[0] - a[0]) ** 2) ** 0.5

        def _dis3d(a, b):
            return ((b[1] - a[1]) ** 2 + (b[0] - a[0]) ** 2 + (b[2] - a[2]) ** 2) ** 0.5

        def _location(x):
            from carla import Location
            return Location(x[0], x[1], 0)

        def _cos(vector_a, vector_b):
            ab = vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1]
            a_b = (vector_a[0] ** 2 + vector_a[1] ** 2) ** 0.5 * (vector_b[0] ** 2 + vector_b[1] ** 2) ** 0.5 + 1e-8
            return ab / a_b

        def _cos3d(vector_a, vector_b):
            ab = vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1] + vector_a[2] * vector_b[2]
            a_b = (vector_a[0] ** 2 + vector_a[1] ** 2 + vector_a[2] ** 2) ** 0.5 * (
                        vector_b[0] ** 2 + vector_b[1] ** 2 + vector_b[2] ** 2) ** 0.5 + 1e-8
            return ab / a_b

        # choose function related to dimensions
        if self.dim == '3d':
            _pos = _pos3d
            _transform_car = _transform_car3d
            _dis = _dis3d
            _rotate_car = _rotate_car3d
        elif self.dim == '2d':
            pass
        else:
            raise NotImplementedError

        def _retrieve_options(list_waypoints, current_waypoint):
            """
            Compute type of roads option that current_waypoint and
            each of the wp in list_waypoint forms
            """
            options = []
            for p in list_waypoints:
                next_next_waypoint = p.next(3.0)[0]
                n = next_next_waypoint.transform.rotation.yaw
                n = n % 360.0
                c = current_waypoint.transform.rotation.yaw
                c = c % 360.0
                diff_angle = (n - c) % 180.0
                if diff_angle < 1.0:
                    link = RoadOption.STRAIGHT
                elif diff_angle > 90.0:
                    link = RoadOption.LEFT
                else:
                    link = RoadOption.RIGHT
                options.append(link)
            return options

        def _is_ahead(wp, target_pos):
            """
            Test if a target pos is ahead of the waypoint
            """
            wp_pos = _pos(wp)
            orientation = math.radians(wp.transform.rotation.yaw)
            target_vector = np.array([target_pos[0] - wp_pos[0], target_pos[1] - wp_pos[1]])
            forward_vector = np.array([np.cos(orientation), np.sin(orientation)])
            d_angle = math.degrees(math.acos(_cos(forward_vector, target_vector)))
            return d_angle < 90

        def _lateral_shift(transform, shift):
            transform.rotation.yaw += 90
            return transform.location + shift * transform.get_forward_vector()

        def _total_lanes(wp):
            # obtain the numbers of the lanes
            left_lanes = 0
            right_lanes = 0
            d = 3.5
            while True:
                left_lanes_tmp = left_lanes + 1
                shift = left_lanes_tmp * d
                w_l = self._map.get_waypoint(_lateral_shift(wp.transform, shift * -1.1), project_to_road=False)
                if w_l is None or w_l.road_id != wp.road_id:
                    break
                elif (w_l.lane_id * wp.lane_id < 0) or w_l.lane_id == wp.lane_id:
                    break
                else:
                    if w_l.lane_change.values[0] == w_l.lane_change:
                        break
                    else:
                        left_lanes = left_lanes + 1
            while True:
                right_lanes_tmp = right_lanes + 1
                shift = right_lanes_tmp * d
                w_r = self._map.get_waypoint(_lateral_shift(wp.transform, shift * 1.1), project_to_road=False)
                if w_r is None or w_r.road_id != wp.road_id:
                    break
                elif (w_r.lane_id * wp.lane_id < 0) or w_r.lane_id == wp.lane_id:
                    break
                else:
                    if w_r.lane_change.values[0] == w_r.lane_change:
                        break
                    else:
                        right_lanes = right_lanes + 1
            return left_lanes, right_lanes

        def _find_lane_pos(wp):
            """
            Find lane point position given a lane point
            """
            ang = math.radians(wp.transform.rotation.yaw)
            sin_alpha, cos_alpha = np.sin(ang), np.cos(ang)
            pos = _pos(wp)
            x, y, z = pos[0], pos[1], pos[2]
            d = 3.5 / 2.0

            # The immedia left and right
            lane_pos_right_world = [x - d * sin_alpha, y + d * cos_alpha, z]
            lane_pos_left_world = [x + d * sin_alpha, y - d * cos_alpha, z]

            # The solid lanes
            d = 3.5
            total_left_lanes, total_right_lanes = _total_lanes(wp)

            d_left = (total_left_lanes + 0.5) * d
            d_right = (total_right_lanes + 0.5) * d
            solid_lane_pos_left_world = [x + d_left * sin_alpha, y - d_left * cos_alpha, z]
            solid_lane_pos_right_world = [x - d_right * sin_alpha, y + d_right * cos_alpha, z]

            return [lane_pos_left_world, lane_pos_right_world, solid_lane_pos_left_world, solid_lane_pos_right_world]

        def _purge_obsolete(wp_q, wp_pos_q, lp_q):
            """
            Purge the waypoins in a que that is after the car
            Note here we assume wp_q holds the RoadOption as well
            """
            if len(wp_q) > 1:
                wp = wp_q[0][0]
                while _is_ahead(wp, current_pos):
                    wp_q.popleft()
                    wp_pos_q.popleft()
                    lp_q.popleft()
                    wp = wp_q[0][0]
            return wp_q, wp_pos_q, lp_q

        interval = self._visual / self._waypoints_queue_l[0].maxlen

        def _fill_future_recursive(wp_q, wp_pos_q, lp_q):
            '''
            Used when autopilot is set. Shall split into more
            Input: 3 ques with initial values: waypoint_que,waypoint_pos_que,lanepoints_pos_q
            Output: 3 lists containing all branched out deques.
            '''
            assert len(wp_q) > 0
            if len(wp_q) == wp_q.maxlen:  # basecase
                return [wp_q], [wp_pos_q], [lp_q]

            else:  # recursive case
                last_waypoint = wp_q[-1][0]
                next_waypoints = list(last_waypoint.next(interval))
                road_options = _retrieve_options(next_waypoints, last_waypoint)
                if len(next_waypoints) == 1:  # single choice; seperate it to save copy space
                    wp_q.append((next_waypoints[0], road_options[0]))
                    wp_pos_q.append(_pos(next_waypoints[0]))
                    lp_q.append(_find_lane_pos(next_waypoints[0]))
                    return _fill_future_recursive(wp_q, wp_pos_q, lp_q)

                elif self.past_intersect:
                    if self.farther_features == 'rand':  # randomly choose after second intersection
                        return _fill_future(wp_q, wp_pos_q, lp_q)
                    elif self.farther_features == 'end':  # features cannot extend beyond seoncd intersection
                        while len(wp_q) < wp_q.maxlen:
                            wp_q.append(wp_q[-1])
                            wp_pos_q.append(wp_pos_q[-1])
                            lp_q.append(lp_q[-1])
                        return [wp_q], [wp_pos_q], [lp_q]

                else:  # multple choice as the first intersection
                    self.past_intersect = True
                    wp_q_l, wp_pos_q_l, lp_q_l = [], [], []
                    for i in range(len(next_waypoints)):
                        n_wp_q = copy.copy(wp_q)
                        n_wp_pos_q = copy.copy(wp_pos_q)
                        n_lp_q = copy.copy(lp_q)
                        n_wp_q.append((next_waypoints[i], road_options[i]))
                        n_wp_pos_q.append(_pos(next_waypoints[i]))
                        n_lp_q.append(_find_lane_pos(next_waypoints[i]))
                        n_wp_q, n_wp_pos_q, n_lp_q = _fill_future_recursive(n_wp_q, n_wp_pos_q, n_lp_q)
                        wp_q_l.extend(n_wp_q)
                        wp_pos_q_l.extend(n_wp_pos_q)
                        lp_q_l.extend(n_lp_q)
                    return wp_q_l, wp_pos_q_l, lp_q_l

        def _fill_future(wp_q, wp_pos_q, lp_q):
            """
            Used during training. When there is an intersection, randomly select one to follow
            Duing behavioral cloning, the policy only need to make choices. No need to interact with env
            """
            assert len(wp_q) > 0
            # import pdb; pdb.set_trace()
            while len(wp_q) < wp_q.maxlen:
                last_waypoint = wp_q[-1][0]
                next_waypoints = list(last_waypoint.next(interval))

                if len(next_waypoints) == 1:
                    next_waypoint = next_waypoints[0]
                    road_option = RoadOption.LANEFOLLOW
                else:
                    road_options_list = _retrieve_options(next_waypoints, last_waypoint)
                    if hasattr(self, "road_option"):
                        road_option = self.road_option
                    else:
                        if self.scenario:
                            if self.scenario_name == 'Cross_Join':
                                road_option = RoadOption(2)  # random.choice(road_options_list)
                            elif self.scenario_name == 'Ring_Join':
                                road_option = road_options_list[1]  # random.choice(road_options_list)
                            elif self.scenario_name == 'Straight_Follow_Single':
                                road_option = RoadOption(2)  # road_options_list[1]  # random.choice(road_options_list)
                            elif self.scenario_name == 'Straight_Follow_Double':
                                road_option = RoadOption(3)  # road_options_list[1]  # random.choice(road_options_list)
                            elif self.scenario_name == 'Cross_Follow':
                                road_option = RoadOption(3)
                            elif self.scenario_name == 'Cross_Turn_Left':
                                road_option = road_options_list[0]  # RoadOption(1)
                            elif self.scenario_name == 'Cross_Turn_Right':
                                road_option = RoadOption(2)
                            else:
                                print('\nself.scenario-self.scenario_name: ', self.scenario_name, 'wp_q.maxlen: ', wp_q.maxlen)
                        else:
                            print('\nself.scenario_name: ', self.scenario_name)
                            road_option = random.choice(road_options_list)
                        # road_option = random.choice(road_options_list)
                    next_waypoint = next_waypoints[road_options_list.index(road_option)]
                    # if self.scenario_name == 'Cross_Turn_Left':
                    #     next_waypoint = next_waypoints[2]

                    self.intersection_wp = next_waypoint  # record the nearest waypoint

                if _pos(next_waypoint)[2] != 0 and self.scenario_name == 'Cross_Turn_Left' and self.scenario:
                    wp_q[-1] = (next_waypoint, road_option)
                    wp_pos_q[-1] = _pos(next_waypoint)
                    lp_q[-1] = _find_lane_pos(next_waypoint)
                    continue

                wp_q.append((next_waypoint, road_option))
                wp_pos_q.append(_pos(next_waypoint))
                lp_q.append(_find_lane_pos(next_waypoint))
            return [wp_q], [wp_pos_q], [lp_q]

        def _update_wp_que(wp_q_l, wp_pos_q_l, lp_q_l):
            """
            wp_q_s_l: waypoint_que_secondary_list: an intermediate result. The list that is returned from _fill_future_recursive
            Input: 3 lists
            Output: 3 lists (expert data) / 3 deques (training)

            """
            wp_q_f_l, wp_pos_q_f_l, lp_q_f_l = [], [], []
            for i in range(len(wp_q_l)):  # update all existing wp_ques inside list given
                # purge obsolete
                wp_q_l[i], wp_pos_q_l[i], lp_q_l[i] = _purge_obsolete(wp_q_l[i], wp_pos_q_l[i], lp_q_l[i])
                # update ques
                if self.autopilot:
                    wp_q_s_l, wp_pos_q_s_l, lp_q_s_l = _fill_future_recursive(wp_q_l[i], wp_pos_q_l[i], lp_q_l[i])
                else:
                    wp_q_s_l, wp_pos_q_s_l, lp_q_s_l = _fill_future(wp_q_l[i], wp_pos_q_l[i], lp_q_l[i])
                wp_q_f_l += wp_q_s_l
                wp_pos_q_f_l += wp_pos_q_s_l
                lp_q_f_l += lp_q_s_l
            return wp_q_f_l, wp_pos_q_f_l, lp_q_f_l

        def _downsize(wp_q_l, wp_pos_q_l, lp_q_l):
            """
            Input given three lists, downsize it using slice to form features
            """
            if self.obs_wp == 'equal':  # equal distance
                maxlen = wp_q_l.maxlen
                _slice = tuple([range(0, maxlen, int(maxlen / self._visual))])
            elif self.obs_wp == 'exp':  # exponentially decaying
                _slice = tuple([np.array([int((1.3 ** i) / self._visual * self._maxlen) for i in range(1, 17)])])
                _slice_equal = tuple([range(0, self._maxlen, int(self._maxlen / self._visual))])
            else:
                raise NotImplementedError

            waypoints_forward, waypoints_pos_forward_world, lanepoints_pos_forward_world = [], [], []
            self.waypoints_queue_equal_l = []

            for i in range(len(wp_q_l)):
                # import pdb; pdb.set_trace()
                # print('i: {0}, len(wp_q_l):{1}, wp_q_l[i]:{2}, _slice: {3}, self.scenario_name:{4}'.format(i, len(wp_q_l), len(wp_q_l[i]), _slice, self.scenario_name))
                waypoints_forward.append(np.array(wp_q_l[i])[_slice])
                self.waypoints_queue_equal_l.append(np.array(wp_q_l[i])[_slice_equal])
                waypoints_pos_forward_world.append(np.array(wp_pos_q_l[i])[_slice])
                # Now landpoints_pos_forward_world is in shape like
                # [[[leftx,lefty,leftz],[rightx,righty,rightz]],[...],[...]]. Have to downsize one dimension
                lanepoints_pos_forward = np.array(lp_q_l[i])[_slice]
                lanepoints_pos_forward_world.append(np.reshape(lanepoints_pos_forward, (-1, 3)))

            return waypoints_forward, waypoints_pos_forward_world, lanepoints_pos_forward_world

        # the functions below must be called after the assignment for current_wp
        def _to_intersection():
            if hasattr(self, "intersection_wp"):
                intersection_pos = _pos3d(self.intersection_wp)
                if _is_ahead(self.current_wp, intersection_pos):
                    return _dis3d(_pos3d(self.intersection_wp), _pos3d(self.current_wp))
                else:
                    return 70.
            else:
                return 70.

        def _valid_shift(wp_select_shift, wp_select):
            return not (wp_select_shift.lane_id * wp_select.lane_id < 0 or wp_select_shift.lane_id == wp_select.lane_id \
                        or wp_select_shift.road_id != wp_select.road_id)

        def _existed_and_valid(current_waypoint, shift):
            if shift == 0.:
                return True, current_waypoint
            else:
                current_waypoint_shift = self._map.get_waypoint(_lateral_shift(current_waypoint.transform, shift * 1.1))
                return _valid_shift(current_waypoint_shift, current_waypoint), current_waypoint_shift

        def _generate_wp_features(current_waypoint_shift, current_pos, wps_queue, wps_pos_queue, lane_pos_queue):
            current_waypoint_pos_shift_car = _transform_car(current_pos, _pos(current_waypoint_shift))
            # # catch extreme case where autopilot does not follow the waypoint under big angle
            # # if 1 branch only but past intersect is true, Then just merged and need to re-see
            if (len(wps_pos_queue) == 0) or (len(wps_queue) == 1 and self.autopilot and self.just_merged):
                wps_pos_queue = [deque(maxlen=self._maxlen)]
                wps_queue = [deque(maxlen=self._maxlen)]
                lane_pos_queue = [deque(maxlen=self._maxlen)]
            if len(wps_queue[0]) == 0:
                wps_queue[0].append((current_waypoint_shift, RoadOption.LANEFOLLOW))
                wps_pos_queue[0].append(_pos(current_waypoint_shift))
                lane_pos_queue[0].append(_find_lane_pos(current_waypoint_shift))
            wps_queue, wps_pos_queue, lane_pos_queue = \
                _update_wp_que(wps_queue, wps_pos_queue, lane_pos_queue)
            # Downsize ques and pick features # Note waypoints_forward_l is still holds the RoadOption Information
            wps_forward, wps_pos_forward_world, lanepoints_pos_forward_world = \
                _downsize(wps_queue, wps_pos_queue, lane_pos_queue)
            # generate car reference lane positions
            waypoints_pos_forward_car = [[_transform_car(current_pos, pos) for pos in wp_f_w] for wp_f_w in
                                         wps_pos_forward_world]
            lane_poss_car = [[_transform_car(current_pos, pos) for pos in lp_p_f_w] for lp_p_f_w in
                             lanepoints_pos_forward_world]
            return current_waypoint_pos_shift_car, wps_queue, wps_pos_queue, lane_pos_queue

        def _clear_global_map(idx):
            end_idx = len(self._global_waypoints_queue_l[idx])
            for i in range(end_idx):
                self._global_waypoints_queue_l[idx][i].clear()
                self._global_waypoints_pos_queue_l[idx][i].clear()
                self._global_lanepoints_pos_queue_l[idx][i].clear()

        def _update_reflines(idx):
            self._global_reflines[idx] = [[_transform_car(current_pos, pos) for pos in wp_f_w] for wp_f_w in
                                          self._global_waypoints_pos_queue_l[idx]]

        def _update_lanes(idx_global, shift):
            update, current_waypoint_shift = _existed_and_valid(current_waypoint, shift)
            idx = idx_global
            if update:
                wps_queue, wps_pos_queue, lane_pos_queue = self._global_waypoints_queue_l[idx], \
                                                           self._global_waypoints_pos_queue_l[idx], \
                                                           self._global_lanepoints_pos_queue_l[idx]
                self._global_current_waypoint_pos_car[idx], self._global_waypoints_queue_l[idx], \
                self._global_waypoints_pos_queue_l[idx], self._global_lanepoints_pos_queue_l[idx] = \
                    _generate_wp_features(current_waypoint_shift, current_pos, wps_queue, wps_pos_queue, lane_pos_queue)
                _update_reflines(idx)
            else:
                _clear_global_map(idx)

        def _parse_status():
            # return the lane
            wp = self.current_wp
            for i in range(self.max_lanes):
                if len(self._global_waypoints_queue_l[i][0]) > 0 and self._global_waypoints_queue_l[i][0][0][
                    0].road_id == wp.road_id \
                        and self._global_waypoints_queue_l[i][0][0][
                    0].lane_id == wp.lane_id:  # and self._global_waypoints_queue_l[i][0][0][0].lane_change == wp.lane_change:
                    return i
                if len(self._global_waypoints_queue_l[i][0]) > 10 and self._global_waypoints_queue_l[i][0][10][
                    0].road_id == wp.road_id \
                        and self._global_waypoints_queue_l[i][0][10][
                    0].lane_id == wp.lane_id:  # and self._global_waypoints_queue_l[i][0][0][0].lane_change == wp.lane_change:
                    return i
            #return int(self.max_lanes // 2)
            return self.last_status

        current_pos = self.current_pos = _pos(self.vehicle)
        current_waypoint = self._map.get_waypoint(self.vehicle.get_location())

        self.last_waypoint = current_waypoint
        self.current_wp = current_waypoint

        # -------------------------------------------------------------------------
        # For debugging
        # -------------------------------------------------------------------------
        self.current_laneid = self.current_wp.lane_id
        self.current_lanetype = self.current_wp.lane_type
        self.current_lanechange = self.current_wp.lane_change
        self.current_roadid = self.current_wp.road_id

        mid = int(self.max_lanes // 2)

        if len(self._global_waypoints_queue_l[mid - 1][0]) > 0:
            self.left_roadid = self._global_waypoints_queue_l[mid - 1][0][0][0].road_id
            self.left_laneid = self._global_waypoints_queue_l[mid - 1][0][0][0].lane_id
            self.left_lanetype = self._global_waypoints_queue_l[mid - 1][0][0][0].lane_type
            self.left_lanechange = self._global_waypoints_queue_l[mid - 1][0][0][0].lane_change

        if len(self._global_waypoints_queue_l[mid + 1][0]) > 0:
            self.right_roadid = self._global_waypoints_queue_l[mid + 1][0][0][0].road_id
            self.right_laneid = self._global_waypoints_queue_l[mid + 1][0][0][0].lane_id
            self.right_lanetype = self._global_waypoints_queue_l[mid + 1][0][0][0].lane_type
            self.right_lanechange = self._global_waypoints_queue_l[mid + 1][0][0][0].lane_change

        if len(self._global_waypoints_queue_l[mid][0]) > 0:
            self.mid_roadid = self._global_waypoints_queue_l[mid][0][0][0].road_id
            self.mid_laneid = self._global_waypoints_queue_l[mid][0][0][0].lane_id
            self.mid_lanetype = self._global_waypoints_queue_l[mid][0][0][0].lane_type
            self.mid_lanechange = self._global_waypoints_queue_l[mid][0][0][0].lane_change

        # In case of expert collect data, We are keeping all data if encoutering intersection
        # Thus, should remove the branched out sequence and save if it is passed the intersection
        # Notice that Using road ID to test does not work because of 1. Big Roudn Turn. 2. Two consecutive short turns.
        if self.autopilot and len(self._waypoints_queue_l[0]) > 0:
            branches = len(self._waypoints_queue_l)
            branch_to_del_from = [i for i in range(branches)]
            for i in range(branches):
                first_wp = self._waypoints_queue_l[i][0][0]
                road_yaw = first_wp.transform.rotation.yaw % 360.0
                yaw = self.yaw % 360.0
                deg = abs(yaw - road_yaw) % 360.0
                if deg > 30 and deg < 330:
                    branch_to_del_from.remove(i)

            self.just_merged = False
            self.past_intersect = False
            if len(self._waypoints_queue_l) > 1:
                self.past_intersect = True

            # make sure branches are deleted simultaneously
            # When current waypoint is at intersection, If it deletes all branches, then it may
            # generate another branch that we do not want at all at another direction starting from this point
            # Thus here we test the one branch case as well
            if len(branch_to_del_from) < 2:
                for i in sorted(range(branches), reverse=True):
                    if i not in branch_to_del_from:
                        del self._waypoints_queue_l[i]
                        del self._waypoints_pos_queue_l[i]
                        del self._lanepoints_pos_queue_l[i]
                self.right_branch = branch_to_del_from[0] if len(branch_to_del_from) == 1 else 0
                self.just_merged = True if branches > 1 else False
                self.past_intersect = False

        # -----------------------------------------------------------------------------------------------------------------
        # store waypoints if left-lane or right-lane is a dotted lane
        left_lanes, right_lanes = _total_lanes(self.current_wp)

        self.left_lanes, self.right_lanes = left_lanes, right_lanes
        self.status = status = _parse_status()

        # update left lanes observation
        if left_lanes >= 1:
            valid_left_lanes_idx = [[i, status - i] for i in range(1, left_lanes + 1)]
            for idx, idx_global in valid_left_lanes_idx:
                shift = -3.5 * idx
                try:
                    _update_lanes(idx_global, shift)
                except:
                    pass
        # update middle lane
        try:
            _update_lanes(status, 0.)
        except:
            pass
        # udpate right lanes observation
        if right_lanes >= 1:
            valid_right_lanes_idx = [[i, status + i] for i in range(1, right_lanes + 1)]
            for idx, idx_global in valid_right_lanes_idx:
                shift = 3.5 * idx
                try:
                    _update_lanes(idx_global, shift)
                except:
                    pass
        self.left_lanes, self.right_lanes = left_lanes, right_lanes
        # Downsize ques and pick features # Note waypoints_forward_l is still holds the RoadOption Information
        self._waypoints_queue_l, self._waypoints_pos_queue_l, self._lanepoints_pos_queue_l = \
            self._global_waypoints_queue_l[status], self._global_waypoints_pos_queue_l[status], \
            self._global_lanepoints_pos_queue_l[status]
        # Downsize ques and pick features # Note waypoints_forward_l is still holds the RoadOption Information
        self.waypoints_forward_l, self.waypoints_pos_forward_world_l, self.lanepoints_pos_forward_world_l = \
            _downsize(self._waypoints_queue_l, self._waypoints_pos_queue_l, self._lanepoints_pos_queue_l)
        # -----------------------------------------------------------------------------------------------------------------

        self.to_intersection = _to_intersection()

        nearest_waypoint = self.waypoints_forward_l[0][0][0]
        nearest_waypoint_pos_world = _pos(nearest_waypoint)

        self.waypoint_vec = [nearest_waypoint_pos_world[0] - self.current_pos[0],
                             nearest_waypoint_pos_world[1] - self.current_pos[1],
                             nearest_waypoint_pos_world[2] - self.current_pos[2]]

        # Generate car reference frame
        waypoints_pos_forward_car = [[_transform_car(current_pos, pos) for pos in wp_f_w] for wp_f_w in
                                     self.waypoints_pos_forward_world_l]
        self.lane_poss_car = [[_transform_car(current_pos, pos) for pos in lp_p_f_w] for lp_p_f_w in
                              self.lanepoints_pos_forward_world_l]

        self.lane_pos0_car_left, self.lane_pos0_car_right = self.lane_poss_car[0][0][1], self.lane_poss_car[0][1][1]
        self.lane_pos0_solid_car_left, self.lane_pos0_solid_car_right = self.lane_poss_car[0][2][1], \
                                                                        self.lane_poss_car[0][3][1]
        self.lane_pos0_dotted_car_left, self.lane_pos0_dotted_car_right = self.lane_poss_car[0][0][1], \
                                                                        self.lane_poss_car[0][1][1]
        self.current_waypoint_pos_car = _transform_car(current_pos, _pos(current_waypoint))
        self.lane_width = current_waypoint.lane_width
        self.road_direction = self._waypoints_queue_l[0][0][1]

        #  Gather Road Structure Info
        # road_structure = [self.road_direction, current_roadid, current_laneid, next_roadid, intersection, 0] # fixed crossed_lane_markings
        road_structure = []
        for wp_q in self.waypoints_forward_l:
            road_info = [wp_q[0][1], current_waypoint.road_id, current_waypoint. \
                lane_id, wp_q[1][0].road_id, current_waypoint.is_intersection]
            road_structure.append(road_info)

        def _idx(values, target):
            for idx, value in values.items():
                if value == target:
                    return idx

        # search the nearest traffic lights
        def _search_light(goal, trafficlights):
            dis_to_lights = []
            goal = goal.next(5)[0]
            goal_pos = _pos(goal)
            for i in range(len(trafficlights)):
                light_pos = _pos(trafficlights[i])
                dis_to_lights.append(_dis(goal_pos, light_pos))
            min_id = dis_to_lights.index(min(dis_to_lights))
            return trafficlights[min_id]

        if self.all_green_light:
            if self.vehicle.is_at_traffic_light():
                tl = self.vehicle.get_traffic_light()
                if tl is not None:
                    for x in tl.get_group_traffic_lights():
                        x.freeze(True)
                        x.set_state(carla.TrafficLightState.Green)

        nearest_trafficlight_state = 0
        road_structure.append(nearest_trafficlight_state)

        # agent car's information
        def _get_v(_object):
            return (_object.get_velocity().x, _object.get_velocity().y)

        def _get_a(_object):
            return (_object.get_acceleration().x, _object.get_acceleration().y)

        def _get_v_3d(_object):
            return (_object.get_velocity().x, _object.get_velocity().y, _object.get_velocity().z)

        def _get_a_3d(_object):
            return (_object.get_acceleration().x, _object.get_acceleration().y, _object.get_acceleration().z)

        def _get_r(_object):
            return (_object.get_transform().rotation.yaw, _object.get_transform().rotation.pitch,
                    _object.get_transform().rotation.roll)

        def _bound(vehicle, current_pos, fake=False):  # transform the vehicle into 8 points bounding box
            if not fake:
                vehicle_center = _transform_car(current_pos, _pos(vehicle))
                self.zombie_car_ref_pos.append(vehicle_center)
                extent_length = _pos(vehicle.bounding_box.extent)
            else:
                vehicle_center = [-70, 0, 0]
                extent_length = [0, 0, 0]

            return [(vehicle_center[0] - extent_length[0], vehicle_center[1] - extent_length[1],
                     vehicle_center[2] - extent_length[2]), \
                    (vehicle_center[0] + extent_length[0], vehicle_center[1] - extent_length[1],
                     vehicle_center[2] - extent_length[2]), \
                    (vehicle_center[0] - extent_length[0], vehicle_center[1] + extent_length[1],
                     vehicle_center[2] - extent_length[2]), \
                    (vehicle_center[0] + extent_length[0], vehicle_center[1] + extent_length[1],
                     vehicle_center[2] - extent_length[2])]

        def _norm(vec):
            return (vec[0] ** 2 + vec[1] ** 2) ** 0.5

        def _norm3d(vec):
            return (vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2) ** 0.5

        if self.dim == '2d':
            pass
        elif self.dim == '3d':
            _get_v = _get_v_3d
            _get_a = _get_a_3d
            _norm = _norm3d

        def _get_v_car(_object):
            v_world = _get_v(_object)
            v_car = _rotate_car(v_world)
            return list(v_car)

        def _get_acc_car(_object):
            acc_world = _get_a(_object)
            acc_world = _rotate_car(acc_world)
            return list(acc_world)

        def _update_zombie_cars_status():
            current_pos = _pos(self.vehicle)
            self.visible_zombie_cars.clear()
            for i in range(len(self.zombie_cars)):
                vehicle_pos = _pos(self.zombie_cars[i])
                dis = _dis(vehicle_pos, current_pos)
                if dis < 70:
                    self.visible_zombie_cars.append([dis, i])
            self.visible_zombie_cars = sorted(self.visible_zombie_cars)

        def _zombie_cars_info(current_pos, total_cars=6):
            vehicles_bounding_box, vehicles_v, vehicles_acc = [], [], []
            for index in self.visible_zombie_cars:
                vehicle = self.zombie_cars[index[1]]
                vehicles_bounding_box.append(_bound(vehicle, current_pos))
                vehicles_v.append(_get_v_car(vehicle))
                vehicles_acc.append(_get_acc_car(vehicle))
                if len(vehicles_bounding_box) >= total_cars:
                    break
            while len(vehicles_bounding_box) < total_cars:
                vehicles_bounding_box.append(_bound(None, current_pos, fake=True))
                vehicles_v.append([0, 0, 0])
                vehicles_acc.append([0, 0, 0])
            return vehicles_bounding_box, vehicles_v, vehicles_acc

        def _get_angular_v(_object):
            return (_object.get_angular_velocity().x, _object.get_angular_velocity().y, _object.get_angular_velocity().z)

        # agent car's information
        agent_velocity_world = self.agent_velocity_world = _get_v(self.vehicle)
        self.agent_velocity_car = agent_velocity_car = _rotate_car(agent_velocity_world)
        self.v_norm_world = _norm(agent_velocity_world)
        self.v_hour_world = 3.6 * self.v_norm_world
        self.agent_acc_world = agent_acc_world = _get_a(self.vehicle)
        self.agent_acc_car = agent_acc_car = _rotate_car(agent_acc_world)
        self.acc_norm_world = _norm(agent_acc_world)
        agent_steer = self.steer = self.vehicle.get_control().steer
        agent_throttle = self.throttle = self.vehicle.get_control().throttle
        agent_brake = self.brake = self.vehicle.get_control().brake
        agent_throttle_brake = self.agent_throttle_brake = agent_throttle + -1 * agent_brake
        agent_control = self.agent_control = [agent_steer, agent_throttle_brake]

        self.agent_angular_velocity = _get_angular_v(self.vehicle)

        # zombie car's feature:including bounding box, velocity, and acceleration
        # visible region for the ego car is 70 meters forward and backward
        _update_zombie_cars_status()
        zombie_cars_bounding_box, zombie_cars_v, zombie_cars_acc = _zombie_cars_info(current_pos, self.other_cars)

        object_recovery = {  # 'trafficlight': nearest_trafficlight_state,
            'agent_velocity_car': list(agent_velocity_car),
            'agent_acc_car': list(agent_acc_car),
        }
        if self.other_cars > 0:
            object_recovery.update({
                'zombie_cars_bounding_box': zombie_cars_bounding_box,
                'zombie_cars_v': zombie_cars_v,
                'zombie_cars_acc': zombie_cars_acc})
            
            # add zombie cars global positions and velocites for debug
            self.zombiecar_worldpos = [_pos(self.zombie_cars[i]) for i in range(len(self.zombie_cars))]
            self.zombiecar_worldv = [_get_v(self.zombie_cars[i]) for i in range(len(self.zombie_cars))]
            self.zombiecar_worldacc = [_get_a(self.zombie_cars[i]) for i in range(len(self.zombie_cars))]
            self.zombiecar_transform = [[self.zombie_cars[i].get_transform().rotation.yaw, self.zombie_cars[i].get_transform().rotation.pitch, self.zombie_cars[i].get_transform().rotation.roll] for i in range(len(self.zombie_cars))]
            self.zombiecar_speed = [_norm(_get_v(self.zombie_cars[i])) for i in range(len(self.zombie_cars))]
            self.zombiecar_angular_velocity = [_get_angular_v(self.zombie_cars[i]) for i in range(len(self.zombie_cars))]

        if self.autopilot and self.draw_features:
            waypoints = {'lane_pos': self.lanepoints_pos_forward_world_l}
        else:
            waypoints = {'lane_pos': self.lane_poss_car}
        if self.feature != "lane_car":
            status = _parse_status()
            waypoints['waypoints_pos_forward_car'] = self._global_waypoints_pos_forward_car[status]

        observations = {'object_recovery': object_recovery,
                        'waypoints': waypoints}

        def _flat_list(ls):
            if type(ls) == list or type(ls) == tuple:
                output = []
                for item in ls:
                    output += _flat_list(item)
                return output
            else:
                return [ls]

        def _flat(dicts, keys):
            """
            Inner keys for object recovery:
            ['agent_acc_car', 'agent_velocity_car', 'zombie_cars_acc', 'zombie_cars_bounding_box', 'zombie_cars_v']
            """
            output = []
            for key in keys:
                inner_keys = sorted(dicts[key].keys())
                for inner_key in inner_keys:
                    inner_item = dicts[key][inner_key]
                    if type(inner_item) == list or type(inner_item) == tuple or type(inner_item) == type(np.array([])):
                        output += _flat_list(inner_item)
                    else:
                        output.append(inner_item)
            return output

        def _process_output(observations, road_structure, agent_control):
            waypoints = _flat(observations, ['waypoints'])
            object_recovery = _flat(observations, ['object_recovery'])
            if self.mode == 'wp':  # Be Warry Of this Part. The IDXs that spread across all files. Is Dangerours!!!
                output = waypoints + agent_control
            elif self.mode == 'wp_obj':
                output = waypoints + object_recovery + agent_control
            elif self.mode == 'all':
                output = waypoints + object_recovery + road_structure + agent_control
            if not hasattr(logger, "wp_len"):
                logger.wp_len = len(waypoints)
                logger.obj_len = len(object_recovery)
                logger.road_len = len(road_structure)
                logger.ctrl_len = len(agent_control)

                # here are the indexes for different parts in the output
                # ['agent_acc_car', 'agent_velocity_car', 'zombie_cars_acc', 'zombie_cars_bounding_box', 'zombie_cars_v']
                obj_start_idx = len(waypoints)
                logger.egoa_idx = [obj_start_idx, obj_start_idx+3]
                logger.egov_idx = [obj_start_idx+3, obj_start_idx+6]
                logger.zombiea_idx = [obj_start_idx+6, obj_start_idx+24]
                logger.zombiev_idx = [obj_start_idx+96, obj_start_idx+114]
                logger.zombiebx_idx = [obj_start_idx+24, obj_start_idx+96]

            output = np.array(output)
            self.shape = (1, len(output))
            return output

        if not self.autopilot:  # In training everything one dimensional.
            return _process_output(observations, road_structure, agent_control)
        else:  # autopilot.
            if len(observations['waypoints'][
                       'lane_pos']) == 1 or self.just_merged:  # Either did not split, or Split and successfully merged already
                output = []
                # Draw features instead of draw lane points
                if self.render and self.draw_features:
                    self._draw_waypoints(observations['waypoints']['lane_pos'][0])
                while self.history:
                    obs, road_struct, agent_cont = self.history.popleft()
                    for key in obs['waypoints']:
                        obs['waypoints'][key] = obs['waypoints'][key][self.right_branch]
                        # Draw features instead of draw lane points
                        if self.render and self.draw_features:
                            self._draw_waypoints(obs['waypoints'][key])
                    road_struct = road_struct[self.right_branch]
                    output.append(_process_output(obs, road_struct, agent_cont))

                for key in observations['waypoints']:
                    observations['waypoints'][key] = observations['waypoints'][key][0]
                output.append(_process_output(observations, road_structure[0], agent_control))
                return output

            else:  # push it to buffer and return a False Flag
                self.history.append((observations, road_structure, agent_control))
                return False

    def _reward(self, lateral_goal=1):
        def _cos(vector_a, vector_b):
            ab = np.dot(vector_a, vector_b)
            a_b = np.linalg.norm(vector_a) * np.linalg.norm(vector_b) + 1e-10
            return ab / a_b

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

        def _dis3d(a, b):
            return ((b[1] - a[1]) ** 2 + (b[0] - a[0]) ** 2 + (b[2] - a[2]) ** 2) ** 0.5

        def _is_ahead(wp, target_pos):
            """
            Test if a target pos is ahead of the waypoint
            """
            wp_pos = _pos(wp)
            orientation = math.radians(wp.transform.rotation.yaw)
            target_vector = np.array([target_pos[0] - wp_pos[0], target_pos[1] - wp_pos[1]])
            forward_vector = np.array([np.cos(orientation), np.sin(orientation)])
            d_angle = math.degrees(math.acos(_cos(forward_vector, target_vector)))
            return d_angle < 90

        def _get_ref(status, lateral_goal):
            ref = status + lateral_goal - 1
            return ref
            # position reward

        # sigma_pos = 0.3
        sigma_pos = self.sigmas["sigma_pos"]

        # retrieve refline idx
        ref = _get_ref(self.status, lateral_goal)
        last_ref = _get_ref(self.last_status, lateral_goal)
        self.ref = ref
        if self.status == last_ref:
            ref = last_ref

        try:
            track_pos = abs(self._global_current_waypoint_pos_car[ref][1]) / self.lane_width
        except:
            track_pos = 100
        if ref < int(self.max_lanes / 2):
            # left overtaking
            scale_rew = 0.4
            scale_expected_velocity = 1.6
        elif ref == int(self.max_lanes / 2):
            scale_rew = 1.0
            scale_expected_velocity = 1.0
        else:
            # right overtaking
            scale_rew = 0.1
            scale_expected_velocity = 0.4
        self.track_rew = np.exp(-np.power(track_pos, 2) / 2 / sigma_pos / sigma_pos)

        # velocity reward
        sigma_vel_upper, sigma_vel_lower = self.sigmas["sigma_vel_upper"], self.sigmas["sigma_vel_lower"]
        # sigma_vel = 3 if self.v_abs <= self.expected_velocity else 0.6
        # sigma_vel_upper, sigma_vel_lower = 0.6, 1.0
        sigma_vel = sigma_vel_upper if self.v_norm_world <= self.expected_velocity else sigma_vel_lower
        self.v_rew = np.exp(-np.power(self.v_norm_world - self.expected_velocity * scale_expected_velocity,
                                      2) / 2 / sigma_vel / sigma_vel)

        # angle reward
        # sigma_ang = 0.4
        sigma_ang = self.sigmas["sigma_ang"]
        waypoint_vec = [self.waypoints_pos_forward_world_l[0][1][0] -
                        self.waypoints_pos_forward_world_l[0][0][0],
                        self.waypoints_pos_forward_world_l[0][1][1] - self.waypoints_pos_forward_world_l[0][0][1]]
        self.ang = math.acos(_cos(waypoint_vec, self.agent_velocity_world[:2]))
        self.ang_rew = np.exp(-np.power(self.ang, 2) / 2 / sigma_ang / sigma_ang)

        self.rew_now = self.track_rew * self.v_rew * self.ang_rew * scale_rew
        if self.collision():
            self.rew_now = -10

        self.ep_len += 1
        self.last_status = self.status
        return self.rew_now

    def _reward_cl(self):
        def _cos(vector_a, vector_b):
            ab = np.dot(vector_a, vector_b)
            a_b = np.linalg.norm(vector_a) * np.linalg.norm(vector_b) + 1e-10
            return ab / a_b

        # sigma_pos = 0.3
        sigma_pos = self.sigmas["sigma_pos"]

        ref = self.status
        try:
            track_pos = abs(self._global_current_waypoint_pos_car[ref][1]) / self.lane_width
        except:
            track_pos = 100
        self.track_rew = np.exp(-np.power(track_pos, 2) / 2 / sigma_pos / sigma_pos)

        # velocity reward
        sigma_vel_upper, sigma_vel_lower = self.sigmas["sigma_vel_upper"], self.sigmas["sigma_vel_lower"]
        # sigma_vel = 3 if self.v_abs <= self.expected_velocity else 0.6
        # sigma_vel_upper, sigma_vel_lower = 0.6, 1.0
        sigma_vel = sigma_vel_upper if self.v_norm_world <= self.expected_velocity else sigma_vel_lower
        self.v_rew = np.exp(-np.power(self.v_norm_world - self.expected_velocity, 2) / 2 / sigma_vel / sigma_vel)

        # angle reward
        # sigma_ang = 0.4
        sigma_ang = self.sigmas["sigma_ang"]
        waypoint_vec = [self.waypoints_pos_forward_world_l[0][1][0] -
                        self.waypoints_pos_forward_world_l[0][0][0],
                        self.waypoints_pos_forward_world_l[0][1][1] - self.waypoints_pos_forward_world_l[0][0][1]]
        self.ang = math.acos(_cos(waypoint_vec, self.agent_velocity_world[:2]))
        self.ang_rew = np.exp(-np.power(self.ang, 2) / 2 / sigma_ang / sigma_ang)

        self.rew_now = self.track_rew * self.v_rew * self.ang_rew
        if self.collision():
            self.rew_now = -10

        self.ep_len += 1
        self.last_status = self.status
        return self.rew_now

# ==============================================================================
# -- Gym Wrapper -----------------------------------------------------------
# ==============================================================================
import gym
from gym import spaces
from collections import deque
from baselines import logger


class Carla(gym.Env):
    def __init__(self, world, max_length=400, stack=1, train_mode="all", test=True, region=0.8, start_v=6.4,
                 scenario_name="Cross_Join"):
        self.world = world
        self._control = carla.VehicleControl()
        self._stack = stack
        self._ob = deque([], maxlen=stack)
        self.v_history = deque([], maxlen=5)
        self.start_v = start_v
        self.train_mode = train_mode
        self.v_ep = []
        self.v_eps = deque([], maxlen=40)
        self.acc_ep = []
        self.acc_eps = deque([], maxlen=40)
        self.left_offset_ep = []
        self.left_offset_eps = deque([], maxlen=40)
        self.right_offset_ep = []
        self.right_offset_eps = deque([], maxlen=40)
        if train_mode == "steer":
            self.action_space = spaces.Box(np.array([-1]), np.array([1]))
        else:
            self.action_space = spaces.Box(np.array([-1, -1]), np.array([1, 1]))
        high = np.array([np.inf] * (self.world.shape[1] - 2))
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self._max_rollout_steps = max_length
        self._frames = 0
        self.test = test
        self.region = region
        self._eps = 0
        self.throttle_brake = 0.
        self.scenario_name = scenario_name
        self.acc_prev = 0.
        self.ep_jerk = 0.
        self.abs_ep_jerk = 0.
        self.ep_acc = 0. 

    def _detect_reset(self):
        def _dis(a, b):
            return ((b[1] - a[1]) ** 2 + (b[0] - a[0]) ** 2) ** 0.5

        v_norm_mean = np.mean(self.v_history)
        if len(self.v_history) == 5:
            if self.world.lane_pos0_dotted_car_left > self.region or self.world.lane_pos0_dotted_car_right < -self.region + 0.1:
                return True
            elif self.world.scenario:
                if v_norm_mean < 0.05 or v_norm_mean > 11:
                    return True
            else:
                if v_norm_mean < 4. or v_norm_mean > 10.:
                    return True
        else:
            return False

    def _get_scene(self):
        return self.world.road_direction

    def reset(self):
        try:
            self.world.restart()
            self._ob.clear()
            self.v_history.clear()
            if len(self.v_ep) != 0:
                self.v_eps.append(np.mean(self.v_ep))
                self.acc_eps.append(np.mean(self.acc_ep))
                self.left_offset_eps.append(np.mean(self.left_offset_ep))
                self.right_offset_eps.append(np.mean(self.right_offset_ep))
                self.v_ep = []
                self.acc_ep = []
                self.left_offset_ep = []
                self.right_offset_ep = []

            # self.fake_step(start_v=self.start_v)
            if self.world.scenario:
                if self.scenario_name != 'OtherLeadingVehicle_FullMap':
                    if self.scenario_name == 'Cross_Join' or self.scenario_name == 'Cross_Turn_Right':
                        self.fake_step(start_v=7.0, wait_steps=200)
                    elif self.scenario_name == 'Cross_Follow':
                        self.fake_step(start_v=7.0, wait_steps=50)
                    elif self.scenario_name == 'Ring_Join':
                        self.fake_step(start_v=7.0, wait_steps=150)
                    elif self.scenario_name == 'Cross_Turn_Left':
                        self.fake_step(start_v=7.0, wait_steps=150)
                    elif self.scenario_name == 'OverTake':
                        if self._eps < self.g_scratch_eps:
                            self.fake_step(start_v=9.3)
                        else:
                            self.fake_step_overtake(start_v=9.3)
                    elif self.scenario_name == 'OtherLeadingVehicle':
                        self.fake_step(start_v=9.3)
                    elif self.scenario_name == 'Straight_Follow_Double':
                        self.fake_step(start_v=9.0)
                    elif self.scenario_name == 'Straight_Follow_Single':
                        self.fake_step(start_v=7.0)
                    else:
                        raise NotImplementedError
            else:
                self.fake_step(start_v=self.start_v)
            ## make sure it is a valid starting point
            ## if it is not, remove this starting point
            if self.world.lane_pos0_solid_car_left > self.region or self.world.lane_pos0_solid_car_right < -self.region + 0.1:
                self.reset()

            for i in range(self._stack):
                self._ob.append(self.world._observation()[:-2])
           
            #print(self.ep_acc, "EP JERK")
            #print(self.abs_ep_jerk, "EP JERK")
            #print('self.ep_jerk: ', self.ep_jerk)
            #print('self.abs_ep_jerk: ', self.abs_ep_jerk)
            self.ep_acc = 0. 
            self.acc_prev = 0.
            self.ep_jerk = 0.
            self.abs_ep_jerk = 0.

            self._frames = 0
            self._eps = self._eps + 1
            return np.concatenate(self._ob, axis=0)
        except:
            try:
                print("RESET FAILED")
                self.wait_for_reset = True
                self.world.force_restart()
                if self.world.scenario:
                    if self.scenario_name != 'OtherLeadingVehicle_FullMap':
                        if self.scenario_name == 'Cross_Join' or self.scenario_name == 'Cross_Turn_Right':
                            self.fake_step(start_v=7.0, wait_steps=200)
                        elif self.scenario_name == 'Cross_Follow':
                            self.fake_step(start_v=7.0, wait_steps=50)
                        elif self.scenario_name == 'Ring_Join':
                            self.fake_step(start_v=7.0, wait_steps=150)
                        elif self.scenario_name == 'Cross_Turn_Left':
                            self.fake_step(start_v=7.0, wait_steps=150)
                        elif self.scenario_name == 'OverTake':
                            if self._eps < self.g_scratch_eps:
                                self.fake_step(start_v=9.3)
                            else:
                                self.fake_step_overtake(start_v=9.3)
                        elif self.scenario_name == 'OtherLeadingVehicle':
                            self.fake_step(start_v=9.3)
                        elif self.scenario_name == 'Straight_Follow_Double':
                            self.fake_step(start_v=9.0)
                        elif self.scenario_name == 'Straight_Follow_Single':
                            self.fake_step(start_v=7.0)
                        else:
                            raise NotImplementedError
                else:
                    self.fake_step(start_v=self.start_v)
                for i in range(self._stack):
                    self._ob.append(self.world._observation()[:-2])

                self._frames = 0
                return np.concatenate(self._ob, axis=0)
            except:
                print("RESET FAILED AGAIN")
                self.wait_for_reset = True
                self.world.force_restart()
                self._eps = self._eps + 1
                return self.reset()

    def step(self, action):
        """
        The max steer is 1, which corresponds to 70 degrees
        Assuming human could turn 70 degrees in 1.5 seconds
        Then max_steer per frame is delta_time/1.5
        Apply similar constraint to throttle movements
        Assuming human could press the throttle all down in 0.2 seconds
        """
        try:
            control = action
            max_steer = self.world.delta_time / 1.5
            max_throttle = self.world.delta_time / 0.2
            if self.train_mode == "all":
                steer, throttle_brake = action[0], action[1]
                time_steer = np.clip(steer, -max_steer, max_steer)
                time_throttle = np.clip(throttle_brake, -max_throttle, max_throttle)
                steer = np.clip(time_steer + self.world.steer, -1, 1)
                throttle_brake = np.clip(time_throttle + self.world.agent_throttle_brake, -1, 1)
                self.throttle_brake = throttle_brake
                if throttle_brake < 0:
                    throttle = 0
                    brake = 1
                else:
                    throttle = np.clip(throttle_brake, 0, 1)
                    brake = 0
            elif self.train_mode == "steer":
                steer = action[0]
                time_steer = np.clip(steer, -max_steer, max_steer)
                steer = np.clip(time_steer + self.world.steer, -1, 1)
                throttle = 0.5
                brake = 0
            else:
                raise NotImplementedError
            self.world.steer = steer
            action = np.array([steer, throttle, brake])
            terminal = False

            self.v_history.append(self.world.v_norm_world)
            self.v_ep.append(self.world.v_norm_world)
            self.acc_ep.append(self.world.acc_norm_world)
            self.left_offset_ep.append(self.world.lane_pos0_solid_car_left)
            self.right_offset_ep.append(self.world.lane_pos0_car_right)

            self._control.steer = float(action[0])
            self._control.throttle = float(action[1])
            self._control.brake = float(action[2])
            self.world.apply_control(self._control)

            acc_now = self.world.acc_norm_world
            jerk = (acc_now - self.acc_prev) / 0.2
            self.ep_jerk = self.ep_jerk + jerk
            self.abs_ep_jerk = self.abs_ep_jerk + abs(jerk)
            self.acc_prev = acc_now
            self.ep_acc += acc_now
            self._frames += 1
            if self._detect_reset():
                done = True
                terminal = True
            elif self._frames >= self._max_rollout_steps:
                done = True
            elif self.world.collision():  # _sensor._collision:
                terminal = True
                done = True
            else:
                done = False
            self.terminal = terminal

            v_now = self.world.v_norm_world
            acc_now = self.world.acc_norm_world
            left_offset = self.world.lane_pos0_solid_car_left
            right_offset = self.world.lane_pos0_car_right
            rew = self.world._reward_cl()
            ob = self.get_ob(self.world._observation()[:-2])
            return ob, rew, done, {'scene': self._get_scene(), 'terminal': terminal,
                                   'v': v_now, 'acc': acc_now, 'left_offset': left_offset,
                                   'right_offset': right_offset, 'control': control,
                                   'current_pos': self.world.current_pos, 'yaw': np.array([self.world.yaw])}
        except:
            print("STEP FAILED")
            self.world.force_restart()
            return self.step_reset(action)

    def step_reset(self, action):
        control = self.world.agent_control
        v_now = self.world.v_norm_world
        acc_now = self.world.acc_norm_world
        left_offset = self.world.lane_pos0_solid_car_left
        right_offset = self.world.lane_pos0_car_right
        rew = 0.
        ob = self.get_ob(self.world._observation()[:-2])
        done, terminal = True, True
        return ob, rew, done, {'scene': self._get_scene(), 'terminal': terminal,
                               'v': v_now, 'acc': acc_now, 'left_offset': left_offset,
                               'right_offset': right_offset, 'control': control,
                               'current_pos': self.world.current_pos, 'yaw': np.array([self.world.yaw])}

    def fake_step(self, action=[0, 1], start_v=6.4, wait_steps=0):
        self.world.steer = action[0]
        fake_step_time_limit = 100 if self.world.sync else 500

        # if self.scenario_name != 'OtherLeadingVehicle':
        if self.scenario_name != 'OtherLeadingVehicle':
            fake_step_time_limit = 300 if self.world.sync else 500
        else:
            fake_step_time_limit = 100 if self.world.sync else 500

        fake_step_time = 0
        wait_steps_count = 0
        _control = carla.VehicleControl()
        while self.world.v_norm_world < start_v and fake_step_time < fake_step_time_limit:
            if wait_steps_count > wait_steps or wait_steps == 0:
                if self.train_mode == "all":
                    steer, throttle_brake = action[0], action[1]
                    steer = np.clip(steer, -1, 1)
                    throttle_brake = np.clip(throttle_brake, -1, 1)
                    if throttle_brake < 0:
                        throttle = 0
                        brake = 1
                    else:
                        throttle = np.clip(throttle_brake, 0, 1)
                        brake = 0
                elif self.train_mode == "steer":
                    steer = action[0]
                    steer = np.clip(steer, -1, 1)
                    throttle = 0.5
                    brake = 0
                else:
                    raise NotImplementedError
                action = np.array([steer, throttle, brake])
                terminal = False

                self._control.steer = float(action[0])
                self._control.throttle = float(action[1])
                self._control.brake = float(action[2])
                self.world.apply_control(self._control)
                fake_step_time += 1
            else:
                steer, throttle, brake = 0., 0., 0.
                _action = np.array([steer, throttle, brake])
                _control.steer = float(_action[0])
                _control.throttle = float(_action[1])
                _control.brake = float(_action[2])
                self.world.apply_control(_control)
                wait_steps_count += 1
                fake_step_time += 1
        if fake_step_time >= 500:
            self.reset()

    def get_ob(self, ob):
        self._ob.append(ob)
        return np.concatenate(self._ob, axis=0)

    def eprew(self):
        return self.world.ep_rew


class RuleBased(object):
    def __init__(self, C=10, a=2):
        self.C = C
        self.a = a

        # overtake hyperparameter
        self.overtake_firststage = 1.5
        self.overtake_secondstage = 3.0
        self.overtake_thirdstage = 1.0

        # overtake hyperparameter
        self.carfollow_c1 = 1.5
        self.carfollow_c2 = 2
        self.carfollow_c3 = 1

    def _init_condition(self, world, coordinator):
        self.v0 = v0 = world.v_norm_world
        def get_roadid(world, vehicle):
            wp = world._map.get_waypoint(vehicle.get_location())
            return [wp.road_id, wp.lane_id]
        zombie_cars_on_sameroad = [car for car in world.zombie_cars if get_roadid(world, car)==get_roadid(world, world.vehicle)]
        if len(zombie_cars_on_sameroad) > 0:
            zombie_car = zombie_cars_on_sameroad[0]
            pos_car = coordinator.transform_car3d(utils._pos3d(zombie_car))
            L = pos_car[0]
            vf = utils._norm2d(utils._get_v(zombie_car))
        else:
            L = 10
            vf = 7 
        self.vf = vf
        self.L = L
        #crash_index = ((v0)**2/(2*self.a) + self.C)/(L+(vf**2/(2*self.a)))
        self.crash_index = ((v0)**2/(2*self.a) + self.C)/(L+(vf**2/(2*self.a)))/4 #the range here for overtake
 
    def decision(self, world, frames, coordinator):
        self.v0 = v0 = world.v_norm_world
        def get_roadid(world, vehicle):
            wp = world._map.get_waypoint(vehicle.get_location())
            return [wp.road_id, wp.lane_id]
        zombie_cars_on_sameroad = [car for car in world.zombie_cars if get_roadid(world, car)==get_roadid(world, world.vehicle)]
        if len(zombie_cars_on_sameroad) > 0:
            zombie_car = zombie_cars_on_sameroad[0]
            pos_car = coordinator.transform_car3d(utils._pos3d(zombie_car))
            L = pos_car[0]
            vf = utils._norm2d(utils._get_v(zombie_car))
        else:
            L = 100
            vf = 7 
        self.vf = vf
        self.L = L
        crash_index = self.crash_index
        
        #if crash_index <= 0.2:
        #    world.rulebased_stage = "Full Stream"
        #    return self.full_stream()
        #elif crash_index >= 0.3 and crash_index < 0.4:
        #    world.rulebased_stage = "Car Following"
        #    return self.car_following()
        #elif crash_index >= 0.4 and crash_index < 0.7:
        #    world.rulebased_stage = "OverTake"
        #    return self.overtake(world, frames)
        #elif crash_index >= 0.7:
        #    world.rulebased_stage = "Stop"
        #    return self.stop()
        if "Join" in world.scenario_name or "Follow" in world.scenario_name or "Cross" in world.scenario_name:
            world.rulebased_stage = "Car Following"
            return self.car_following()
        elif "OverTake" in world.scenario_name or "Other" in world.scenario_name:
            world.rulebased_stage = "OverTake"
            return self.overtake(world, frames)


    def full_stream(self):
        return [1, 10, 40]

    def car_following(self):
        s0 = self.L
        v0 = self.vf
        v1 = self.v0
        S = (self.vf**2)/(2*2)
        lateral_goal = 1
        d1 = s0-self.carfollow_c1*S
        d2 = s0-self.carfollow_c2*S
        c1 = 1
        c2 = 2
        amax = 10
        def calculated_s(v1, v0, s0, c1, c2):
            a_bound1 = 1/2 * (v1**2-v0**2)/d1
            a_bound2 = 1/2 * (v1**2-v0**2)/d2
            alower = min(a_bound1, a_bound2)
            aup = max(a_bound1, a_bound2)
            if alower > amax:
                a = amax
            else:
                alower = max(alower, 0)
                aup = max(aup, amax)
                a = 0.5*(aup+alower)
            delta_t = abs(v1-v0)/a
            delta_S = abs(v1**2-v0**2)/(2*a)
            S = s0+delta_t*v0-delta_S
            return S 
        if v1 > v0:
            if d1 < 0:
                target_vel = v0
                longitudinal_goal = 5
            elif d1 >=0 and d2 < 0:
                a_low = 0.5*(v1**2-v0**2)/d1
                if a_low > amax:
                    target_vel = v0
                    longitudinal_goal = 5
                else:
                    target_vel = v0
                    c1_star = s0/S-(v1-v0)**2/(2*a_low*S)
                    longitudinal_goal = calculated_s(v1, v0, s0, c1, c1_star)
            else:
                target_vel = v0
                longitudinal_goal = calculated_s(v1, v0, s0, c1, c2)
        else:
            if d1 < 0:
                target_vel = v1
                longitudinal_goal = 5
            elif d1 > 0 and d2 < 0:
                a_low = 0.5*(v0**2-v1**2)/(-d2) 
                if a_low > amax:
                    target_vel = v0
                    longitudinal_goal = 5
                else:
                    target_vel = v0
                    c1_star = s0/S-(v1-v0)**2/(2*a_low*S)
                    longitudinal_goal = calculated_s(v1, v0, s0, c2, c1_star)
            else:
                target_vel = v0
                longitudinal_goal = calculated_s(v1, v0, s0, c1, c2) 
        target_vel = target_vel * 3.6              
        if logger.scenario_name == "Cross_Join":
            target_vel = target_vel + 2
        elif logger.scenario_name == "Ring_Join":
            target_vel = target_vel + 3
        elif logger.scenario_name == "Straight_Follow_Single":
            target_vel = np.clip(target_vel, 0, 20)
 
        longitudinal_goal = min(max(0, longitudinal_goal), 70)
        return [lateral_goal, longitudinal_goal, target_vel]

    def overtake(self, world, frames):
        if frames < 100:
            if frames < 23: # first stage
                if world.status == 2:
                    lateral_goal = 0
                    target_vel = self.overtake_firststage*11.25
                elif world.status == 1:
                    lateral_goal = 1
                    target_vel = min(max(14.4, self.overtake_secondstage*11.25), 40)
            elif frames >= 23 and frames < 60: # second stage
                lateral_goal = 1
                target_vel = min(max(14.4, self.overtake_secondstage*11.25), 40)
                self.end_v = world.v_norm_world*3.6
            elif frames >= 60: # third stage
                if world.status == 1:
                    lateral_goal = 2
                elif world.status == 2:
                    lateral_goal = 1
                target_vel = self.end_v
        longitudinal_goal = 10
        return [lateral_goal, longitudinal_goal, target_vel]
        
    def stop(self):
        stop_dis = int(self.L-3.5)
        mod = stop_dis%10
        if mod < 5:
            target_dis = int(stop_dis - mod)
        else:
            target_dis = int(math.ceil(stop_dis/10.0)*10.0)
        target_dis = max(target_dis, 5) 
        return [1, target_dis, 0]

# ==============================================================================
# -- MultiDiscrete action space environment wrapper for the decision output-----
# ==============================================================================
# The environment wrapper for decision output, the action space in the wrapper
# is a multi-discrete class and each dim represents: 
# 1) lateral goal: the lateral goal represents a lane-changing option, including
# left lane, ego car's lane, right lane, total three roads
# 2) longitudinal goal: the distancee between the car position and the goal 
# position ahead of the ego car, including 0, 10, 20, 30, 40, 50, 60, 70 in meter
# 3) target velocity: including 0, 10, 20, 30, 40 in km/h
# Note: if the excute mode is short, decision will update every step, else will
# update after excute all the actions
from baselines.gail.planner.pybpp import BppPlanner
from baselines.gail.planner.commondata import TrajectoryPoint
from baselines.gail.controller.pid_controller import VehiclePIDController
from baselines.gail import utils
import math
class CarlaDM(gym.Env):
    def __init__(self, world, max_length=400, stack=1, train_mode="all", test=True, region=0.8, start_v=6.4,
                 excute_mode='short', interval=0.2, lanes=3, D_skip=1,
                 overtake_curriculum=False, scenario_name=None, g_scratch_eps=120, rule_based=False):
        self.world = world
        self._control = carla.VehicleControl()
        self._stack = stack
        self._ob = deque([], maxlen=stack)
        self.v_history = deque([], maxlen=5)
        self.start_v = start_v
        self.train_mode = train_mode
        self.v_ep = []
        self.v_eps = deque([], maxlen=40)
        self.acc_ep = []
        self.acc_eps = deque([], maxlen=40)
        self.left_offset_ep = []
        self.left_offset_eps = deque([], maxlen=40)
        self.right_offset_ep = []
        self.right_offset_eps = deque([], maxlen=40)
        self.interval = interval
        self.vehicle_planner = BppPlanner(interval, 5)
        self.vehicle_controller = VehiclePIDController(self.world.vehicle)
        self.lanes = lanes
        self._eps = 0
        self._cols = 0
        self.action_space = spaces.MultiDiscrete([lanes, 7, 5])
        high = np.array([np.inf] * (self.world.shape[1] - 2))
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.coordinate = utils.Coordinate(self.world.vehicle)
        self.g_scratch_eps = g_scratch_eps

        self.acc_prev = 0.
        self.ep_jerk = 0.
        self.abs_ep_jerk = 0.

        self._max_rollout_steps = max_length
        self._frames = 0
        self.last_wp = self.world.waypoints_forward_l[0][0][0]
        self.test = test
        self.region = region
        self.excute_mode = excute_mode
        self.D_skip = D_skip
        self.overtake_curriculum = overtake_curriculum
        self.rule_based = rule_based
        self.terminal = False
        if self.rule_based:
            self.rule_based_dm = RuleBased()
            self.rule_based_dm._init_condition(self.world, self.coordinate)
        if logger.scenario_name == "Merge_Env":
            # rules to choose different scenario class here
            self.scenario_name = SCENARIO_NAMES[logger.select_scenario_id]
        else:
            self.scenario_name = scenario_name

    def late_init(self):
        self.vehicle_controller = VehiclePIDController(self.world.vehicle)
        self.coordinate = utils.Coordinate(self.world.vehicle)

    def _detect_reset(self):
        def _dis(a, b):
            return ((b[1] - a[1]) ** 2 + (b[0] - a[0]) ** 2) ** 0.5

        v_norm_mean = np.mean(self.v_history)
        if len(self.v_history) == 5:
            #if self.world.lane_pos0_dotted_car_left > self.region or self.world.lane_pos0_dotted_car_right < -self.region + 0.1:
            if self.world.lane_pos0_solid_car_left > self.region or self.world.lane_pos0_solid_car_right < -self.region + 0.1:
                return True
            elif v_norm_mean < 0.05:
                return True
        else:
            return False

    def _get_scene(self):
        return self.world.road_direction

    def _update_agent_car(self):
        self.coordinate.update_vehicle(self.world.vehicle)
        self.vehicle_controller.update_vehicle(self.world.vehicle)

    def _recover_world_coordinate(self, traj, mode="3d"):
        # obtain locations
        locations_car = []
        locations_world = []
        if mode == "3d":
            for point in traj:
                locations_car.append([point.x, point.y, point.z])
            for location_car in locations_car:
                location_world = self.coordinate.transform_world3d(location_car)
                locations_world.append(location_world)
        elif mode == "2d":
            for point in traj:
                locations_car.append([point.x, point.y])
            for location_car in locations_car:
                location_world = self.coordinate.transform_world2d(location_car)
                np.append(location_world, 0.)
                locations_world.append(location_world)
        return locations_world

    def reset(self):
        self.wait_for_reset = True
        try:
            self.world.restart()
            self._ob.clear()
            self.v_history.clear()
            self._update_agent_car()
            if len(self.v_ep) != 0:
                self.v_eps.append(np.mean(self.v_ep))
                self.acc_eps.append(np.mean(self.acc_ep))
                self.left_offset_eps.append(np.mean(self.left_offset_ep))
                self.right_offset_eps.append(np.mean(self.right_offset_ep))
                self.v_ep = []
                self.acc_ep = []
                self.left_offset_ep = []
                self.right_offset_ep = []

            ## make sure it is a valid starting point
            ## if it is not, remove this starting point
            if self.world.lane_pos0_solid_car_left > self.region or self.world.lane_pos0_solid_car_right < -self.region + 0.1:
                self.reset()

            if self.world.scenario:
                if self.scenario_name != 'OtherLeadingVehicle_FullMap':
                    if self.scenario_name == 'Cross_Join' or self.scenario_name == 'Cross_Turn_Right':
                        self.fake_step(start_v=7.0, wait_steps=200)
                    elif self.scenario_name == 'Cross_Follow':
                        self.fake_step(start_v=7.0, wait_steps=50)
                    elif self.scenario_name == 'Ring_Join':
                        self.fake_step(start_v=7.0, wait_steps=150)
                    elif self.scenario_name == 'Cross_Turn_Left':
                        self.fake_step(start_v=7.0, wait_steps=150)
                    elif self.scenario_name == 'OverTake':
                        if self._eps < self.g_scratch_eps:
                            self.fake_step(start_v=9.3)
                        else:
                            self.fake_step_overtake(start_v=9.3)
                            self.world._mode = "Policy"
                    elif self.scenario_name == 'OtherLeadingVehicle':
                        self.fake_step(start_v=9.3)
                    elif self.scenario_name == 'Straight_Follow_Double' or self.scenario_name == 'Straight_Follow_Single':
                        self.fake_step(start_v=7.0)
                    else:
                        raise NotImplementedError
                else:
                    self.fake_step(start_v=self.start_v)
            else:
                self.fake_step(start_v=7.0)

            for i in range(self._stack):
                self._ob.append(self.world._observation()[:-2])
            
            if self.rule_based:
                self.rule_based_dm._init_condition(self.world, self.coordinate)

            self._frames = 0
            self._eps = self._eps + 1
            self.ep_jerk = 0.
            self.abs_ep_jerk = 0.

            self.last_wp = self.world.waypoints_forward_l[0][0][0]
            self.wait_for_reset = False
            return np.concatenate(self._ob, axis=0)
        except:
            print("RESET FAILED")
            try:
                self.wait_for_reset = True
                self.world.force_restart()
                self.late_init()
                # -------------------------------------------------------
                # copy without world.restart
                # -------------------------------------------------------
                self._ob.clear()
                self.v_history.clear()
                self._update_agent_car()
                if len(self.v_ep) != 0:
                    self.v_eps.append(np.mean(self.v_ep))
                    self.acc_eps.append(np.mean(self.acc_ep))
                    self.left_offset_eps.append(np.mean(self.left_offset_ep))
                    self.right_offset_eps.append(np.mean(self.right_offset_ep))
                    self.v_ep = []
                    self.acc_ep = []
                    self.left_offset_ep = []
                    self.right_offset_ep = []

                ## make sure it is a valid starting point
                ## if it is not, remove this starting point
                if self.world.lane_pos0_solid_car_left > self.region or self.world.lane_pos0_solid_car_right < -self.region + 0.1:
                    self.reset()

                # self.fake_step(start_v=self.start_v, wait_steps=self.wait_times)
                if self.world.scenario:
                    if self.scenario_name != 'OtherLeadingVehicle_FullMap':
                        if self.scenario_name == 'Cross_Join' or self.scenario_name == 'Cross_Turn_Right':
                            self.fake_step(start_v=7.0, wait_steps=200)
                        elif self.scenario_name == 'Cross_Follow':
                            self.fake_step(start_v=7.0, wait_steps=50)
                        elif self.scenario_name == 'Ring_Join':
                            self.fake_step(start_v=7.0, wait_steps=150)
                        elif self.scenario_name == 'Cross_Turn_Left':
                            self.fake_step(start_v=7.0, wait_steps=150)
                        elif self.scenario_name == 'OverTake':
                            if self._eps < self.g_scratch_eps:
                                self.fake_step(start_v=9.3)
                            else:
                                self.fake_step_overtake(start_v=9.3)
                        elif self.scenario_name == 'OtherLeadingVehicle':
                            self.fake_step(start_v=9.3)
                        elif self.scenario_name == 'Straight_Follow_Double' or self.scenario_name == 'Straight_Follow_Single':
                            self.fake_step(start_v=7.0)
                        else:
                            raise NotImplementedError
                    else:
                        self.fake_step(start_v=self.start_v)
                else:
                    self.fake_step(start_v=7.0)

                for i in range(self._stack):
                    self._ob.append(self.world._observation()[:-2])
            
                if self.rule_based:
                    self.rule_based_dm._init_condition(self.world, self.coordinate)

                self._frames = 0
                self._eps = self._eps + 1
                self.last_wp = self.world.waypoints_forward_l[0][0][0]
                self.wait_for_reset = False
                self.terminal = False
                return np.concatenate(self._ob, axis=0)
            except:
                print("RESET FAILED AGAIN")
                self.wait_for_reset = True
                self.world.force_restart()
                self.late_init()
                return self.reset()

    def fake_step(self, action=[0, 1], start_v=6.4, wait_steps=0):
        self.world.steer = action[0]
        fake_step_time_limit = 100 if self.world.sync else 500

        #if self.scenario_name != 'OtherLeadingVehicle':
        if self.scenario_name != 'OtherLeadingVehicle':
            fake_step_time_limit = 300 if self.world.sync else 500
        else:
            fake_step_time_limit = 100 if self.world.sync else 500

        fake_step_time = 0
        wait_steps_count = 0
        _control = carla.VehicleControl()
        while self.world.v_norm_world < start_v and fake_step_time < fake_step_time_limit:
            if wait_steps_count > wait_steps or wait_steps == 0:
                if self.train_mode == "all":
                    steer, throttle_brake = action[0], action[1]
                    steer = np.clip(steer, -1, 1)
                    throttle_brake = np.clip(throttle_brake, -1, 1)
                    if throttle_brake < 0:
                        throttle = 0
                        brake = 1
                    else:
                        throttle = np.clip(throttle_brake, 0, 1)
                        brake = 0
                elif self.train_mode == "steer":
                    steer = action[0]
                    steer = np.clip(steer, -1, 1)
                    throttle = 0.5
                    brake = 0
                else:
                    raise NotImplementedError
                action = np.array([steer, throttle, brake])
                self.terminal = False

                self._control.steer = float(action[0])
                self._control.throttle = float(action[1])
                self._control.brake = float(action[2])
                self.world.apply_control(self._control)
                fake_step_time += 1
            else:
                steer, throttle, brake = 0., 0., 0.
                _action = np.array([steer, throttle, brake])
                _control.steer = float(_action[0])
                _control.throttle = float(_action[1])
                _control.brake = float(_action[2])
                self.world.apply_control(_control)
                wait_steps_count += 1
                fake_step_time += 1
        if fake_step_time >= 500:
            self.reset()

    def fake_step_overtake(self, action=[0, 1], start_v=6.4, wait_steps=0):
        self.world.steer = action[0]
        self.world._mode = "Fake"
        fake_step_time_limit = 100 if self.world.sync else 500

        if self.scenario_name != 'OtherLeadingVehicle':
            fake_step_time_limit = 300 if self.world.sync else 500
        else:
            fake_step_time_limit = 100 if self.world.sync else 500

        fake_step_time = 0
        wait_steps_count = 0
        _control = carla.VehicleControl()
        while self.world.v_norm_world < start_v and fake_step_time < fake_step_time_limit:
            if wait_steps_count > wait_steps or wait_steps == 0:
                if self.train_mode == "all":
                    steer, throttle_brake = action[0], action[1]
                    steer = np.clip(steer, -1, 1)
                    throttle_brake = np.clip(throttle_brake, -1, 1)
                    if throttle_brake < 0:
                        throttle = 0
                        brake = 1
                    else:
                        throttle = np.clip(throttle_brake, 0, 1)
                        brake = 0
                elif self.train_mode == "steer":
                    steer = action[0]
                    steer = np.clip(steer, -1, 1)
                    throttle = 0.5
                    brake = 0
                else:
                    raise NotImplementedError
                action = np.array([steer, throttle, brake])
                terminal = False

                self._control.steer = float(action[0])
                self._control.throttle = float(action[1])
                self._control.brake = float(action[2])
                self.world.apply_control(self._control)
                fake_step_time += 1
            else:
                steer, throttle, brake = 0., 0., 0.
                _action = np.array([steer, throttle, brake])
                _control.steer = float(_action[0])
                _control.throttle = float(_action[1])
                _control.brake = float(_action[2])
                self.world.apply_control(_control)
                wait_steps_count += 1
                fake_step_time += 1
       
        continue_steps = logger.keypoints[self.world.scenario_now.cur_checkpoint]        
        for i in range(continue_steps[0]):
            action = logger.expert_acs[i]
            steer, throttle_brake = action[0], action[1]
            steer = np.clip(steer, -1, 1)
            throttle_brake = np.clip(throttle_brake, -1, 1)
            if throttle_brake < 0:
                throttle = 0
                brake = 1
            else:
                throttle = np.clip(throttle_brake, 0, 1)
                brake = 0
            action = np.array([steer, throttle, brake])
            self._control.steer = float(action[0])
            self._control.throttle = float(action[1])
            self._control.brake = float(action[2])
            self.world.apply_control(self._control)

    def step(self, decision):
        """
        The input of decisions are transferred to a squecne of trajectory
        points by the BPP planner. Then the sequence of trajectory points
        are transferred to a squence of control signals by the PID controller
        :param decision: a list of discrete values
        :output: steering and throttle_brake values
        """

        def lateral_shift_wp(transform, shift):
            transform.rotation.yaw += 90
            shift_location = transform.location + shift * transform.get_forward_vector()
            w = self.world._map.get_waypoint(shift_location, project_to_road=False)
            return w

        def _valid_wp(wp, wp_ref):
            return not (wp.lane_id * wp_ref.lane_id < 0 or wp.lane_id == wp_ref.lane_id or wp.road_id != wp_ref.road_id)

        def _is_ahead(wp, target_pos):
            """
            Test if a target pos is ahead of the waypoint
            """
            wp_pos = _pos(wp)
            orientation = math.radians(wp.transform.rotation.yaw)
            target_vector = np.array([target_pos[0] - wp_pos[0], target_pos[1] - wp_pos[1]])
            forward_vector = np.array([np.cos(orientation), np.sin(orientation)])
            d_angle = math.degrees(math.acos(_cos(forward_vector, target_vector)))
            return d_angle < 90

        def _retrieve_goal_wp(longitudinal_goal, lateral_goal):
            def _get_ref(lateral_goal):
                ref = self.world.status + lateral_goal - 1
                return ref
                # wp_select = self.world.current_wp.next(int(longitudinal_goal))[0]

            wp_select = self.world.waypoints_queue_equal_l[0][int(longitudinal_goal)][0]
            self.world.wp_select = wp_select
            current_wp = self.world.current_wp
            # wp_refline = self.world.waypoints
            need_reset = False
            # retrieve possible lane-changing goals
            shifts = [-3.5, 0, 3.5]
            wp_select_possible = []
            current_wps_shift = []
            for shift in shifts:
                wp_select_possible.append(lateral_shift_wp(wp_select.transform, shift))
                current_wps_shift.append(lateral_shift_wp(current_wp.transform, shift))
            wp_select_shift = wp_select_possible[lateral_goal]
            current_wp_shift = current_wps_shift[lateral_goal]
            self.world.wp_select_shift = wp_select_shift
            self.world.current_wp_shift = current_wp_shift
           
            # candidates for visualization 
            #choice_candidates = [[0, 5, 40], [1, 5, 50], [0, 10, 10], [1, 10, 40], [2, 5, 40], [2, 10, 50], [1, 15, 30], [2, 15, 30]]
            #def generate_wp_candidates(choices):
            #    wp_candidates = []
            #    for choice in choices:
            #        wp_select = self.world.waypoints_queue_equal_l[0][int(choice[1])][0]
            #        shift = shifts[choice[0]]
            #        wp_shift = lateral_shift_wp(wp_select.transform, shift)
            #        wp_candidates.append(wp_shift)
            #    return wp_candidates
            #self.world.wp_candidates = generate_wp_candidates(choice_candidates)

            if current_wp_shift is None or wp_select_shift is None:
                need_reset = True
                reset_type = "WP IS NONE"
                return None, need_reset, reset_type, current_wps_shift
            else:
                # if lateral goal is not 1 (current lane), the choosing lane must not be a solid lane
                if lateral_goal != 1 and (
                        not _valid_wp(current_wp, current_wp_shift) or not _valid_wp(wp_select, wp_select_shift)):
                    need_reset = True
                    reset_type = "INVALID LANE-CHANGING"
                    wp_goal = wp_select_shift
                elif self.world.to_intersection < 20 and lateral_goal != 1:
                    need_reset = True
                    reset_type = "INVALID LANE-CHANGING BEFORE INTERSECTION"
                    wp_goal = wp_select_shift
                elif len(self.world._global_reflines[self.world.status + (lateral_goal - 1)][0]) == 0:
                    need_reset = True
                    reset_type = "INVALID REFLINE"
                    wp_goal = wp_select_shift
                #elif _get_ref(lateral_goal) not in [1, 2, 3] and self.scenario_name == "OtherLeadingVehicle":
                elif _get_ref(lateral_goal) not in [1, 2, 3] and (self.scenario_name == "OtherLeadingVehicle" or self.scenario_name == "OverTake"):
                    reset_type = "INVALID CHOICE IN OVERTAKE SCENARIO"
                    need_reset = True
                    wp_goal = wp_select_shift
                else:
                    reset_type = "VALID"
                    wp_goal = wp_select_shift
                return wp_goal, need_reset, reset_type, current_wps_shift

        def parse_goals(lateral_goal, longitudinal_goal, target_vel):
            # here target velocity is in km/h
            longitudinal_goal = longitudinal_goal * 10.0 + 10.
            if longitudinal_goal == 0.:
                longitudinal_goal = 5.0
            target_vel = target_vel * 10.0 + 10.
            return lateral_goal, longitudinal_goal, target_vel

        def generate_refline(refline_pos):
            refline = []
            for i in range(len(refline_pos) - 1):
                pos = refline_pos[i]
                pos_next = refline_pos[i + 1]
                x, y, z = pos[0], pos[1], pos[2]
                dir_x, dir_y, dir_z = (pos_next[0] - pos[0]) / 0.05, (pos_next[1] - pos[1]) / 0.05, (
                            pos_next[2] - pos[2]) / 0.05
                dir_xyz = [dir_x, dir_y, dir_z]
                dir_norm = utils._norm3d(dir_xyz)
                dir_x, dir_y, dir_z = dir_x / dir_norm, dir_y / dir_norm, dir_z / dir_norm
                point = TrajectoryPoint()
                point.x, point.y, point.z, point.dir_x, point.dir_y, point.dir_z = x, y, z, dir_x, dir_y, dir_z
                refline.append(point)
            return refline

        # ====================================================================
        # timeout handler
        # ====================================================================
        try:
            self.decision = decision
            lateral_goal, longitudinal_goal, target_vel = decision
            lateral_goal = int(lateral_goal)
            # here we implement a curriculum
            # -----------------------------------------------------
            # standard curriculum
            # -----------------------------------------------------
            if self.overtake_curriculum == 1:
                if self._eps % 1 == 0:
                    if self._frames < 100:
                        if self._frames < 23:
                            if self.world.status == 2:
                                lateral_goal = 0
                                target_vel = 3
                            elif self.world.status == 1:
                                lateral_goal = 1
                                target_vel = 3
                        elif self._frames >= 23 and self._frames < 60:
                            lateral_goal = 1
                            target_vel = 2
                        elif self._frames >= 60:
                            if self.world.status == 1:
                                lateral_goal = 2
                            elif self.world.status == 2:
                                lateral_goal = 1
                            target_vel = 2
                    else:
                        lateral_goal = 1
                        target_vel = 2
                    longitudinal_goal = random.choice([1])
            # -----------------------------------------------------
            # turn left with 25km/h and 25km/h back to the original lane
            # -----------------------------------------------------
            if self.overtake_curriculum == 2:
                if self.world.scenario_now.cur_checkpoint == "KL40" or self.world.scenario_now.cur_checkpoint == "TR":
                    lateral_goal = 2
                else:
                    lateral_goal = 1
            # -----------------------------------------------------
            # turn right curriculum
            # -----------------------------------------------------
            if self.overtake_curriculum == 3:
                if self._eps % 1 == 0:
                    if self._frames < 100:
                        if self._frames < 23:
                            if self.world.status == 2:
                                lateral_goal = 2
                                target_vel = 4
                            elif self.world.status == 1:
                                lateral_goal = 1
                                target_vel = 3
                        elif self._frames >= 23 and self._frames < 60:
                            lateral_goal = 1
                            target_vel = 3
                        elif self._frames >= 60:
                            if self.world.status == 3:
                                lateral_goal = 0
                                target_vel = 2
                            elif self.world.status == 2:
                                lateral_goal = 1
                                target_vel = 2
                    else:
                        lateral_goal = 1
                        target_vel = 2
                    longitudinal_goal = random.choice([1])
            #if self._frames > 160:
            #    lateral_goal = 2
            # -----------------------------------------------------
            lateral_goal, longitudinal_goal, target_vel = parse_goals(lateral_goal, longitudinal_goal, target_vel)
            if self.rule_based:
                lateral_goal, longitudinal_goal, target_vel = self.rule_based_dm.decision(self.world, self._frames, self.coordinate)
            self.lateral_goal = lateral_goal
            self.world.longitudinal_goal, self.world.lateral_goal, self.world.target_vel = longitudinal_goal, lateral_goal, target_vel
            # generate goal position for bpp planner
            # the lateral goal is a lane-changing option
            current_wp_now = self.world.waypoints_forward_l[0][0][0]
            goal_wp, need_reset, reset_type, current_wps_shift = _retrieve_goal_wp(longitudinal_goal, lateral_goal)
            if need_reset:
                print(reset_type)
                return self.step_reset(decision)
            else:
                # ---------------------------------------------------------
                # lane-chaning detection
                lane_changing = not (
                (current_wp_now.lane_id * self.last_wp.lane_id < 0 or current_wp_now.lane_id == self.last_wp.lane_id)) \
                                and current_wp_now.road_id == self.last_wp.road_id
                if lane_changing:
                    self.world.lane_change = lane_changing
                # retrieve waypoints on the target lane
                end_idx = int(longitudinal_goal * 10)
                interval_idx = int(self.interval * 10)
                refline_pos = self.world._global_reflines[self.world.status + (lateral_goal - 1)]
                refline_car_pos = refline_pos
                refline_pos = refline_pos[0][:end_idx:interval_idx]
                refline = generate_refline(refline_pos)
                self.last_wp = current_wp_now
                # ---------------------------------------------------------
                # execute decision
                for i in range(self.D_skip):
                    self.world.goal_wp_now = goal_wp
                    goal_pos_world = utils._pos3d(goal_wp)
                    goal_pos_car = self.coordinate.transform_car3d(goal_pos_world)
                    goal_pos_car = refline_car_pos[0][-1]
                    goal_pos = goal_pos_car
                    goal_pos_last = refline_car_pos[0][-2]
                    # compute goal direction
                    goal_dir = [(goal_pos_car[0] - goal_pos_last[0]) / 0.05,
                                (goal_pos_car[1] - goal_pos_last[1]) / 0.05,
                                (goal_pos_car[2] - goal_pos_last[2]) / 0.05]
                    goal_pos_car_norm = utils._norm3d(goal_dir)
                    goal_dir = [goal_pos_car[0] / goal_pos_car_norm, goal_pos_car[1] / goal_pos_car_norm,
                                goal_pos_car[2] / goal_pos_car_norm]

                    start_pos = [0., 0., 0.]
                    if self.world.v_norm_world < 0.001:
                        start_dir = [0.00001, 0., 0.]
                    else:
                        start_dir = [1., 0., 0.]

                    # bpp planner, note the target_vel and current_vel must be in m/s
                    cur_vel = self.world.v_norm_world
                    tgt_vel = target_vel / 3.6
                    cur_acc = self.world.acc_norm_world

                    traj_ret = self.vehicle_planner.run_step(start_pos, start_dir, refline, cur_vel, tgt_vel, cur_acc)
                    #for p in traj_ret:
                    #    print(p.x, p.y)
                    # Debug
                    # traj_ret_lst = [[point.x, point.y, point.z, point.dir_x, point.dir_y, point.dir_z, point.theta, point.velocity, point.acceleration, point.curvature, point.sumdistance] for point in traj_ret]
                    # if self._frames > 20 and self._frames <= 70:
                    #     pickle.dump([start_pos, start_dir, refline, cur_vel, tgt_vel, cur_acc, traj_ret_lst], self._file)
                    # if self._frames > 80:
                    #     self._file.close()
                    #     exit()
                    self.traj_ret_world = self.world.traj_ret = self._recover_world_coordinate(traj_ret)

                    # pid controller, note the target_vel_pid must be in km/h
                    if self.excute_mode == "short":
                        # get reference position and target velocity
                        idx = 5
                        target_ref_point = traj_ret[idx]
                        target_vel_pid = traj_ret[idx].velocity * 3.6
                        self.world.pp_cur_vel, self.world.pp_tgt_vel, self.world.pp_target_vel_pid = cur_vel*3.6, tgt_vel*3.6, target_vel_pid
                        ref_pos = [self.traj_ret_world[idx][0], self.traj_ret_world[idx][1],
                                   self.traj_ret_world[idx][2]]
                        ref_location = carla.Location(x=ref_pos[0], y=ref_pos[1], z=ref_pos[2])
                        self.world.ref_location = ref_location

                        # get reference orientation
                        idx = 30
                        if self.rule_based:
                            idx = min(30, len(traj_ret)-1)                    
                        target_ref_point = traj_ret[idx]
                        self.world.pp_ref_vel = target_ref_point.velocity*3.6
                        target_pos_pid = [self.traj_ret_world[idx][0], self.traj_ret_world[idx][1],
                                          self.traj_ret_world[idx][2]]
                        target_location_pid = carla.Location(x=target_pos_pid[0], y=target_pos_pid[1],
                                                             z=target_pos_pid[2])
                        target_wp_pid = self.world._map.get_waypoint(target_location_pid)
                        self.world.target_wp_pid = target_wp_pid

                        # pid control
                        control = self.vehicle_controller.run_step(target_vel_pid, target_wp_pid, target_ref_point)
                        steer = control.steer
                        throttle = control.throttle
                        brake = control.brake
                        throttle_brake = throttle + -1 * brake
                        action = [steer, throttle_brake]
                        return self.step_control(action)
                    elif self.excute_mode == "long":
                        raise NotImplementedError
                    else:
                        raise NotImplementedError
        except:
            print("DECISION STEP FAILED")
            self.world.force_restart()
            self.late_init()
            return self.step_reset(self.decision)

    def step_reset(self, decision):
        control = self.world.agent_control
        v_now = self.world.v_norm_world
        acc_now = self.world.acc_norm_world
        left_offset = self.world.lane_pos0_solid_car_left
        right_offset = self.world.lane_pos0_car_right
        rew = 0.
        ob = self.get_ob(self.world._observation()[:-2])
        done, terminal = True, True
        self.terminal = terminal
        return ob, rew, done, {'scene': self._get_scene(), 'terminal': terminal,
                               'v': v_now, 'acc': acc_now, 'left_offset': left_offset,
                               'right_offset': right_offset, 'control': control,
                               'current_pos': self.world.current_pos, 'yaw': np.array([self.world.yaw])}

    def step_control(self, action):
        max_steer = self.world.delta_time / 1.5
        max_throttle = self.world.delta_time / 0.2

        if self.train_mode == "all":
            steer, throttle_brake = action[0], action[1]
            if throttle_brake < 0:
                throttle = 0
                brake = 1
            else:
                throttle = np.clip(throttle_brake, 0, 1)
                brake = 0
        elif self.train_mode == "steer":
            steer = action[0]
            time_steer = np.clip(steer, -max_steer, max_steer)
            steer = np.clip(time_steer + self.world.steer, -1, 1)
            throttle = 0.5
            brake = 0
        else:
            raise NotImplementedError
        self.world.steer = steer
        action = np.array([steer, throttle, brake])
        terminal = False

        self.v_history.append(self.world.v_norm_world)
        self.v_ep.append(self.world.v_norm_world)
        self.acc_ep.append(self.world.acc_norm_world)
        self.left_offset_ep.append(self.world.lane_pos0_solid_car_left)
        self.right_offset_ep.append(self.world.lane_pos0_car_right)

        self._control.steer = float(action[0])
        self._control.throttle = float(action[1])
        self._control.brake = float(action[2])

        self._frames += 1

        self.world.apply_control(self._control)
        rew = self.world._reward(self.lateral_goal)
        self.world.ep_rew += self.world.rew_now
        if self._detect_reset():
            done = True
            terminal = True
        elif self._frames >= self._max_rollout_steps:
            done = True
        elif self.world.collision():#_sensor._collision:
            terminal = True
            done = True
        else:
            done = False
        self.terminal = terminal

        v_now = self.world.v_norm_world
        acc_now = self.world.acc_norm_world
        jerk = (acc_now - self.acc_prev) / 0.2
        self.ep_jerk = self.ep_jerk + jerk
        self.abs_ep_jerk = self.abs_ep_jerk + abs(jerk)
        self.acc_prev = acc_now
        # print('self.ep_jerk: ', self.ep_jerk)
        # print('self.abs_ep_jerk: ', self.abs_ep_jerk)
        left_offset = self.world.lane_pos0_solid_car_left
        right_offset = self.world.lane_pos0_car_right
        ob = self.get_ob(self.world._observation()[:-2])
        control = action[:2]
        return ob, rew, done, {'scene': self._get_scene(), 'terminal': terminal,
                               'v': v_now, 'acc': acc_now, 'left_offset': left_offset,
                               'right_offset': right_offset, 'control': control,
                               'current_pos': self.world.current_pos, 'yaw': np.array([self.world.yaw])}

    def get_ob(self, ob):
        self._ob.append(ob)
        return np.concatenate(self._ob, axis=0)

    def eprew(self):
        return self.world.ep_rew


# ==============================================================================
# -- DrivingForceControl -----------------------------------------------------------
# ==============================================================================


class DrivingForceControl(object):
    def __init__(self, world, start_in_autopilot):
        # init joystick
        pygame.joystick.init()
        self.world = world

        self._autopilot_enabled = start_in_autopilot
        if isinstance(world.vehicle, carla.Vehicle):
            self._control = carla.VehicleControl()
            # world.vehicle.set_autopilot(self._autopilot_enabled)
        else:
            raise NotImplementedError("Actor type not supported")
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        self.update_button()
        self.name = self.joystick.get_name()
        # self.world.throttle_brake = 0.
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def update_button(self):
        self.steering_axises = self.joystick.get_axis(0)
        self.throttle_axises = self.joystick.get_axis(2)
        self.brake_axises = self.joystick.get_axis(3)
        self.reverse_button = self.joystick.get_button(5)
        self.reset_button = self.joystick.get_button(3)
        self.camera_control_button = self.joystick.get_button(4)
        self.weather_control_button = self.joystick.get_button(2)
        self.sensor_control_button = self.joystick.get_button(0)
        self.recording_button = self.joystick.get_button(1)

    def parse_events(self, client, world, clock):
        self.last_throttle_axises = self.throttle_axises
        self.update_button()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if self.reset_button == 1:
                world.restart()
            elif self.camera_control_button == 1:
                world.camera_manager.toggle_camera()
            elif self.weather_control_button == 1:
                world.next_weather()
            elif self.sensor_control_button == 1:
                world.camera_manager.next_sensor()
            elif self.reverse_button == 1:
                self._control.gear = 1 if self._control.reverse else -1
        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(clock.get_time())
                self._control.gear = 1 if self._control.reverse else -1
            self.world.vehicle.apply_control(self._control)

    def _parse_vehicle_keys(self, milliseconds):
        # throttle brake processing
        if self.last_throttle_axises == 0 and self.throttle_axises == 0:
            throttle = 0
        else:
            throttle = self.throttle_axises * -0.5 + 0.5
        if self.brake_axises < 0:
            brake = 1
        else:
            brake = 0
        steer = self.steering_axises
        self._control.steer = steer
        self._control.throttle = throttle
        self._control.brake = brake
        self._control.reverse = 0


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        fonts = [x for x in pygame.font.get_fonts()]
        # mono is not installed in the cluster.
        # but it was there inside the docker. So...
        default_font = 'dejavusans'  # 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame_number = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame_number = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.simulation_time = timestamp.elapsed_seconds

        if not self._show_info:
            return
        t = world.vehicle.get_transform()
        v = world.vehicle.get_velocity()
        c = world.vehicle.get_control()
        heading = 'N' if abs(t.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(t.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > t.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > t.rotation.yaw > -179.5 else ''
        # colhist = world.collision_sensor.get_collision_history()
        # self.colhist = colhist
        # self.collision_now = colhist[-1]
        # collision = [colhist[x + self.frame_number - 200] for x in range(0, 200)]
        # max_col = max(1.0, max(collision))
        # collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        current_waypoint = world._map.get_waypoint(world.vehicle.get_location())
        recommanded_waypoints = world._map.get_spawn_points()  # recommanded waypoints

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

        def _dis(a, b):
            return ((b[1] - a[1]) ** 2 + (b[0] - a[0]) ** 2) ** 0.5

        current_pos = _pos(world.vehicle)
        self._info_text = [
            #'Server:  % 16f FPS' % self.server_fps,
            #'Client:  % 16f FPS' % world.clock.get_fps(),
            #'',
            #'Vehicle: % 20s' % get_actor_display_name(world.vehicle, truncate=20),
            #'Map:     % 20s' % world._map.name,
            #'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            #'',
            # 'LANE pos LOG reference car LEFT:% 20s' %('(%5.3f, %5.3f, %5.3f)' % (world.lane_poss_car[0][0][0], world.lane_poss_car[0][0][1], world.lane_poss_car[0][0][2])),
            # 'LANE pos LOG reference car RIGHT:% 20s' %('(%5.3f, %5.3f, %5.3f)' % (world.lane_poss_car[0][1][0], world.lane_poss_car[0][1][1], world.lane_poss_car[0][1][2])),
            # 'LANE pos LOG reference car solid_LEFT:% 20s' %('(%5.3f, %5.3f, %5.3f)' % (world.lane_poss_car[0][2][0], world.lane_poss_car[0][2][1], world.lane_poss_car[0][2][2])),
            # 'LANE pos LOG reference car solid RIGHT:% 20s' %('(%5.3f, %5.3f, %5.3f)' % (world.lane_poss_car[0][3][0], world.lane_poss_car[0][3][1], world.lane_poss_car[0][3][2])),
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)),
            #u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (t.rotation.yaw, heading),
            #'Current Pos:% 20s' % ('(% 5.3f, % 5.3f, % 5.3f)' % (
            #world.current_pos[0], world.current_pos[1], world.current_pos[2])),
            #'YAW:%20s' % ('%f' % (world.yaw)),
            #'PITCH:%20s' % ('%f' % (world.pitch)),
            #'ROLL:%20s' % ('%f' % (world.roll)),
            ##'Velocity reference car % 20s' % ('(%5.3f, %5.3f, %5.3f)' % (
            ##world.agent_velocity_car[0], world.agent_velocity_car[1], world.agent_velocity_car[2])),
            ##'Velocity world % 20s' % ('(%5.3f, %5.3f, %5.3f)' % (
            ##world.agent_velocity_world[0], world.agent_velocity_world[1], world.agent_velocity_world[2])),
            ##'ACC reference car % 20s' % ('(%5.3f, %5.3f, %5.3f)' % (
            ##world.agent_acc_car[0], world.agent_acc_car[1], world.agent_acc_car[2])),
            ##'ACC world % 20s' % ('(%5.3f, %5.3f, %5.3f)' % (
            ##world.agent_acc_world[0], world.agent_acc_world[1], world.agent_acc_world[2])),
            ## 'Road direction :% 20s' %('%s' % (world.road_direction)),
            ## 'Height:  % 18.3f m' % t.location.z,
            ## 'Cos:  % 18.3f m' % world.ang,
            ## 'Lane Width:  % 18.3f m' % world.lane_width,
            ## 'Dis now: % 18.3f m' % world.dis_now,
            'REW velocity:  % 18.3f' % world.v_rew,
            'REW angle:  % 18.3f' % world.ang_rew,
            'REW track:  % 18.3f' % world.track_rew,
            'REW final:  % 18.3f' % world.rew_now,
            'EP rew this episode:  % 18.3f' % world.ep_rew,
            'EP length this episode:  % 18.3f' % world.ep_len,
            ## 'AVG EP rew in 100 episodes % 18.3f' % np.mean(world.ep_rew_buffer),
            ## 'AVG EP len in 100 episodes % 18.3f' % np.mean(world.ep_len_buffer),
            ## 'Current Waypoint reference car :% 20s' %('(%5.3f, %5.3f)' % (world.current_waypoint_pos_car[0],world.current_waypoint_pos_car[1])),
            #'',
            ('Throttle:', c.throttle, 0.0, 1.0),
            ('Steer:', c.steer, -1.0, 1.0),
            ('Brake:', c.brake, 0.0, 1.0),
            #('Reverse:', c.reverse),
            #('Hand brake:', c.hand_brake),
            # '',
            # 'Collision:',
            # collision,
            # '',
            #'Number of vehicles: % 8d' % len(vehicles),
            #'V: {0}'.format(world.v_value),
            #'Fake_reward: {0}'.format(world.fake_reward),
            #'Zombie Speed: {0} km/h'.format(world.scenario_now.speed),
            #'Speed Freq Statistic: {0}'.format(world.speed_freq_statistic)
        ]

        if hasattr(world, "lateral_goal"):
            _addition_text = [
                'lateral goal % 3.0f ' % world.lateral_goal,
                'longitudinal goal % 3.0f ' % world.longitudinal_goal,
                'target velocity % 3.0f' % world.target_vel,
            ]
            self._info_text = self._info_text + _addition_text
        #if hasattr(world, "left_lanes"):
        #    _addition_text = [
        #        'left_lanes % 5d' % world.left_lanes,
        #        'right_lanes % 5d' % world.right_lanes,
        #    ]
        #    self._info_text = self._info_text + _addition_text
        #if hasattr(logger, "_dot"):
        #    _addition_text = [
        #        'Radian :  % 3.6f' % logger._dot,
        #        'Theta :  % 3.6f' % logger._theta
        #    ]
        #    self._info_text = self._info_text + _addition_text
        if hasattr(world, "status"):
            _addition_text = [
                'STATUS: % 3d' % world.status,
            ]
            self._info_text = self._info_text + _addition_text
        if hasattr(world, "ref"):
            _addition_text = [
                'REF: % 4d' % world.ref,
            ]
            self._info_text = self._info_text + _addition_text
        if hasattr(world, "_mode"):
            _addition_text = [
                'Mode: %s' % world._mode,
            ]
            self._info_text = self._info_text + _addition_text
        #if hasattr(world, "scenario_now") and hasattr(world.scenario_now, "cur_checkpoint"):
        #    _addition_text = [
        #        'Current Stage: %s' % world.scenario_now.cur_checkpoint,
        #    ]
        #    self._info_text = self._info_text + _addition_text
        #if hasattr(world, "rulebased_stage"):
        #    _addition_text = [
        #        'Rule-Based Stage: %s' % world.rulebased_stage,
        #    ]
        #    self._info_text = self._info_text + _addition_text
        #if hasattr(world, "pp_cur_vel"):
        #    _addition_text = [
        #        'PP current velocity: %s' % round(world.pp_cur_vel, 2),
        #        'PP target velocity: %s' % round(world.pp_tgt_vel, 2),
        #        'PP ref velocity 5m: %s' % round(world.pp_ref_vel, 2),
        #        'PP ref velocity 1m: %s' % round(world.pp_target_vel_pid, 2)
        #    ]
        #    self._info_text = self._info_text + _addition_text
        #if world.scenario_name == "OtherLeadingVehicle" or world.scenario_name == "OverTake":
        #    if hasattr(world, "zombiecar_worldpos"):
        #        _addition_text = [
        #           'Zombie Car I pos: % 20s' %('(%5.3f, %5.3f, %5.3f)' % (world.zombiecar_worldpos[0][0], world.zombiecar_worldpos[0][1], world.zombiecar_worldpos[0][2])),
        #           'Zombie Car II pos: % 20s' %('(%5.3f, %5.3f, %5.3f)' % (world.zombiecar_worldpos[1][0], world.zombiecar_worldpos[1][1], world.zombiecar_worldpos[1][2])),
        #           'Zombie Car I Velocity: % 20s' %('(%5.3f, %5.3f, %5.3f)' % (world.zombiecar_worldv[0][0], world.zombiecar_worldv[0][1], world.zombiecar_worldv[0][2])),
        #           'Zombie Car II Velocity: % 20s' %('(%5.3f, %5.3f, %5.3f)' % (world.zombiecar_worldv[1][0], world.zombiecar_worldv[1][1], world.zombiecar_worldv[1][2])),
        #           'Zombie Car I Acc: % 20s' %('(%5.3f, %5.3f, %5.3f)' % (world.zombiecar_worldacc[0][0], world.zombiecar_worldacc[0][1], world.zombiecar_worldacc[0][2])),
        #           'Zombie Car II Acc: % 20s' %('(%5.3f, %5.3f, %5.3f)' % (world.zombiecar_worldacc[1][0], world.zombiecar_worldacc[1][1], world.zombiecar_worldacc[1][2])),
        #           'Zombie Car I Transform: % 20s' %('(%5.3f, %5.3f, %5.3f)' % (world.zombiecar_transform[0][0], world.zombiecar_transform[0][1], world.zombiecar_transform[0][2])),
        #           'Zombie Car II Transform: % 20s' %('(%5.3f, %5.3f, %5.3f)' % (world.zombiecar_transform[1][0], world.zombiecar_transform[1][1], world.zombiecar_transform[1][2])),
        #           'Zombie Car I Speed: % 15.0f km/h' % world.zombiecar_speed[0],
        #           'Zombie Car II Speed: % 15.0f km/h' % world.zombiecar_speed[1],
        #        ]
        #        self._info_text = self._info_text + _addition_text
        #        if world.ep_len in logger.keyframes:
        #            keypointsinfo = [world.ep_len, {"Agent": [world.current_pos, world.agent_velocity_world, world.agent_acc_world, (world.yaw, world.pitch, world.roll), world.agent_angular_velocity], 
        #                                            "Zombie Cars I": [world.zombiecar_worldpos[0], world.zombiecar_worldv[0], world.zombiecar_worldacc[0], world.zombiecar_transform[0], world.zombiecar_angular_velocity[0]],
        #                                            "Zombie Cars II": [world.zombiecar_worldpos[1], world.zombiecar_worldv[1], world.zombiecar_worldacc[1], world.zombiecar_transform[1], world.zombiecar_angular_velocity[1]],}]
        #            if world.ep_len > logger.keyframes[-1]:
        #                pass
        #            else:
        #                for k, v in logger.keypoints.items():
        #                    if world.ep_len in v:
        #                        world.keypointsinfos[k].append(keypointsinfo)
        #                if world.ep_len == logger.keyframes[-1]:
        #                    print(world.keypointsinfos, "SAVING")
        #                    pickle.dump(world.keypointsinfos, world.keypoints_saver)
        #                    world.keypoints_saver.close()
        #                    print("KEYPOINTS INFO STORED")
        #if hasattr(world, "current_laneid"):
        #   _addition_text = [
        #       'RoadID: % 4d' % world.current_roadid,
        #       'LaneID: % 4d' % world.current_laneid,
        #       'LaneTYPE: % 10s' % str(world.current_lanetype),
        #       'LaneCHANGE: % 10s' %str(world.current_lanechange),
        #   ]
        #   self._info_text = self._info_text + _addition_text
        #if hasattr(world, "left_laneid"):
        #   _addition_text = [
        #       '',
        #       'Left RoadID: % 4d' % world.left_roadid,
        #       'Left LaneID: % 4d' % world.left_laneid,
        #       'Left LaneTYPE: % 10s' % str(world.left_lanetype),
        #       'Left LaneCHANGE: % 10s' %str(world.left_lanechange),
        #   ]
        #   self._info_text = self._info_text + _addition_text
        #if hasattr(world, "right_laneid"):
        #   _addition_text = [
        #       '',
        #       'Right RoadID: % 4d' % world.right_roadid,
        #       'Right LaneID: % 4d' % world.right_laneid,
        #       'Right LaneTYPE: % 10s' % str(world.right_lanetype),
        #       'Right LaneCHANGE: % 10s' %str(world.right_lanechange),
        #   ]
        #   self._info_text = self._info_text + _addition_text
        #if hasattr(world, "mid_laneid"):
        #   _addition_text = [
        #       '',
        #       'Mid RoadID: % 4d' % world.mid_roadid,
        #       'Mid LaneID: % 4d' % world.mid_laneid,
        #       'Mid LaneTYPE: % 10s' % str(world.mid_lanetype),
        #       'Mid LaneCHANGE: % 10s' %str(world.mid_lanechange),
        #   ]
        #   self._info_text = self._info_text + _addition_text
        # if hasattr(logger, "steering_from_pos"):
        #     _addition_text = [
        #         'REF DIR: % 20s' %('(%5.3f, %5.3f, %5.3f)' % (logger._ref_dir[0], logger._ref_dir[1], logger._ref_dir[2])),
        #         'CAR DIR: % 20s' %('(%5.3f, %5.3f, %5.3f)' % (logger._car_dir[0], logger._car_dir[1], logger._car_dir[2])),
        #         'REF POS: % 20s' %('(%5.3f, %5.3f, %5.3f)' % (logger._ref_pos[0], logger._ref_pos[1], logger._ref_pos[2])),
        #         'THETA :  % 3.6f' % logger._ref_theta,
        #         'CROSS: % 20s' %('(%5.3f, %5.3f, %5.3f)' % (logger._cross[0], logger._cross[1], logger._cross[2])),
        #         'CTE :  % 3.6f' % logger._cte,
        #         'Steer(POS): % 3.6f' % logger.steering_from_pos,
        #         'Steer(ANGLE): % 3.6f' % logger.steering_from_angle
        #     ]
        #     self._info_text = self._info_text + _addition_text
        #if len(vehicles) > 1:
        #    self._info_text += ['Nearby vehicles:']
        #    distance = lambda l: math.sqrt(
        #        (l.x - t.location.x) ** 2 + (l.y - t.location.y) ** 2 + (l.z - t.location.z) ** 2)
        #    vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.vehicle.id]
        #    for index in world.visible_zombie_cars:
        #        vehicle = world.zombie_cars[index[1]]
        #        d = distance(world.zombie_cars[index[1]].get_location())
        #        vehicle_type = get_actor_display_name(vehicle, truncate=22)
        #        self._info_text.append('% 4dm %s' % (d, vehicle_type))
        self._notifications.tick(world, self._server_clock)

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)

    #def render(self, display):
    #    if self._show_info:
    #        info_surface = pygame.Surface((230, self.dim[1]//8))
    #        info_surface.set_alpha(100)
    #        display.blit(info_surface, (80+100, 0))
    #        v_offset = 4
    #        bar_h_offset = 100+80+100
    #        bar_width = 106
    #        for item in self._info_text:
    #            if v_offset + 18 > self.dim[1]:
    #                break
    #            if isinstance(item, list):
    #                if len(item) > 1:
    #                    points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
    #                    pygame.draw.lines(display, (255, 136, 0), False, points, 2)
    #                item = None
    #                v_offset += 18
    #            elif isinstance(item, tuple):
    #                if isinstance(item[1], bool):
    #                    rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
    #                    pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
    #                else:
    #                    rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
    #                    pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
    #                    f = (item[1] - item[2]) / (item[3] - item[2])
    #                    if item[2] < 0.0:
    #                        rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
    #                    else:
    #                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
    #                    pygame.draw.rect(display, (255, 255, 255), rect)
    #                item = item[0]
    #            if item:  # At this point has to be a str.
    #                surface = self._font_mono.render(item, True, (255, 255, 255))
    #                display.blit(surface, (8+80+100, v_offset))
    #            v_offset += 18
    #    self._notifications.render(display)
    #    self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._history = []
        self._parent = parent_actor
        self._hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        self._collision = 0
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self._history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        self._collision = bool(self)
        actor_type = get_actor_display_name(event.other_actor)
        self._hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self._history.append((event.frame_number, intensity))
        if len(self._history) > 4000:
            self._history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self._hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_detector')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        self.crossed_lane_markings = 2
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.crossed_lane_markings = event.crossed_lane_markings
        text = ['%r' % str(x).split()[-1] for x in set(event.crossed_lane_markings)]
        self.crossed_lane_markings = [x for x in set(event.crossed_lane_markings)]
        self._hud.notification('Crossed line %s' % ' and '.join(text))


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._surface = None
        self._parent = parent_actor
        self._hud = hud
        self._recording = False
        self._draw_waypoints = True
        self._camera_transforms = [
            carla.Transform(carla.Location(x=1.6, z=1.7)),
            carla.Transform(carla.Location(x=0, z=90), carla.Rotation(pitch=-90)),
            carla.Transform(carla.Location(x=0, z=50), carla.Rotation(pitch=-90)),
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15))]
        self._transform_index = 2
        self._sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
             'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self._sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
            item.append(bp)
        self._index = None

    def toggle_waypoint(self):
        self._draw_waypoints = not self._draw_waypoints

    def toggle_camera(self):
        self._transform_index = (self._transform_index + 1) % len(self._camera_transforms)
        self.sensor.set_transform(self._camera_transforms[self._transform_index])

    def set_sensor(self, index, notify=True):
        index = index % len(self._sensors)
        needs_respawn = True if self._index is None \
            else self._sensors[index][0] != self._sensors[self._index][0]
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self._surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self._sensors[index][-1],
                self._camera_transforms[self._transform_index],
                attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self._hud.notification(self._sensors[index][2])
        self._index = index

    def next_sensor(self):
        self.set_sensor(self._index + 1)

    def toggle_recording(self):
        self._recording = not self._recording
        self._hud.notification('Recording %s' % ('On' if self._recording else 'Off'))

    def render(self, display):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == K_TAB:
                    self.toggle_camera()
                if event.key == K_w:
                    self.toggle_waypoint()

        if self._surface is not None:
            display.blit(self._surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self._sensors[self._index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 3), 3))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self._hud.dim) / 100.0
            lidar_data += (0.5 * self._hud.dim[0], 0.5 * self._hud.dim[1])
            lidar_data = np.fabs(lidar_data)
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self._hud.dim[0], self._hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self._surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self._sensors[self._index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self._surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self._recording:
            image.save_to_disk('_out/%08d' % image.frame_number)


class ActionNoise(object):
    def reset(self):
        pass


class NormalActionNoise(ActionNoise):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        return np.random.normal(self.mu, self.sigma)

    def __repr__(self):
        return 'NormalActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================
def _retrieve_goal_wp(world, longitudinal_goal, lateral_goal):
    def lateral_shift_wp(world, transform, shift):
        transform.rotation.yaw += 90
        shift_location = transform.location + shift * transform.get_forward_vector()
        w = world._map.get_waypoint(shift_location, project_to_road=False)
        return w

    def _valid_wp(wp, wp_ref):
        return not (wp.lane_id * wp_ref.lane_id < 0 or wp.lane_id == wp_ref.lane_id or wp.road_id != wp_ref.road_id)

    wp_select = world.waypoints_queue_equal_l[0][int(longitudinal_goal)][0]
    current_wp = world.current_wp
    # wp_refline = self.world.waypoints
    need_reset = False
    # retrieve possible lane-changing goals
    shifts = [-3.5, 0, 3.5]
    wp_select_possible = []
    current_wps_shift = []
    for shift in shifts:
        wp_select_possible.append(lateral_shift_wp(world, wp_select.transform, shift))
        current_wps_shift.append(lateral_shift_wp(world, current_wp.transform, shift))
    wp_select_shift = wp_select_possible[lateral_goal]
    current_wp_shift = current_wps_shift[lateral_goal]
    if current_wp_shift is None or wp_select_shift is None:
        need_reset = True
        reset_type = "No waypoint"
        return None, need_reset, reset_type, current_wps_shift
    else:
        # if lateral goal is not 1 (current lane), the choosing lane must not be a solid lane
        if not world.scenario:
            if lateral_goal != 1 and (
                    not _valid_wp(current_wp, current_wp_shift) or not _valid_wp(wp_select, wp_select_shift)):
                need_reset = True
                reset_type = "INVALID LANE-CHANGING"
                wp_goal = wp_select_shift
            elif world.to_intersection < 20 and lateral_goal != 1:
                need_reset = True
                reset_type = "INVALID LANE-CHANGING BEFORE INTERSECTION"
                wp_goal = wp_select_shift
            else:
                reset_type = "VALID"
                wp_goal = wp_select_shift
        else:
            wp_goal = wp_select_shift
            reset_type = "VALID"
        return wp_goal, need_reset, reset_type, current_wps_shift


def _recover_world_coordinate(coordinate, traj, mode="3d"):
    # obtain locations
    locations_car = []
    locations_world = []
    if mode == "3d":
        for point in traj:
            locations_car.append([point.x, point.y, point.z])
        for location_car in locations_car:
            location_world = coordinate.transform_world3d(location_car)
            locations_world.append(location_world)
    elif mode == "2d":
        for point in traj:
            locations_car.append([point.x, point.y])
        for location_car in locations_car:
            location_world = coordinate.transform_world2d(location_car)
            np.append(location_world, 0.)
            locations_world.append(location_world)
    return locations_world


def fake_step(world, action=[0, 1], start_v=6.4, wait_steps=0):
    world.steer = action[0]
    fake_step_time_limit = 300 if world.sync else 500
    fake_step_time = 0
    _control = carla.VehicleControl()
    wait_steps_count = 0
    while world.v_norm_world < start_v and fake_step_time < fake_step_time_limit:
        if wait_steps_count > wait_steps or wait_steps == 0:
            steer, throttle_brake = action[0], action[1]
            steer = np.clip(steer, -1, 1)
            throttle_brake = np.clip(throttle_brake, -1, 1)
            if throttle_brake < 0:
                throttle = 0
                brake = 1
            else:
                throttle = np.clip(throttle_brake, 0, 1)
                brake = 0
            action = np.array([steer, throttle, brake])
            terminal = False
            _control.steer = float(action[0])
            _control.throttle = float(action[1])
            _control.brake = float(action[2])
            world.apply_control(_control)
            fake_step_time += 1
        else:
            steer, throttle, brake = 0., 0., 0.
            _action = np.array([steer, throttle, brake])
            _control.steer = float(_action[0])
            _control.throttle = float(_action[1])
            _control.brake = float(_action[2])
            world.apply_control(_control)
            wait_steps_count += 1
            fake_step_time += 1


def parse_goals(lateral_goal, longitudinal_goal, target_vel):
    # here target velocity is in km/h
    longitudinal_goal = longitudinal_goal * 10.0 + 10.
    if longitudinal_goal == 0.:
        longitudinal_goal = 5.0
    target_vel = target_vel * 10.0 + 10.
    return lateral_goal, longitudinal_goal, target_vel


def generate_refline(refline_pos):
    refline = []
    for i in range(len(refline_pos) - 1):
        pos = refline_pos[i]
        pos_next = refline_pos[i + 1]
        x, y, z = pos[0], pos[1], pos[2]
        dir_x, dir_y, dir_z = (pos_next[0] - pos[0]) / 0.05, (pos_next[1] - pos[1]) / 0.05, (
                    pos_next[2] - pos[2]) / 0.05
        dir_xyz = [dir_x, dir_y, dir_z]
        dir_norm = utils._norm3d(dir_xyz)
        dir_x, dir_y, dir_z = dir_x / dir_norm, dir_y / dir_norm, dir_z / dir_norm
        point = TrajectoryPoint()
        point.x, point.y, point.z, point.dir_x, point.dir_y, point.dir_z = x, y, z, dir_x, dir_y, dir_z
        refline.append(point)
    return refline


def game_loop(args):
    world = None
    timenow = time.strftime("%Y%m%d%H%M%S", time.localtime())
    filename = '../log/' + "%s" % (timenow) + '_' + str(args.num_trajectories) + 'traj' + '_' + str(
        args.length) + 'len' + '_' + str(args.dim) + '_' + str(args.dis_max)
    outfile = open(filename + '.pkl', 'wb')

    ckpt_dir = '../log/'
    tfboard = TensorBoardOutputFormat(ckpt_dir)

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(5.0)

        sigmas = {'sigma_pos': 0.3, 'sigma_vel_upper': 3, 'sigma_vel_lower': 0.6, 'sigma_ang': 0.4}
        world = World(client.get_world(), args.sync, sigmas=sigmas, A_skip=args.A_skip, other_cars=args.other_cars,
                      autopilot=args.autopilot, camera=args.render,
                      maxlen=args.maxlen, dim=args.dim, dis_max=args.dis_max, render=args.render, width=args.width,
                      height=args.height, draw_features=args.draw_features,
                      farther_features=args.farther_features, all_green_light=args.all_green_light,
                      curriculumn_threshold=args.curriculumn_threshold, mode=args.mode,
                      max_lanes=args.max_lanes, scenario_name=args.scenario_name)
        controller = DrivingForceControl(world, args.autopilot)
        clock = pygame.time.Clock()
        frames = 0
        ep_frames = 0
        total_times = args.length * args.num_trajectories
        print("Max length: ", args.length)
        print("Max trajectories: ", args.num_trajectories)
        if args.scenario_name != 'OtherLeadingVehicle_FullMap':
            if args.scenario_name == 'Cross_Join' or args.scenario_name == 'Cross_Turn_Right':
                fake_step(world, start_v=7, wait_steps=200)
            elif args.scenario_name == 'Cross_Follow':
                fake_step(world, start_v=7, wait_steps=50)
            elif args.scenario_name == 'OtherLeadingVehicle':
                fake_step(world, start_v=9.3)
            elif args.scenario_name == 'Cross_Turn_Left' or args.scenario_name == 'Ring_Join':
                fake_step(world, start_v=7, wait_steps=150)
            elif args.scenario_name == 'Cross_Follow_Double':
                fake_step(world, start_v=9, wait_steps=0)
            else:
                fake_step(world, start_v=7)
        else:
            fake_step(world, start_v=7)
        print(frames, "FRAMES INIT")

        obs_ep = []
        while True:
            if controller.parse_events(client, world, clock):
                return
            try:
                observation_ac = world.next_frame()
            except:
                continue
            observation, ac = observation_ac[:-2], observation_ac[-2:]
            current_pos = world.current_pos
            # observation_ac = observation+current_pos+ac
            observation_ac = np.concatenate((observation, current_pos, ac))
            # print(len(observation), len(ac), len(current_pos))
            if isinstance(observation_ac, bool):
                pass
            else:
                obs_ep.append(observation_ac)
                if world.collision():#_sensor._collision:
                    print("COLLISION DETECTION")
                    world.restart()
                    if args.scenario_name != 'OtherLeadingVehicle_FullMap':
                        if args.scenario_name == 'Cross_Join' or args.scenario_name == 'Cross_Turn_Right':
                            fake_step(world, start_v=7, wait_steps=200)
                        elif args.scenario_name == 'Cross_Follow':
                            fake_step(world, start_v=7, wait_steps=50)
                        elif args.scenario_name == 'OtherLeadingVehicle':
                            fake_step(world, start_v=9.3)
                        elif args.scenario_name == 'Cross_Turn_Left' or args.scenario_name == 'Ring_Join':
                            fake_step(world, start_v=7, wait_steps=150)
                        elif args.scenario_name == 'Cross_Follow_Double':
                            fake_step(world, start_v=9, wait_steps=0)
                        else:
                            fake_step(world, start_v=7)
                    else:
                        fake_step(world, start_v=7)
                    obs_ep = []
                    ep_frames = 0
                    continue
                frames += 1
                ep_frames += 1
                world.ep_len = ep_frames
                world.ep_rew += world._reward()
                if frames == total_times:
                    break
                if args.fixed_length:
                    if ep_frames % args.length == 0:
                        if world.ep_rew > 50.:
                            for obs in obs_ep:
                                pickle.dump(obs, outfile)
                        obs_ep = []
                        ep_frames = 0
                        if args.scenario_name != 'OtherLeadingVehicle_FullMap':
                            world.restart()
                            if args.scenario_name == 'Cross_Join' or args.scenario_name == 'Cross_Turn_Right':
                                fake_step(world, start_v=7, wait_steps=200)
                            elif args.scenario_name == 'Cross_Follow':
                                fake_step(world, start_v=7, wait_steps=50)
                            elif args.scenario_name == 'OtherLeadingVehicle':
                                fake_step(world, start_v=9.3)
                            elif args.scenario_name == 'Cross_Turn_Left' or args.scenario_name == 'Ring_Join':
                                fake_step(world, start_v=7, wait_steps=150)
                            elif args.scenario_name == 'Cross_Follow_Double':
                                fake_step(world, start_v=9, wait_steps=0)
                            else:
                                fake_step(world, start_v=7)
                        else:
                            fake_step(world, start_v=7)
                else:
                    if frames % 1000 == 0:
                        for obs in obs_ep:
                            pickle.dump(obs, outfile)
                        obs_ep = []
                        if args.scenario_name != 'OtherLeadingVehicle_FullMap':
                            world.restart()
                            if args.scenario_name == 'Cross_Join' or args.scenario_name == 'Cross_Turn_Right':
                                fake_step(world, start_v=7, wait_steps=200)
                            elif args.scenario_name == 'Cross_Follow':
                                fake_step(world, start_v=7, wait_steps=50)
                            elif args.scenario_name == 'OtherLeadingVehicle':
                                fake_step(world, start_v=9.3)
                            elif args.scenario_name == 'Cross_Turn_Left' or args.scenario_name == 'Ring_Join':
                                fake_step(world, start_v=7, wait_steps=150)
                            elif args.scenario_name == 'Cross_Follow_Double':
                                fake_step(world, start_v=9, wait_steps=0)
                            else:
                                fake_step(world, start_v=7)
                        else:
                            fake_step(world, start_v=7)
                        ep_frames = 0
                        print(frames, " Frames")

        print("----------------------")
        print("Frames are: ", frames)
        print("----------------------")
        outfile.close()

    finally:
        print("inside python finally block")
        if world is not None:
            world.destroy()
        pygame.quit()


def game_loop_replay(args):
    world = None
    # timenow = time.strftime("%Y%m%d%H%M%S", time.localtime())
    # filename = '../log/' + "%s" % (timenow) + '_' + str(args.num_trajectories) + 'traj' + '_' + str(args.dim) + '_' + str(args.dis_max)
    # outfile = open(filename + '.pkl', 'wb')

    ckpt_dir = '../log/'
    tfboard = TensorBoardOutputFormat(ckpt_dir)

    logger.actor_id = 0

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(5.0)

        sigmas = {'sigma_pos': 0.4, 'sigma_vel_upper': 3.6, 'sigma_vel_lower': 3.6, 'sigma_ang': 0.4}
        world = World(client.get_world(), args.sync, sigmas=sigmas, A_skip=args.A_skip, other_cars=args.other_cars,
                      autopilot=args.autopilot, camera=args.render,
                      maxlen=args.maxlen, dim=args.dim, dis_max=args.dis_max, render=args.render, width=args.width,
                      height=args.height, draw_features=args.draw_features,
                      farther_features=args.farther_features, all_green_light=args.all_green_light,
                      curriculumn_threshold=args.curriculumn_threshold, mode=args.mode,
                      max_lanes=args.lanes, scenario_name=args.scenario_name)
        # controller = DrivingForceControl(world, args.autopilot)
        clock = pygame.time.Clock()
        frames = 0
        ep_frames = 0
        total_times = args.num_length * args.num_trajectories
        print("Max length: ", args.num_length)
        print("Max trajectories: ", args.num_trajectories)
        if args.scenario_name != 'OtherLeadingVehicle_FullMap':
            if args.scenario_name == 'Cross_Join' or args.scenario_name == 'Cross_Turn_Right':
                fake_step(world, start_v=7, wait_steps=200)
            elif args.scenario_name == 'Cross_Follow':
                fake_step(world, start_v=7, wait_steps=50)
            elif args.scenario_name == 'OtherLeadingVehicle':
                fake_step(world, start_v=9.3)
            elif args.scenario_name == 'Cross_Turn_Left' or args.scenario_name == 'Ring_Join':
                fake_step(world, start_v=7, wait_steps=150)
            elif args.scenario_name == 'Cross_Follow_Double':
                fake_step(world, start_v=9, wait_steps=0)
            else:
                fake_step(world, start_v=7)
        else:
            fake_step(world, start_v=7)
        print(frames, "FRAMES INIT")

        obs_ep = []
        if args.replay:
            #file = open("/data/Program/log/deterministic_trpo_overtake100.pkl", "rb")
            file = open(args.expert_path, "rb")
            #file = open("/data/Program/log/Ring_Join_TRPO_curriculum.pkl", "rb")
        while True:
            # if controller.parse_events(client, world, clock):
            #    return
            try:
                observation_ac = world.next_frame()
            except:
                continue

            if args.replay:
                try:
                    item = pickle.load(file)
                except:
                    break
                control = carla.VehicleControl()
                control.steer = item[-2] * 1.05  # + np.random.normal(item[-2], item[-2]/10)
                control.throttle = item[-1] * 1.05  # + np.random.normal(item[-1], item[-1]/10)
                # control.brake = 0
                # control.reverse = 0
                # print('control.steer: {0}, control.throttle: {1}'.format(item[-2], item[-1]))
                world.vehicle.apply_control(control)

            observation, ac = observation_ac[:-2], observation_ac[-2:]
            current_pos = world.current_pos
            observation_ac = np.concatenate((observation, current_pos, ac))
            # print(len(observation), len(ac), len(current_pos))
            if isinstance(observation_ac, bool):
                pass
            else:
                obs_ep.append(observation_ac)
                if world.collision():#_sensor._collision:
                    print("COLLISION DETECTION")
                    while ep_frames != 199:
                        item = pickle.load(file)
                        ep_frames += 1
                        frames += 1

                    world.restart()
                    if args.scenario_name != 'OtherLeadingVehicle_FullMap':
                        if args.scenario_name == 'Cross_Join' or args.scenario_name == 'Cross_Turn_Right':
                            fake_step(world, start_v=7, wait_steps=200)
                        elif args.scenario_name == 'Cross_Follow':
                            fake_step(world, start_v=7, wait_steps=50)
                        elif args.scenario_name == 'OtherLeadingVehicle':
                            fake_step(world, start_v=9.3)
                        elif args.scenario_name == 'Cross_Turn_Left' or args.scenario_name == 'Ring_Join':
                            fake_step(world, start_v=7, wait_steps=150)
                        elif args.scenario_name == 'Cross_Follow_Double':
                            fake_step(world, start_v=9, wait_steps=0)
                        else:
                            fake_step(world, start_v=7)
                    else:
                        fake_step(world, start_v=7)
                    obs_ep = []
                    ep_frames = 0
                    continue
                frames += 1
                ep_frames += 1
                world.ep_len = ep_frames
                world.ep_rew += world._reward()
                if frames == total_times:
                    break
                if args.fixed_length:
                    if ep_frames % args.num_length == 0:
                        # if world.ep_rew > 50.:
                        #     for obs in obs_ep:
                        #         pickle.dump(obs, outfile)
                        obs_ep = []
                        ep_frames = 0
                        if args.scenario_name != 'OtherLeadingVehicle_FullMap':
                            world.restart()
                            if args.scenario_name == 'Cross_Join' or args.scenario_name == 'Cross_Turn_Right':
                                fake_step(world, start_v=7, wait_steps=200)
                            elif args.scenario_name == 'Cross_Follow':
                                fake_step(world, start_v=7, wait_steps=50)
                            elif args.scenario_name == 'OtherLeadingVehicle':
                                fake_step(world, start_v=9.3)
                            elif args.scenario_name == 'Cross_Turn_Left' or args.scenario_name == 'Ring_Join':
                                fake_step(world, start_v=7, wait_steps=150)
                            elif args.scenario_name == 'Cross_Follow_Double':
                                fake_step(world, start_v=9, wait_steps=0)
                            else:
                                fake_step(world, start_v=7)
                        else:
                            fake_step(world, start_v=7)
                else:
                    if frames % 1000 == 0:
                        # for obs in obs_ep:
                        #     pickle.dump(obs, outfile)
                        obs_ep = []
                        if args.scenario_name != 'OtherLeadingVehicle_FullMap':
                            world.restart()
                            if args.scenario_name == 'Cross_Join' or args.scenario_name == 'Cross_Turn_Right':
                                fake_step(world, start_v=7, wait_steps=200)
                            elif args.scenario_name == 'Cross_Follow':
                                fake_step(world, start_v=7, wait_steps=50)
                            elif args.scenario_name == 'OtherLeadingVehicle':
                                fake_step(world, start_v=9.3)
                            elif args.scenario_name == 'Cross_Turn_Left' or args.scenario_name == 'Ring_Join':
                                fake_step(world, start_v=7, wait_steps=150)
                            elif args.scenario_name == 'Cross_Follow_Double':
                                fake_step(world, start_v=9, wait_steps=0)
                            else:
                                fake_step(world, start_v=7)
                        else:
                            fake_step(world, start_v=7)
                        ep_frames = 0
                        print(frames, " Frames")

        print("----------------------")
        print("Frames are: ", frames)
        print("----------------------")
        # outfile.close()

    finally:
        print("inside python finally block")
        if world is not None:
            world.destroy()
        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(description='CARLA Manual Control Client')
    argparser.add_argument('-v', '--verbose', action='store_true', dest='debug')
    argparser.add_argument('--host', metavar='H', default='127.0.0.1', help='IP of the host server')
    argparser.add_argument('-p', '--port', metavar='P', default=2000, type=int, help='TCP port to listen to')
    argparser.add_argument('-a', '--autopilot', action='store_true', help='enable autopilot')
    argparser.add_argument('--res', metavar='WIDTHxHEIGHT', default='1000x1000', help='window resolution')
    argparser.add_argument('--num_length', metavar='LENGTH', default=200, type=int, help='trajectory length')
    argparser.add_argument('--num_trajectories', metavar='TRA', default=100, type=int, help='maximum trajectory number')
    argparser.add_argument('--other_cars', metavar='0', default=0, type=int, help='other vehicle''s type')
    argparser.add_argument('--maxlen', metavar='M', default=100, type=int, help='Max len history')
    argparser.add_argument('--debug', action='store_true', help='show the debug info')
    argparser.add_argument('--debug_key', metavar='K', default='current_pos', type=str, help='key of the debug info\
            , choices are current_pos, waypoints_pos_forward,waypoint_nearest,\
            waypoint_last,stopline_pos,light')
    argparser.add_argument('--dim', metavar='D', default='2d', type=str, help='observations dim, 2d and 3d')
    argparser.add_argument('--dis_max', metavar='DM', default=1.3, type=float, help='max distances')
    argparser.add_argument('--fixed_length', action='store_true', help='fixed sample length')
    argparser.add_argument('--action_noise', action='store_true', help='add action noise')
    argparser.add_argument('--acstd', metavar='STD', default=0.1, type=float, help='action noise std')
    argparser.add_argument('--acmean', metavar='MEAN', default=0., type=float, help='action noise mean')
    argparser.add_argument('--sync', action='store_true', help='sync mode server')
    argparser.add_argument('--A_skip', default=1, type=int, help='sync mode server')
    argparser.add_argument('--D_skip', default=1, type=int, help='sync mode server')
    argparser.add_argument('--render', action='store_true')
    argparser.add_argument('--draw_features', action='store_true',
                           help='Save and Draw world-frame features instead of Car-frame, for testing')
    argparser.add_argument('--all_green_light', action='store_true', help='all green light')
    argparser.add_argument('--farther_features', choices=['end', 'rand'], default='rand',
                           help='expert see randomly after 2nd intersection or noting')
    argparser.add_argument('--curriculumn_threshold', type=int, default=2000,
                           help='num of episode where curriculumn learning kicks in')
    argparser.add_argument('--mode', choices=['all', 'wp', 'wp_obj'], default='all',
                           help='num of episode where curriculumn learning kicks in')
    argparser.add_argument('--controller', type=str, default='none', help='controller for collecting data')
    argparser.add_argument('--lanes', type=int, default=3, help='max lanes')
    # argparser.add_argument('--scenario_name', default='OtherLeadingVehicle', type=str,
    #                        choices=['OtherLeadingVehicle', 'CarFollowing', 'OtherLeadingVehicle_FullMap', 'Cross_Join',
    #                                 'Ring_Join', 'Straight_Follow_Single', 'Straight_Follow_Double', 'Cross_Follow',
    #                                 'Cross_Turn_Left', 'Cross_Turn_Right'], help='Scenarios')
    argparser.add_argument('--scenario_name', default='OtherLeadingVehicle', type=str,
                           help='Scenarios')
    argparser.add_argument('--expert_path', default='', type=str,
                           help='expert_data')
    argparser.add_argument('--replay', action='store_true',
                           help='Replay the supervised data')

    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', args.host, args.port)

    # print(__doc__)
    # print(args)

    try:
        if args.replay:
            game_loop_replay(args)
        else:
            game_loop(args)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
    except Exception as error:
        logging.exception(error)


if __name__ == '__main__':
    main()
