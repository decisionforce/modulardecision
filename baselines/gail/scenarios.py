import carla
import time
import numpy as np
import shapely.geometry
import shapely.affinity
from utils import _dis3d, _pos3d
from agents.navigation.basic_agent import *
from agents.navigation.roaming_agent import *
from agents.navigation.controller import VehiclePIDController
import random
from collections import deque
from enum import Enum
from agents.tools.misc import distance_vehicle, draw_waypoints
import pickle
from baselines import logger

# import pdb


# try:
#     from agents.navigation.local_planner import _retrieve_options as retrieve_options
# except:
#     print('0.9.4')
#     pass

def detect_lane_obstacle(world, actor, extension_factor=3, margin=1.02):
    """
    This function identifies if an obstacle is present in front of the reference actor
    """
    # world = CarlaDataProvider.get_world()
    world_actors = world.get_actors().filter('vehicle.*')
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


class RotatedRectangle(object):
    """
    This class contains method to draw rectangle and find intersection point.
    """

    def __init__(self, c_x, c_y, width, height, angle):
        self.c_x = c_x
        self.c_y = c_y
        self.w = width  # pylint: disable=invalid-name
        self.h = height  # pylint: disable=invalid-name
        self.angle = angle

    def get_contour(self):
        """
        create contour
        """
        w = self.w
        h = self.h
        c = shapely.geometry.box(-w / 2.0, -h / 2.0, w / 2.0, h / 2.0)
        rc = shapely.affinity.rotate(c, self.angle)
        return shapely.affinity.translate(rc, self.c_x, self.c_y)

    def intersection(self, other):
        """
        Obtain a intersection point between two contour.
        """
        return self.get_contour().intersection(other.get_contour())


class RoadOption(Enum):
    """
    RoadOption represents the possible topological configurations when moving from a segment of lane to other.
    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4


def retrieve_options(list_waypoints, current_waypoint):
    """
    Compute the type of connection between the current active waypoint and the multiple waypoints present in
    list_waypoints. The result is encoded as a list of RoadOption enums.

    :param list_waypoints: list with the possible target waypoints in case of multiple options
    :param current_waypoint: current active waypoint
    :return: list of RoadOption enums representing the type of connection from the active waypoint to each
             candidate in list_waypoints
    """
    options = []
    for next_waypoint in list_waypoints:
        # this is needed because something we are linking to
        # the beggining of an intersection, therefore the
        # variation in angle is small
        next_next_waypoint = next_waypoint.next(3.0)[0]
        link = _compute_connection(current_waypoint, next_next_waypoint)
        options.append(link)

    return options


def _compute_connection(current_waypoint, next_waypoint):
    """
    Compute the type of topological connection between an active waypoint (current_waypoint) and a target waypoint
    (next_waypoint).

    :param current_waypoint: active waypoint
    :param next_waypoint: target waypoint
    :return: the type of topological connection encoded as a RoadOption enum:
             RoadOption.STRAIGHT
             RoadOption.LEFT
             RoadOption.RIGHT
    """
    n = next_waypoint.transform.rotation.yaw
    n = n % 360.0

    c = current_waypoint.transform.rotation.yaw
    c = c % 360.0

    diff_angle = (n - c) % 180.0
    if diff_angle < 1.0:
        return RoadOption.STRAIGHT
    elif diff_angle > 90.0:
        return RoadOption.LEFT
    else:
        return RoadOption.RIGHT


# helper functions, mostly are planners
class WaypointFollower(object):
    """
    This is an atomic behavior to follow waypoints indefinitely
    while maintaining a given speed or if given a waypoint plan,
    follows the given plan
    """

    def __init__(self, actor, target_speed, plan=None,
                 avoid_collision=False, name="FollowWaypoints", map=None):
        """
        Set up actor and local planner
        """
        self._actor_list = []
        self._actor_list.append(actor)
        # print('\n\ninit_actor: ', actor)
        self._target_speed = target_speed
        self._local_planner_list = []
        self._plan = plan
        self._args_lateral_dict = {'K_P': 1.0, 'K_D': 0.01, 'K_I': 0.0, 'dt': 0.05}
        self._avoid_collision = avoid_collision
        self._map = map

    def setup(self, timeout=5):
        """
        Delayed one-time initialization
        """
        for actor in self._actor_list:
            # print('\n\nactor: ', actor)
            self._apply_local_planner(actor, self._map)

        return True

    def _apply_local_planner(self, actor, map):
        local_planner = WpFollowplanner(
            actor=actor,
            map=map,
            opt_dict={
                'target_speed': self._target_speed,
                'lateral_control_dict': self._args_lateral_dict})
        if self._plan is not None:
            local_planner.set_global_plan(self._plan)
        self._local_planner_list.append(local_planner)

    def update(self):
        """
        Run local planner, obtain and apply control to actor
        """
        for actor, local_planner in zip(self._actor_list, self._local_planner_list):
            if actor is not None and actor.is_alive and local_planner is not None:
                control = local_planner.run_step(debug=False)
                # if self._avoid_collision and detect_lane_obstacle(actor):
                #     control.throttle = 0.0
                #     control.brake = 1.0
                actor.apply_control(control)


class WpFollowplanner(object):
    MIN_DISTANCE_PERCENTAGE = 0.9

    def __init__(self, actor, map, opt_dict):
        self._vehicle = actor
        self._map = map

        self._dt = None
        self._target_speed = None
        self._sampling_radius = None
        self._min_distance = None
        self._current_waypoint = None
        self._target_road_option = None
        self._next_waypoints = None
        self._target_waypoint = None
        self._vehicle_controller = None
        self._global_plan = None
        # queue with tuples of (waypoint, RoadOption)
        self._waypoints_queue = deque(maxlen=600)
        self._buffer_size = 5
        self._waypoint_buffer = deque(maxlen=self._buffer_size)

        # initializing controller
        self.init_controller(opt_dict)

    def init_controller(self, opt_dict):
        """
        Controller initialization.

        :param opt_dict: dictionary of arguments.
        :return:
        """
        # default params
        self._dt = 1.0 / 20.0
        self._target_speed = 20.0  # Km/h
        self._sampling_radius = self._target_speed * 0.5 / 3.6  # 0.5 seconds horizon
        self._min_distance = self._sampling_radius * self.MIN_DISTANCE_PERCENTAGE

        args_lateral_dict = {
            'K_P': 1.95,
            'K_D': 0.01,
            'K_I': 1.4,
            'dt': self._dt}
        args_longitudinal_dict = {
            'K_P': 1.0,
            'K_D': 0,
            'K_I': 1,
            'dt': self._dt}

        # parameters overload
        if 'dt' in opt_dict:
            self._dt = opt_dict['dt']
        if 'target_speed' in opt_dict:
            self._target_speed = opt_dict['target_speed']
        if 'sampling_radius' in opt_dict:
            self._sampling_radius = self._target_speed * \
                                    opt_dict['sampling_radius'] / 3.6
        if 'lateral_control_dict' in opt_dict:
            args_lateral_dict = opt_dict['lateral_control_dict']
        if 'longitudinal_control_dict' in opt_dict:
            args_longitudinal_dict = opt_dict['longitudinal_control_dict']

        self._current_waypoint = self._map.get_waypoint(
            self._vehicle.get_location())
        self._vehicle_controller = VehiclePIDController(self._vehicle,
                                                        args_lateral=args_lateral_dict,
                                                        args_longitudinal=args_longitudinal_dict)

        self._global_plan = False

        # compute initial waypoints
        self._waypoints_queue.append((self._current_waypoint.next(self._sampling_radius)[0], RoadOption.LANEFOLLOW))
        self._target_road_option = RoadOption.LANEFOLLOW
        # fill waypoint trajectory queue
        self._compute_next_waypoints(k=200)

    def _compute_next_waypoints(self, k=1):
        """
        Add new waypoints to the trajectory queue.

        :param k: how many waypoints to compute
        :return:
        """
        # check we do not overflow the queue
        available_entries = self._waypoints_queue.maxlen - len(self._waypoints_queue)
        k = min(available_entries, k)

        for _ in range(k):
            last_waypoint = self._waypoints_queue[-1][0]
            next_waypoints = list(last_waypoint.next(self._sampling_radius))

            if len(next_waypoints) == 1:
                # only one option available ==> lanefollowing
                next_waypoint = next_waypoints[0]
                road_option = RoadOption.LANEFOLLOW
            else:
                # random choice between the possible options
                road_options_list = retrieve_options(
                    next_waypoints, last_waypoint)
                road_option = random.choice(road_options_list)
                next_waypoint = next_waypoints[road_options_list.index(
                    road_option)]

            self._waypoints_queue.append((next_waypoint, road_option))

    def run_step(self, debug=True):
        """
        Execute one step of local planning which involves running the longitudinal and lateral PID controllers to
        follow the waypoints trajectory.

        :param debug: boolean flag to activate waypoints debugging
        :return:
        """

        # not enough waypoints in the horizon? => add more!
        if len(self._waypoints_queue) < int(self._waypoints_queue.maxlen * 0.5):
            if not self._global_plan:
                self._compute_next_waypoints(k=100)

        if len(self._waypoints_queue) == 0:
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 0.0
            control.hand_brake = False
            control.manual_gear_shift = False

            return control

        # Buffering the waypoints
        if not self._waypoint_buffer:
            for i in range(self._buffer_size):
                if self._waypoints_queue:
                    self._waypoint_buffer.append(
                        self._waypoints_queue.popleft())
                else:
                    break

        # current vehicle waypoint
        self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        # target waypoint
        # self._target_waypoint, self._target_road_option = self._waypoint_buffer[0]
        # move using PID controllers
        self._target_waypoint = self._current_waypoint.next(5.0)[0]
        control = self._vehicle_controller.run_step(self._target_speed, self._target_waypoint)
        # purge the queue of obsolete waypoints
        vehicle_transform = self._vehicle.get_transform()
        max_index = -1

        for i, (waypoint, _) in enumerate(self._waypoint_buffer):
            if distance_vehicle(
                    waypoint, vehicle_transform) < self._min_distance:
                max_index = i
        if max_index >= 0:
            for i in range(max_index + 1):
                self._waypoint_buffer.popleft()
        if debug:
            draw_waypoints(self._vehicle.get_world(), [self._target_waypoint], self._vehicle.get_location().z + 1.0)
        return control


class OtherLeadingVehicle(object):
    def __init__(self, name, map, world):
        self.name = name
        self._map = map
        self.world = world
        self.speed = 0
        self.blueprint_library = self.world.get_blueprint_library()
        self._scenario_init()

    def _scenario_init(self):
        # init hero car
        # --------------------------------------------------------
        # setup cars on a given waypoint
        hero_car_pos = [388.9, -140, 0]
        wp_location = carla.Location(x=hero_car_pos[0], y=hero_car_pos[1], z=hero_car_pos[2])
        wp = self._map.get_waypoint(wp_location)
        hero_vehicle_transform = wp.transform
        hero_model = 'vehicle.lincoln.mkz2017'
        blueprint = random.choice(self.blueprint_library.filter(hero_model))
        blueprint.set_attribute('role_name', 'hero')
        self.hero_car = self.world.try_spawn_actor(blueprint, hero_vehicle_transform)

        # init zombie cars
        first_vehicle_location = 25
        second_vehicle_location = first_vehicle_location + 8
        first_vehicle_waypoint = wp.next(first_vehicle_location)[0]
        second_vehicle_waypoint = wp.next(second_vehicle_location)[0].get_left_lane()
        first_vehicle_transform = carla.Transform(first_vehicle_waypoint.transform.location,
                                                  first_vehicle_waypoint.transform.rotation)
        second_vehicle_transform = carla.Transform(second_vehicle_waypoint.transform.location,
                                                   second_vehicle_waypoint.transform.rotation)

        models = ['vehicle.nissan.patrol', 'vehicle.audi.tt']
        blueprints = [random.choice(self.world.get_blueprint_library().filter(model)) for model in models]
        for blueprint in blueprints:
            blueprint.set_attribute('role_name', 'scenario')
        self.first_vehicle = self.world.try_spawn_actor(blueprints[0], first_vehicle_transform)
        self.second_vehicle = self.world.try_spawn_actor(blueprints[1], second_vehicle_transform)
        self.zombie_cars = [self.first_vehicle, self.second_vehicle]
        # --------------------------------------------------------
        # --------------------------------------------------------
        # setup local planners for zombie cars
        self._first_vehicle_speed = 36 / 3.2
        self._second_vehicle_speed = 45
        first_vehicle_planner = WaypointFollower(self.zombie_cars[0], self._first_vehicle_speed,map=self._map,
                                                         avoid_collision=True)
        second_vehicle_planner = WaypointFollower(self.zombie_cars[1], self._second_vehicle_speed,map=self._map,
                                                          avoid_collision=True)
        self.vehicle_planners = [first_vehicle_planner, second_vehicle_planner]
        for planner in self.vehicle_planners:
            planner.setup()

    def _update(self):
        # update action for two local planners
        if _dis3d(_pos3d(self.hero_car), _pos3d(self.first_vehicle)) > 26.:
            pass
        else:
            for planner in self.vehicle_planners:
                planner.update()

    def restart(self):
        self._remove_all_actors()
        self._scenario_init()

    def _remove_all_actors(self):
        actors = [self.hero_car] + self.zombie_cars
        for actor in actors:
            if actor.is_alive:
                actor.destroy()

    def _remove_zombie_cars(self):
        actors = self.zombie_cars
        for actor in actors:
            if actor.is_alive:
                actor.destroy()

class OverTake(object):
    def __init__(self, name, map, world, checkkeys):   
        self.name = name
        self._map = map
        self.world = world
        self.speed = 0
        self.keypointsinfos = logger.keypoints
        self.checkpoints = {k:v for k,v in self.keypointsinfos.items() if k in checkkeys}
        self.checkkeys = checkkeys
        self.checkframes = 1
        self.keychoice = 0
        self.framechoice = 0
        self.blueprint_library = self.world.get_blueprint_library()
        self._scenario_init()

    def _scenario_init(self):
        # init hero car
        # --------------------------------------------------------
        # setup cars on a given waypoint
        hero_car_pos = [388.9, -140, 0]
        wp_location = carla.Location(x=hero_car_pos[0], y=hero_car_pos[1], z=hero_car_pos[2])
        wp = self._map.get_waypoint(wp_location)
        hero_vehicle_transform = wp.transform
        hero_model = 'vehicle.lincoln.mkz2017'
        blueprint = random.choice(self.blueprint_library.filter(hero_model))
        blueprint.set_attribute('role_name', 'hero')
        self.hero_car = self.world.try_spawn_actor(blueprint, hero_vehicle_transform)

        # init zombie cars
        first_vehicle_location = 25
        second_vehicle_location = first_vehicle_location + 8
        first_vehicle_waypoint = wp.next(first_vehicle_location)[0]
        second_vehicle_waypoint = wp.next(second_vehicle_location)[0].get_left_lane()
        first_vehicle_transform = carla.Transform(first_vehicle_waypoint.transform.location,
                                                  first_vehicle_waypoint.transform.rotation)
        second_vehicle_transform = carla.Transform(second_vehicle_waypoint.transform.location,
                                                   second_vehicle_waypoint.transform.rotation)

        models = ['vehicle.nissan.patrol', 'vehicle.audi.tt']
        blueprints = [random.choice(self.world.get_blueprint_library().filter(model)) for model in models]
        for blueprint in blueprints:
            blueprint.set_attribute('role_name', 'scenario')
        self.first_vehicle = self.world.try_spawn_actor(blueprints[0], first_vehicle_transform)
        self.second_vehicle = self.world.try_spawn_actor(blueprints[1], second_vehicle_transform)
        self.zombie_cars = [self.first_vehicle, self.second_vehicle]


        checkinfo = self.checkpoints[self.checkkeys[self.keychoice%len(self.checkkeys)]][self.framechoice%self.checkframes]
        self.cur_checkpoint = self.checkkeys[self.keychoice%len(self.checkkeys)]
        self.framechoice += 1
        if self.framechoice%self.checkframes == 0:
            self.keychoice += 1

        # --------------------------------------------------------
        # --------------------------------------------------------
        # setup local planners for zombie cars
        self._first_vehicle_speed = 36 / 3.2
        self._second_vehicle_speed = 45
        first_vehicle_planner = WaypointFollower(self.zombie_cars[0], self._first_vehicle_speed,map=self._map,
                                                         avoid_collision=True)
        second_vehicle_planner = WaypointFollower(self.zombie_cars[1], self._second_vehicle_speed,map=self._map,
                                                          avoid_collision=True)
        self.vehicle_planners = [first_vehicle_planner, second_vehicle_planner]
        for planner in self.vehicle_planners:
            planner.setup()

    #def _scenario_init(self):
    #    # init hero car
    #    # --------------------------------------------------------
    #    # setup cars on a given waypoint
    #    hero_car_pos = [388.9, -140, 0]
    #    wp_location = carla.Location(x=hero_car_pos[0], y=hero_car_pos[1], z=hero_car_pos[2])
    #    wp = self._map.get_waypoint(wp_location)
    #    hero_vehicle_transform = wp.transform
    #    hero_model = 'vehicle.lincoln.mkz2017'
    #    blueprint = random.choice(self.blueprint_library.filter(hero_model))
    #    blueprint.set_attribute('role_name', 'hero')
    #    self.hero_car = self.world.try_spawn_actor(blueprint, hero_vehicle_transform)

    #    # init zombie cars
    #    first_vehicle_location = 25
    #    second_vehicle_location = first_vehicle_location + 8
    #    first_vehicle_waypoint = wp.next(first_vehicle_location)[0]
    #    second_vehicle_waypoint = wp.next(second_vehicle_location)[0].get_left_lane()
    #    first_vehicle_transform = carla.Transform(first_vehicle_waypoint.transform.location,
    #                                              first_vehicle_waypoint.transform.rotation)
    #    second_vehicle_transform = carla.Transform(second_vehicle_waypoint.transform.location,
    #                                               second_vehicle_waypoint.transform.rotation)

    #    models = ['vehicle.nissan.patrol', 'vehicle.audi.tt']
    #    blueprints = [random.choice(self.world.get_blueprint_library().filter(model)) for model in models]
    #    for blueprint in blueprints:
    #        blueprint.set_attribute('role_name', 'scenario')
    #    self.first_vehicle = self.world.try_spawn_actor(blueprints[0], first_vehicle_transform)
    #    self.second_vehicle = self.world.try_spawn_actor(blueprints[1], second_vehicle_transform)
    #    
    #    checkinfo = self.checkpoints[self.checkkeys[self.keychoice%len(self.checkkeys)]][self.framechoice%self.checkframes]
    #    checkinfo = checkinfo[1]
    #    self.cur_checkpoint = self.checkkeys[self.keychoice%len(self.checkkeys)]
    #    hero_car_info, first_vehicle_info, second_vehicle_info = checkinfo["Agent"], checkinfo["Zombie Cars I"], checkinfo["Zombie Cars II"]
    #    self._scenario_init_teleport(hero_car_info, first_vehicle_info, second_vehicle_info)
    #    self.framechoice += 1
    #    if self.framechoice%self.checkframes == 0:
    #        self.keychoice += 1
    
    def _scenario_init_teleport(self, hero_car_info, first_vehicle_info, second_vehicle_info):
        hero_car_pos, hero_car_v, hero_car_rotation, hero_car_angular_v = hero_car_info[0], hero_car_info[1], hero_car_info[3], hero_car_info[4]
        first_vehicle_pos, first_vehicle_v, first_vehicle_rotation, first_vehicle_angular_v = first_vehicle_info[0], first_vehicle_info[1], first_vehicle_info[3], first_vehicle_info[4]
        second_vehicle_pos, second_vehicle_v, second_vehicle_rotation, second_vehicle_angular_v = second_vehicle_info[0], second_vehicle_info[1], second_vehicle_info[3], second_vehicle_info[4]
        
        hero_car_impulse, first_vehicle_impulse, second_vehicle_impulse = hero_car_info[2], first_vehicle_info[2], second_vehicle_info[2] 

        # setup hero car        
        wp_location = carla.Location(x=hero_car_pos[0], y=hero_car_pos[1], z=hero_car_pos[2])
        wp_rotation = carla.Rotation()
        wp_rotation.yaw, wp_rotation.pitch, wp_rotation.roll = hero_car_rotation[0], hero_car_rotation[1], hero_car_rotation[2]
        hero_vehicle_transform = carla.Transform()
        hero_vehicle_transform.location, hero_vehicle_transform.rotation = wp_location, wp_rotation
        hero_vehicle_velocity = carla.Vector3D()
        hero_vehicle_velocity.x, hero_vehicle_velocity.y, hero_vehicle_velocity.z = hero_car_v[0], hero_car_v[1], hero_car_v[2]       
        hero_vehicle_angular_velocity = carla.Vector3D()
        hero_vehicle_angular_velocity.x, hero_vehicle_angular_velocity.y, hero_vehicle_angular_velocity.z = hero_car_angular_v[0], hero_car_angular_v[1], hero_car_angular_v[2]

        #self.hero_car.set_simulate_physics(False)
        self.hero_car.set_transform(hero_vehicle_transform)        
        self.hero_car.set_velocity(hero_vehicle_velocity)
        self.hero_car.set_angular_velocity(hero_vehicle_angular_velocity)

        hero_vehicle_impulse = carla.Vector3D()
        #hero_vehicle_impulse.x, hero_vehicle_impulse.y, hero_vehicle_impulse.z = hero_car_impulse[0], hero_car_impulse[1], hero_car_impulse[2]
        #self.hero_car.add_impulse(hero_vehicle_impulse)

        # setup zombie cars
        wp_location = carla.Location(x=first_vehicle_pos[0], y=first_vehicle_pos[1], z=first_vehicle_pos[2])
        wp_rotation = carla.Rotation()
        wp_rotation.yaw, wp_rotation.pitch, wp_rotation.roll = first_vehicle_rotation[0], first_vehicle_rotation[1], first_vehicle_rotation[2]
        first_vehicle_transform = carla.Transform()
        first_vehicle_transform.location, first_vehicle_transform.rotation = wp_location, wp_rotation
        first_vehicle_velocity = carla.Vector3D()
        first_vehicle_velocity.x, first_vehicle_velocity.y, first_vehicle_velocity.z = first_vehicle_v[0], first_vehicle_v[1], first_vehicle_v[2]
       
        first_vehicle_angular_velocity = carla.Vector3D()
        first_vehicle_angular_velocity.x, first_vehicle_angular_velocity.y, first_vehicle_angular_velocity.z = first_vehicle_angular_v[0], first_vehicle_angular_v[1], first_vehicle_angular_v[2] 
 
        #self.first_vehicle.set_simulate_physics(False)
        self.first_vehicle.set_transform(first_vehicle_transform)
        self.first_vehicle.set_velocity(first_vehicle_velocity)
        self.first_vehicle.set_angular_velocity(first_vehicle_angular_velocity)
        
        wp_location = carla.Location(x=second_vehicle_pos[0], y=second_vehicle_pos[1], z=second_vehicle_pos[2])
        wp_rotation = carla.Rotation()
        wp_rotation.yaw, wp_rotation.pitch, wp_rotation.roll = second_vehicle_rotation[0], second_vehicle_rotation[1], second_vehicle_rotation[2]
        second_vehicle_transform = carla.Transform()
        second_vehicle_transform.location, second_vehicle_transform.rotation = wp_location, wp_rotation
        second_vehicle_velocity = carla.Vector3D()
        second_vehicle_velocity.x, second_vehicle_velocity.y, second_vehicle_velocity.z = second_vehicle_v[0], second_vehicle_v[1], second_vehicle_v[2]

        second_vehicle_angular_velocity = carla.Vector3D()
        second_vehicle_angular_velocity.x, second_vehicle_angular_velocity.y, second_vehicle_angular_velocity.z = second_vehicle_angular_v[0], second_vehicle_angular_v[1], second_vehicle_angular_v[2] 
        self.second_vehicle.set_angular_velocity(second_vehicle_angular_velocity)
        
        #self.second_vehicle.set_simulate_physics(False)
        self.second_vehicle.set_transform(second_vehicle_transform)
        self.second_vehicle.set_velocity(second_vehicle_velocity)

        #self.first_vehicle.set_simulate_physics(True)
        #self.second_vehicle.set_simulate_physics(True)

        self.zombie_cars = [self.first_vehicle, self.second_vehicle]
        # --------------------------------------------------------
        # --------------------------------------------------------
        # setup local planners for zombie cars
        self._first_vehicle_speed = 36 / 3.2
        self._second_vehicle_speed = 45
        first_vehicle_planner = WaypointFollower(self.zombie_cars[0], self._first_vehicle_speed,map=self._map,
                                                         avoid_collision=True)
        second_vehicle_planner = WaypointFollower(self.zombie_cars[1], self._second_vehicle_speed,map=self._map,
                                                          avoid_collision=True)
        self.vehicle_planners = [first_vehicle_planner, second_vehicle_planner]
        for planner in self.vehicle_planners:
            planner.setup()
    
    def _update(self):
        # update action for two local planners
        if _dis3d(_pos3d(self.hero_car), _pos3d(self.first_vehicle)) > 26.:
            pass
        else:
            for planner in self.vehicle_planners:
                planner.update()

    def restart(self):
        self._remove_all_actors()
        self._scenario_init()

    def _remove_all_actors(self):
        actors = [self.hero_car] + self.zombie_cars
        for actor in actors:
            if actor.is_alive:
                actor.destroy()

    def _remove_zombie_cars(self):
        actors = self.zombie_cars
        for actor in actors:
            if actor.is_alive:
                actor.destroy()

class CarFollowing(object):
    def __init__(self, name, map, world):
        self.name = name
        self._map = map
        self.world = world
        self.blueprint_library = self.world.get_blueprint_library()
        self._scenario_init()

    def _scenario_init(self):
        # init hero car
        hero_car_pos = [388.9, -140, 0]
        wp_location = carla.Location(x=hero_car_pos[0], y=hero_car_pos[1], z=hero_car_pos[2])
        wp = self._map.get_waypoint(wp_location)
        hero_vehicle_transform = wp.transform
        hero_model = 'vehicle.lincoln.mkz2017'
        blueprint = random.choice(self.blueprint_library.filter(hero_model))
        blueprint.set_attribute('role_name', 'heroxxx')
        self.hero_car = self.world.try_spawn_actor(blueprint, hero_vehicle_transform)
        # self.hero_car.set_autopilot(enabled=True)

        # init zombie cars
        first_vehicle_location = 25
        first_vehicle_waypoint = wp.next(first_vehicle_location)[0]
        second_vehicle_waypoint = wp.next(first_vehicle_location)[0].get_left_lane()
        third_vehicle_waypoint = wp.next(first_vehicle_location)[0].get_right_lane()
        first_vehicle_transform = carla.Transform(first_vehicle_waypoint.transform.location,
                                                  first_vehicle_waypoint.transform.rotation)
        second_vehicle_transform = carla.Transform(second_vehicle_waypoint.transform.location,
                                                   second_vehicle_waypoint.transform.rotation)
        third_vehicle_transform = carla.Transform(third_vehicle_waypoint.transform.location,
                                                  third_vehicle_waypoint.transform.rotation)

        models = ['vehicle.nissan.patrol', 'vehicle.audi.tt', 'vehicle.lincoln.mkz2017']
        blueprints = [random.choice(self.world.get_blueprint_library().filter(model)) for model in models]
        for blueprint in blueprints:
            blueprint.set_attribute('role_name', 'scenario')
        self.first_vehicle = self.world.try_spawn_actor(blueprints[0], first_vehicle_transform)
        self.second_vehicle = self.world.try_spawn_actor(blueprints[1], second_vehicle_transform)
        self.third_vehicle = self.world.try_spawn_actor(blueprints[2], third_vehicle_transform)
        self.zombie_cars = [self.first_vehicle, self.second_vehicle, self.third_vehicle]

        # setup local planners for zombie cars
        self._first_vehicle_speed = 25
        self._second_vehicle_speed = 23
        self._third_vehicle_speed = 23
        first_vehicle_planner = WaypointFollower(self.zombie_cars[0], self._first_vehicle_speed,map=self._map,
                                                         avoid_collision=True)
        second_vehicle_planner = WaypointFollower(self.zombie_cars[1], self._second_vehicle_speed,map=self._map,
                                                          avoid_collision=True)
        third_vehicle_planner = WaypointFollower(self.zombie_cars[2], self._second_vehicle_speed,map=self._map,
                                                         avoid_collision=True)
        self.vehicle_planners = [first_vehicle_planner, second_vehicle_planner, third_vehicle_planner]
        for planner in self.vehicle_planners:
            planner.setup()

    def _update(self):
        # update action for two local planners
        for planner in self.vehicle_planners:
            planner.update()

    def restart(self):
        self._remove_all_actors()
        self._scenario_init()

    def _remove_all_actors(self):
        actors = [self.hero_car] + self.zombie_cars
        for actor in actors:
            if actor.is_alive:
                actor.destroy()

    def _remove_zombie_cars(self):
        actors = self.zombie_cars
        for actor in actors:
            if actor.is_alive:
                actor.destroy()

# helper functions, mostly are planners
class WaypointFollower_FullMap(object):
    """
    This is an atomic behavior to follow waypoints indefinitely
    while maintaining a given speed or if given a waypoint plan,
    follows the given plan
    """

    def __init__(self, actor, target_speed, map, world, pattern_1=None, pattern_2=None, plan=None,
                 avoid_collision=False, actor_location=None, name="FollowWaypoints"):
        """
        Set up actor and local planner
        """
        self._actor_list = []
        self._actor_list.append(actor)
        # print('\n\ninit_actor: ', actor)
        self._target_speed = target_speed
        self._local_planner_list = []
        self._plan = plan
        self._args_lateral_dict = {'K_P': 1.0, 'K_D': 0.0, 'K_I': 0.0, 'dt': 0.03}
        self._args_longitudinal_dict = {'K_P': 1.0, 'K_D': 0.0, 'K_I': 0.0, 'dt': 0.03}
        self._avoid_collision = avoid_collision
        self.pattern_1 = pattern_1
        self.pattern_2 = pattern_2
        self.map = map
        self.world = world
        self.actor_location = actor_location

    def setup(self, timeout=5):
        """
        Delayed one-time initialization
        """
        for actor in self._actor_list:
            self._apply_local_planner(actor)

        return True

    def _apply_local_planner(self, actor):
        local_planner = WpFollowplanner_FullMap(
            map=self.map,
            actor=actor,
            actor_location=self.actor_location,
            opt_dict={
                'target_speed': self._target_speed,
                'lateral_control_dict': self._args_lateral_dict},
            pattern_1=self.pattern_1,
            pattern_2=self.pattern_2
        )
        if self._plan is not None:
            local_planner.set_global_plan(self._plan)
        self._local_planner_list.append(local_planner)

    def update(self):
        """
        Run local planner, obtain and apply control to actor
        """
        # print('Update ...')
        for actor, local_planner in zip(self._actor_list, self._local_planner_list):
            # print(actor is not None, actor.is_alive, local_planner is not None)
            if actor is not None and actor.is_alive and local_planner is not None:
                control = local_planner.run_step(debug=False)
                if self._avoid_collision and detect_lane_obstacle(world=self.world, actor=actor):
                    control.throttle = 0.0
                    control.brake = 1.0
                actor.apply_control(control)


class WpFollowplanner_FullMap(object):
    MIN_DISTANCE_PERCENTAGE = 0.9

    def __init__(self, actor, opt_dict, map, pattern_1=None, pattern_2=None, actor_location=None):
        self.pattern_1 = pattern_1
        self.pattern_2 = pattern_2
        self._vehicle = actor
        self._map = map  # self._vehicle.get_world().get_map()

        self._dt = None
        self._target_speed = None
        self._sampling_radius = None
        self._min_distance = None
        self._current_waypoint = None
        self._target_road_option = None
        self._next_waypoints = None
        self._target_waypoint = None
        self._vehicle_controller = None
        self._global_plan = None
        self._road_options_list_prev = None
        self._index = None
        self.actor_location = actor_location
        # queue with tuples of (waypoint, RoadOption)
        self._waypoints_queue = deque(maxlen=600)
        self._buffer_size = 5
        self._waypoint_buffer = deque(maxlen=self._buffer_size)

        # initializing controller
        self.init_controller(opt_dict)

    def init_controller(self, opt_dict):
        """
        Controller initialization.

        :param opt_dict: dictionary of arguments.
        :return:
        """
        # default params
        self._dt = 1.0 / 20.0
        self._target_speed = 20.0  # Km/h
        self._sampling_radius = self._target_speed * 0.5 / 3.6  # 0.5 seconds horizon
        self._min_distance = self._sampling_radius * self.MIN_DISTANCE_PERCENTAGE

        args_lateral_dict = {
            'K_P': 1.95,
            'K_D': 0.01,
            'K_I': 1.4,
            'dt': self._dt}
        args_longitudinal_dict = {
            'K_P': 1.0,
            'K_D': 0,
            'K_I': 1,
            'dt': self._dt}

        # parameters overload
        if 'dt' in opt_dict:
            self._dt = opt_dict['dt']
        if 'target_speed' in opt_dict:
            self._target_speed = opt_dict['target_speed']
        if 'sampling_radius' in opt_dict:
            self._sampling_radius = self._target_speed * \
                                    opt_dict['sampling_radius'] / 3.6
        if 'lateral_control_dict' in opt_dict:
            args_lateral_dict = opt_dict['lateral_control_dict']
        if 'longitudinal_control_dict' in opt_dict:
            args_longitudinal_dict = opt_dict['longitudinal_control_dict']

        # self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        self._current_waypoint = self._map.get_waypoint(self.actor_location)
        # print('self._vehicle.get_location(): ', self._current_waypoint.transform.location, self._vehicle.get_transform().location, self._vehicle, self._current_waypoint.next(1.5))
        self._vehicle_controller = VehiclePIDController(self._vehicle,
                                                        args_lateral=args_lateral_dict,
                                                        args_longitudinal=args_longitudinal_dict)

        self._global_plan = False

        self._waypoints_queue.append((self._current_waypoint.next(1.5)[0], RoadOption.LANEFOLLOW))
        self._target_road_option = RoadOption.LANEFOLLOW
        # fill waypoint trajectory queue
        self._compute_next_waypoints(k=200)

    def _compute_next_waypoints(self, k=1):
        """
        Add new waypoints to the trajectory queue.

        :param k: how many waypoints to compute
        :return:
        """
        # check we do not overflow the queue
        available_entries = self._waypoints_queue.maxlen - len(self._waypoints_queue)
        k = min(available_entries, k)

        for _ in range(k):
            last_waypoint = self._waypoints_queue[-1][0]
            next_waypoints = last_waypoint.next(1.5)
            # print('next_waypoints: ', last_waypoint, next_waypoints)
            if len(next_waypoints) == 1:
                # only one option available ==> lanefollowing
                next_waypoint = next_waypoints[0]
                road_option = RoadOption.LANEFOLLOW
            else:
                # random choice between the possible options
                road_options_list = retrieve_options(
                    next_waypoints, self._current_waypoint)
                if self.pattern_1:
                    index = self.pattern_1.pop(0)
                    road_option = road_options_list[index]
                    next_waypoint = next_waypoints[index]
                    self.pattern_1.append(index)
                elif self.pattern_2:
                    index = self.pattern_2.pop(0)
                    if isinstance(index, int):
                        index = road_options_list.index(RoadOption(index))
                        road_option = RoadOption(index)
                        next_waypoint = next_waypoints[road_options_list.index(
                            road_option)]
                    elif isinstance(index, list):
                        next_waypoint = self._map.get_waypoint(
                            carla.Location(x=index[0], y=index[1], z=index[2]))
                        road_option = RoadOption.LANEFOLLOW
                    else:
                        raise NotImplementedError('index must be type `int` or `list`')
                    self.pattern_2.append(index)
                    print(road_options_list)
                else:  # self.pattern_1 is None and self.pattern_2 is None
                    print('self.pattern_1 is None and self.pattern_2 is None')
                # print(next_waypoint.transform.location)
            self._waypoints_queue.append((next_waypoint, road_option))

    def run_step(self, debug=True):
        """
        Execute one step of local planning which involves running the longitudinal and lateral PID controllers to
        follow the waypoints trajectory.

        :param debug: boolean flag to activate waypoints debugging
        :return:
        """
        # not enough waypoints in the horizon? => add more!
        if len(self._waypoints_queue) < int(self._waypoints_queue.maxlen * 0.5):
            if not self._global_plan:
                self._compute_next_waypoints(k=100)

        if len(self._waypoints_queue) == 0:
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 0.0
            control.hand_brake = False
            control.manual_gear_shift = False

            return control

        # Buffering the waypoints
        if not self._waypoint_buffer:
            for i in range(self._buffer_size):
                if self._waypoints_queue:
                    self._waypoint_buffer.append(
                        self._waypoints_queue.popleft())
                else:
                    break

        self._target_waypoint, self._target_road_option = self._waypoint_buffer[0]

        control = self._vehicle_controller.run_step(self._target_speed, self._target_waypoint)

        # purge the queue of obsolete waypoints
        vehicle_transform = self._vehicle.get_transform()
        max_index = -1

        for i, (waypoint, _) in enumerate(self._waypoint_buffer):
            if distance_vehicle(
                    waypoint, vehicle_transform) < self._min_distance:
                max_index = i
        if max_index >= 0:
            for i in range(max_index + 1):
                self._waypoint_buffer.popleft()
        if debug:
            draw_waypoints(self._vehicle.get_world(), [self._target_waypoint], self._vehicle.get_location().z + 1.0)
        return control


class OtherLeadingVehicle_FullMap(object):
    def __init__(self, name, map, world, only_reset_hero=True):
        self.name = name
        self._map = map
        self.world = world
        self.only_reset_hero = only_reset_hero
        self.speed = 0
        self.blueprint_library = self.world.get_blueprint_library()
        self._scenario_init()

    def _scenario_init(self):
        # init hero car
        # --------------------------------------------------------
        # setup cars on a given waypoint
        self.hero_car_pos_candidate = [[93.75690460205078, -132.76296997070312, 9.84310531616211],
                       [143.19349670410156, -204.4090118408203, 1.8431016206741333],
                       [-100.46805572509766, 16.266956329345703, 1.8431016206741333],
                       [-74.38717651367188, 99.71611022949219, 1.8052573204040527],
                       [-2.410623788833618, 207.50567626953125, 1.8431040048599243],
                       [244.31658935546875, 53.67372131347656, 1.8431016206741333],
                       # [245.8651123046875, -9.9967041015625, 1.8431016206741333],
                       [-6.594831466674805, -208.17323303222656, 1.8431016206741333],
                       [4.926102638244629, 91.77217864990234, 1.8432115316390991],
                       [4.926102638244629, 40.57860565185547, 1.8431016206741333],
                       #[5.430785179138184, 122.2763442993164, 1.8431016206741333],
                       [-77.88716888427734, 40.30692672729492, 1.8052647113800049],
                       [-149.06358337402344, 107.70558166503906, 1.8431016206741333]
                       ]
        self.hero_car_pos = [-77.88716888427734, 40.30692672729492, 1.8052647113800049]
        wp_location = carla.Location(x=self.hero_car_pos[0], y=self.hero_car_pos[1], z=self.hero_car_pos[2])
        wp = self._map.get_waypoint(wp_location)
        hero_vehicle_transform = wp.transform
        hero_model = 'vehicle.lincoln.mkz2017'
        blueprint = random.choice(self.blueprint_library.filter(hero_model))
        blueprint.set_attribute('role_name', 'hero')
        self.hero_car = self.world.try_spawn_actor(blueprint, hero_vehicle_transform)

        models = ['vehicle.nissan.patrol', 'vehicle.audi.tt',
                  'vehicle.lincoln.mkz2017', 'vehicle.volkswagen.t2',
                  'vehicle.tesla.model3', 'vehicle.nissan.micra',
                  'vehicle.audi.a2',
                  ]

        # models = ['vehicle.nissan.patrol', 'vehicle.audi.tt',
        #           'vehicle.lincoln.mkz2017', 'vehicle.volkswagen.t2',
        #           'vehicle.toyota.prius', 'vehicle.tesla.model3',
        #           'vehicle.seat.leon', 'vehicle.nissan.micra',
        #           'vehicle.mini.cooperst', 'vehicle.jeep.wrangler_rubicon',
        #           'vehicle.dodge_charger.police', 'vehicle.citroen.c3',
        #           'vehicle.chevrolet.impala', 'vehicle.mercedes-benz.coupe',
        #           'vehicle.bmw.isetta', 'vehicle.bmw.grandtourer',
        #           'vehicle.audi.a2', 'vehicle.ford.mustang',
        #           ]

        blueprints = [random.choice(self.world.get_blueprint_library().filter(model)) for model in models]
        for blueprint in blueprints:
            blueprint.set_attribute('role_name', 'scenario')

        self.blueprints = blueprints
        self.models = models
        # Not available: 135, 160, 6, 10, 11
        first_car_pos = [93.75690460205078, -132.76296997070312, 9.84310531616211]  # 88
        first_wp_location = carla.Location(x=first_car_pos[0], y=first_car_pos[1], z=first_car_pos[2])
        first_vehicle_waypoint = self._map.get_waypoint(first_wp_location)
        first_vehicle_transform = carla.Transform(first_vehicle_waypoint.transform.location,
                                                  first_vehicle_waypoint.transform.rotation)
        self.first_vehicle = self.world.try_spawn_actor(blueprints[0 % len(models)], first_vehicle_transform)
        self._first_vehicle_speed = 25

        # print('\n\nself.first_vehicle: ', first_wp_location, first_vehicle_waypoint, first_vehicle_transform,
        #       self.first_vehicle, self.first_vehicle.get_location())

        # for actor in self.world.get_actors():
        #     print(actor)
        #     if 'vehicle' in actor.type_id:
        #         print('vehicle', actor.get_location())

        next_second_car_pos = [143.19349670410156, -204.4090118408203, 1.8431016206741333]  # 77
        next_second_wp_location = carla.Location(x=next_second_car_pos[0], y=next_second_car_pos[1],
                                                 z=next_second_car_pos[2])
        next_second_vehicle_waypoint = self._map.get_waypoint(next_second_wp_location)
        next_second_vehicle_transform = carla.Transform(next_second_vehicle_waypoint.transform.location,
                                                        next_second_vehicle_waypoint.transform.rotation)
        self.next_second_vehicle = self.world.try_spawn_actor(blueprints[1 % len(models)],
                                                              next_second_vehicle_transform)
        self._next_second_vehicle_speed = 25

        second_vehicle_waypoint = next_second_vehicle_waypoint.next(16)[0].get_left_lane()
        second_vehicle_transform = carla.Transform(second_vehicle_waypoint.transform.location,
                                                   second_vehicle_waypoint.transform.rotation)
        self.second_vehicle = self.world.try_spawn_actor(blueprints[2 % len(models)], second_vehicle_transform)
        self._second_vehicle_speed = 26

        third_car_pos = [-100.46805572509766, 16.266956329345703, 1.8431016206741333]  # 189
        third_wp_location = carla.Location(x=third_car_pos[0], y=third_car_pos[1], z=third_car_pos[2])
        third_vehicle_waypoint = self._map.get_waypoint(third_wp_location)
        third_vehicle_transform = carla.Transform(third_vehicle_waypoint.transform.location,
                                                  third_vehicle_waypoint.transform.rotation)
        self.third_vehicle = self.world.try_spawn_actor(blueprints[3 % len(models)], third_vehicle_transform)
        # setup local planners for zombie cars
        self._third_vehicle_speed = 25

        fourth_car_pos = [-74.38717651367188, 99.71611022949219, 1.8052573204040527]  # 27
        fourth_wp_location = carla.Location(x=fourth_car_pos[0], y=fourth_car_pos[1], z=fourth_car_pos[2])
        fourth_vehicle_waypoint = self._map.get_waypoint(fourth_wp_location)
        fourth_vehicle_transform = carla.Transform(fourth_vehicle_waypoint.transform.location,
                                                   fourth_vehicle_waypoint.transform.rotation)
        self.fourth_vehicle = self.world.try_spawn_actor(blueprints[4 % len(models)], fourth_vehicle_transform)
        # setup local planners for zombie cars
        self._fourth_vehicle_speed = 20

        next_fifth_car_pos = [-2.410623788833618, 207.50567626953125, 1.8431040048599243]  # 4
        next_fifth_wp_location = carla.Location(x=next_fifth_car_pos[0], y=next_fifth_car_pos[1],
                                                z=next_fifth_car_pos[2])
        next_fifth_vehicle_waypoint = self._map.get_waypoint(next_fifth_wp_location)
        next_fifth_vehicle_transform = carla.Transform(next_fifth_vehicle_waypoint.transform.location,
                                                       next_fifth_vehicle_waypoint.transform.rotation)
        self.next_fifth_vehicle = self.world.try_spawn_actor(blueprints[6 % len(models)], next_fifth_vehicle_transform)
        # setup local planners for zombie cars
        self._next_fifth_vehicle_speed = 25

        fifth_vehicle_waypoint = next_fifth_vehicle_waypoint.next(16)[0].get_left_lane()
        fifth_vehicle_transform = carla.Transform(fifth_vehicle_waypoint.transform.location,
                                                  fifth_vehicle_waypoint.transform.rotation)
        self.fifth_vehicle = self.world.try_spawn_actor(blueprints[7 % len(models)], fifth_vehicle_transform)
        # setup local planners for zombie cars
        self._fifth_vehicle_speed = 26

        next_sixth_car_pos = [244.31658935546875, 53.67372131347656, 1.8431016206741333]  # 162
        next_sixth_wp_location = carla.Location(x=next_sixth_car_pos[0], y=next_sixth_car_pos[1],
                                                z=next_sixth_car_pos[2])
        next_sixth_vehicle_waypoint = self._map.get_waypoint(next_sixth_wp_location)
        next_sixth_vehicle_transform = carla.Transform(next_sixth_vehicle_waypoint.transform.location,
                                                       next_sixth_vehicle_waypoint.transform.rotation)
        self.next_sixth_vehicle = self.world.try_spawn_actor(blueprints[8 % len(models)], next_sixth_vehicle_transform)
        # setup local planners for zombie cars
        self._next_sixth_vehicle_speed = 25

        sixth_vehicle_waypoint = next_sixth_vehicle_waypoint.next(16)[0].get_left_lane()
        sixth_vehicle_transform = carla.Transform(sixth_vehicle_waypoint.transform.location,
                                                  sixth_vehicle_waypoint.transform.rotation)
        self.sixth_vehicle = self.world.try_spawn_actor(blueprints[9 % len(models)], sixth_vehicle_transform)
        # setup local planners for zombie cars
        self._sixth_vehicle_speed = 26

        next_seventh_car_pos = [245.8651123046875, -9.9967041015625, 1.8431016206741333]  # 134
        next_seventh_wp_location = carla.Location(x=next_seventh_car_pos[0], y=next_seventh_car_pos[1],
                                                  z=next_seventh_car_pos[2])
        next_seventh_vehicle_waypoint = self._map.get_waypoint(next_seventh_wp_location)
        next_seventh_vehicle_transform = carla.Transform(next_seventh_vehicle_waypoint.transform.location,
                                                         next_seventh_vehicle_waypoint.transform.rotation)
        self.next_seventh_vehicle = self.world.try_spawn_actor(blueprints[10 % len(models)],
                                                               next_seventh_vehicle_transform)
        # setup local planners for zombie cars
        self._next_seventh_vehicle_speed = 25

        seventh_vehicle_waypoint = next_seventh_vehicle_waypoint.next(16)[0].get_left_lane()
        seventh_vehicle_transform = carla.Transform(seventh_vehicle_waypoint.transform.location,
                                                    seventh_vehicle_waypoint.transform.rotation)
        self.seventh_vehicle = self.world.try_spawn_actor(blueprints[11 % len(models)], seventh_vehicle_transform)
        # setup local planners for zombie cars
        self._seventh_vehicle_speed = 26

        next_eighth_car_pos = [-6.594831466674805, -208.17323303222656, 1.8431016206741333]  # 68
        next_eighth_wp_location = carla.Location(x=next_eighth_car_pos[0], y=next_eighth_car_pos[1],
                                                 z=next_eighth_car_pos[2])
        next_eighth_vehicle_waypoint = self._map.get_waypoint(next_eighth_wp_location)
        next_eighth_vehicle_transform = carla.Transform(next_eighth_vehicle_waypoint.transform.location,
                                                        next_eighth_vehicle_waypoint.transform.rotation)
        self.next_eighth_vehicle = self.world.try_spawn_actor(blueprints[12 % len(models)],
                                                              next_eighth_vehicle_transform)
        # setup local planners for zombie cars
        self._next_eighth_vehicle_speed = 25

        eighth_vehicle_waypoint = next_eighth_vehicle_waypoint.next(16)[0].get_left_lane()
        eighth_vehicle_transform = carla.Transform(eighth_vehicle_waypoint.transform.location,
                                                   eighth_vehicle_waypoint.transform.rotation)
        self.eighth_vehicle = self.world.try_spawn_actor(blueprints[13 % len(models)], eighth_vehicle_transform)
        # setup local planners for zombie cars
        self._eighth_vehicle_speed = 26

        no_12_car_pos = [4.926102638244629, 91.77217864990234, 1.8432115316390991]  # 53
        no_12_wp_location = carla.Location(x=no_12_car_pos[0], y=no_12_car_pos[1], z=no_12_car_pos[2])
        no_12_vehicle_waypoint = self._map.get_waypoint(no_12_wp_location)
        no_12_vehicle_transform = carla.Transform(no_12_vehicle_waypoint.transform.location,
                                                  no_12_vehicle_waypoint.transform.rotation)
        self.no_12_vehicle = self.world.try_spawn_actor(blueprints[17 % len(models)], no_12_vehicle_transform)
        # setup local planners for zombie cars
        self.no_12_vehicle_speed = 25

        no_13_car_pos = [4.926102638244629, 40.57860565185547, 1.8431016206741333]  # 145
        no_13_wp_location = carla.Location(x=no_13_car_pos[0], y=no_13_car_pos[1], z=no_13_car_pos[2])
        no_13_vehicle_waypoint = self._map.get_waypoint(no_13_wp_location)
        no_13_vehicle_transform = carla.Transform(no_13_vehicle_waypoint.transform.location,
                                                  no_13_vehicle_waypoint.transform.rotation)
        self.no_13_vehicle = self.world.try_spawn_actor(blueprints[18 % len(models)], no_13_vehicle_transform)
        # setup local planners for zombie cars
        self.no_13_vehicle_speed = 25

        no_14_car_pos = [5.430785179138184, 122.2763442993164, 1.8431016206741333]  # 98
        no_14_wp_location = carla.Location(x=no_14_car_pos[0], y=no_14_car_pos[1], z=no_14_car_pos[2])
        no_14_vehicle_waypoint = self._map.get_waypoint(no_14_wp_location)
        no_14_vehicle_transform = carla.Transform(no_14_vehicle_waypoint.transform.location,
                                                  no_14_vehicle_waypoint.transform.rotation)
        self.no_14_vehicle = self.world.try_spawn_actor(blueprints[19 % len(models)], no_14_vehicle_transform)
        # setup local planners for zombie cars
        self.no_14_vehicle_speed = 25

        self.zombie_cars = [self.first_vehicle, self.second_vehicle, self.next_second_vehicle, self.third_vehicle,
                            self.fourth_vehicle,
                            self.fifth_vehicle, self.next_fifth_vehicle, self.sixth_vehicle,
                            self.next_sixth_vehicle, self.seventh_vehicle, self.next_seventh_vehicle,
                            self.eighth_vehicle, self.next_eighth_vehicle,
                            self.no_12_vehicle,
                            self.no_13_vehicle, self.no_14_vehicle
                            ]

        first_vehicle_planner = WaypointFollower_FullMap(actor=self.first_vehicle, map=self._map,
                                                         target_speed=self._first_vehicle_speed,
                                                         actor_location=first_wp_location,
                                                         avoid_collision=True, pattern_1=[1, 0, 2],
                                                         world=self.world)  # [1,3,1,1,2,1]
        second_vehicle_planner = WaypointFollower_FullMap(actor=self.second_vehicle,
                                                          target_speed=self._second_vehicle_speed,
                                                          actor_location=second_vehicle_waypoint.transform.location,
                                                          map=self._map,
                                                          avoid_collision=True, pattern_1=[1, 0, 0, 0, 1],
                                                          world=self.world)
        next_second_vehicle_planner = WaypointFollower_FullMap(actor=self.next_second_vehicle,
                                                               target_speed=self._next_second_vehicle_speed,
                                                               actor_location=next_second_wp_location,
                                                               map=self._map, avoid_collision=True, pattern_1=[0, 1, 0],
                                                               world=self.world)
        third_vehicle_planner = WaypointFollower_FullMap(actor=self.third_vehicle,
                                                         target_speed=self._third_vehicle_speed,
                                                         actor_location=third_wp_location,
                                                         map=self._map,
                                                         avoid_collision=True, pattern_1=[0, 0, 0],
                                                         world=self.world)
        fourth_vehicle_planner = WaypointFollower_FullMap(actor=self.fourth_vehicle,
                                                          target_speed=self._fourth_vehicle_speed,
                                                          actor_location=fourth_wp_location,
                                                          map=self._map,
                                                          avoid_collision=True,
                                                          pattern_1=[0, 0, 0, 0, 1],
                                                          world=self.world)

        fifth_vehicle_planner = WaypointFollower_FullMap(actor=self.fifth_vehicle,
                                                         target_speed=self._fifth_vehicle_speed,
                                                         actor_location=fifth_vehicle_waypoint.transform.location,
                                                         map=self._map, avoid_collision=True,
                                                         pattern_1=[1, 1, 0, 0, 0],
                                                         world=self.world)
        next_fifth_vehicle_planner = WaypointFollower_FullMap(actor=self.next_fifth_vehicle,
                                                              target_speed=self._next_fifth_vehicle_speed,
                                                              actor_location=next_fifth_wp_location,
                                                              map=self._map,
                                                              avoid_collision=True,
                                                              pattern_1=[0, 1, 0],
                                                              world=self.world)

        sixth_vehicle_planner = WaypointFollower_FullMap(actor=self.sixth_vehicle,
                                                         target_speed=self._sixth_vehicle_speed,
                                                         actor_location=sixth_vehicle_waypoint.transform.location,
                                                         map=self._map,
                                                         avoid_collision=True,
                                                         pattern_1=[1, 0, 0, 0, 1],
                                                         world=self.world)
        next_sixth_vehicle_planner = WaypointFollower_FullMap(actor=self.next_sixth_vehicle,
                                                              target_speed=self._next_sixth_vehicle_speed,
                                                              actor_location=next_sixth_wp_location,
                                                              map=self._map,
                                                              avoid_collision=True,
                                                              pattern_1=[0, 1, 0],
                                                              world=self.world)

        seventh_vehicle_planner = WaypointFollower_FullMap(actor=self.seventh_vehicle,
                                                           target_speed=self._seventh_vehicle_speed,
                                                           actor_location=seventh_vehicle_waypoint.transform.location,
                                                           map=self._map,
                                                           avoid_collision=True, pattern_1=[1, 0, 0, 0, 1],
                                                           world=self.world)
        next_seventh_vehicle_planner = WaypointFollower_FullMap(actor=self.next_seventh_vehicle,
                                                                target_speed=self._next_seventh_vehicle_speed,
                                                                actor_location=next_seventh_wp_location,
                                                                map=self._map,
                                                                avoid_collision=True, pattern_1=[0, 1, 0],
                                                                world=self.world)

        eighth_vehicle_planner = WaypointFollower_FullMap(actor=self.eighth_vehicle,
                                                          target_speed=self._eighth_vehicle_speed,
                                                          actor_location=eighth_vehicle_waypoint.transform.location,
                                                          map=self._map,
                                                          avoid_collision=True, pattern_1=[0, 0, 0, 1, 1],
                                                          world=self.world)
        next_eighth_vehicle_planner = WaypointFollower_FullMap(self.next_eighth_vehicle,
                                                               target_speed=self._next_eighth_vehicle_speed,
                                                               actor_location=next_eighth_wp_location,
                                                               map=self._map,
                                                               avoid_collision=True, pattern_1=[0, 1, 0],
                                                               world=self.world)

        no_12_vehicle_planner = WaypointFollower_FullMap(actor=self.no_12_vehicle,
                                                         target_speed=self.no_12_vehicle_speed,
                                                         actor_location=no_12_wp_location,
                                                         map=self._map, avoid_collision=True,
                                                         pattern_1=[1, 1, 0, 3, 1, 1, 1, 1, 1, 0, 1, 0],
                                                         world=self.world)

        no_13_vehicle_planner = WaypointFollower_FullMap(actor=self.no_13_vehicle,
                                                         target_speed=self.no_13_vehicle_speed,
                                                         actor_location=no_13_wp_location,
                                                         map=self._map, avoid_collision=True,
                                                         pattern_1=[1, 1, 0, 3, 1, 1, 1, 1, 1, 0, 1, 0],
                                                         world=self.world)

        no_14_vehicle_planner = WaypointFollower_FullMap(actor=self.no_14_vehicle,
                                                         target_speed=self.no_14_vehicle_speed,
                                                         actor_location=no_14_wp_location,
                                                         map=self._map, avoid_collision=True,
                                                         pattern_1=[1, 1, 0, 3, 1, 1, 1, 1, 1, 0, 1, 0],
                                                         world=self.world)

        self.vehicle_planners = [first_vehicle_planner, second_vehicle_planner, next_second_vehicle_planner,
                                 third_vehicle_planner, fourth_vehicle_planner,
                                 fifth_vehicle_planner, next_fifth_vehicle_planner, sixth_vehicle_planner,
                                 next_sixth_vehicle_planner, seventh_vehicle_planner,
                                 next_seventh_vehicle_planner, eighth_vehicle_planner, next_eighth_vehicle_planner,
                                 no_12_vehicle_planner, no_13_vehicle_planner, no_14_vehicle_planner,
                                 ]

        for planner in self.vehicle_planners:
            planner.setup()

    def generate_car(self):
        additional_zombie_car = list()
        additional_zombie_car_speed = list()
        additional_pattern = list()
        additional_actor_location = list()
        all_car_pos = [[93.75690460205078, -132.76296997070312, 9.84310531616211],
                       [143.19349670410156, -204.4090118408203, 1.8431016206741333],
                       [-2, -2, -2],
                       [-100.46805572509766, 16.266956329345703, 1.8431016206741333],
                       [-74.38717651367188, 99.71611022949219, 1.8052573204040527],
                       [-2.410623788833618, 207.50567626953125, 1.8431040048599243],
                       [-2, -2, -2],
                       [244.31658935546875, 53.67372131347656, 1.8431016206741333],
                       [-2, -2, -2],
                       [245.8651123046875, -9.9967041015625, 1.8431016206741333],
                       [-2, -2, -2],
                       [-6.594831466674805, -208.17323303222656, 1.8431016206741333],
                       [-2, -2, -2],
                       [4.926102638244629, 91.77217864990234, 1.8432115316390991],
                       [4.926102638244629, 40.57860565185547, 1.8431016206741333],
                       [5.430785179138184, 122.2763442993164, 1.8431016206741333]
                       ]
        all_pattern = [[1, 0, 2],
                       [0, 1, 0],
                       [1, 0, 0, 0, 1],
                       [0, 0, 0],
                       [0, 0, 0, 0, 1],
                       [0, 1, 0],
                       [1, 1, 0, 0, 0],
                       [0, 1, 0],
                       [1, 0, 0, 0, 1],
                       [0, 1, 0],
                       [1, 0, 0, 0, 1],
                       [0, 1, 0],
                       [0, 0, 0, 1, 1],
                       [1, 1, 0, 3, 1, 1, 1, 1, 1, 0, 1, 0],
                       [1, 1, 0, 3, 1, 1, 1, 1, 1, 0, 1, 0],
                       [1, 1, 0, 3, 1, 1, 1, 1, 1, 0, 1, 0],
                       ]

        for i, car_pos in enumerate(all_car_pos):
            if car_pos == [-1, -1, -1] or car_pos == [-2, -2, -2]:
                # car_pos == [-2, -2, -2]: get_left_lane(), speed=26
                # car_pos == [-1, -1, -1]: get_left_lane()
                car_pos = all_car_pos[i - 1]
                orig_actor_location = carla.Location(x=car_pos[0], y=car_pos[1], z=car_pos[2])
                vehicle_waypoint = self._map.get_waypoint(orig_actor_location).next(16)[0].get_left_lane()
                actor_location = vehicle_waypoint.transform.location
            else:
                actor_location = carla.Location(x=car_pos[0], y=car_pos[1], z=car_pos[2])
                vehicle_waypoint = self._map.get_waypoint(actor_location)

            world_actors = self.world.get_actors().filter('vehicle.*')
            flag_spawn = True
            for adversary in world_actors:
                if actor_location.distance(adversary.get_location()) < 30:
                    flag_spawn = False
            if flag_spawn:
                vehicle_transform = carla.Transform(vehicle_waypoint.transform.location,
                                                    vehicle_waypoint.transform.rotation)
                try:
                    # vehicle = self.world.try_spawn_actor(self.blueprints[np.random.randint(0, len(self.blueprints))],
                    #                                      vehicle_transform)
                    vehicle = self.world.spawn_actor(self.blueprints[np.random.randint(0, len(self.blueprints))],
                                                         vehicle_transform)
                    if car_pos == [-2, -2, -2]:
                        _vehicle_speed = 26
                    else:
                        _vehicle_speed = 25
                    additional_zombie_car.append(vehicle)
                    additional_zombie_car_speed.append(_vehicle_speed)
                    additional_pattern.append(all_pattern[i])
                    additional_actor_location.append(actor_location)
                    self.zombie_cars.append(vehicle)
                except:
                    print('generate_car() Failed!', actor_location)

        for i, (one_zombie_car, one_zombie_car_speed, one_pattern, one_actor_location) in enumerate(
                zip(additional_zombie_car, additional_zombie_car_speed, additional_pattern,
                    additional_actor_location)):
            vehicle_planner = WaypointFollower_FullMap(actor=one_zombie_car, map=self._map,
                                                       actor_location=one_actor_location,
                                                       target_speed=one_zombie_car_speed,
                                                       avoid_collision=True, pattern_1=one_pattern,
                                                       world=self.world)

            self.vehicle_planners.append(vehicle_planner)
            vehicle_planner.setup()

    def _update(self):
        # update action for two local planners
        # if _dis3d(_pos3d(self.hero_car), _pos3d(self.first_vehicle)) > 26.:
        #     pass
        # else:
        #     for planner in self.vehicle_planners:
        #         planner.update()
        self.generate_car()
        for planner in self.vehicle_planners:
            planner.update()

    def restart(self):
        if self.only_reset_hero:
            random.shuffle(self.hero_car_pos_candidate)
            world_actors = self.world.get_actors().filter('vehicle.*')

            for hero_car_pos in self.hero_car_pos_candidate:
                wp_location = carla.Location(x=hero_car_pos[0], y=hero_car_pos[1], z=hero_car_pos[2])
                flag_spawn = True
                for adversary in world_actors:
                    if wp_location.distance(adversary.get_location()) < 10:
                        flag_spawn = False
                if flag_spawn:
                    wp = self._map.get_waypoint(wp_location)
                    hero_vehicle_transform = wp.transform
                    hero_model = 'vehicle.lincoln.mkz2017'
                    blueprint = random.choice(self.blueprint_library.filter(hero_model))
                    blueprint.set_attribute('role_name', 'hero')
                    self.hero_car = self.world.try_spawn_actor(blueprint, hero_vehicle_transform)
                    break
        else:
            self._remove_all_actors()
            self.zombie_cars = list()
            self.vehicle_planners = list()
            self._scenario_init()

    def _remove_all_actors(self):
        actors = [self.hero_car] + self.zombie_cars
        # actors = self.zombie_cars
        for actor in actors:
            if actor.is_alive:
                actor.destroy()

    def _remove_zombie_cars(self):
        actors = self.zombie_cars
        for actor in actors:
            if actor.is_alive:
                actor.destroy()


class Cross_Join(object):
    def __init__(self, name, map, world, only_reset_hero=False):
        self.name = name
        self._map = map
        self.world = world
        self.speed = 0
        self.only_reset_hero = only_reset_hero
        self.blueprint_library = self.world.get_blueprint_library()
        self._scenario_init()

    def _scenario_init(self):
        # init hero car
        # --------------------------------------------------------
        # setup cars on a given waypoint
        self.hero_car_pos = [-42.350990295410156, -2.835118293762207, 1.8431016206741333]
        # self.hero_car_pos = [-74.38717651367188, 57.531620025634766, 1.805267095565796]  # 13
        wp_location = carla.Location(x=self.hero_car_pos[0], y=self.hero_car_pos[1], z=self.hero_car_pos[2]+10)
        wp = self._map.get_waypoint(wp_location)
        hero_vehicle_transform = wp.transform
        hero_model = 'vehicle.lincoln.mkz2017'
        blueprint = random.choice(self.blueprint_library.filter(hero_model))
        blueprint.set_attribute('role_name', 'hero')
        self.hero_car = self.world.try_spawn_actor(blueprint, hero_vehicle_transform)

        models = ['vehicle.nissan.patrol', 'vehicle.audi.tt',
                  'vehicle.lincoln.mkz2017', 'vehicle.volkswagen.t2',
                  'vehicle.tesla.model3', 'vehicle.nissan.micra',
                  'vehicle.audi.a2',
                  ]

        blueprints = [random.choice(self.world.get_blueprint_library().filter(model)) for model in models]
        for blueprint in blueprints:
            blueprint.set_attribute('role_name', 'scenario')

        self.blueprints = blueprints
        self.models = models
        # Not available: 135, 160
        fourth_car_pos = [-74.38717651367188, 57.531620025634766, 1.805267095565796]  # 15
        fourth_wp_location = carla.Location(x=fourth_car_pos[0], y=fourth_car_pos[1], z=fourth_car_pos[2]+10)
        fourth_vehicle_waypoint = self._map.get_waypoint(fourth_wp_location)
        fourth_vehicle_transform = carla.Transform(fourth_vehicle_waypoint.transform.location,
                                                   fourth_vehicle_waypoint.transform.rotation)
        self.fourth_vehicle = self.world.try_spawn_actor(blueprints[4 % len(models)], fourth_vehicle_transform)
        # setup local planners for zombie cars
        speed_list = [21, 25]   #, 31]
        # speed = random.choice([21, 25, 31])
        self.speed = speed_list[logger.select_scenario_id]
        print('Velocity: ', self.speed)
        self._fourth_vehicle_speed = self.speed  #random.choice([21, 27, 31])

        fifth_car_pos = [-74.38717651367188, 77.64903259277344, 1.8052573204040527]  # 25
        fifth_wp_location = carla.Location(x=fifth_car_pos[0], y=fifth_car_pos[1], z=fifth_car_pos[2]+10)
        fifth_vehicle_waypoint = self._map.get_waypoint(fifth_wp_location)
        fifth_vehicle_transform = carla.Transform(fifth_vehicle_waypoint.transform.location,
                                                  fifth_vehicle_waypoint.transform.rotation)
        self.fifth_vehicle = self.world.try_spawn_actor(blueprints[4 % len(models)], fifth_vehicle_transform)
        # setup local planners for zombie cars
        self._fifth_vehicle_speed = self.speed-1 #random.choice([21, 27, 31])

        sixth_car_pos = [-74.38717651367188, 97.71611022949219, 1.8052573204040527]  # 27
        sixth_wp_location = carla.Location(x=sixth_car_pos[0], y=sixth_car_pos[1], z=sixth_car_pos[2]+10)
        sixth_vehicle_waypoint = self._map.get_waypoint(sixth_wp_location)
        sixth_vehicle_transform = carla.Transform(sixth_vehicle_waypoint.transform.location,
                                                  sixth_vehicle_waypoint.transform.rotation)
        self.sixth_vehicle = self.world.try_spawn_actor(blueprints[4 % len(models)], sixth_vehicle_transform)
        # setup local planners for zombie cars
        self._sixth_vehicle_speed = self.speed-1 #random.choice([21, 27, 31])

        self.zombie_cars = [self.fourth_vehicle, self.fifth_vehicle, self.sixth_vehicle]

        fourth_vehicle_planner = WaypointFollower_FullMap(actor=self.fourth_vehicle,
                                                          actor_location=fourth_wp_location,
                                                          target_speed=self._fourth_vehicle_speed,
                                                          map=self._map,
                                                          avoid_collision=False,
                                                          pattern_1=[1, 1, 1, 1, 1, 1, 0, 0, 1, ],
                                                          world=self.world)
        fifth_vehicle_planner = WaypointFollower_FullMap(actor=self.fifth_vehicle,
                                                         actor_location=fifth_wp_location,
                                                         target_speed=self._fifth_vehicle_speed,
                                                         map=self._map,
                                                         avoid_collision=False,
                                                         pattern_1=[1, 1, 1, 1, 1, 1, 0, 0, 1, ],
                                                         world=self.world)
        sixth_vehicle_planner = WaypointFollower_FullMap(actor=self.sixth_vehicle,
                                                         actor_location=sixth_wp_location,
                                                         target_speed=self._sixth_vehicle_speed,
                                                         map=self._map,
                                                         avoid_collision=False,
                                                         pattern_1=[1, 1, 1, 1, 1, 1, 0, 0, 1, ],
                                                         world=self.world)

        self.vehicle_planners = [fourth_vehicle_planner, fifth_vehicle_planner, sixth_vehicle_planner]

        for planner in self.vehicle_planners:
            planner.setup()

    def generate_car(self):
        additional_zombie_car = list()
        additional_zombie_car_speed = list()
        additional_pattern = list()
        additional_actor_location = list()
        # all_car_pos = [[-74.38717651367188, 57.531620025634766, 1.805267095565796],
        #                [-74.38717651367188, 75.64903259277344, 1.8052573204040527],
        #                [-74.38717651367188, 99.71611022949219, 1.8052573204040527]]
        all_car_pos = []
        all_pattern = [[1, 1, 1, 1, 1, 1, 0, 0, 1, ], [1, 1, 1, 1, 1, 1, 0, 0, 1, ], [1, 1, 1, 1, 1, 1, 0, 0, 1, ], ]

        for i, car_pos in enumerate(all_car_pos):
            if car_pos == [-1, -1, -1] or car_pos == [-2, -2, -2]:
                # car_pos == [-2, -2, -2]: get_left_lane(), speed=26
                # car_pos == [-1, -1, -1]: get_left_lane()
                car_pos = all_car_pos[i - 1]
                orig_actor_location = carla.Location(x=car_pos[0], y=car_pos[1], z=car_pos[2])
                vehicle_waypoint = self._map.get_waypoint(orig_actor_location).next(16)[0].get_left_lane()
                actor_location = vehicle_waypoint.transform.location
            else:
                actor_location = carla.Location(x=car_pos[0], y=car_pos[1], z=car_pos[2])
                vehicle_waypoint = self._map.get_waypoint(actor_location)

            world_actors = self.world.get_actors().filter('vehicle.*')
            flag_spawn = True
            for adversary in world_actors:
                if actor_location.distance(adversary.get_location()) < 5:
                    flag_spawn = False
            if flag_spawn:
                vehicle_transform = carla.Transform(vehicle_waypoint.transform.location,
                                                    vehicle_waypoint.transform.rotation)
                vehicle = self.world.try_spawn_actor(self.blueprints[np.random.randint(0, len(self.blueprints))],
                                                     vehicle_transform)
                if car_pos == [-2, -2, -2]:
                    _vehicle_speed = 26
                else:
                    _vehicle_speed = 25
                self.speed = _vehicle_speed
                additional_zombie_car.append(vehicle)
                additional_zombie_car_speed.append(_vehicle_speed)
                additional_pattern.append(all_pattern[i])
                additional_actor_location.append(actor_location)
                self.zombie_cars.append(vehicle)

        for i, (one_zombie_car, one_zombie_car_speed, one_pattern, one_actor_location) in enumerate(
                zip(additional_zombie_car, additional_zombie_car_speed, additional_pattern, additional_actor_location)):
            vehicle_planner = WaypointFollower_FullMap(actor=one_zombie_car, map=self._map,
                                                       actor_location=one_actor_location,
                                                       target_speed=one_zombie_car_speed,
                                                       avoid_collision=True, pattern_1=one_pattern,
                                                       world=self.world)

            self.vehicle_planners.append(vehicle_planner)
            vehicle_planner.setup()

    def _update(self):
        self.generate_car()
        for planner in self.vehicle_planners:
            planner.update()

    def restart(self):
        if self.only_reset_hero:
            wp_location = carla.Location(x=self.hero_car_pos[0], y=self.hero_car_pos[1], z=self.hero_car_pos[2])
            wp = self._map.get_waypoint(wp_location)
            hero_vehicle_transform = wp.transform
            hero_model = 'vehicle.lincoln.mkz2017'
            blueprint = random.choice(self.blueprint_library.filter(hero_model))
            blueprint.set_attribute('role_name', 'hero')
            self.hero_car = self.world.try_spawn_actor(blueprint, hero_vehicle_transform)
        else:
            self._remove_all_actors()
            # self.zombie_cars = list()
            # self.vehicle_planners = list()
            self._scenario_init()

    def _remove_all_actors(self):
        actors = [self.hero_car] + self.zombie_cars
        # actors = self.zombie_cars
        for actor in actors:
            if actor.is_alive:
                actor.destroy()

    def _remove_zombie_cars(self):
        actors = self.zombie_cars
        for actor in actors:
            if actor.is_alive:
                actor.destroy()


class Ring_Join(object):
    def __init__(self, name, map, world, only_reset_hero=False):
        self.name = name
        self._map = map
        self.world = world
        self.only_reset_hero = only_reset_hero
        self.speed = 0
        self.blueprint_library = self.world.get_blueprint_library()
        self._scenario_init()

    def _scenario_init(self):
        # init hero car
        # --------------------------------------------------------
        # setup cars on a given waypoint
        self.hero_car_pos = [52.61453628540039, -7.843905448913574, 1.8431028127670288]  # 55
        wp_location = carla.Location(x=self.hero_car_pos[0], y=self.hero_car_pos[1], z=self.hero_car_pos[2])
        wp = self._map.get_waypoint(wp_location)
        hero_vehicle_transform = wp.transform
        hero_model = 'vehicle.lincoln.mkz2017'
        blueprint = random.choice(self.blueprint_library.filter(hero_model))
        blueprint.set_attribute('role_name', 'hero')
        self.hero_car = self.world.try_spawn_actor(blueprint, hero_vehicle_transform)

        models = ['vehicle.nissan.patrol', 'vehicle.audi.tt',
                  'vehicle.lincoln.mkz2017', 'vehicle.volkswagen.t2',
                  'vehicle.tesla.model3', 'vehicle.nissan.micra',
                  'vehicle.audi.a2',
                  ]

        blueprints = [random.choice(self.world.get_blueprint_library().filter(model)) for model in models]
        for blueprint in blueprints:
            blueprint.set_attribute('role_name', 'scenario')

        self.blueprints = blueprints
        self.models = models
        # Not available: 135, 160
        fourth_car_pos = [4.926102638244629, 40.57860565185547, 1.8431016206741333]  # 145
        fourth_wp_location = carla.Location(x=fourth_car_pos[0], y=fourth_car_pos[1], z=fourth_car_pos[2])
        fourth_vehicle_waypoint = self._map.get_waypoint(fourth_wp_location)
        fourth_vehicle_transform = carla.Transform(fourth_vehicle_waypoint.transform.location,
                                                   fourth_vehicle_waypoint.transform.rotation)
        self.fourth_vehicle = self.world.try_spawn_actor(blueprints[4 % len(models)], fourth_vehicle_transform)
        # setup local planners for zombie cars
        speed_list = [21, 25]  # , 35]
        # self.speed = random.choice([23,])  # default is 21, 27, 31
        self.speed = speed_list[logger.select_scenario_id]
        print('velocity: ', self.speed)
        self._fourth_vehicle_speed = self.speed

        fifth_car_pos = [4.926102638244629, 59.08685302734375, 1.8430894613265991]  # 47
        fifth_wp_location = carla.Location(x=fifth_car_pos[0], y=fifth_car_pos[1], z=fifth_car_pos[2])
        fifth_vehicle_waypoint = self._map.get_waypoint(fifth_wp_location)
        fifth_vehicle_transform = carla.Transform(fifth_vehicle_waypoint.transform.location,
                                                   fifth_vehicle_waypoint.transform.rotation)
        self.fifth_vehicle = self.world.try_spawn_actor(blueprints[4 % len(models)], fifth_vehicle_transform)
        # setup local planners for zombie cars
        self._fifth_vehicle_speed = self.speed-1

        sixth_car_pos = [4.926102638244629, 72.03030395507812, 1.843079686164856]  # 49
        sixth_wp_location = carla.Location(x=sixth_car_pos[0], y=sixth_car_pos[1], z=sixth_car_pos[2])
        sixth_vehicle_waypoint = self._map.get_waypoint(sixth_wp_location)
        sixth_vehicle_transform = carla.Transform(sixth_vehicle_waypoint.transform.location,
                                                   sixth_vehicle_waypoint.transform.rotation)
        self.sixth_vehicle = self.world.try_spawn_actor(blueprints[4 % len(models)], sixth_vehicle_transform)
        # setup local planners for zombie cars
        self._sixth_vehicle_speed = self.speed-1

        seventh_car_pos = [4.926102638244629, 91.77217864990234, 1.8432115316390991]  # 53
        seventh_wp_location = carla.Location(x=seventh_car_pos[0], y=seventh_car_pos[1], z=seventh_car_pos[2])
        seventh_vehicle_waypoint = self._map.get_waypoint(seventh_wp_location)
        seventh_vehicle_transform = carla.Transform(seventh_vehicle_waypoint.transform.location,
                                                   seventh_vehicle_waypoint.transform.rotation)
        self.seventh_vehicle = self.world.try_spawn_actor(blueprints[4 % len(models)], seventh_vehicle_transform)
        # setup local planners for zombie cars
        self._seventh_vehicle_speed = self.speed-1


        self.zombie_cars = [self.fourth_vehicle, self.fifth_vehicle, self.sixth_vehicle, self.seventh_vehicle]

        fourth_vehicle_planner = WaypointFollower_FullMap(actor=self.fourth_vehicle,
                                                          target_speed=self._fourth_vehicle_speed,
                                                          actor_location=fourth_wp_location,
                                                          map=self._map,
                                                          avoid_collision=False,
                                                          pattern_1=[1, 1, 0, 3, 1, 1, 1, 1, 1, 0, 1, 0],
                                                          world=self.world)

        fifth_vehicle_planner = WaypointFollower_FullMap(actor=self.fifth_vehicle,
                                                          target_speed=self._fifth_vehicle_speed,
                                                          actor_location=fifth_wp_location,
                                                          map=self._map,
                                                          avoid_collision=False,
                                                          pattern_1=[1, 1, 0, 3, 1, 1, 1, 1, 1, 0, 1, 0],
                                                          world=self.world)
        sixth_vehicle_planner = WaypointFollower_FullMap(actor=self.sixth_vehicle,
                                                          target_speed=self._sixth_vehicle_speed,
                                                          actor_location=sixth_wp_location,
                                                          map=self._map,
                                                          avoid_collision=False,
                                                          pattern_1=[1, 1, 0, 3, 1, 1, 1, 1, 1, 0, 1, 0],
                                                          world=self.world)

        seventh_vehicle_planner = WaypointFollower_FullMap(actor=self.seventh_vehicle,
                                                          target_speed=self._seventh_vehicle_speed,
                                                          actor_location=seventh_wp_location,
                                                          map=self._map,
                                                          avoid_collision=False,
                                                          pattern_1=[1, 1, 0, 3, 1, 1, 1, 1, 1, 0, 1, 0],
                                                          world=self.world)

        self.vehicle_planners = [fourth_vehicle_planner, fifth_vehicle_planner, sixth_vehicle_planner, seventh_vehicle_planner]

        for planner in self.vehicle_planners:
            planner.setup()

    def generate_car(self):
        additional_zombie_car = list()
        additional_zombie_car_speed = list()
        additional_pattern = list()
        additional_actor_location = list()
        all_car_pos = [[4.926102638244629, 40.57860565185547, 1.8431016206741333]]
        all_pattern = [[1, 1, 0, 3, 1, 1, 1, 1, 1, 0, 1, 0]]

        for i, car_pos in enumerate(all_car_pos):
            if car_pos == [-1, -1, -1] or car_pos == [-2, -2, -2]:
                # car_pos == [-2, -2, -2]: get_left_lane(), speed=26
                # car_pos == [-1, -1, -1]: get_left_lane()
                car_pos = all_car_pos[i - 1]
                orig_actor_location = carla.Location(x=car_pos[0], y=car_pos[1], z=car_pos[2])
                vehicle_waypoint = self._map.get_waypoint(orig_actor_location).next(16)[0].get_left_lane()
                actor_location = vehicle_waypoint.transform.location
            else:
                actor_location = carla.Location(x=car_pos[0], y=car_pos[1], z=car_pos[2])
                vehicle_waypoint = self._map.get_waypoint(actor_location)

            world_actors = self.world.get_actors().filter('vehicle.*')
            flag_spawn = True
            for adversary in world_actors:
                #if actor_location.distance(adversary.get_location()) < 15:
                if actor_location.distance(adversary.get_location()) < 5:
                    flag_spawn = False
            if flag_spawn:
                vehicle_transform = carla.Transform(vehicle_waypoint.transform.location,
                                                    vehicle_waypoint.transform.rotation)
                vehicle = self.world.try_spawn_actor(self.blueprints[np.random.randint(0, len(self.blueprints))],
                                                     vehicle_transform)
                if car_pos == [-2, -2, -2]:
                    _vehicle_speed = 26
                else:
                    _vehicle_speed = 30
                    #_vehicle_speed = 20
                    #_vehicle_speed = random.choice([23, 21, 22])
                self.speed = _vehicle_speed
                additional_zombie_car.append(vehicle)
                additional_zombie_car_speed.append(_vehicle_speed)
                additional_pattern.append(all_pattern[i])
                additional_actor_location.append(actor_location)
                self.zombie_cars.append(vehicle)

        for i, (one_zombie_car, one_zombie_car_speed, one_pattern, one_actor_location) in enumerate(
                zip(additional_zombie_car, additional_zombie_car_speed, additional_pattern,
                    additional_actor_location)):
            vehicle_planner = WaypointFollower_FullMap(actor=one_zombie_car, map=self._map,
                                                       actor_location=one_actor_location,
                                                       target_speed=one_zombie_car_speed,
                                                       avoid_collision=False, pattern_1=one_pattern,
                                                       world=self.world)

            self.vehicle_planners.append(vehicle_planner)
            vehicle_planner.setup()

    def _update(self):
        # update action for two local planners
        # if _dis3d(_pos3d(self.hero_car), _pos3d(self.first_vehicle)) > 26.:
        #     pass
        # else:
        #     for planner in self.vehicle_planners:
        #         planner.update()
        # self.generate_car()
        for planner in self.vehicle_planners:
            planner.update()

    def restart(self):
        if self.only_reset_hero:
            wp_location = carla.Location(x=self.hero_car_pos[0], y=self.hero_car_pos[1], z=self.hero_car_pos[2])
            wp = self._map.get_waypoint(wp_location)
            hero_vehicle_transform = wp.transform
            hero_model = 'vehicle.lincoln.mkz2017'
            blueprint = random.choice(self.blueprint_library.filter(hero_model))
            blueprint.set_attribute('role_name', 'hero')
            self.hero_car = self.world.try_spawn_actor(blueprint, hero_vehicle_transform)
        else:
            self._remove_all_actors()
            self.zombie_cars = list()
            self.vehicle_planners = list()
            self._scenario_init()

    def _remove_all_actors(self):
        actors = [self.hero_car] + self.zombie_cars
        # actors = self.zombie_cars
        for actor in actors:
            if actor.is_alive:
                actor.destroy()

    def _remove_zombie_cars(self):
        actors = self.zombie_cars
        for actor in actors:
            if actor.is_alive:
                actor.destroy()


class Straight_Follow_Single(object):
    def __init__(self, name, map, world, only_reset_hero=False):
        self.name = name
        self._map = map
        self.world = world
        self.only_reset_hero = only_reset_hero
        self.speed = 0
        self.blueprint_library = self.world.get_blueprint_library()
        self._scenario_init()
        self.speed = 15

    def _scenario_init(self):
        # init hero car
        # --------------------------------------------------------
        # setup cars on a given waypoint
        # 16.876914978027344, -134.40997314453125, 1.8707298040390015  # 177
        self.hero_car_pos = [93.75690460205078, -132.76296997070312, 9.84310531616211]  # 88
        wp_location = carla.Location(x=self.hero_car_pos[0], y=self.hero_car_pos[1], z=self.hero_car_pos[2])
        wp = self._map.get_waypoint(wp_location)
        hero_vehicle_transform = wp.transform
        hero_model = 'vehicle.lincoln.mkz2017'
        blueprint = random.choice(self.blueprint_library.filter(hero_model))
        blueprint.set_attribute('role_name', 'hero')
        self.hero_car = self.world.try_spawn_actor(blueprint, hero_vehicle_transform)

        models = ['vehicle.nissan.patrol', 'vehicle.audi.tt',
                  'vehicle.lincoln.mkz2017', 'vehicle.volkswagen.t2',
                  'vehicle.tesla.model3', 'vehicle.nissan.micra',
                  'vehicle.audi.a2',
                  ]

        blueprints = [random.choice(self.world.get_blueprint_library().filter(model)) for model in models]
        for blueprint in blueprints:
            blueprint.set_attribute('role_name', 'scenario')

        self.blueprints = blueprints
        self.models = models
        # Not available: 135, 160
        fourth_car_pos = [93.75690460205078, -132.76296997070312, 9.84310531616211]  # 88 + 10
        fourth_wp_location = carla.Location(x=fourth_car_pos[0], y=fourth_car_pos[1], z=fourth_car_pos[2])
        fourth_vehicle_waypoint = self._map.get_waypoint(fourth_wp_location)
        fourth_vehicle_waypoint = fourth_vehicle_waypoint.next(25)[0]
        fourth_vehicle_transform = carla.Transform(fourth_vehicle_waypoint.transform.location,
                                                   fourth_vehicle_waypoint.transform.rotation)
        self.fourth_vehicle = self.world.try_spawn_actor(blueprints[4 % len(models)], fourth_vehicle_transform)
        # setup local planners for zombie cars
        #self._fourth_vehicle_speed = np.random.choice([10, 15, 20, 25])
        self._fourth_vehicle_speed = np.random.choice([15])
        # print('\n\nself._fourth_vehicle_speed: ', self._fourth_vehicle_speed)

        self.zombie_cars = [self.fourth_vehicle]

        fourth_vehicle_planner = WaypointFollower_FullMap(actor=self.fourth_vehicle,
                                                          target_speed=self._fourth_vehicle_speed,
                                                          actor_location=fourth_vehicle_waypoint.transform.location,
                                                          map=self._map,
                                                          avoid_collision=True,
                                                          pattern_1=[1, 0, 2],
                                                          world=self.world)

        self.vehicle_planners = [fourth_vehicle_planner]

        for planner in self.vehicle_planners:
            planner.setup()

    def generate_car(self):
        additional_zombie_car = list()
        additional_zombie_car_speed = list()
        additional_pattern = list()
        additional_actor_location = list()
        all_car_pos = [[93.75690460205078, -132.76296997070312, 9.84310531616211]]  # 88 + 10
        all_pattern = [[1, 0, 2]]

        for i, car_pos in enumerate(all_car_pos):
            if car_pos == [-1, -1, -1] or car_pos == [-2, -2, -2]:
                # car_pos == [-2, -2, -2]: get_left_lane(), speed=26
                # car_pos == [-1, -1, -1]: get_left_lane()
                car_pos = all_car_pos[i - 1]
                orig_actor_location = carla.Location(x=car_pos[0], y=car_pos[1], z=car_pos[2])
                vehicle_waypoint = self._map.get_waypoint(orig_actor_location).next(16)[0].get_left_lane()
                actor_location = vehicle_waypoint.transform.location
            else:
                orig_actor_location = carla.Location(x=car_pos[0], y=car_pos[1], z=car_pos[2])
                vehicle_waypoint = self._map.get_waypoint(orig_actor_location)
                vehicle_waypoint = vehicle_waypoint.next(25)[0]
                actor_location = vehicle_waypoint.transform.location

            world_actors = self.world.get_actors().filter('vehicle.*')
            flag_spawn = True
            for adversary in world_actors:
                if actor_location.distance(adversary.get_location()) < 15:
                    flag_spawn = False
            if flag_spawn:
                vehicle_transform = carla.Transform(vehicle_waypoint.transform.location,
                                                    vehicle_waypoint.transform.rotation)
                vehicle = self.world.try_spawn_actor(self.blueprints[np.random.randint(0, len(self.blueprints))],
                                                     vehicle_transform)
                if car_pos == [-2, -2, -2]:
                    _vehicle_speed = 26
                else:
                    _vehicle_speed = 20
                self.speed = _vehicle_speed
                additional_zombie_car.append(vehicle)
                additional_zombie_car_speed.append(_vehicle_speed)
                additional_actor_location.append(actor_location)
                additional_pattern.append(all_pattern[i])
                self.zombie_cars.append(vehicle)

        for i, (one_zombie_car, one_zombie_car_speed, one_pattern, one_actor_location) in enumerate(
                zip(additional_zombie_car, additional_zombie_car_speed, additional_pattern,
                    additional_actor_location)):
            vehicle_planner = WaypointFollower_FullMap(actor=one_zombie_car, map=self._map,
                                                       actor_location=one_actor_location,
                                                       target_speed=one_zombie_car_speed,
                                                       avoid_collision=True, pattern_1=one_pattern,
                                                       world=self.world)

            self.vehicle_planners.append(vehicle_planner)
            vehicle_planner.setup()

    def _update(self):
        # self.generate_car()
        for planner in self.vehicle_planners:
            planner.update()

    def restart(self):
        if self.only_reset_hero:
            wp_location = carla.Location(x=self.hero_car_pos[0], y=self.hero_car_pos[1], z=self.hero_car_pos[2])
            wp = self._map.get_waypoint(wp_location)
            wp = wp.next(8)[0]
            hero_vehicle_transform = wp.transform
            hero_model = 'vehicle.lincoln.mkz2017'
            blueprint = random.choice(self.blueprint_library.filter(hero_model))
            blueprint.set_attribute('role_name', 'hero')
            self.hero_car = self.world.try_spawn_actor(blueprint, hero_vehicle_transform)
        else:
            self._remove_all_actors()
            self.zombie_cars = list()
            self.vehicle_planners = list()
            self._scenario_init()

    def _remove_all_actors(self):
        actors = [self.hero_car] + self.zombie_cars
        # actors = self.zombie_cars
        for actor in actors:
            if actor.is_alive:
                actor.destroy()

    def _remove_zombie_cars(self):
        actors = self.zombie_cars
        for actor in actors:
            if actor.is_alive:
                actor.destroy()


class Straight_Follow_Double(object):
    def __init__(self, name, map, world, only_reset_hero=False):
        self.name = name
        self._map = map
        self.world = world
        self.speed = 0
        self.only_reset_hero = only_reset_hero
        self.blueprint_library = self.world.get_blueprint_library()
        self._scenario_init()

    def _scenario_init(self):
        # init hero car
        # --------------------------------------------------------
        # setup cars on a given waypoint
        # self.hero_car_pos = [-2.410623788833618, 207.50567626953125, 1.8431040048599243]  # 88
        self.hero_car_pos = [-15, 207.50567626953125, 1.8431040048599243]  # 88
        wp_location = carla.Location(x=self.hero_car_pos[0], y=self.hero_car_pos[1], z=self.hero_car_pos[2])
        wp = self._map.get_waypoint(wp_location)
        hero_vehicle_transform = wp.transform
        hero_model = 'vehicle.lincoln.mkz2017'
        blueprint = random.choice(self.blueprint_library.filter(hero_model))
        blueprint.set_attribute('role_name', 'hero')
        self.hero_car = self.world.try_spawn_actor(blueprint, hero_vehicle_transform)

        models = ['vehicle.nissan.patrol', 'vehicle.audi.tt',
                  'vehicle.lincoln.mkz2017', 'vehicle.volkswagen.t2',
                  'vehicle.tesla.model3', 'vehicle.nissan.micra',
                  'vehicle.audi.a2',
                  ]

        blueprints = [random.choice(self.world.get_blueprint_library().filter(model)) for model in models]
        for blueprint in blueprints:
            blueprint.set_attribute('role_name', 'scenario')

        self.blueprints = blueprints
        self.models = models
        # Not available: 135, 160
        fourth_car_pos = [10.190117835998535, 207.50567626953125, 1.8431016206741333]  # 4
        fourth_wp_location = carla.Location(x=fourth_car_pos[0], y=fourth_car_pos[1], z=fourth_car_pos[2])
        fourth_vehicle_waypoint = self._map.get_waypoint(fourth_wp_location)
        fourth_vehicle_transform = carla.Transform(fourth_vehicle_waypoint.transform.location,
                                                   fourth_vehicle_waypoint.transform.rotation)
        self.fourth_vehicle = self.world.try_spawn_actor(blueprints[4 % len(models)], fourth_vehicle_transform)
        # setup local planners for zombie cars
        # self._fourth_vehicle_speed = 20
        self._fourth_vehicle_speed = 18
        self.speed = self._fourth_vehicle_speed

        next_fourth_car_pos = [10.181385040283203, 204.00567626953125, 1.8431016206741333]  # 3
        next_fourth_wp_location = carla.Location(x=next_fourth_car_pos[0], y=next_fourth_car_pos[1], z=next_fourth_car_pos[2])
        next_fourth_vehicle_waypoint = self._map.get_waypoint(next_fourth_wp_location)
        next_fourth_vehicle_transform = carla.Transform(next_fourth_vehicle_waypoint.transform.location,
                                                   next_fourth_vehicle_waypoint.transform.rotation)
        self.next_fourth_vehicle = self.world.try_spawn_actor(blueprints[4 % len(models)], next_fourth_vehicle_transform)
        # setup local planners for zombie cars
        # self._next_fourth_vehicle_speed = 20
        self._next_fourth_vehicle_speed = 18

        self.zombie_cars = [self.fourth_vehicle, self.next_fourth_vehicle]

        fourth_vehicle_planner = WaypointFollower_FullMap(actor=self.fourth_vehicle,
                                                          target_speed=self._fourth_vehicle_speed,
                                                          actor_location=fourth_wp_location,
                                                          map=self._map,
                                                          avoid_collision=True,
                                                          pattern_1=[0, 1, 0,],
                                                          world=self.world)

        next_fourth_vehicle_planner = WaypointFollower_FullMap(actor=self.next_fourth_vehicle,
                                                          target_speed=self._next_fourth_vehicle_speed,
                                                          actor_location=next_fourth_wp_location,
                                                          map=self._map,
                                                          avoid_collision=True,
                                                          pattern_1=[1, 1, 0, 0, 0, ],
                                                          world=self.world)

        self.vehicle_planners = [fourth_vehicle_planner, next_fourth_vehicle_planner]

        for planner in self.vehicle_planners:
            planner.setup()

    def generate_car(self):
        additional_zombie_car = list()
        additional_zombie_car_speed = list()
        additional_pattern = list()
        additional_actor_location = list()
        all_car_pos = [[10.190117835998535, 207.50567626953125, 1.8431016206741333], [10.181385040283203, 204.00567626953125, 1.8431016206741333]]
        all_pattern = [[0, 1, 0,], [1, 1, 0, 0, 0, ]]

        for i, car_pos in enumerate(all_car_pos):
            if car_pos == [-1, -1, -1] or car_pos == [-2, -2, -2]:
                # car_pos == [-2, -2, -2]: get_left_lane(), speed=26
                # car_pos == [-1, -1, -1]: get_left_lane()
                car_pos = all_car_pos[i - 1]
                orig_actor_location = carla.Location(x=car_pos[0], y=car_pos[1], z=car_pos[2])
                vehicle_waypoint = self._map.get_waypoint(orig_actor_location).next(16)[0].get_left_lane()
                actor_location = vehicle_waypoint.transform.location
            else:
                actor_location = carla.Location(x=car_pos[0], y=car_pos[1], z=car_pos[2])
                vehicle_waypoint = self._map.get_waypoint(actor_location)
                vehicle_waypoint = vehicle_waypoint.next(8)[0]

            world_actors = self.world.get_actors().filter('vehicle.*')
            flag_spawn = True
            for adversary in world_actors:
                if actor_location.distance(adversary.get_location()) < 15:
                    flag_spawn = False
            if flag_spawn:
                vehicle_transform = carla.Transform(vehicle_waypoint.transform.location,
                                                    vehicle_waypoint.transform.rotation)
                vehicle = self.world.try_spawn_actor(self.blueprints[np.random.randint(0, len(self.blueprints))],
                                                     vehicle_transform)
                if car_pos == [-2, -2, -2]:
                    _vehicle_speed = 26
                else:
                    _vehicle_speed = 20
                self.speed = _vehicle_speed
                additional_zombie_car.append(vehicle)
                additional_zombie_car_speed.append(_vehicle_speed)
                additional_pattern.append(all_pattern[i])
                additional_actor_location.append(actor_location)
                self.zombie_cars.append(vehicle)

        for i, (one_zombie_car, one_zombie_car_speed, one_pattern, one_actor_location) in enumerate(
                zip(additional_zombie_car, additional_zombie_car_speed, additional_pattern,
                    additional_actor_location)):
            vehicle_planner = WaypointFollower_FullMap(actor=one_zombie_car, map=self._map,
                                                       actor_location=one_actor_location,
                                                       target_speed=one_zombie_car_speed,
                                                       avoid_collision=True, pattern_1=one_pattern,
                                                       world=self.world)

            self.vehicle_planners.append(vehicle_planner)
            vehicle_planner.setup()

    def _update(self):
        # update action for two local planners
        # if _dis3d(_pos3d(self.hero_car), _pos3d(self.first_vehicle)) > 26.:
        #     pass
        # else:
        #     for planner in self.vehicle_planners:
        #         planner.update()
        # self.generate_car()
        for planner in self.vehicle_planners:
            planner.update()

    def restart(self):
        if self.only_reset_hero:
            wp_location = carla.Location(x=self.hero_car_pos[0], y=self.hero_car_pos[1], z=self.hero_car_pos[2])
            wp = self._map.get_waypoint(wp_location)
            wp = wp.next(8)[0]
            hero_vehicle_transform = wp.transform
            hero_model = 'vehicle.lincoln.mkz2017'
            blueprint = random.choice(self.blueprint_library.filter(hero_model))
            blueprint.set_attribute('role_name', 'hero')
            self.hero_car = self.world.try_spawn_actor(blueprint, hero_vehicle_transform)
        else:
            self._remove_all_actors()
            self.zombie_cars = list()
            self.vehicle_planners = list()
            self._scenario_init()

    def _remove_all_actors(self):
        actors = [self.hero_car] + self.zombie_cars
        # actors = self.zombie_cars
        for actor in actors:
            if actor.is_alive:
                actor.destroy()

    def _remove_zombie_cars(self):
        actors = self.zombie_cars
        for actor in actors:
            if actor.is_alive:
                actor.destroy()

class Cross_Follow(object):
    def __init__(self, name, map, world, only_reset_hero=False):
        self.name = name
        self._map = map
        self.world = world
        self.only_reset_hero = only_reset_hero
        self.speed = 0
        self.blueprint_library = self.world.get_blueprint_library()
        self._scenario_init()

    def _scenario_init(self):
        # init hero car
        # --------------------------------------------------------
        # setup cars on a given waypoint
        # self.hero_car_pos = [-74.38717651367188, 75.64903259277344, 1.8052573204040527]  # 15
        self.hero_car_pos = [-74.38717651367188, 48, 1.8052573204040527]  # 15
        wp_location = carla.Location(x=self.hero_car_pos[0], y=self.hero_car_pos[1], z=self.hero_car_pos[2])
        wp = self._map.get_waypoint(wp_location)
        hero_vehicle_transform = wp.transform
        hero_model = 'vehicle.lincoln.mkz2017'
        blueprint = random.choice(self.blueprint_library.filter(hero_model))
        blueprint.set_attribute('role_name', 'hero')
        self.hero_car = self.world.try_spawn_actor(blueprint, hero_vehicle_transform)

        models = ['vehicle.nissan.patrol', 'vehicle.audi.tt',
                  'vehicle.lincoln.mkz2017', 'vehicle.volkswagen.t2',
                  'vehicle.tesla.model3', 'vehicle.nissan.micra',
                  'vehicle.audi.a2',
                  ]

        blueprints = [random.choice(self.world.get_blueprint_library().filter(model)) for model in models]
        for blueprint in blueprints:
            blueprint.set_attribute('role_name', 'scenario')

        self.blueprints = blueprints
        self.models = models
        # Not available: 135, 160
        fourth_car_pos = [-74.38717651367188, 40.29738235473633, 1.8052647113800049]  # 13
        fourth_wp_location = carla.Location(x=fourth_car_pos[0], y=fourth_car_pos[1], z=fourth_car_pos[2])
        fourth_vehicle_waypoint = self._map.get_waypoint(fourth_wp_location)
        fourth_vehicle_transform = carla.Transform(fourth_vehicle_waypoint.transform.location,
                                                   fourth_vehicle_waypoint.transform.rotation)
        self.fourth_vehicle = self.world.try_spawn_actor(blueprints[4 % len(models)], fourth_vehicle_transform)
        # setup local planners for zombie cars
        self._fourth_vehicle_speed = 20

        # Not available: 135, 160
        next_fourth_car_pos = [-74.38717651367188, 57.531620025634766, 1.8052573204040527]  # 15
        next_fourth_wp_location = carla.Location(x=next_fourth_car_pos[0], y=next_fourth_car_pos[1], z=next_fourth_car_pos[2])
        next_fourth_vehicle_waypoint = self._map.get_waypoint(next_fourth_wp_location)
        next_fourth_vehicle_transform = carla.Transform(next_fourth_vehicle_waypoint.transform.location,
                                                   next_fourth_vehicle_waypoint.transform.rotation)
        self.next_fourth_vehicle = self.world.try_spawn_actor(blueprints[4 % len(models)], next_fourth_vehicle_transform)
        # setup local planners for zombie cars
        self._next_fourth_vehicle_speed = 20

        self.zombie_cars = [self.fourth_vehicle, self.next_fourth_vehicle]

        fourth_vehicle_planner = WaypointFollower_FullMap(actor=self.fourth_vehicle,
                                                          target_speed=self._fourth_vehicle_speed,
                                                          actor_location=fourth_wp_location,
                                                          map=self._map,
                                                          avoid_collision=False,
                                                          pattern_1=[1, 1, 1, 1, 1, 1, 0, 0, 1, ],
                                                          world=self.world)

        next_fourth_vehicle_planner = WaypointFollower_FullMap(actor=self.next_fourth_vehicle,
                                                          target_speed=self._next_fourth_vehicle_speed,
                                                          actor_location=next_fourth_wp_location,
                                                          map=self._map,
                                                          avoid_collision=False,
                                                          pattern_1=[1, 1, 1, 1, 1, 1, 0, 0, 1, ],
                                                          world=self.world)

        self.vehicle_planners = [fourth_vehicle_planner, next_fourth_vehicle_planner]

        for planner in self.vehicle_planners:
            planner.setup()

    def generate_car(self):
        additional_zombie_car = list()
        additional_zombie_car_speed = list()
        additional_pattern = list()
        additional_actor_location = list()
        all_car_pos = [[-74.38717651367188, 57.531620025634766, 1.805267095565796]]
        all_pattern = [[1, 1, 1, 1, 1, 1, 0, 0, 1, ]]

        for i, car_pos in enumerate(all_car_pos):
            if car_pos == [-1, -1, -1] or car_pos == [-2, -2, -2]:
                # car_pos == [-2, -2, -2]: get_left_lane(), speed=26
                # car_pos == [-1, -1, -1]: get_left_lane()
                car_pos = all_car_pos[i - 1]
                orig_actor_location = carla.Location(x=car_pos[0], y=car_pos[1], z=car_pos[2])
                vehicle_waypoint = self._map.get_waypoint(orig_actor_location).next(16)[0].get_left_lane()
                actor_location = vehicle_waypoint.transform.location
            else:
                actor_location = carla.Location(x=car_pos[0], y=car_pos[1], z=car_pos[2])
                vehicle_waypoint = self._map.get_waypoint(actor_location)

            world_actors = self.world.get_actors().filter('vehicle.*')
            flag_spawn = True
            for adversary in world_actors:
                if actor_location.distance(adversary.get_location()) < 15:
                    flag_spawn = False
            if flag_spawn:
                vehicle_transform = carla.Transform(vehicle_waypoint.transform.location,
                                                    vehicle_waypoint.transform.rotation)
                vehicle = self.world.try_spawn_actor(self.blueprints[np.random.randint(0, len(self.blueprints))],
                                                     vehicle_transform)
                if car_pos == [-2, -2, -2]:
                    _vehicle_speed = 26
                else:
                    _vehicle_speed = 20
                additional_zombie_car.append(vehicle)
                additional_zombie_car_speed.append(_vehicle_speed)
                additional_pattern.append(all_pattern[i])
                additional_actor_location.append(actor_location)
                self.zombie_cars.append(vehicle)

        for i, (one_zombie_car, one_zombie_car_speed, one_pattern, one_actor_location) in enumerate(
                zip(additional_zombie_car, additional_zombie_car_speed, additional_pattern,
                    additional_actor_location)):
            vehicle_planner = WaypointFollower_FullMap(actor=one_zombie_car, map=self._map,
                                                       actor_location=one_actor_location,
                                                       target_speed=one_zombie_car_speed,
                                                       avoid_collision=False, pattern_1=one_pattern,
                                                       world=self.world)

            self.vehicle_planners.append(vehicle_planner)
            vehicle_planner.setup()

    def _update(self):
        # update action for two local planners
        # if _dis3d(_pos3d(self.hero_car), _pos3d(self.first_vehicle)) > 26.:
        #     pass
        # else:
        #     for planner in self.vehicle_planners:
        #         planner.update()
        # self.generate_car()
        for planner in self.vehicle_planners:
            planner.update()

    def restart(self):
        if self.only_reset_hero:
            wp_location = carla.Location(x=self.hero_car_pos[0], y=self.hero_car_pos[1], z=self.hero_car_pos[2])
            wp = self._map.get_waypoint(wp_location)
            wp = wp.next(8)[0]
            hero_vehicle_transform = wp.transform
            hero_model = 'vehicle.lincoln.mkz2017'
            blueprint = random.choice(self.blueprint_library.filter(hero_model))
            blueprint.set_attribute('role_name', 'hero')
            self.hero_car = self.world.try_spawn_actor(blueprint, hero_vehicle_transform)
        else:
            self._remove_all_actors()
            self.zombie_cars = list()
            self.vehicle_planners = list()
            self._scenario_init()

    def _remove_all_actors(self):
        actors = [self.hero_car] + self.zombie_cars
        # actors = self.zombie_cars
        for actor in actors:
            if actor.is_alive:
                actor.destroy()

    def _remove_zombie_cars(self):
        actors = self.zombie_cars
        for actor in actors:
            if actor.is_alive:
                actor.destroy()

class Cross_Turn_Left(object):
    def __init__(self, name, map, world, only_reset_hero=False):
        self.name = name
        self._map = map
        self.world = world
        self.only_reset_hero = only_reset_hero
        self.speed = 0
        self.blueprint_library = self.world.get_blueprint_library()
        self._scenario_init()

    def _scenario_init(self):
        # init hero car
        # --------------------------------------------------------
        # setup cars on a given waypoint
        #self.speed = np.random.choice([21, 26, 42])
        speed_list = [26, 42]
        # self.speed = random.choice([42])  # default is 21, 27, 31
        self.speed = speed_list[logger.select_scenario_id]
        print('Velocity: ', self.speed)
        self.hero_car_pos = [-42.350990295410156, -2.835118293762207, 1.8431016206741333]
        wp_location = carla.Location(x=self.hero_car_pos[0], y=self.hero_car_pos[1], z=self.hero_car_pos[2])
        wp = self._map.get_waypoint(wp_location)
        hero_vehicle_transform = wp.transform
        hero_model = 'vehicle.lincoln.mkz2017'
        blueprint = random.choice(self.blueprint_library.filter(hero_model))
        blueprint.set_attribute('role_name', 'hero')
        self.hero_car = self.world.try_spawn_actor(blueprint, hero_vehicle_transform)

        models = ['vehicle.nissan.patrol', 'vehicle.audi.tt',
                  'vehicle.lincoln.mkz2017', 'vehicle.volkswagen.t2',
                  'vehicle.tesla.model3', 'vehicle.nissan.micra',
                  'vehicle.audi.a2',
                  ]

        blueprints = [random.choice(self.world.get_blueprint_library().filter(model)) for model in models]
        for blueprint in blueprints:
            blueprint.set_attribute('role_name', 'scenario')

        self.blueprints = blueprints
        self.models = models
        # Not available: 135, 160
        fourth_car_pos = [-74.38717651367188, 57.531620025634766, 1.805267095565796]  # 15
        fourth_wp_location = carla.Location(x=fourth_car_pos[0], y=fourth_car_pos[1], z=fourth_car_pos[2])
        fourth_vehicle_waypoint = self._map.get_waypoint(fourth_wp_location)
        fourth_vehicle_transform = carla.Transform(fourth_vehicle_waypoint.transform.location,
                                                   fourth_vehicle_waypoint.transform.rotation)
        self.fourth_vehicle = self.world.try_spawn_actor(blueprints[4 % len(models)], fourth_vehicle_transform)
        # setup local planners for zombie cars
        self._fourth_vehicle_speed = self.speed

        no_2_car_pos = [-95.79371643066406, 0.17835818231105804, 1.8431016206741333]  # 191 below
        no_2_wp_location = carla.Location(x=no_2_car_pos[0], y=no_2_car_pos[1], z=no_2_car_pos[2])
        no_2_vehicle_waypoint = self._map.get_waypoint(no_2_wp_location)
        no_2_vehicle_transform = carla.Transform(no_2_vehicle_waypoint.transform.location,
                                                   no_2_vehicle_waypoint.transform.rotation)
        self.no_2_vehicle = self.world.try_spawn_actor(blueprints[4 % len(models)], no_2_vehicle_transform)
        # setup local planners for zombie cars
        self._no_2_vehicle_speed = self.speed

        # no_3_car_pos = [-84.8062973022461, -25, 1.7985864877700806]  # 27
        # no_3_wp_location = carla.Location(x=no_3_car_pos[0], y=no_3_car_pos[1], z=no_3_car_pos[2])
        # no_3_vehicle_waypoint = self._map.get_waypoint(no_3_wp_location)
        # no_3_vehicle_transform = carla.Transform(no_3_vehicle_waypoint.transform.location,
        #                                          no_3_vehicle_waypoint.transform.rotation)
        # self.no_3_vehicle = self.world.try_spawn_actor(blueprints[4 % len(models)], no_3_vehicle_transform)
        # # setup local planners for zombie cars
        # self._no_3_vehicle_speed = 20

        self.zombie_cars = [self.fourth_vehicle, self.no_2_vehicle] # , self.no_3_vehicle]

        fourth_vehicle_planner = WaypointFollower_FullMap(actor=self.fourth_vehicle,
                                                          target_speed=self._fourth_vehicle_speed,
                                                          actor_location=fourth_wp_location,
                                                          map=self._map,
                                                          avoid_collision=False,
                                                          pattern_1=[1, 1, 1, 1, 1, 1, 0, 0, 1, ],
                                                          world=self.world)
        no_2_vehicle_planner = WaypointFollower_FullMap(actor=self.no_2_vehicle,
                                                          target_speed=self._no_2_vehicle_speed,
                                                        actor_location=no_2_wp_location,
                                                          map=self._map,
                                                          avoid_collision=False,
                                                          pattern_1=[0, 0, 3, ],
                                                          world=self.world)
        # no_3_vehicle_planner = WaypointFollower_FullMap(actor=self.no_3_vehicle,
        #                                                   target_speed=self._no_3_vehicle_speed,
        #                                                 actor_location=no_3_wp_location,
        #                                                   map=self._map,
        #                                                   avoid_collision=False,
        #                                                   pattern_1=[0, 0, 1, 1, 0, 0, 0, 0, 0,],
        #                                                   world=self.world)

        self.vehicle_planners = [fourth_vehicle_planner, no_2_vehicle_planner] # , no_3_vehicle_planner]

        for planner in self.vehicle_planners:
            planner.setup()

    def generate_car(self):
        additional_zombie_car = list()
        additional_zombie_car_speed = list()
        additional_pattern = list()
        additional_actor_location = list()
        all_car_pos = [[-74.38717651367188, 57.531620025634766, 1.805267095565796], [-95.79371643066406, 0.17835818231105804, 1.8431016206741333]]
        all_pattern = [[1, 1, 1, 1, 1, 1, 0, 0, 1, ], [0, 0, 3]]
        # 199: [-85.21101379394531, -126.87477111816406, 1.7985864877700806], [0, 0, 1, 1, 0, 0, 0, 0, 0,]
        for i, car_pos in enumerate(all_car_pos):
            if car_pos == [-1, -1, -1] or car_pos == [-2, -2, -2]:
                # car_pos == [-2, -2, -2]: get_left_lane(), speed=26
                # car_pos == [-1, -1, -1]: get_left_lane()
                car_pos = all_car_pos[i - 1]
                orig_actor_location = carla.Location(x=car_pos[0], y=car_pos[1], z=car_pos[2])
                vehicle_waypoint = self._map.get_waypoint(orig_actor_location).next(16)[0].get_left_lane()
                actor_location = vehicle_waypoint.transform.location
            else:
                actor_location = carla.Location(x=car_pos[0], y=car_pos[1], z=car_pos[2])
                vehicle_waypoint = self._map.get_waypoint(actor_location)

            world_actors = self.world.get_actors().filter('vehicle.*')
            flag_spawn = True
            for adversary in world_actors:
                if actor_location.distance(adversary.get_location()) < 8:
                    flag_spawn = False
            if flag_spawn:
                vehicle_transform = carla.Transform(vehicle_waypoint.transform.location,
                                                    vehicle_waypoint.transform.rotation)
                vehicle = self.world.try_spawn_actor(self.blueprints[np.random.randint(0, len(self.blueprints))],
                                                     vehicle_transform)
                if car_pos == [-2, -2, -2]:
                    _vehicle_speed = 26
                else:
                    _vehicle_speed = self.speed
                additional_zombie_car.append(vehicle)
                additional_zombie_car_speed.append(_vehicle_speed)
                additional_pattern.append(all_pattern[i])
                additional_actor_location.append(actor_location)
                self.zombie_cars.append(vehicle)

        for i, (one_zombie_car, one_zombie_car_speed, one_pattern, one_actor_location) in enumerate(
                zip(additional_zombie_car, additional_zombie_car_speed, additional_pattern,
                    additional_actor_location)):
            vehicle_planner = WaypointFollower_FullMap(actor=one_zombie_car, map=self._map,
                                                       actor_location=one_actor_location,
                                                       target_speed=one_zombie_car_speed,
                                                       avoid_collision=False, pattern_1=one_pattern,
                                                       world=self.world)

            self.vehicle_planners.append(vehicle_planner)
            vehicle_planner.setup()

    def _update(self):
        # update action for two local planners
        # if _dis3d(_pos3d(self.hero_car), _pos3d(self.first_vehicle)) > 26.:
        #     pass
        # else:
        #     for planner in self.vehicle_planners:
        #         planner.update()
        self.generate_car()
        for planner in self.vehicle_planners:
            planner.update()

    def restart(self):
        if self.only_reset_hero:
            wp_location = carla.Location(x=self.hero_car_pos[0], y=self.hero_car_pos[1], z=self.hero_car_pos[2])
            wp = self._map.get_waypoint(wp_location)
            hero_vehicle_transform = wp.transform
            hero_model = 'vehicle.lincoln.mkz2017'
            blueprint = random.choice(self.blueprint_library.filter(hero_model))
            blueprint.set_attribute('role_name', 'hero')
            self.hero_car = self.world.try_spawn_actor(blueprint, hero_vehicle_transform)
        else:
            self._remove_all_actors()
            self.zombie_cars = list()
            self.vehicle_planners = list()
            self._scenario_init()

    def _remove_all_actors(self):
        actors = [self.hero_car] + self.zombie_cars
        # actors = self.zombie_cars
        for actor in actors:
            if actor.is_alive:
                actor.destroy()

    def _remove_zombie_cars(self):
        actors = self.zombie_cars
        for actor in actors:
            if actor.is_alive:
                actor.destroy()

class Cross_Turn_Right(object):
    def __init__(self, name, map, world, only_reset_hero=False):
        self.name = name
        self._map = map
        self.world = world
        self.speed = 0
        self.only_reset_hero = only_reset_hero
        self.blueprint_library = self.world.get_blueprint_library()
        self._scenario_init()

    def _scenario_init(self):
        # init hero car
        # --------------------------------------------------------
        # setup cars on a given waypoint
        self.hero_car_pos = [-42.350990295410156, -2.835118293762207, 1.8431016206741333]
        wp_location = carla.Location(x=self.hero_car_pos[0], y=self.hero_car_pos[1], z=self.hero_car_pos[2])
        wp = self._map.get_waypoint(wp_location)
        hero_vehicle_transform = wp.transform
        hero_model = 'vehicle.lincoln.mkz2017'
        blueprint = random.choice(self.blueprint_library.filter(hero_model))
        blueprint.set_attribute('role_name', 'hero')
        self.hero_car = self.world.try_spawn_actor(blueprint, hero_vehicle_transform)

        models = ['vehicle.nissan.patrol', 'vehicle.audi.tt',
                  'vehicle.lincoln.mkz2017', 'vehicle.volkswagen.t2',
                  'vehicle.tesla.model3', 'vehicle.nissan.micra',
                  'vehicle.audi.a2',
                  ]

        blueprints = [random.choice(self.world.get_blueprint_library().filter(model)) for model in models]
        for blueprint in blueprints:
            blueprint.set_attribute('role_name', 'scenario')

        self.blueprints = blueprints
        self.models = models
        # Not available: 135, 160
        fourth_car_pos = [-74.38717651367188, 57.531620025634766, 1.805267095565796]  # 13
        fourth_wp_location = carla.Location(x=fourth_car_pos[0], y=fourth_car_pos[1], z=fourth_car_pos[2])
        fourth_vehicle_waypoint = self._map.get_waypoint(fourth_wp_location)
        fourth_vehicle_transform = carla.Transform(fourth_vehicle_waypoint.transform.location,
                                                   fourth_vehicle_waypoint.transform.rotation)
        self.fourth_vehicle = self.world.try_spawn_actor(blueprints[4 % len(models)], fourth_vehicle_transform)
        # setup local planners for zombie cars
        self._fourth_vehicle_speed = 20

        no_2_car_pos = [-95.79371643066406, 0.17835818231105804, 1.8431016206741333]  # 191 below
        no_2_wp_location = carla.Location(x=no_2_car_pos[0], y=no_2_car_pos[1], z=no_2_car_pos[2])
        no_2_vehicle_waypoint = self._map.get_waypoint(no_2_wp_location)
        no_2_vehicle_transform = carla.Transform(no_2_vehicle_waypoint.transform.location,
                                                   no_2_vehicle_waypoint.transform.rotation)
        self.no_2_vehicle = self.world.try_spawn_actor(blueprints[4 % len(models)], no_2_vehicle_transform)
        # setup local planners for zombie cars
        self._no_2_vehicle_speed = 20

        no_3_car_pos = [-84.8062973022461, -25, 1.7985864877700806]  # 27
        no_3_wp_location = carla.Location(x=no_3_car_pos[0], y=no_3_car_pos[1], z=no_3_car_pos[2])
        no_3_vehicle_waypoint = self._map.get_waypoint(no_3_wp_location)
        no_3_vehicle_transform = carla.Transform(no_3_vehicle_waypoint.transform.location,
                                                 no_3_vehicle_waypoint.transform.rotation)
        self.no_3_vehicle = self.world.try_spawn_actor(blueprints[4 % len(models)], no_3_vehicle_transform)
        # setup local planners for zombie cars
        self._no_3_vehicle_speed = 20

        self.zombie_cars = [self.fourth_vehicle, self.no_2_vehicle, self.no_3_vehicle]

        fourth_vehicle_planner = WaypointFollower_FullMap(actor=self.fourth_vehicle,
                                                          target_speed=self._fourth_vehicle_speed,
                                                          actor_location=fourth_wp_location,
                                                          map=self._map,
                                                          avoid_collision=True,
                                                          pattern_1=[1, 1, 1, 1, 1, 1, 0, 0, 1, ],
                                                          world=self.world)
        no_2_vehicle_planner = WaypointFollower_FullMap(actor=self.no_2_vehicle,
                                                          target_speed=self._no_2_vehicle_speed,
                                                        actor_location=no_2_wp_location,
                                                          map=self._map,
                                                          avoid_collision=True,
                                                          pattern_1=[0, 0, 3, ],
                                                          world=self.world)
        no_3_vehicle_planner = WaypointFollower_FullMap(actor=self.no_3_vehicle,
                                                          target_speed=self._no_3_vehicle_speed,
                                                        actor_location=no_3_wp_location,
                                                          map=self._map,
                                                          avoid_collision=True,
                                                          pattern_1=[0, 0, 1, 1, 0, 0, 0, 0, 0,],
                                                          world=self.world)

        self.vehicle_planners = [fourth_vehicle_planner, no_2_vehicle_planner, no_3_vehicle_planner]

        for planner in self.vehicle_planners:
            planner.setup()

    def generate_car(self):
        additional_zombie_car = list()
        additional_zombie_car_speed = list()
        additional_pattern = list()
        additional_actor_location = list()
        all_car_pos = [[-74.38717651367188, 57.531620025634766, 1.805267095565796], [-95.79371643066406, 0.17835818231105804, 1.8431016206741333], [-85.21101379394531, -126.87477111816406, 1.7985864877700806]]
        all_pattern = [[1, 1, 1, 1, 1, 1, 0, 0, 1, ], [0, 0, 3], [0, 0, 1, 1, 0, 0, 0, 0, 0,]]

        for i, car_pos in enumerate(all_car_pos):
            if car_pos == [-1, -1, -1] or car_pos == [-2, -2, -2]:
                # car_pos == [-2, -2, -2]: get_left_lane(), speed=26
                # car_pos == [-1, -1, -1]: get_left_lane()
                car_pos = all_car_pos[i - 1]
                orig_actor_location = carla.Location(x=car_pos[0], y=car_pos[1], z=car_pos[2])
                vehicle_waypoint = self._map.get_waypoint(orig_actor_location).next(16)[0].get_left_lane()
                actor_location = vehicle_waypoint.transform.location
            else:
                actor_location = carla.Location(x=car_pos[0], y=car_pos[1], z=car_pos[2])
                vehicle_waypoint = self._map.get_waypoint(actor_location)

            world_actors = self.world.get_actors().filter('vehicle.*')
            flag_spawn = True
            for adversary in world_actors:
                if actor_location.distance(adversary.get_location()) < 15:
                    flag_spawn = False
            if flag_spawn:
                vehicle_transform = carla.Transform(vehicle_waypoint.transform.location,
                                                    vehicle_waypoint.transform.rotation)
                vehicle = self.world.try_spawn_actor(self.blueprints[np.random.randint(0, len(self.blueprints))],
                                                     vehicle_transform)
                if car_pos == [-2, -2, -2]:
                    _vehicle_speed = 26
                else:
                    _vehicle_speed = 25
                additional_zombie_car.append(vehicle)
                additional_zombie_car_speed.append(_vehicle_speed)
                additional_pattern.append(all_pattern[i])
                additional_actor_location.append(actor_location)
                self.zombie_cars.append(vehicle)

        for i, (one_zombie_car, one_zombie_car_speed, one_pattern, one_actor_location) in enumerate(
                zip(additional_zombie_car, additional_zombie_car_speed, additional_pattern,
                    additional_actor_location)):
            vehicle_planner = WaypointFollower_FullMap(actor=one_zombie_car, map=self._map,
                                                       actor_location=one_actor_location,
                                                       target_speed=one_zombie_car_speed,
                                                       avoid_collision=True, pattern_1=one_pattern,
                                                       world=self.world)

            self.vehicle_planners.append(vehicle_planner)
            vehicle_planner.setup()

    def _update(self):
        # update action for two local planners
        # if _dis3d(_pos3d(self.hero_car), _pos3d(self.first_vehicle)) > 26.:
        #     pass
        # else:
        #     for planner in self.vehicle_planners:
        #         planner.update()
        self.generate_car()
        for planner in self.vehicle_planners:
            planner.update()

    def restart(self):
        if self.only_reset_hero:
            wp_location = carla.Location(x=self.hero_car_pos[0], y=self.hero_car_pos[1], z=self.hero_car_pos[2])
            wp = self._map.get_waypoint(wp_location)
            hero_vehicle_transform = wp.transform
            hero_model = 'vehicle.lincoln.mkz2017'
            blueprint = random.choice(self.blueprint_library.filter(hero_model))
            blueprint.set_attribute('role_name', 'hero')
            self.hero_car = self.world.try_spawn_actor(blueprint, hero_vehicle_transform)
        else:
            self._remove_all_actors()
            self.zombie_cars = list()
            self.vehicle_planners = list()
            self._scenario_init()

    def _remove_all_actors(self):
        actors = [self.hero_car] + self.zombie_cars
        # actors = self.zombie_cars
        for actor in actors:
            if actor.is_alive:
                actor.destroy()

    def _remove_zombie_cars(self):
        actors = self.zombie_cars
        for actor in actors:
            if actor.is_alive:
                actor.destroy()
