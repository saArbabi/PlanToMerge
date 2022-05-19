import numpy as np
from importlib import reload
from vehicles.tracked_vehicle import TrackedVehicle
# import time

class EnvInitializor():
    def __init__(self, config):
        self.lanes_n = config['lanes_n']
        self.lane_length = config['lane_length']
        self.lane_width = config['lane_width']
        self.lane_length = config['lane_length']
        self.merge_lane_start = config['merge_lane_start']

        self.min_v = 10
        self.max_v = 25
        self.desired_v_range = self.max_v-self.min_v

    def get_init_speed(self, aggressiveness):
        init_speed = self.rng.uniform(self.min_v, \
                            self.min_v+aggressiveness*self.desired_v_range)
        return init_speed

    def create_main_lane_vehicle(self, lead_vehicle, lane_id, glob_x, agg):
        init_speed = self.get_init_speed(agg)
        new_vehicle = TrackedVehicle(\
                    self.next_vehicle_id, lane_id, glob_x,\
                                                init_speed, agg)
        new_vehicle.set_driver_params(self.rng)
        init_action = new_vehicle.idm_action(new_vehicle, lead_vehicle)
        while init_action < -new_vehicle.driver_params['min_act']:
            new_vehicle.glob_x -= 10
            init_action = new_vehicle.idm_action(new_vehicle, lead_vehicle)
            if new_vehicle.glob_x < 0:
                return
        else:
            self.next_vehicle_id += 1
            return new_vehicle

    def create_ramp_merge_vehicle(self, lane_id, glob_x, agg):
        lead_vehicle = self.dummy_stationary_car
        init_speed = self.get_init_speed(agg)
        new_vehicle = TrackedVehicle(\
                    self.next_vehicle_id, lane_id, glob_x,\
                                                init_speed, agg)

        new_vehicle.set_driver_params(self.rng)
        init_action = new_vehicle.idm_action(new_vehicle, lead_vehicle)
        if init_action >= new_vehicle.driver_params['safe_braking']:
            self.next_vehicle_id += 1
            return new_vehicle

    def init_env(self, episode_id):
        print('episode_id: ', episode_id)
        self.rng = np.random.RandomState(episode_id)

        # main road vehicles
        lane_id = 1
        vehicles = []
        traffic_density = 2

        available_spacing = self.lane_length - self.merge_lane_start-50
        glob_x = available_spacing
        avg_spacing = available_spacing/traffic_density
        aggs = [0.5, 0.2]

        lead_vehicle = None
        new_vehicle = self.create_main_lane_vehicle(lead_vehicle, \
                                        lane_id, glob_x, aggs[0])
        vehicles.append(new_vehicle)

        lead_vehicle = vehicles[-1]
        glob_x = 50
        new_vehicle = self.create_main_lane_vehicle(lead_vehicle, \
                                        lane_id, glob_x, aggs[1])

        vehicles.append(new_vehicle)

        lead_car_speed = self.rng.uniform(self.min_v, self.max_v)
        vehicles[0].speed = vehicles[0].driver_params['desired_v'] = lead_car_speed

        # ramp vehicles
        lane_id = 2
        aggs = 0.9
        glob_x = 100
        new_vehicle = self.create_ramp_merge_vehicle(lane_id, glob_x, aggs)
        vehicles.append(new_vehicle)
        return vehicles
