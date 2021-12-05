from datetime import date, timedelta, datetime
from calendar import monthrange
from random import uniform
import logging
import json
import pandas as pd
import numpy as np
import networkx as nx
import os
import random
from scipy.stats import skewnorm
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

from Airport import Airport
from Airline import Airline
from Airplane import Airplane
from Country import Country


class Environment:
    PASSENGER_VARIABILITY_PERCENTAGE_PER_DAY = 5
    PASSENGER_PERSONAL_BOOKING_SKEW = 0.2  # Negative values are left skewed, positive values are right skewed.
    PASSENGER_BUSINESS_BOOKING_SKEW = 8  # Negative values are left skewed, positive values are right skewed.
    PASSENGER_PERSONAL_PROBABILITY = 0.5  # Split for personal passengers
    DAYS_BEFORE_BOOKING_START = 365

    def __init__(self, today=date.today(), world_loc=None, route_loc=None, flow_loc=None):
        self.date = today
        self.countries = {}
        self.airports = {}
        self.airlines = {}
        self.airplanes = {}
        self.distance_graph = {}
        self.route_graph = nx.MultiGraph()
        self.route_df = pd.DataFrame()
        self.passenger_flow_df = pd.DataFrame()
        self.passenger_flow_monthly_sum = []
        self.passenger_flow_airport_list = []
        self.passenger_flow_monthly_weights = []
        logger.info("Started initializing ENVIRONMENT...")
        if world_loc is not None and route_loc is not None:
            self.init_env(world_loc)
            self.init_routes(route_loc)
            self.init_passenger_data(flow_loc)
        else:
            logger.error("Environment data file not found")
            raise NotImplemented
        logger.info("Finished initializing ENVIRONMENT.")

    def init_passenger_data(self, file_location):
        logger.info("Started initializing passenger demand data...")
        self.passenger_flow_df = pd.read_csv(file_location, index_col=0, header=0).astype('int32')
        self.passenger_flow_monthly_sum = list(self.passenger_flow_df.sum())
        self.passenger_flow_airport_list = list(self.passenger_flow_df.index)
        self.passenger_flow_monthly_weights = pd.DataFrame()
        for month in range(1, 12 + 1):
            self.passenger_flow_monthly_weights[str(month)] = self.passenger_flow_df[str(month)] / \
                                                              self.passenger_flow_monthly_sum[month - 1] * 100
        logger.info("Finished initializing passenger demand data.")

    def init_routes(self, file_location):
        logger.info("Started initializing world Routes...")
        self.route_df = pd.read_csv(file_location, index_col=0, header=0)
        self.route_graph = nx.from_pandas_edgelist(self.route_df, 'source', 'target', 'Distance_Km')
        logger.info("Finished initializing world Routes.")

    def init_env(self, file_location):
        logger.info("Started initializing world...")
        world = {}
        with open(file_location) as world_json:
            world = json.load(world_json)

        for airport in world["airports"]:
            airport_data = world["airports"][airport]
            new_airport = Airport(iata=airport_data["iata"],
                                  name=airport_data["name"],
                                  city=airport_data["city"],
                                  country=airport_data["country"],
                                  lat=airport_data["lat"],
                                  long=airport_data["long"])
            self.airports[airport] = new_airport
        logger.info("Finished initializing world airports.")

        for country in world["countries"]:
            new_country = Country(country)
            for airport in world["countries"][country]["airports"]:
                new_country.add_airport(airport, self.airports[airport])
            for airline in world["countries"][country]["airlines"]:
                airline_data = world["countries"][country]["airlines"][airline]
                new_airline = Airline(name=airline_data["name"],
                                      country=country)
                self.airlines[airline] = new_airline
                new_country.add_airline(airline, new_airline)
                for airplane in world["countries"][country]["airlines"][airline]["airplanes"]:
                    airplane_data = world["countries"][country]["airlines"][airline]["airplanes"][airplane]
                    new_airplane = Airplane(identifier=airplane,
                                            airline=airline,
                                            start_airport=airplane_data["source_airport"],
                                            end_airport=airplane_data["destination_airport"],
                                            distance=airplane_data["distance"])
                    new_airline.add_airplane(airplane, new_airplane)
                    airplane_tuple = (airplane_data["source_airport"], airplane_data["destination_airport"])
                    if airplane_tuple not in self.airplanes:
                        self.airplanes[airplane_tuple] = {airplane: new_airplane}
                    else:
                        self.airplanes[airplane_tuple][airplane] = new_airplane
            self.countries[country] = new_country
        logger.info("Finished initializing world country data.")
        logger.info("Finished initializing world.")

    def get_demand(self):
        if self.date.month in [1, 7, 8, 9, 12]:
            return uniform(0.8, 1)
        if self.date.month in [4, 5, 6, 10, 11]:
            return uniform(0.3, 0.8)
        if self.date.month in [2, 3]:
            return uniform(0.1, 0.3)

    def increment_ticker(self):
        self.date += timedelta(1)

    def get_month(self):
        return self.date.month

    def get_number_of_passenger_today(self):
        month = self.get_month()
        return self.passenger_flow_monthly_sum[month - 1]

    def get_transit_airports(self):
        return self.passenger_flow_airport_list

    def get_transit_airports_weights(self):
        month = str(self.get_month())
        return self.passenger_flow_monthly_weights[month]

    def get_random_path(self):
        airports = self.get_transit_airports()
        airports_weight = self.get_transit_airports_weights()
        return random.choices(airports, weights=airports_weight, k=2)

    def generate_passenger_path(self):
        path = self.get_random_path()
        while path[0] == path[1]:
            path = self.get_random_path()
        return path

    # def do__(self):
    #     path = self.get_random_path()
    #     comp_path = nx.dijkstra_path(self.route_graph, source=path[0], target=path[1])
    #     print(comp_path, path)

    @staticmethod
    def get_skewed_data(skew, max_value, size):
        random_skew = skewnorm.rvs(a=skew, loc=max_value, size=size)
        if size != 0:
            random_skew -= min(random_skew)
            random_skew /= max(random_skew)
            random_skew *= random_skew
            # plt.hist(random_skew, 365, color='red', alpha=0.1)
            # plt.show()
        return random_skew

    def build_passenger_booking_pattern(self, number_of_days):
        timestamp = datetime.now().timestamp()
        logger.info("Started building passenger booking pattern...")
        day_data = []
        normalised_max_value = 1
        total_passenger_count = 0
        logger.info("Started creating passenger source and destination airport...")
        day = 0
        month = 1
        while day < number_of_days:
            passenger_count = self.get_number_of_passenger_today()
            for passenger_index in range(passenger_count):
                path = self.generate_passenger_path()
                path.append(day + 1)
                day_data.append(path)
            total_passenger_count += passenger_count
            logger.info(f"Finished passenger path for day: {day + 1}")

            month_end_day = monthrange(self.date.year, self.date.month)[1]
            if month_end_day == self.date.day:
                logger.info(f"Started saving passenger data for month {month}...")
                personal_passenger_count = round(total_passenger_count * self.PASSENGER_PERSONAL_PROBABILITY)
                business_passenger_count = total_passenger_count - personal_passenger_count

                personal_passenger_skew_booking_day = self.get_skewed_data(skew=self.PASSENGER_PERSONAL_BOOKING_SKEW,
                                                                           max_value=normalised_max_value,
                                                                           size=personal_passenger_count)
                business_passenger_skew_booking_day = self.get_skewed_data(skew=self.PASSENGER_BUSINESS_BOOKING_SKEW,
                                                                           max_value=normalised_max_value,
                                                                           size=business_passenger_count)

                prebooked_days_norm = np.append(personal_passenger_skew_booking_day,
                                                business_passenger_skew_booking_day)
                is_personal = np.append(np.ones((1, personal_passenger_count)), np.zeros((1, business_passenger_count)))

                month_array = np.full(shape=personal_passenger_count+business_passenger_count, fill_value=month, dtype=np.int)

                prebooked_days_norm, is_personal = shuffle(prebooked_days_norm, is_personal)

                df = pd.DataFrame(day_data, columns=['s', 'd', 'day_of_flight'])
                df['prebooked_days_norm'] = prebooked_days_norm
                df = df.assign(
                    prebooked_days=lambda row: round(row.prebooked_days_norm * self.DAYS_BEFORE_BOOKING_START))
                df = df.assign(
                    day_of_booking=lambda row: self.DAYS_BEFORE_BOOKING_START + row.day_of_flight - row.prebooked_days)
                df['is_passenger'] = is_personal
                df['month'] = month_array
                df.to_csv("./gen_dat/passenger_route_" + str(month) + "_" + str(timestamp) + ".csv")
                logger.info(f"Finished saving passenger data for month {month}.")

                del df
                del prebooked_days_norm
                del is_personal
                del business_passenger_skew_booking_day
                del personal_passenger_skew_booking_day
                del day_data
                day_data = []
                total_passenger_count = 0
                month += 1

            self.increment_ticker()
            day += 1

        logger.info("Finished creating passenger source and destination airport.")
        logger.info("Finished building passenger booking pattern.")


#
# e = Environment()
# high_demand_date = date.fromisoformat('2021-07-01')
# mid_demand_date = date.fromisoformat('2021-05-01')
# low_demand_date = date.fromisoformat('2021-03-01')
#
# print("\nhigh")
# for i in range(10):
#     print(e.get_demand(high_demand_date), end=",")
# print("\nmid")
# for i in range(10):
#     print(e.get_demand(mid_demand_date), end=",")
# print("\nlow")
# for i in range(10):
#     print(e.get_demand(low_demand_date), end=",")


curr_file_path = os.path.realpath(__file__)
log_file_path = os.path.dirname(curr_file_path) + os.sep + os.pardir + os.sep + "log" + os.sep + "environment_log.log"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

log_file_handler = logging.FileHandler(log_file_path, mode='w')
log_file_handler.setFormatter(log_formatter)

log_stream_handler = logging.StreamHandler()
log_stream_handler.setFormatter(log_formatter)

logger.addHandler(log_file_handler)
logger.addHandler(log_stream_handler)


base_file_path = os.path.dirname(curr_file_path) + os.sep + os.pardir + os.sep + "data" + os.sep + "cleaned" + os.sep
world_file_loc = base_file_path + "eu_world_data_2019.json"
edge_file_loc = base_file_path + "eu_cleaned_airlines_routes.csv"
flow_file_loc = base_file_path + "eu_cleaned_airports_2019_passenger_data_complete.csv"

simulation_start_day = date(2019, 1, 1)

# print(base_file_path)
e = Environment(simulation_start_day, world_file_loc, edge_file_loc, flow_file_loc)


# print(e.passenger_flow_df)
# print(e.passenger_flow_airport_list)
# print(e.passenger_flow_monthly_weights)
# print(e.passenger_flow_monthly_sum)


# print(e.get_transit_airports())
# print(e.passenger_flow_airport_list)

start_time = datetime.now()
e.build_passenger_booking_pattern(number_of_days=365)
end_time = datetime.now()
print(end_time - start_time)


# date = datetime.now()
# d = date.replace(day=monthrange(date.year, date.month)[1])
# print(monthrange(date.year, date.month)[1], date.day)



# total_planes = 0
# for route in e.airplanes:
#     airplane_count = len(e.airplanes[route])
#     total_planes += airplane_count
#     print(route, airplane_count)
# print(total_planes, len(e.airplanes))

# print(e.countries["Belgium"].airlines["SN"].airplanes["SN-BRU-FCO"])
# print(e.countries["Canada"].airports["YAZ"])
# print(e.countries["Canada"].airlines["AC"])
# e.countries["Canada"].airlines["AC"].airplanes["AC-YYZ-ISL"].capacity = 100
# print(e.countries["Canada"].airlines["AC"].airplanes["AC-YYZ-ISL"].capacity)
# print(e.airlines["AC"].airplanes["AC-YYZ-ISL"].capacity)
# print(e.airplanes[("ISL","YYZ")]["AC-ISL-YYZ"].capacity)
# print(e.airplanes[("ISL","YYZ")]["AC-ISL-YYZ"].get_price(date.today(), date.today()+timedelta(3), 0.1))
# print(e.airplanes[("ISL","YYZ")]["AC-ISL-YYZ"].get_price(date.today(), date.today()+timedelta(5), 0.1))
# print(e.airplanes[("ISL","YYZ")]["AC-ISL-YYZ"].get_price(date.today(), date.today()+timedelta(3), 0.5))
# print(e.airplanes[("ISL","YYZ")]["AC-ISL-YYZ"].get_price(date.today(), date.today()+timedelta(5), 0.5))
# print(e.airplanes[("ISL","YYZ")]["AC-ISL-YYZ"].get_price(date.today(), date.today()+timedelta(3), 0.8))
# print(e.airplanes[("ISL","YYZ")]["AC-ISL-YYZ"].get_price(date.today(), date.today()+timedelta(5), 0.8))
# print(e.airports["KEF"])
# print(e.airports["KFS"])
# # print(e.airplanes[("KEF","KFS")])#["AC-ISL-YYZ"].capacity)
# x = nx.dijkstra_path(e.route_graph, source="KEF", target="KFS")
# print(x)
