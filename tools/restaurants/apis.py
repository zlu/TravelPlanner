import pandas as pd
from pandas import DataFrame
from typing import Optional
import numpy as np
from z3 import Array, ArraySort, BoolSort, IntSort, Select, Store, If

from utils.func import extract_before_parenthesis

class Restaurants:
    def __init__(self, path="../database/restaurants/clean_restaurant_2022.csv"):
        self.path = path
        self.data = pd.read_csv(self.path).dropna()[['Name','Average Cost','Cuisines','Aggregate Rating','City']]
        print("Restaurants loaded.")

    def load_db(self):
        self.data = pd.read_csv(self.path).dropna()

    def run(self,
            city: str,
            ) -> DataFrame:
        """Search for restaurant ."""
        results = self.data[self.data["City"] == city]
        # results = results[results["date"] == date]
        # if price_order == "asc":
        #     results = results.sort_values(by=["Average Cost"], ascending=True)
        # elif price_order == "desc": 
        #     results = results.sort_values(by=["Average Cost"], ascending=False)

        # if rating_order == "asc":
        #     results = results.sort_values(by=["Aggregate Rating"], ascending=True)
        # elif rating_order == "desc":
        #     results = results.sort_values(by=["Aggregate Rating"], ascending=False)
        if len(results) == 0:
            return "There is no restaurant in this city."
        non_repeat = []
        drop_index = []
        for index in range(np.array(results).shape[0]):
            name = np.array(results)[:,0][index]
            if name not in non_repeat:
                non_repeat.append(name)
            else:
                drop_index.append(results.index.to_numpy()[index])
        return results.drop(drop_index)

    def run_for_all_cities(self, all_cities: list, cities: list) -> DataFrame:
        """Search for flights by origin, destination, and departure date."""
        results = Array('restaurant', IntSort(), IntSort(), ArraySort(IntSort(), IntSort()))
        results_cuisines = Array(
            'restaurant cuisines',
            IntSort(),
            ArraySort(IntSort(), IntSort(), BoolSort()),
        )
        cuisines_list = ['Chinese', 'American', 'Italian', 'Mexican', 'Indian', 'Mediterranean', 'French']
        for i, city in enumerate(cities):
            result = self.data[self.data["City"] == city]
            if len(result) != 0:
                price = Array('Price', IntSort(), IntSort())
                cuisines = Array('Cuisines', IntSort(), IntSort(), BoolSort())
                length = Array('Length', IntSort(), IntSort())
                non_repeat = []
                non_repeat_index = []
                for index in range(np.array(result).shape[0]):
                    name = np.array(result)[:,0][index]
                    if name not in non_repeat:
                        non_repeat.append(name)
                        non_repeat_index.append(index)
                for order, index in enumerate(non_repeat_index):
                    price = Store(price, order, np.array(result)[:,1][index])
                    types = np.array(result)[:,2][index]
                    for j in range(len(cuisines_list)):
                        cuisines = Store(cuisines, order, j, cuisines_list[j] in types)

                length = Store(length, 0, len(non_repeat_index))
                results = Store(results, all_cities.index(city), 0, price)
                results = Store(results, all_cities.index(city), 1, length)
                results_cuisines = Store(results_cuisines, all_cities.index(city), cuisines)
            else:
                length = Array('Length', IntSort(), IntSort())
                length = Store(length, 0, -1)
                results = Store(results, all_cities.index(city), 1, length)
        return results, results_cuisines

    def get_info(self, info, i, key):
        if key == 'Cuisines':
            info_key = Select(info, i)
            return info_key, None
        element = ['Price', 'Length']
        info_key = Select(info, i, element.index(key))
        info_length = Select(info, i, 1)
        length = Select(info_length, 0)
        return info_key, length

    def get_info_for_index(self, price_list, index):
        return Select(price_list, index)

    def eat_in_which_city(self, arrives, origin, cities, departure_dates, days):
        result = []
        origin = -1
        cities = [origin] + cities + [origin]
        arrives_array = Array('arrives', IntSort(), IntSort())
        cities_array = Array('cities', IntSort(), IntSort())
        departure_dates_array = Array('departure_dates', IntSort(), IntSort())
        for index, arrive in enumerate(arrives):
            arrives_array = Store(arrives_array, index, arrive)
        for index, city in enumerate(cities):
            cities_array = Store(cities_array, index, city)
        for index, date in enumerate(departure_dates):
            departure_dates_array = Store(departure_dates_array, index, date)
        i = 0
        for day in range(days):
            arrtime = Select(arrives_array, i)
            result.append(
                If(
                    day == Select(departure_dates_array, i),
                    If(arrtime > 10, Select(cities_array, i), Select(cities_array, i + 1)),
                    Select(cities_array, i + 1),
                )
            )
            result.append(
                If(
                    day == Select(departure_dates_array, i),
                    If(arrtime > 13, Select(cities_array, i), Select(cities_array, i + 1)),
                    Select(cities_array, i + 1),
                )
            )
            result.append(
                If(
                    day == Select(departure_dates_array, i),
                    If(arrtime > 20, Select(cities_array, i), Select(cities_array, i + 1)),
                    Select(cities_array, i + 1),
                )
            )
            i += If(day == Select(departure_dates_array, i), 1, 0)
        print("Having eat in which info for {} restaurants".format(len(result)))
        return result

    def check_exists(self, cuisine, restaurant_cuisines_list, restaurant_index):
        cuisines_list = ['Chinese', 'American', 'Italian', 'Mexican', 'Indian', 'Mediterranean', 'French']
        exists = Select(restaurant_cuisines_list, restaurant_index, cuisines_list.index(cuisine))
        return If(restaurant_index != -1, exists, False)

    def run_for_annotation(self,
            city: str,
            ) -> DataFrame:
        """Search for restaurant ."""
        results = self.data[self.data["City"] == extract_before_parenthesis(city)]
        # results = results[results["date"] == date]
        # if price_order == "asc":
        #     results = results.sort_values(by=["Average Cost"], ascending=True)
        # elif price_order == "desc":
        #     results = results.sort_values(by=["Average Cost"], ascending=False)

        # if rating_order == "asc":
        #     results = results.sort_values(by=["Aggregate Rating"], ascending=True)
        # elif rating_order == "desc":
        #     results = results.sort_values(by=["Aggregate Rating"], ascending=False)

        return results
