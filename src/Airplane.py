from math import pow
from datetime import date, timedelta


class Airplane:
    PRICE_PER_KILOMETER = 0.325
    MAX_BOOKING_START_DAYS = 60
    DEMAND_COEFFICIENT = 3
    SALE_COEFFICIENT = 2
    capacity = 250

    def __init__(self, identifier, airline, start_airport, end_airport, distance):
        self.airplane_id = identifier
        self.airline = airline
        self.start_airport = start_airport
        self.end_airport = end_airport
        self.travel_distance = distance
        self.departure_datetime_list = [date.today() + timedelta(3),date.today() + timedelta(5)]

    def __str__(self):
        return self.airline + " : " + self.start_airport + ">" + self.end_airport

    def get_price(self, today, departure_date, demand):
        if isinstance(today, date) and isinstance(departure_date, date) and isinstance(demand, float):
            if departure_date in self.departure_datetime_list:
                days_before_departure = (departure_date - today).days
                if days_before_departure >= 0:
                    time = days_before_departure / self.MAX_BOOKING_START_DAYS
                    price_inc = demand - pow((- pow((time - 1), self.SALE_COEFFICIENT) - demand),
                                             self.DEMAND_COEFFICIENT)

                    price = (self.PRICE_PER_KILOMETER * self.travel_distance)
                    price += price_inc * price
                    return price
                else:
                    raise Exception("Flight already departed")
            else:
                raise Exception("No fight for the the given date")
        else:
            raise Exception("Type miss match")
