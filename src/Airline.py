class Airline:

    base_price = 10000

    def __init__(self, name, country):
        self.name = name
        self.country = country
        self.airplanes = {}

    def __str__(self):
        return self.name + "(" + str(len(self.airplanes)) + "-" + self.country + ")"

    def add_airplane(self, airplane_id, airplane):
        self.airplanes[airplane_id] = airplane

