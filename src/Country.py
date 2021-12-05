class Country:

    def __init__(self, name ):
        self.country_name = name
        self.airports = {}
        self.airlines = {}
        self.population = 2000

    def __str__(self):
        return self.country_name + "(" + str(len(self.airports)) + "," + str(len(self.airlines)) + "," + str(self.population) + ")"

    def add_airport(self, airport_id, airport):
        self.airports[airport_id] = airport

    def add_airline(self, airline_id, airline):
        self.airlines[airline_id] = airline



