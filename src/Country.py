class Country:

    def __init__(self, airports):
        self.country_name = None
        self.airports = []
        self.restricted_countries = []

    def add_airport(self, airport):
        self.airports.append(airport)

    def set_country_name(self, name):
        if self.country_name is None:
            self.country_name = name

    def add_restricted_countries(self, country):
        self.restricted_countries.append(country)
