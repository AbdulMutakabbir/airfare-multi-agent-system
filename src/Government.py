class Government:

    def __init__(self, country):
        self.country = country
        self.restricted_countries = []

    def add_restricted_country(self, country):
        self.restricted_countries.append(country)

    def del_restricted_country(self, country):
        self.restricted_countries.remove(country)

