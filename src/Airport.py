class Airport:

    def __init__(self, iata, name, city, country, lat, long):
        self.iata = iata
        self.name = name
        self.city = city
        self.country = country
        self.lat = lat
        self.long = long

    def __str__(self):
        return self.iata + " (" + self.city + "," + self.country + ")"
