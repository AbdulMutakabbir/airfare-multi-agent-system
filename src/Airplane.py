class Airplane:

    def __init__(self, identifier, capacity, price_economy_class, price_business_class, price_first_class):
        self.airplane_id = identifier
        self.airplane_capacity = capacity
        self.economy_class_price = price_economy_class
        self.business_class_price = price_business_class
        self.firs_class_price = price_first_class

    def increment_prices(self):
        self.economy_class_price += 10
        self.business_class_price += 50
        self.firs_class_price += 100

    def decrement_prices(self):
        self.economy_class_price -= 10
        self.business_class_price -= 50
        self.firs_class_price -= 100