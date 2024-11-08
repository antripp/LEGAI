import os
import hashlib
import json
import datetime
import logging
import random
import requests
from cryptography.fernet import Fernet # type: ignore

class User:
    def __init__(self, user_id, username, email):
        self.user_id = user_id
        self.username = username
        self.email = email
        self.created_at = datetime.datetime.now()
        self.data = {}

    def collect_user_data(self):
        self.data['name'] = input("Enter your name: ")
        self.data['address'] = input("Enter your address: ")
        self.data['phone'] = input("Enter your phone number: ")

    def save_user_data(self):
        with open(f"user_{self.user_id}.json", "w") as f:
            json.dump(self.data, f)

    def log_user_activity(self, activity):
        logging.info(f"User {self.user_id}: {activity}")

class Payment:
    def __init__(self, payment_id, user_id, amount):
        self.payment_id = payment_id
        self.user_id = user_id
        self.amount = amount
        self.timestamp = datetime.datetime.now()

    def process_payment(self, payment_details):
        if self.validate_payment(payment_details):
            encrypted_details = self.encrypt_payment_details(payment_details)
            print(f"Payment of {self.amount} for user {self.user_id} processed.")
        else:
            self.log_payment_error("Invalid payment details")

    def validate_payment(self, payment_details):
        return bool(random.getrandbits(1))  # Simulate a random pass/fail for payment validation

    def encrypt_payment_details(self, payment_details):
        key = Fernet.generate_key()
        cipher = Fernet(key)
        encrypted_data = cipher.encrypt(payment_details.encode())
        return encrypted_data

    def log_payment_error(self, error_message):
        logging.error(f"Payment {self.payment_id}: {error_message}")

class Product:
    def __init__(self, product_id, name, price, stock_quantity):
        self.product_id = product_id
        self.name = name
        self.price = price
        self.stock_quantity = stock_quantity

    def update_stock(self, quantity):
        if quantity <= self.stock_quantity:
            self.stock_quantity -= quantity
        else:
            logging.warning(f"Product {self.product_id}: Not enough stock available.")

class Order:
    def __init__(self, order_id, user, products):
        self.order_id = order_id
        self.user = user
        self.products = products
        self.total_amount = sum(product.price for product in products)
        self.status = "Pending"

    def confirm_order(self):
        payment = Payment(payment_id=random.randint(1000, 9999), user_id=self.user.user_id, amount=self.total_amount)
        payment_details = input("Enter payment details for order confirmation: ")
        payment.process_payment(payment_details)
        self.status = "Confirmed"

    def ship_order(self):
        if self.status == "Confirmed":
            self.status = "Shipped"
            print(f"Order {self.order_id} has been shipped.")
        else:
            print(f"Order {self.order_id} cannot be shipped. Current status: {self.status}")

class Review:
    def __init__(self, review_id, user_id, product_id, rating, comment):
        self.review_id = review_id
        self.user_id = user_id
        self.product_id = product_id
        self.rating = rating
        self.comment = comment
        self.created_at = datetime.datetime.now()

    def save_review(self):
        review_data = {
            "review_id": self.review_id,
            "user_id": self.user_id,
            "product_id": self.product_id,
            "rating": self.rating,
            "comment": self.comment,
            "created_at": self.created_at.isoformat()
        }
        with open(f"review_{self.review_id}.json", "w") as f:
            json.dump(review_data, f)

class Notification:
    def __init__(self, notification_id, user_id, message):
        self.notification_id = notification_id
        self.user_id = user_id
        self.message = message
        self.sent_at = None

    def send_notification(self):
        # Simulate sending a notification
        self.sent_at = datetime.datetime.now()
        print(f"Notification sent to user {self.user_id}: {self.message}")

class ExternalAPI:
    @staticmethod
    def fetch_exchange_rate():
        try:
            response = requests.get("https://api.exchangerate-api.com/v4/latest/USD")
            if response.status_code == 200:
                return response.json().get("rates", {})
            else:
                logging.error("Failed to fetch exchange rate.")
                return {}
        except requests.RequestException as e:
            logging.error(f"Error occurred while fetching exchange rate: {e}")
            return {}

class AuditTrail:
    def __init__(self, audit_id, user_id, action, details):
        self.audit_id = audit_id
        self.user_id = user_id
        self.action = action
        self.details = details
        self.timestamp = datetime.datetime.now()

    def log_audit_trail(self):
        audit_entry = {
            "audit_id": self.audit_id,
            "user_id": self.user_id,
            "action": self.action,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }
        with open(f"audit_{self.audit_id}.json", "w") as f:
            json.dump(audit_entry, f)

if __name__ == "__main__":
    user = User(user_id=1, username="john_doe", email="john@example.com")
    user.collect_user_data()
    user.save_user_data()

    product1 = Product(product_id=101, name="Laptop", price=1200, stock_quantity=10)
    product2 = Product(product_id=102, name="Mouse", price=20, stock_quantity=50)
    product1.update_stock(2)

    order = Order(order_id=5001, user=user, products=[product1, product2])
    order.confirm_order()
    order.ship_order()

    review = Review(review_id=3001, user_id=user.user_id, product_id=product1.product_id, rating=4, comment="Great product!")
    review.save_review()

    notification = Notification(notification_id=4001, user_id=user.user_id, message="Your order has been shipped!")
    notification.send_notification()

    audit = AuditTrail(audit_id=6001, user_id=user.user_id, action="Order Shipped", details=f"Order {order.order_id} was shipped.")
    audit.log_audit_trail()

    exchange_rate = ExternalAPI.fetch_exchange_rate()
    print("Exchange Rate:", exchange_rate)