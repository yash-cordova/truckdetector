from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()  # Load .env file variables

MONGO_URL = os.getenv("MONGO_URL")

client = MongoClient(MONGO_URL)
db = client["dock_management"]

# Define the collections
users_collection = db["users"]
settings_collection = db["settings"]  # new collection for system settings
dock1_collection = db["dock1"]  # dock1 collection for real-time updates
