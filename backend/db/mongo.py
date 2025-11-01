from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure
from dotenv import load_dotenv
import os
import logging

logger = logging.getLogger(__name__)

load_dotenv()  # Load .env file variables

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017/")

# Create MongoDB client with timeout settings
try:
    client = MongoClient(
        MONGO_URL,
        serverSelectionTimeoutMS=5000,  # 5 seconds timeout
        connectTimeoutMS=5000,
        socketTimeoutMS=5000,
        maxPoolSize=10,
        retryWrites=True
    )
    # Test connection
    client.server_info()
    logger.info(f"Successfully connected to MongoDB at {MONGO_URL}")
    db = client["dock_management"]
    
    # Define the collections
    users_collection = db["users"]
    settings_collection = db["settings"]  # new collection for system settings
    dock1_collection = db["dock1"]  # dock1 collection for real-time updates
    
except (ServerSelectionTimeoutError, ConnectionFailure) as e:
    logger.error(f"Failed to connect to MongoDB at {MONGO_URL}: {e}")
    logger.warning("Application will continue but database operations will fail")
    # Create dummy collections to prevent import errors
    client = None
    db = None
    users_collection = None
    settings_collection = None
    dock1_collection = None
except Exception as e:
    logger.error(f"Unexpected error connecting to MongoDB: {e}")
    client = None
    db = None
    users_collection = None
    settings_collection = None
    dock1_collection = None
