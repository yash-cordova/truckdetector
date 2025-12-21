from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "your-secret-key")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 1))
    SERVER_HOST: str = os.getenv("SERVER_HOST", "192.168.0.110")
    SERVER_PORT: int = int(os.getenv("SERVER_PORT", 3041))

settings = Settings()
