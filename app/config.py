import os 
from dotenv import load_dotenv

load_dotenv()

CAMRERA_IP = os.getenv("CAMERA_IP")
CAMERA_USERNAME = os.getenv("CAMERA_USERNAME")
CAMERA_PASSWORD = os.getenv("CAMERA_PASSWORD")
CAMERA_RTSP_PORT = os.getenv("CAMERA_RTSP_PORT")

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
