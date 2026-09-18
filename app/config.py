import os 
from dotenv import load_dotenv

load_dotenv()

CAMRERA_IP = os.getenv("CAMERA_IP")
CAMERA_USERNAME = os.getenv("CAMERA_USERNAME")
CAMERA_PASSWORD = os.getenv("CAMERA_PASSWORD")
CAMERA_RTSP_PORT = os.getenv("CAMERA_RTSP_PORT")