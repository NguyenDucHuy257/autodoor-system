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

YAW_THRESHOLD = float(os.getenv("YAW_THRESHOLD"))
DOWN_THRESHOLD = float(os.getenv("DOWN_THRESHOLD"))
FAR_FACE_RATIO = float(os.getenv("FAR_FACE_RATIO"))

FACE_MIN_CONFIDENCE = float(os.getenv("FACE_MIN_CONFIDENCE")) 
FACE_MIN_RATIO = float(os.getenv("FACE_MIN_RATIO")) 
FACE_MIN_SHARPNESS = float(os.getenv("FACE_MIN_SHARPNESS")) 
FACE_MIN_BRIGHTNESS = float(os.getenv("FACE_MIN_BRIGHTNESS")) 
FACE_MAX_BRIGHTNESS = float(os.getenv("FACE_MAX_BRIGHTNESS")) 
