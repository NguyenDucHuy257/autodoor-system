"""Face boxes only; web UI owns names and details."""
import base64
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def face_thumbnail(frame, box):
    height, width = frame.shape[:2]
    x, y, w, h = (int(box[k]) for k in ("x", "y", "width", "height"))
    left, top, right, bottom = max(0, x), max(0, y), min(width, x+w), min(height, y+h)
    if right <= left or bottom <= top: return None
    crop = frame[top:bottom, left:right]
    scale = min(1.0, 128/max(crop.shape[:2]))
    if scale < 1:
        crop = cv2.resize(crop, (max(1, int(crop.shape[1]*scale)), max(1, int(crop.shape[0]*scale))))
    ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64,"+base64.b64encode(encoded).decode("ascii") if ok else None

def draw_faces(frame, faces, age_ms=0, model="AdaFace"):
    output = frame.copy()
    for result in faces:
        x, y, w, h = (int(result["bbox"][k]) for k in ("x", "y", "width", "height"))
        color = (40, 220, 40) if result.get("verdict") == "RECOGNIZED" else (30, 30, 240)
        cv2.rectangle(output, (x, y), (x+w, y+h), color, 2)
    return output


@lru_cache(maxsize=1)
def caption_font():
    for path in (Path("C:/Windows/Fonts/arial.ttf"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")):
        if path.is_file():
            return ImageFont.truetype(str(path), 14)
    return ImageFont.load_default()


def recording_frame(frame, faces):
    """Export the recognition image and its bottom-right crop/name strip."""
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    right = image.width-8
    labels = {"UNKNOWN": "Không xác định", "REJECTED": "Chất lượng không đạt", "NO_EMBEDDING": "Chưa có embedding"}
    for face in reversed(faces):
        if not face.get("crop") or right < 80:
            continue
        raw = base64.b64decode(face["crop"].split(",", 1)[1])
        crop = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if crop is None:
            continue
        size, caption_height = 80, 46
        left, top = right-size, max(0, image.height-size-caption_height-8)
        crop = cv2.resize(crop, (size, size))
        image.paste(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)), (left, top))
        draw.rectangle((left, top+size, right, top+size+caption_height), fill=(0,0,0))
        name = (face.get("identity") or {}).get("name") or labels.get(face["verdict"], face["verdict"])
        lines, line = [], ""
        for char in name:
            if line and draw.textlength(line+char, font=caption_font()) > size-6:
                lines.append(line)
                line = ""
            line += char
        if line: lines.append(line)
        for index, line in enumerate(lines[:2]):
            draw.text((left+3, top+size+3+index*18), line, font=caption_font(), fill=(255,255,255))
        color = (40,220,40) if face.get("verdict") == "RECOGNIZED" else (240,30,30)
        draw.rectangle((left, top, right, top+size+caption_height), outline=color, width=2)
        right = left-6
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
