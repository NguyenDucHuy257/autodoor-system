import cv2
import time
import unicodedata
import numpy as np

from sqlalchemy import select

from app.camera.rtsp_camera import RTSPCamera

from app.database.database import SessionLocal
from app.database.models import (
    Member,
    FaceEmbedding,
)

from app.recognition.yunet_detector import YuNetDetector
from app.recognition.sface_embedder import SFaceEmbedder
from app.recognition.face_matcher import FaceMatcher

from app.config import (
    CAMRERA_IP,
    CAMERA_USERNAME,
    CAMERA_PASSWORD,
    CAMERA_RTSP_PORT,

)


# ============================================================
# CONFIG
# ============================================================

WINDOW_NAME = "FACE RECOGNITION DEBUG"

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

MATCH_THRESHOLD = 0.4

LAPTOP_CAMERA_INDEX = 0
LAPTOP_MIRROR = True


# ============================================================
# TEXT
# ============================================================

def to_ascii(text: str | None) -> str:

    if text is None:
        return ""

    normalized = unicodedata.normalize(
        "NFD",
        str(text)
    )

    return "".join(
        c
        for c in normalized
        if unicodedata.category(c) != "Mn"
    )


def draw_text(
    image,
    text,
    x,
    y,
    color=(255, 255, 255),
    scale=0.45,
    thickness=1,
):

    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


# ============================================================
# CAMERA SOURCE MANAGER
# ============================================================

class CameraSource:

    def __init__(self):

        self.mode = None

        self.channel = None

        self.laptop_camera = None

        self.rtsp_camera = None


    def stop(self):

        if self.laptop_camera is not None:

            self.laptop_camera.release()

            self.laptop_camera = None


        if self.rtsp_camera is not None:

            self.rtsp_camera.stop()

            self.rtsp_camera = None


    def use_laptop(self):

        self.stop()

        print("Switch -> LAPTOP CAMERA")

        camera = cv2.VideoCapture(
            LAPTOP_CAMERA_INDEX
        )

        if not camera.isOpened():

            print("Cannot open laptop camera")

            return

        self.laptop_camera = camera

        self.mode = "LAPTOP"

        self.channel = None


    def use_rtsp(self, channel: int):

        self.stop()

        print(
            f"Switch -> RTSP CHANNEL {channel}"
        )

        rtsp_url = (
            f"rtsp://{CAMERA_USERNAME}:"
            f"{CAMERA_PASSWORD}"
            f"@{CAMRERA_IP}:"
            f"{CAMERA_RTSP_PORT}"
            f"/cam/realmonitor"
            f"?channel={channel}"
            f"&subtype=0"
        )

        camera = RTSPCamera(
            url=rtsp_url
        )

        camera.start()

        self.rtsp_camera = camera

        self.mode = "RTSP"

        self.channel = channel


    def get_frame(self):

        if self.mode == "LAPTOP":

            if self.laptop_camera is None:
                return None

            ret, frame = (
                self.laptop_camera.read()
            )

            if not ret or frame is None:
                return None

            if LAPTOP_MIRROR:

                frame = cv2.flip(
                    frame,
                    1
                )

            return frame


        if self.mode == "RTSP":

            if self.rtsp_camera is None:
                return None

            return (
                self.rtsp_camera
                .get_latest_frame()
            )


        return None


# ============================================================
# AI
# ============================================================

detector = YuNetDetector(
    model_path=(
        "models/"
        "face_detection_yunet_2023mar.onnx"
    ),
    score_threshold=0.6,
)

embedder = SFaceEmbedder(
    model_path=(
        "models/"
        "face_recognition_sface_2021dec.onnx"
    )
)

matcher = FaceMatcher(
    match_threshold=MATCH_THRESHOLD
)




# ============================================================
# LOAD DATABASE
# ============================================================

session = SessionLocal()

stmt = (
    select(
        Member,
        FaceEmbedding
    )
    .join(
        FaceEmbedding,
        FaceEmbedding.member_id
        == Member.id
    )
    .where(
        Member.active.is_(True)
    )
)

rows = session.execute(
    stmt
).all()


person_database = []

for member, face_embedding in rows:

    embedding = np.frombuffer(
        face_embedding.embedding,
        dtype=np.float32
    )

    person_database.append(
        {
            "member_id": member.id,
            "code": member.code,
            "name": member.full_name,
            "room": member.room_number,
            "pose": face_embedding.pose,
            "embedding": embedding,
        }
    )

session.close()


print(
    f"Loaded {len(person_database)} embeddings"
)


# ============================================================
# BUTTONS
# ============================================================

buttons = []

switch_request = None


def create_buttons():

    result = []

    # LAPTOP
    result.append(
        {
            "label": "LAPTOP",
            "x1": 10,
            "y1": 10,
            "x2": 125,
            "y2": 45,
            "type": "LAPTOP",
            "channel": None,
        }
    )

    # CHANNEL 1 -> 20
    start_x = 145

    button_width = 48
    gap = 5

    for channel in range(1, 21):

        x1 = (
            start_x
            + (channel - 1)
            * (button_width + gap)
        )

        result.append(
            {
                "label": str(channel),
                "x1": x1,
                "y1": 10,
                "x2": x1 + button_width,
                "y2": 45,
                "type": "RTSP",
                "channel": channel,
            }
        )

    return result


buttons = create_buttons()


def mouse_callback(
    event,
    x,
    y,
    flags,
    param,
):

    global switch_request

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    for button in buttons:

        if (
            button["x1"] <= x <= button["x2"]
            and
            button["y1"] <= y <= button["y2"]
        ):

            if button["type"] == "LAPTOP":

                switch_request = {
                    "mode": "LAPTOP"
                }

            else:

                switch_request = {
                    "mode": "RTSP",
                    "channel": button["channel"]
                }

            return


# ============================================================
# DRAW BUTTONS
# ============================================================

def draw_buttons(
    image,
    source: CameraSource,
):

    for button in buttons:

        active = False

        if (
            button["type"] == "LAPTOP"
            and source.mode == "LAPTOP"
        ):
            active = True

        if (
            button["type"] == "RTSP"
            and source.mode == "RTSP"
            and source.channel
            == button["channel"]
        ):
            active = True


        if active:

            fill_color = (
                0,
                180,
                0
            )

        else:

            fill_color = (
                50,
                50,
                50
            )


        cv2.rectangle(
            image,
            (
                button["x1"],
                button["y1"]
            ),
            (
                button["x2"],
                button["y2"]
            ),
            fill_color,
            -1
        )


        cv2.rectangle(
            image,
            (
                button["x1"],
                button["y1"]
            ),
            (
                button["x2"],
                button["y2"]
            ),
            (
                220,
                220,
                220
            ),
            1
        )


        text_size = cv2.getTextSize(
            button["label"],
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            1
        )[0]


        text_x = (
            button["x1"]
            + (
                button["x2"]
                - button["x1"]
                - text_size[0]
            ) // 2
        )

        text_y = (
            button["y1"]
            + (
                button["y2"]
                - button["y1"]
                + text_size[1]
            ) // 2
        )


        draw_text(
            image,
            button["label"],
            text_x,
            text_y,
            (
                255,
                255,
                255
            ),
            0.45,
            1
        )


# ============================================================
# SOURCE
# ============================================================

source = CameraSource()

# Mặc định mở camera laptop
source.use_laptop()


# ============================================================
# WINDOW
# ============================================================

cv2.namedWindow(
    WINDOW_NAME
)

cv2.setMouseCallback(
    WINDOW_NAME,
    mouse_callback
)


# ============================================================
# FPS
# ============================================================

fps = 0.0

fps_counter = 0

fps_started_at = time.monotonic()


# ============================================================
# MAIN
# ============================================================

try:

    while True:

        # ----------------------------------------------------
        # SOURCE SWITCH
        # ----------------------------------------------------

        if switch_request is not None:

            request = switch_request

            switch_request = None


            if request["mode"] == "LAPTOP":

                source.use_laptop()


            elif request["mode"] == "RTSP":

                source.use_rtsp(
                    request["channel"]
                )


        # ----------------------------------------------------
        # FRAME
        # ----------------------------------------------------

        frame = source.get_frame()

        if frame is None:

            canvas = np.zeros(
                (
                    DISPLAY_HEIGHT,
                    DISPLAY_WIDTH,
                    3
                ),
                dtype=np.uint8
            )

            draw_buttons(
                canvas,
                source
            )

            draw_text(
                canvas,
                "Waiting for camera...",
                20,
                80,
                (
                    0,
                    255,
                    255
                ),
                0.7,
                2
            )

            cv2.imshow(
                WINDOW_NAME,
                canvas
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord("q"):
                break

            continue


        # ----------------------------------------------------
        # YUNET
        # ----------------------------------------------------

        detect_started = (
            time.perf_counter()
        )

        faces = detector.detect(
            frame
        )

        yunet_ms = (
            time.perf_counter()
            - detect_started
        ) * 1000


        face_results = []


        # ====================================================
        # MULTI FACE
        # ====================================================

        for face_index, face in enumerate(
            faces,
            start=1
        ):

            sface_started = (
                time.perf_counter()
            )

            embedding = None

            try:

                embedding = embedder.extract(
                    frame=frame,
                    face=face
                )

            except Exception as error:

                print(
                    "SFace error:",
                    error
                )


            sface_ms = (
                time.perf_counter()
                - sface_started
            ) * 1000


            # -----------------------------------------------
            # MATCH
            # -----------------------------------------------

            best_person = None

            best_score = -1.0


            if embedding is not None:

                for person in person_database:

                    score = matcher.similarity(
                        person["embedding"],
                        embedding
                    )

                    if score > best_score:

                        best_score = score

                        best_person = person


            # Best score dưới threshold
            # => UNKNOWN
            if best_score < MATCH_THRESHOLD:

                best_person = None


            # -----------------------------------------------
            # DRAW FACE
            #
            # USER REQUEST:
            # CHỈ KHUNG ĐỎ QUANH MẶT
            # -----------------------------------------------

            cv2.rectangle(
                frame,
                (
                    face.x,
                    face.y
                ),
                (
                    face.x + face.width,
                    face.y + face.height
                ),
                (
                    0,
                    0,
                    255
                ),
                2
            )


            # -----------------------------------------------
            # RESULT
            # -----------------------------------------------

            if best_person is None:

                identity = "UNKNOWN"

                code = "-"

                room = "-"

                best_pose = "-"

            else:

                identity = to_ascii(
                    best_person["name"]
                )

                code = best_person["code"]

                room = best_person["room"]

                best_pose = (
                    best_person["pose"]
                )


            embedding_dim = 0

            embedding_norm = 0.0

            if embedding is not None:

                embedding_dim = (
                    embedding.size
                )

                embedding_norm = float(
                    np.linalg.norm(
                        embedding
                    )
                )


            face_results.append(
                {
                    "index": face_index,

                    "identity": identity,

                    "code": code,

                    "room": room,

                    "best_pose": best_pose,

                    "similarity": best_score,

                    "yunet_confidence":
                        face.confidence,

                    "bbox_width":
                        face.width,

                    "bbox_height":
                        face.height,

                    "embedding_dim":
                        embedding_dim,

                    "embedding_norm":
                        embedding_norm,

                    "sface_ms":
                        sface_ms,
                }
            )


        # ====================================================
        # FPS
        # ====================================================

        fps_counter += 1

        now = time.monotonic()

        elapsed = (
            now
            - fps_started_at
        )

        if elapsed >= 1.0:

            fps = (
                fps_counter
                / elapsed
            )

            fps_counter = 0

            fps_started_at = now


        # ====================================================
        # DISPLAY RESIZE
        # ====================================================

        display_frame = cv2.resize(
            frame,
            (
                DISPLAY_WIDTH,
                DISPLAY_HEIGHT
            )
        )


        # ====================================================
        # CONTROL BUTTONS
        # ====================================================

        draw_buttons(
            display_frame,
            source
        )


        # ====================================================
        # DEBUG PANEL - TOP LEFT
        # ====================================================

        panel_width = 520

        panel_height = min(
            DISPLAY_HEIGHT - 65,
            120
            + len(face_results) * 82
        )


        overlay = (
            display_frame.copy()
        )


        cv2.rectangle(
            overlay,
            (
                10,
                55
            ),
            (
                panel_width,
                55 + panel_height
            ),
            (
                0,
                0,
                0
            ),
            -1
        )


        display_frame = cv2.addWeighted(
            overlay,
            0.65,
            display_frame,
            0.35,
            0
        )


        y = 80


        # ----------------------------------------------------
        # SOURCE INFO
        # ----------------------------------------------------

        if source.mode == "LAPTOP":

            source_text = (
                "SOURCE: LAPTOP"
            )

        else:

            source_text = (
                f"SOURCE: RTSP"
                f" | CHANNEL: "
                f"{source.channel}"
            )


        draw_text(
            display_frame,
            source_text,
            20,
            y,
            (
                0,
                255,
                255
            ),
            0.55,
            2
        )

        y += 22


        draw_text(
            display_frame,
            (
                f"Faces: {len(faces)}"
                f" | FPS: {fps:.1f}"
                f" | YuNet: {yunet_ms:.1f} ms"
                f" | DB embeddings: "
                f"{len(person_database)}"
            ),
            20,
            y,
            (
                255,
                255,
                255
            ),
            0.45
        )

        y += 22


        draw_text(
            display_frame,
            (
                f"Match threshold: "
                f"{MATCH_THRESHOLD:.2f}"
            ),
            20,
            y,
            (
                255,
                255,
                255
            ),
            0.45
        )

        y += 26


        # ----------------------------------------------------
        # EACH FACE INFO
        # ----------------------------------------------------

        for result in face_results:

            score = result[
                "similarity"
            ]

            if score >= 0:

                score_text = (
                    f"{score:.4f}"
                )

            else:

                score_text = "N/A"


            draw_text(
                display_frame,
                (
                    f"#{result['index']} "
                    f"{result['identity']} "
                    f"| SIM: {score_text}"
                ),
                20,
                y,
                (
                    0,
                    255,
                    0
                )
                if result["identity"] != "UNKNOWN"
                else (
                    0,
                    0,
                    255
                ),
                0.5,
                2
            )

            y += 18


            draw_text(
                display_frame,
                (
                    f"YuNet: "
                    f"{result['yunet_confidence']:.3f}"
                    f" | BBox: "
                    f"{result['bbox_width']}x"
                    f"{result['bbox_height']}"
                ),
                30,
                y,
                (
                    220,
                    220,
                    220
                ),
                0.42
            )

            y += 18


            draw_text(
                display_frame,
                (
                    f"SFace: "
                    f"{result['sface_ms']:.1f}ms"
                ),
                30,
                y,
                (
                    220,
                    220,
                    220
                ),
                0.42
            )

            y += 18


            draw_text(
                display_frame,
                (
                    f"Embedding: "
                    f"{result['embedding_dim']}D"
                    f" | norm: "
                    f"{result['embedding_norm']:.3f}"
                    f" | Best pose: "
                    f"{result['best_pose']}"
                    f" | Room: "
                    f"{result['room']}"
                ),
                30,
                y,
                (
                    220,
                    220,
                    220
                ),
                0.42
            )

            y += 24


        # ====================================================
        # SHOW
        # ====================================================

        cv2.imshow(
            WINDOW_NAME,
            display_frame
        )


        key = (
            cv2.waitKey(1)
            & 0xFF
        )


        if key == ord("q"):
            break


finally:

    source.stop()

    cv2.destroyAllWindows()