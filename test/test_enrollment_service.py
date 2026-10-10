import cv2
import time
import tkinter as tk
import numpy as np

from tkinter import messagebox

from app.database.database import SessionLocal
from app.database.enrollment_repository import EnrollmentRepository

from app.camera.rtsp_camera import RTSPCamera

from app.recognition.yunet_detector import YuNetDetector
from app.recognition.sface_embedder import SFaceEmbedder
from app.recognition.face_quality import FaceQualityGate

from app.services.enrollment_service import EnrollmentService

from app.config import (
    CAMRERA_IP,
    CAMERA_USERNAME,
    CAMERA_PASSWORD,
    CAMERA_RTSP_PORT,

    FACE_MIN_CONFIDENCE,
    FACE_MIN_RATIO,
    FACE_MIN_SHARPNESS,
    FACE_MIN_BRIGHTNESS,
    FACE_MAX_BRIGHTNESS,
)


# ============================================================
# CONFIG
# ============================================================

WINDOW_NAME = "FACE ENROLLMENT"

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

LAPTOP_CAMERA_INDEX = 0
LAPTOP_MIRROR = True

# Thời gian nghỉ giữa 2 pose
POSE_COOLDOWN_SECONDS = 1.5


# ============================================================
# MEMBER INPUT FORM
# ============================================================

def get_member_information():

    result = {
        "name": None,
        "room": None,
    }

    root = tk.Tk()

    root.title(
        "Member Enrollment"
    )

    root.geometry(
        "420x230"
    )

    root.resizable(
        False,
        False
    )


    tk.Label(
        root,
        text="Họ và tên",
        font=("Arial", 11)
    ).pack(
        pady=(20, 5)
    )


    name_entry = tk.Entry(
        root,
        font=("Arial", 12),
        width=35
    )

    name_entry.pack()


    tk.Label(
        root,
        text="Phòng",
        font=("Arial", 11)
    ).pack(
        pady=(15, 5)
    )


    room_entry = tk.Entry(
        root,
        font=("Arial", 12),
        width=35
    )

    room_entry.pack()


    def submit():

        name = (
            name_entry
            .get()
            .strip()
        )

        room = (
            room_entry
            .get()
            .strip()
        )

        if not name:

            messagebox.showerror(
                "Error",
                "Họ tên không được để trống."
            )

            return

        result["name"] = name

        result["room"] = (
            room
            if room
            else None
        )

        root.destroy()


    tk.Button(
        root,
        text="TIẾP TỤC",
        font=("Arial", 11),
        width=15,
        command=submit
    ).pack(
        pady=20
    )


    name_entry.focus()

    root.mainloop()

    return result


# ============================================================
# CAMERA SOURCE
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

        print(
            "Switch -> LAPTOP"
        )

        camera = cv2.VideoCapture(
            LAPTOP_CAMERA_INDEX
        )

        if not camera.isOpened():

            print(
                "Cannot open laptop camera"
            )

            self.mode = None

            return False


        self.laptop_camera = camera

        self.mode = "LAPTOP"

        self.channel = None

        return True


    def use_rtsp(
        self,
        channel: int,
    ):

        self.stop()

        print(
            f"Switch -> RTSP CHANNEL {channel}"
        )

        url = (
            f"rtsp://"
            f"{CAMERA_USERNAME}:"
            f"{CAMERA_PASSWORD}"
            f"@{CAMRERA_IP}:"
            f"{CAMERA_RTSP_PORT}"
            f"/cam/realmonitor"
            f"?channel={channel}"
            f"&subtype=0"
        )

        camera = RTSPCamera(
            url=url
        )

        camera.start()

        self.rtsp_camera = camera

        self.mode = "RTSP"

        self.channel = channel

        return True


    def get_frame(self):

        # ----------------------------------------------------
        # LAPTOP
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # RTSP
        # ----------------------------------------------------

        if self.mode == "RTSP":

            if self.rtsp_camera is None:
                return None

            return (
                self.rtsp_camera
                .get_latest_frame()
            )


        return None


# ============================================================
# DRAW TEXT
# ============================================================

def draw_text(
    image,
    text,
    x,
    y,
    color=(255, 255, 255),
    scale=0.5,
    thickness=1,
):

    cv2.putText(
        image,
        str(text),
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


# ============================================================
# BUTTONS
# ============================================================

def create_buttons():

    result = []

    # --------------------------------------------------------
    # LAPTOP
    # --------------------------------------------------------

    result.append(
        {
            "label": "LAPTOP",
            "x1": 5,
            "y1": 5,
            "x2": 125,
            "y2": 40,
            "type": "LAPTOP",
            "channel": None,
        }
    )


    # --------------------------------------------------------
    # CHANNEL 1 -> 20
    # --------------------------------------------------------

    start_x = 140

    button_width = 45

    gap = 4


    for channel in range(
        1,
        21
    ):

        x1 = (
            start_x
            +
            (channel - 1)
            *
            (button_width + gap)
        )

        result.append(
            {
                "label": str(channel),

                "x1": x1,

                "y1": 5,

                "x2": (
                    x1
                    + button_width
                ),

                "y2": 40,

                "type": "RTSP",

                "channel": channel,
            }
        )


    # --------------------------------------------------------
    # START
    # --------------------------------------------------------

    result.append(
        {
            "label": "START",
            "x1": 5,
            "y1": 50,
            "x2": 125,
            "y2": 90,
            "type": "START",
            "channel": None,
        }
    )


    return result


buttons = create_buttons()


# ============================================================
# UI STATE
# ============================================================

switch_request = None

start_requested = False

enrollment_started = False


# ============================================================
# MOUSE
# ============================================================

def mouse_callback(
    event,
    x,
    y,
    flags,
    param,
):

    global switch_request
    global start_requested


    if event != cv2.EVENT_LBUTTONDOWN:
        return


    for button in buttons:

        clicked = (
            button["x1"]
            <= x
            <= button["x2"]
            and
            button["y1"]
            <= y
            <= button["y2"]
        )

        if not clicked:
            continue


        # ====================================================
        # Đang enrollment:
        # Không cho đổi camera
        # ====================================================

        if (
            enrollment_started
            and
            button["type"]
            in (
                "LAPTOP",
                "RTSP"
            )
        ):
            return


        # ====================================================
        # LAPTOP
        # ====================================================

        if (
            button["type"]
            == "LAPTOP"
        ):

            switch_request = {
                "mode": "LAPTOP"
            }


        # ====================================================
        # RTSP
        # ====================================================

        elif (
            button["type"]
            == "RTSP"
        ):

            switch_request = {
                "mode": "RTSP",

                "channel":
                    button["channel"]
            }


        # ====================================================
        # START
        # ====================================================

        elif (
            button["type"]
            == "START"
        ):

            if not enrollment_started:

                start_requested = True


        return


# ============================================================
# DRAW BUTTONS
# ============================================================

def draw_buttons(
    image,
    source,
):

    for button in buttons:

        active = False


        # ----------------------------------------------------
        # LAPTOP ACTIVE
        # ----------------------------------------------------

        if (
            button["type"]
            == "LAPTOP"
            and
            source.mode
            == "LAPTOP"
        ):

            active = True


        # ----------------------------------------------------
        # RTSP ACTIVE
        # ----------------------------------------------------

        elif (
            button["type"]
            == "RTSP"
            and
            source.mode
            == "RTSP"
            and
            source.channel
            == button["channel"]
        ):

            active = True


        # ----------------------------------------------------
        # START ACTIVE
        # ----------------------------------------------------

        elif (
            button["type"]
            == "START"
            and
            enrollment_started
        ):

            active = True


        # ----------------------------------------------------
        # COLOR
        # ----------------------------------------------------

        color = (
            (0, 180, 0)
            if active
            else (50, 50, 50)
        )


        # Camera buttons bị khóa
        # khi enrollment đã bắt đầu
        if (
            enrollment_started
            and
            button["type"]
            in (
                "LAPTOP",
                "RTSP"
            )
        ):

            color = (
                25,
                25,
                25
            )


        # ----------------------------------------------------
        # BUTTON BACKGROUND
        # ----------------------------------------------------

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
            color,
            -1
        )


        # ----------------------------------------------------
        # BORDER
        # ----------------------------------------------------

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
                200,
                200,
                200
            ),
            1
        )


        # ----------------------------------------------------
        # CENTER TEXT
        # ----------------------------------------------------

        text_size = cv2.getTextSize(
            button["label"],
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            1
        )[0]


        text_x = (
            button["x1"]
            +
            (
                button["x2"]
                - button["x1"]
                - text_size[0]
            ) // 2
        )


        text_y = (
            button["y1"]
            +
            (
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
# POSE GUIDE
# ============================================================

def draw_pose_guide(
    frame,
    next_pose,
):

    height, width = (
        frame.shape[:2]
    )


    center_x = int(
        width * 0.75
    )

    center_y = int(
        height * 0.70
    )


    color = (
        0,
        255,
        0
    )


    # ========================================================
    # FRONT
    # ========================================================

    if next_pose == "FRONT":

        cv2.circle(
            frame,
            (
                center_x,
                center_y
            ),
            14,
            color,
            -1
        )


    # ========================================================
    # LEFT
    # ========================================================

    elif next_pose == "LEFT":

        cv2.arrowedLine(
            frame,
            (
                center_x + 100,
                center_y
            ),
            (
                center_x - 100,
                center_y
            ),
            color,
            7
        )


    # ========================================================
    # RIGHT
    # ========================================================

    elif next_pose == "RIGHT":

        cv2.arrowedLine(
            frame,
            (
                center_x - 100,
                center_y
            ),
            (
                center_x + 100,
                center_y
            ),
            color,
            7
        )


    # ========================================================
    # DOWN
    # ========================================================

    elif next_pose == "DOWN":

        cv2.arrowedLine(
            frame,
            (
                center_x,
                center_y - 100
            ),
            (
                center_x,
                center_y + 100
            ),
            color,
            7
        )


    # ========================================================
    # FAR FRONT
    # ========================================================

    elif next_pose == "FAR_FRONT":

        cv2.circle(
            frame,
            (
                center_x,
                center_y
            ),
            45,
            color,
            3
        )


        # dấu +
        cv2.line(
            frame,
            (
                center_x - 20,
                center_y
            ),
            (
                center_x + 20,
                center_y
            ),
            color,
            3
        )


        cv2.line(
            frame,
            (
                center_x,
                center_y - 20
            ),
            (
                center_x,
                center_y + 20
            ),
            color,
            3
        )


# ============================================================
# MEMBER INFO
# ============================================================

member_info = (
    get_member_information()
)


if not member_info["name"]:

    raise SystemExit


# ============================================================
# AI COMPONENTS
# ============================================================

detector = YuNetDetector(
    model_path=(
        "models/"
        "face_detection_yunet_2023mar.onnx"
    )
)


embedder = SFaceEmbedder(
    model_path=(
        "models/"
        "face_recognition_sface_2021dec.onnx"
    )
)


quality_gate = FaceQualityGate(
    min_confidence=FACE_MIN_CONFIDENCE,

    min_face_ratio=FACE_MIN_RATIO,

    min_sharpness=FACE_MIN_SHARPNESS,

    min_brightness=FACE_MIN_BRIGHTNESS,

    max_brightness=FACE_MAX_BRIGHTNESS,
)


# ============================================================
# DATABASE
# ============================================================

session = SessionLocal()


repository = EnrollmentRepository(
    session=session
)


# ============================================================
# SERVICE
# ============================================================

enrollment = EnrollmentService(
    repository=repository,

    detector=detector,

    embedder=embedder,

    quality_gate=quality_gate
)


# ============================================================
# CAMERA
# ============================================================

source = CameraSource()

# Mặc định laptop
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
# ENROLLMENT STATE
# ============================================================

next_pose = None

last_result = None

success_data = None


# Thời điểm được phép thu pose tiếp theo
next_capture_allowed_at = 0.0


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        # ====================================================
        # CAMERA SWITCH REQUEST
        # ====================================================

        if switch_request is not None:

            request = (
                switch_request
            )

            switch_request = None


            if (
                request["mode"]
                == "LAPTOP"
            ):

                source.use_laptop()


            elif (
                request["mode"]
                == "RTSP"
            ):

                source.use_rtsp(
                    request["channel"]
                )


        # ====================================================
        # START ENROLLMENT
        # ====================================================

        if (
            start_requested
            and
            not enrollment_started
        ):

            start_requested = False


            if source.mode is None:

                print(
                    "Please select camera."
                )


            else:

                enrollment.start_enrollment(
                    full_name=(
                        member_info["name"]
                    ),

                    room_number=(
                        member_info["room"]
                    )
                )


                enrollment_started = True


                next_pose = (
                    enrollment
                    .get_next_required_pose()
                )


                # Cho phép lấy FRONT ngay
                next_capture_allowed_at = 0.0


                print()
                print(
                    "=============================="
                )
                print(
                    "ENROLLMENT STARTED"
                )
                print(
                    "=============================="
                )
                print(
                    "Name:",
                    member_info["name"]
                )
                print(
                    "Room:",
                    member_info["room"]
                )
                print(
                    "Source:",
                    source.mode
                )

                if (
                    source.mode
                    == "RTSP"
                ):

                    print(
                        "Channel:",
                        source.channel
                    )


        # ====================================================
        # GET FRAME
        # ====================================================

        frame = (
            source.get_frame()
        )


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
                130,
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


        # ====================================================
        # PREVIEW BEFORE START
        #
        # Chỉ detect riêng trước khi enrollment.
        # Khi enrollment đã chạy thì không detect lần 2.
        # ====================================================

        preview_face = None


        if not enrollment_started:

            preview_faces = (
                detector.detect(
                    frame
                )
            )


            if len(preview_faces) == 1:

                preview_face = (
                    preview_faces[0]
                )


        # ====================================================
        # ENROLLMENT PROCESS
        # ====================================================

        if enrollment_started:

            now = (
                time.monotonic()
            )


            # ------------------------------------------------
            # CHỈ PROCESS KHI COOLDOWN HẾT
            # ------------------------------------------------

            if (
                now
                >= next_capture_allowed_at
            ):

                before_count = len(
                    enrollment
                    .collected_frames
                )


                result = (
                    enrollment
                    .process_frame(
                        frame
                    )
                )


                after_count = len(
                    enrollment
                    .collected_frames
                )


                if result is not None:

                    last_result = (
                        result
                    )


                    next_pose = (
                        result[
                            "next_pose"
                        ]
                    )


                    # =========================================
                    # VỪA THU ĐƯỢC POSE MỚI
                    # =========================================

                    if (
                        after_count
                        > before_count
                    ):

                        captured_pose = (
                            result[
                                "collected"
                            ][-1]
                        )


                        print(
                            f"Captured: "
                            f"{captured_pose}"
                        )


                        # Nếu chưa complete,
                        # bắt đầu cooldown
                        if not result[
                            "complete"
                        ]:

                            next_capture_allowed_at = (
                                time.monotonic()
                                +
                                POSE_COOLDOWN_SECONDS
                            )


                            print(
                                f"Prepare "
                                f"{next_pose}: "
                                f"{POSE_COOLDOWN_SECONDS}s"
                            )


                    # =========================================
                    # COMPLETE
                    # =========================================

                    if result[
                        "complete"
                    ]:

                        member = (
                            enrollment
                            .finish_enrollment()
                        )


                        # =====================================
                        # COPY ORM DATA TRƯỚC session.close()
                        # =====================================

                        success_data = {
                            "id":
                                member.id,

                            "code":
                                member.code,

                            "name":
                                member.full_name,

                            "room":
                                member.room_number,
                        }


                        print(
                            "All poses captured."
                        )


                        break


        # ====================================================
        # DISPLAY FRAME
        # ====================================================

        display_frame = (
            cv2.resize(
                frame,
                (
                    DISPLAY_WIDTH,
                    DISPLAY_HEIGHT
                )
            )
        )


        # ====================================================
        # PREVIEW FACE BOX
        # ====================================================

        if (
            not enrollment_started
            and
            preview_face is not None
        ):

            original_height, original_width = (
                frame.shape[:2]
            )


            scale_x = (
                DISPLAY_WIDTH
                / original_width
            )

            scale_y = (
                DISPLAY_HEIGHT
                / original_height
            )


            x1 = int(
                preview_face.x
                * scale_x
            )

            y1 = int(
                preview_face.y
                * scale_y
            )

            x2 = int(
                (
                    preview_face.x
                    +
                    preview_face.width
                )
                * scale_x
            )

            y2 = int(
                (
                    preview_face.y
                    +
                    preview_face.height
                )
                * scale_y
            )


            cv2.rectangle(
                display_frame,
                (
                    x1,
                    y1
                ),
                (
                    x2,
                    y2
                ),
                (
                    0,
                    255,
                    0
                ),
                2
            )


            draw_text(
                display_frame,
                (
                    f"Face: "
                    f"{preview_face.confidence * 100:.1f}%"
                ),
                x1,
                max(
                    20,
                    y1 - 10
                ),
                (
                    0,
                    255,
                    0
                ),
                0.55,
                2
            )


        # ====================================================
        # BUTTONS
        # ====================================================

        draw_buttons(
            display_frame,
            source
        )


        # ====================================================
        # INFORMATION PANEL
        # ====================================================

        overlay = (
            display_frame.copy()
        )


        cv2.rectangle(
            overlay,
            (
                10,
                105
            ),
            (
                500,
                350
            ),
            (
                0,
                0,
                0
            ),
            -1
        )


        display_frame = (
            cv2.addWeighted(
                overlay,
                0.65,
                display_frame,
                0.35,
                0
            )
        )


        y = 130


        # ====================================================
        # NAME
        # ====================================================

        draw_text(
            display_frame,
            (
                f"NAME: "
                f"{member_info['name']}"
            ),
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


        y += 25


        # ====================================================
        # ROOM
        # ====================================================

        draw_text(
            display_frame,
            (
                f"ROOM: "
                f"{member_info['room']}"
            ),
            20,
            y,
            (
                255,
                255,
                255
            ),
            0.5
        )


        y += 25


        # ====================================================
        # SOURCE
        # ====================================================

        if (
            source.mode
            == "LAPTOP"
        ):

            source_text = (
                "SOURCE: LAPTOP"
            )


        elif (
            source.mode
            == "RTSP"
        ):

            source_text = (
                f"SOURCE: RTSP"
                f" | CHANNEL: "
                f"{source.channel}"
            )


        else:

            source_text = (
                "SOURCE: NONE"
            )


        draw_text(
            display_frame,
            source_text,
            20,
            y,
            (
                255,
                255,
                255
            ),
            0.5
        )


        y += 30


        # ====================================================
        # BEFORE START
        # ====================================================

        if not enrollment_started:

            draw_text(
                display_frame,
                "Select camera then press START",
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


        # ====================================================
        # ENROLLMENT
        # ====================================================

        else:

            cooldown_remaining = max(
                0.0,
                next_capture_allowed_at
                -
                time.monotonic()
            )


            # ------------------------------------------------
            # NEXT POSE
            # ------------------------------------------------

            draw_text(
                display_frame,
                (
                    f"NEXT POSE: "
                    f"{next_pose}"
                ),
                20,
                y,
                (
                    0,
                    255,
                    0
                ),
                0.6,
                2
            )


            y += 27


            # ------------------------------------------------
            # COLLECTED
            # ------------------------------------------------

            collected = list(
                enrollment
                .collected_frames
                .keys()
            )


            draw_text(
                display_frame,
                (
                    "COLLECTED: "
                    +
                    (
                        ", ".join(
                            collected
                        )
                        if collected
                        else "-"
                    )
                ),
                20,
                y,
                (
                    255,
                    255,
                    255
                ),
                0.44
            )


            y += 25


            # ------------------------------------------------
            # PROGRESS
            # ------------------------------------------------

            draw_text(
                display_frame,
                (
                    f"PROGRESS: "
                    f"{len(collected)}"
                    f"/"
                    f"{len(enrollment.REQUIRED_SLOTS)}"
                ),
                20,
                y,
                (
                    255,
                    255,
                    255
                ),
                0.5
            )


            y += 28


            # ------------------------------------------------
            # COOLDOWN
            # ------------------------------------------------

            if (
                cooldown_remaining
                > 0
            ):

                draw_text(
                    display_frame,
                    (
                        f"GET READY: "
                        f"{cooldown_remaining:.1f}s"
                    ),
                    20,
                    y,
                    (
                        0,
                        255,
                        255
                    ),
                    0.65,
                    2
                )

            else:

                draw_text(
                    display_frame,
                    (
                        f"CAPTURING "
                        f"{next_pose}"
                    ),
                    20,
                    y,
                    (
                        0,
                        255,
                        0
                    ),
                    0.55,
                    2
                )


        # ====================================================
        # POSE GUIDE
        #
        # Hiện hướng dẫn pose tiếp theo ngay cả lúc cooldown
        # để người dùng biết phải quay theo hướng nào.
        # ====================================================

        if (
            enrollment_started
            and
            next_pose is not None
        ):

            draw_pose_guide(
                display_frame,
                next_pose
            )


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


# ============================================================
# CLEANUP
# ============================================================

finally:

    source.stop()

    session.close()

    cv2.destroyAllWindows()


# ============================================================
# FINAL RESULT
# ============================================================

if success_data is not None:

    print()

    print(
        "=============================="
    )

    print(
        "ENROLLMENT SUCCESS"
    )

    print(
        "=============================="
    )

    print(
        "ID:",
        success_data["id"]
    )

    print(
        "Code:",
        success_data["code"]
    )

    print(
        "Name:",
        success_data["name"]
    )

    print(
        "Room:",
        success_data["room"]
    )


else:

    print()

    print(
        "Enrollment cancelled."
    )