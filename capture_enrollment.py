import os
import re
import time

import cv2
import numpy as np
import tkinter as tk

from tkinter import messagebox

from PIL import (
    Image,
    ImageDraw,
    ImageFont,
)

from urllib.parse import quote


from app.camera.rtsp_camera import RTSPCamera

from app.recognition.yunet_detector import YuNetDetector
from app.recognition.sface_embedder import SFaceEmbedder

from app.database.enrollment_store import save_enrollment

from app.config import (
    CAMRERA_IP,
    CAMERA_USERNAME,
    CAMERA_PASSWORD,
    CAMERA_RTSP_PORT,
)


# =========================================================
# CONFIG
# =========================================================

WINDOW_NAME = "DANG KY KHUON MAT - CC HANHPHUC"

FRAME_WIDTH = 960
FRAME_HEIGHT = 540

ENROLLMENT_ROOT = os.path.join(
    "data",
    "enrollment",
)

os.makedirs(
    ENROLLMENT_ROOT,
    exist_ok=True,
)


# =========================================================
# FONT TIẾNG VIỆT
# Dùng font có sẵn của Windows
# =========================================================

FONT_PATH = r"C:\Windows\Fonts\arial.ttf"

FONT_BOLD_PATH = r"C:\Windows\Fonts\arialbd.ttf"


def load_font(
    size: int,
    bold: bool = False,
):
    path = (
        FONT_BOLD_PATH
        if bold
        else FONT_PATH
    )

    return ImageFont.truetype(
        path,
        size=size,
    )


# =========================================================
# TỰ SINH MEMBER CODE
# TEST: dựa trên folder hiện tại
# =========================================================

def generate_member_code():

    max_id = 0

    for folder_name in os.listdir(
        ENROLLMENT_ROOT
    ):

        folder_path = os.path.join(
            ENROLLMENT_ROOT,
            folder_name,
        )

        if not os.path.isdir(
            folder_path
        ):
            continue

        match = re.fullmatch(
            r"M(\d+)",
            folder_name,
        )

        if match is None:
            continue

        number = int(
            match.group(1)
        )

        if number > max_id:
            max_id = number

    return f"M{max_id + 1:04d}"


# =========================================================
# FORM NHẬP THÔNG TIN
# =========================================================

def get_member_info(
    member_code: str,
):

    result = {}

    root = tk.Tk()

    root.title(
        "Đăng ký thành viên"
    )

    root.geometry(
        "460x310"
    )

    root.resizable(
        False,
        False,
    )


    tk.Label(
        root,
        text="ĐĂNG KÝ THÀNH VIÊN",
        font=(
            "Arial",
            18,
            "bold",
        ),
    ).pack(
        pady=20
    )


    form = tk.Frame(
        root
    )

    form.pack(
        padx=20,
        pady=5,
    )


    # =====================================================
    # MEMBER CODE
    # =====================================================

    tk.Label(
        form,
        text="Mã thành viên:",
        font=(
            "Arial",
            12,
        ),
    ).grid(
        row=0,
        column=0,
        sticky="w",
        padx=10,
        pady=12,
    )


    tk.Label(
        form,
        text=member_code,
        font=(
            "Arial",
            12,
            "bold",
        ),
    ).grid(
        row=0,
        column=1,
        sticky="w",
        pady=12,
    )


    # =====================================================
    # FULL NAME
    # =====================================================

    tk.Label(
        form,
        text="Họ và tên:",
        font=(
            "Arial",
            12,
        ),
    ).grid(
        row=1,
        column=0,
        sticky="w",
        padx=10,
        pady=12,
    )


    name_entry = tk.Entry(
        form,
        font=(
            "Arial",
            12,
        ),
        width=25,
    )

    name_entry.grid(
        row=1,
        column=1,
        pady=12,
    )


    # =====================================================
    # ROOM
    # =====================================================

    tk.Label(
        form,
        text="Phòng:",
        font=(
            "Arial",
            12,
        ),
    ).grid(
        row=2,
        column=0,
        sticky="w",
        padx=10,
        pady=12,
    )


    room_entry = tk.Entry(
        form,
        font=(
            "Arial",
            12,
        ),
        width=25,
    )

    room_entry.grid(
        row=2,
        column=1,
        pady=12,
    )


    # =====================================================
    # SUBMIT
    # =====================================================

    def submit():

        full_name = (
            name_entry
            .get()
            .strip()
        )

        room_number = (
            room_entry
            .get()
            .strip()
        )

        if not full_name:

            messagebox.showerror(
                "Lỗi",
                "Vui lòng nhập họ và tên.",
            )

            return

        if not room_number:

            messagebox.showerror(
                "Lỗi",
                "Vui lòng nhập phòng.",
            )

            return


        result["code"] = (
            member_code
        )

        result["full_name"] = (
            full_name
        )

        result["room_number"] = (
            room_number
        )


        root.destroy()


    tk.Button(
        root,
        text="BẮT ĐẦU CHỤP",
        font=(
            "Arial",
            13,
            "bold",
        ),
        width=22,
        height=2,
        command=submit,
    ).pack(
        pady=20
    )


    root.bind(
        "<Return>",
        lambda event: submit(),
    )

    name_entry.focus()

    root.mainloop()

    return result


# =========================================================
# PIL TEXT HELPER
# =========================================================

def draw_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    position: tuple[int, int],
    font,
    color=(255, 255, 255),
):

    draw.text(
        position,
        text,
        font=font,
        fill=color,
    )


def draw_center_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    center_x: int,
    y: int,
    font,
    color=(255, 255, 255),
):

    bbox = draw.textbbox(
        (0, 0),
        text,
        font=font,
    )

    width = (
        bbox[2]
        - bbox[0]
    )

    x = int(
        center_x
        - width / 2
    )

    draw.text(
        (x, y),
        text,
        font=font,
        fill=color,
    )


# =========================================================
# MEMBER
# =========================================================

member_code = (
    generate_member_code()
)


member_info = get_member_info(
    member_code
)


if not member_info:

    print(
        "Đã hủy đăng ký."
    )

    raise SystemExit


# =========================================================
# MEMBER FOLDER
# =========================================================

SAVE_DIR = os.path.join(
    ENROLLMENT_ROOT,
    member_info["code"],
)


os.makedirs(
    SAVE_DIR,
    exist_ok=False,
)


print(
    f"Đã tạo thư mục: "
    f"{SAVE_DIR}"
)


# =========================================================
# RTSP
# =========================================================

safe_password = quote(
    CAMERA_PASSWORD,
    safe="",
)


rtsp_url = (
    f"rtsp://"
    f"{CAMERA_USERNAME}:"
    f"{safe_password}@"
    f"{CAMRERA_IP}:"
    f"{CAMERA_RTSP_PORT}"
    f"/cam/realmonitor"
    f"?channel=15subtype=0"
)


# =========================================================
# MODELS
# =========================================================

detector = YuNetDetector(
    model_path=(
        "models/"
        "face_detection_yunet_2023mar.onnx"
    ),
    score_threshold=0.8
)


embedder = SFaceEmbedder(
    model_path=(
        "models/"
        "face_recognition_sface_2021dec.onnx"
    )
)


# =========================================================
# CAMERA
# =========================================================

camera = RTSPCamera(
    url=rtsp_url
)


# =========================================================
# ENROLLMENT STEPS
# =========================================================

steps = [

    {
        "pose": "FRONT",
        "file": "front.jpg",
        "instruction": (
            "Nhìn thẳng vào camera"
        ),
        "button": (
            "CHỤP MẶT TRƯỚC"
        ),
    },

    {
        "pose": "RIGHT",
        "file": "right.jpg",
        "instruction": (
            "Quay mặt nhẹ sang phải"
        ),
        "button": (
            "CHỤP MẶT BÊN PHẢI"
        ),
    },

    {
        "pose": "LEFT",
        "file": "left.jpg",
        "instruction": (
            "Quay mặt nhẹ sang trái"
        ),
        "button": (
            "CHỤP MẶT BÊN TRÁI"
        ),
    },

    {
        "pose": "DOWN",
        "file": "down.jpg",
        "instruction": (
            "Hơi cúi mặt xuống"
        ),
        "button": (
            "CHỤP MẶT HƠI CÚI"
        ),
    },

]


# =========================================================
# UI STATE
# =========================================================

current_step = 0

latest_clean_frame = None

face_valid = False


# Hiệu ứng nút
button_pressed_until = 0.0

pending_capture = False

pending_frame = None

pending_step = None


# Feedback
feedback_message = ""

feedback_until = 0.0


# DB
database_saved = False

database_error = None

database_member_id = None

database_processing = False


# =========================================================
# BUTTON GEOMETRY
# =========================================================

BUTTON_X1 = 170
BUTTON_Y1 = 450

BUTTON_X2 = 790
BUTTON_Y2 = 525


# =========================================================
# MOUSE CALLBACK
# =========================================================

def mouse_callback(
    event,
    x,
    y,
    flags,
    param,
):

    global button_pressed_until

    global pending_capture
    global pending_frame
    global pending_step

    global feedback_message
    global feedback_until


    if event != cv2.EVENT_LBUTTONDOWN:
        return


    if current_step >= len(steps):
        return


    if pending_capture:
        return


    inside_button = (
        BUTTON_X1 <= x <= BUTTON_X2
        and
        BUTTON_Y1 <= y <= BUTTON_Y2
    )


    if not inside_button:
        return


    if not face_valid:

        feedback_message = (
            "Cần phát hiện đúng 1 khuôn mặt"
        )

        feedback_until = (
            time.time()
            + 1.5
        )

        return


    if latest_clean_frame is None:
        return


    # =====================================================
    # SNAPSHOT NGAY LÚC CLICK
    # =====================================================

    pending_frame = (
        latest_clean_frame.copy()
    )

    pending_step = (
        current_step
    )

    pending_capture = True


    # =====================================================
    # BUTTON PRESS EFFECT
    # =====================================================

    button_pressed_until = (
        time.time()
        + 0.18
    )


# =========================================================
# LƯU FRAME SAU KHI HIỆU ỨNG NÚT KẾT THÚC
# =========================================================

def process_pending_capture():

    global current_step

    global pending_capture
    global pending_frame
    global pending_step

    global feedback_message
    global feedback_until


    if not pending_capture:
        return


    if time.time() < button_pressed_until:
        return


    step = steps[
        pending_step
    ]


    image_path = os.path.join(
        SAVE_DIR,
        step["file"],
    )


    success = cv2.imwrite(
        image_path,
        pending_frame,
    )


    if not success:

        feedback_message = (
            "Không thể lưu ảnh"
        )

        feedback_until = (
            time.time()
            + 2
        )


        pending_capture = False

        pending_frame = None

        pending_step = None

        return


    print(
        f"Đã chụp {step['pose']}: "
        f"{image_path}"
    )


    feedback_message = (
        f"Đã chụp: "
        f"{step['instruction']}"
    )

    feedback_until = (
        time.time()
        + 1
    )


    current_step += 1


    pending_capture = False

    pending_frame = None

    pending_step = None


# =========================================================
# TẠO EMBEDDING TỪ 4 ẢNH
# =========================================================

def create_embeddings_from_folder():

    embeddings = {}


    for step in steps:

        pose = step["pose"]

        image_path = os.path.join(
            SAVE_DIR,
            step["file"],
        )


        frame = cv2.imread(
            image_path
        )


        if frame is None:

            raise ValueError(
                f"Không đọc được ảnh "
                f"{image_path}"
            )


        faces = detector.detect(
            frame
        )


        if len(faces) != 1:

            raise ValueError(
                f"Ảnh {pose} phải có "
                f"đúng 1 khuôn mặt"
            )


        embedding = (
            embedder.extract(
                frame=frame,
                face=faces[0],
            )
        )


        if embedding is None:

            raise ValueError(
                f"Không tạo được "
                f"embedding {pose}"
            )


        embedding = (
            embedding
            .reshape(-1)
            .astype(np.float32)
        )


        embeddings[
            pose
        ] = embedding


    return embeddings


# =========================================================
# LƯU DB
# =========================================================

def save_to_database():

    global database_saved
    global database_error
    global database_member_id
    global database_processing


    if database_saved:
        return


    if database_processing:
        return


    database_processing = True


    try:

        print()
        print(
            "Đang tạo embeddings..."
        )


        embeddings = (
            create_embeddings_from_folder()
        )


        print(
            "Đang lưu PostgreSQL..."
        )


        database_member_id = (
            save_enrollment(
                member_info=member_info,
                embeddings=embeddings,
            )
        )


        database_saved = True

        database_error = None


        print(
            "Đã lưu thành công."
        )

        print(
            f"Member ID: "
            f"{database_member_id}"
        )


    except Exception as error:

        database_error = str(
            error
        )

        print(
            f"Lỗi lưu database: "
            f"{error}"
        )


    finally:

        database_processing = False


# =========================================================
# FONT CACHE
# =========================================================

font_small = load_font(
    20
)

font_normal = load_font(
    25
)

font_medium = load_font(
    30,
    bold=True,
)

font_button = load_font(
    27,
    bold=True,
)

font_success = load_font(
    30,
    bold=True,
)


# =========================================================
# START
# =========================================================

camera.start()


cv2.namedWindow(
    WINDOW_NAME
)


cv2.setMouseCallback(
    WINDOW_NAME,
    mouse_callback,
)


# =========================================================
# LOOP
# =========================================================

try:

    while True:

        frame = (
            camera
            .get_latest_frame()
        )


        if frame is None:

            time.sleep(
                0.01
            )

            continue


        frame = cv2.resize(
            frame,
            (
                FRAME_WIDTH,
                FRAME_HEIGHT,
            ),
        )


        # =================================================
        # FRAME SẠCH
        # =================================================

        latest_clean_frame = (
            frame.copy()
        )


        # =================================================
        # DETECT
        # =================================================

        faces = detector.detect(
            frame
        )


        face_valid = (
            len(faces) == 1
        )


        display_frame = (
            frame.copy()
        )


        # =================================================
        # FACE BOX + LANDMARKS
        # =================================================

        if face_valid:

            face = faces[0]


            cv2.rectangle(
                display_frame,

                (
                    face.x,
                    face.y,
                ),

                (
                    face.x
                    + face.width,

                    face.y
                    + face.height,
                ),

                (
                    0,
                    220,
                    0,
                ),

                2,
            )


            for point in (
                face.landmarks
            ):

                x, y = (
                    point
                    .astype(int)
                )


                cv2.circle(
                    display_frame,

                    (
                        x,
                        y,
                    ),

                    3,

                    (
                        0,
                        0,
                        255,
                    ),

                    -1,
                )


        # =================================================
        # XỬ LÝ CLICK PENDING
        # =================================================

        process_pending_capture()


        # =================================================
        # ĐỦ 4 ẢNH -> DATABASE
        # =================================================

        if (
            current_step
            >= len(steps)
            and
            not database_saved
            and
            database_error is None
            and
            not database_processing
        ):

            save_to_database()


        # =================================================
        # BUTTON BACKGROUND
        # cv2 vẽ hình, Pillow vẽ chữ Unicode
        # =================================================

        is_pressed = (
            pending_capture
            and
            time.time()
            < button_pressed_until
        )


        if current_step < len(steps):

            # Disabled
            if not face_valid:

                button_color = (
                    90,
                    90,
                    90,
                )

            # Press effect
            elif is_pressed:

                button_color = (
                    0,
                    105,
                    0,
                )

            else:

                button_color = (
                    0,
                    175,
                    0,
                )


            # Nút bị ấn xuống 4px
            button_offset = (
                4
                if is_pressed
                else 0
            )


            cv2.rectangle(
                display_frame,

                (
                    BUTTON_X1,
                    BUTTON_Y1
                    + button_offset,
                ),

                (
                    BUTTON_X2,
                    BUTTON_Y2
                    + button_offset,
                ),

                button_color,

                -1,
            )


            # Border
            cv2.rectangle(
                display_frame,

                (
                    BUTTON_X1,
                    BUTTON_Y1
                    + button_offset,
                ),

                (
                    BUTTON_X2,
                    BUTTON_Y2
                    + button_offset,
                ),

                (
                    235,
                    235,
                    235,
                ),

                2,
            )


        # =================================================
        # CHUYỂN SANG PIL ĐỂ VẼ TIẾNG VIỆT
        # =================================================

        rgb_frame = cv2.cvtColor(
            display_frame,
            cv2.COLOR_BGR2RGB,
        )


        pil_image = Image.fromarray(
            rgb_frame
        )


        draw = ImageDraw.Draw(
            pil_image
        )


        # =================================================
        # MEMBER INFO
        # =================================================

        member_text = (
            f"{member_info['code']}  |  "
            f"{member_info['full_name']}  |  "
            f"Phòng {member_info['room_number']}"
        )


        draw_text(
            draw,
            member_text,
            (
                20,
                15,
            ),
            font_small,
            color=(
                255,
                255,
                255,
            ),
        )


        # =================================================
        # CHƯA HOÀN TẤT
        # =================================================

        if current_step < len(steps):

            step = steps[
                current_step
            ]


            draw_text(
                draw,
                (
                    f"Bước "
                    f"{current_step + 1}"
                    f"/"
                    f"{len(steps)}"
                ),
                (
                    20,
                    50,
                ),
                font_small,
                color=(
                    255,
                    255,
                    255,
                ),
            )


            draw_center_text(
                draw,
                step[
                    "instruction"
                ],
                FRAME_WIDTH // 2,
                53,
                font_medium,
                color=(
                    255,
                    230,
                    0,
                ),
            )


            button_y = (
                BUTTON_Y1
                +
                (
                    4
                    if is_pressed
                    else 0
                )
            )


            # chữ cũng di chuyển xuống
            draw_center_text(
                draw,
                step[
                    "button"
                ],
                FRAME_WIDTH // 2,
                button_y + 21,
                font_button,
                color=(
                    255,
                    255,
                    255,
                ),
            )


            # Face state
            if not face_valid:

                if len(faces) == 0:

                    state_text = (
                        "Không tìm thấy khuôn mặt"
                    )

                else:

                    state_text = (
                        "Chỉ được có 1 khuôn mặt"
                    )


                draw_center_text(
                    draw,
                    state_text,
                    FRAME_WIDTH // 2,
                    405,
                    font_normal,
                    color=(
                        255,
                        80,
                        80,
                    ),
                )


        # =================================================
        # ĐÃ CHỤP ĐỦ
        # =================================================

        else:

            if database_saved:

                draw_center_text(
                    draw,
                    "ĐĂNG KÝ THÀNH CÔNG",
                    FRAME_WIDTH // 2,
                    190,
                    font_success,
                    color=(
                        80,
                        255,
                        100,
                    ),
                )


                draw_center_text(
                    draw,
                    (
                        "Đã lưu thông tin và "
                        "4 embedding vào cơ sở dữ liệu"
                    ),
                    FRAME_WIDTH // 2,
                    235,
                    font_normal,
                    color=(
                        255,
                        255,
                        255,
                    ),
                )


                draw_center_text(
                    draw,
                    (
                        f"Member ID: "
                        f"{database_member_id}"
                    ),
                    FRAME_WIDTH // 2,
                    275,
                    font_normal,
                    color=(
                        255,
                        255,
                        255,
                    ),
                )


                draw_center_text(
                    draw,
                    "Nhấn Q để đóng",
                    FRAME_WIDTH // 2,
                    320,
                    font_small,
                    color=(
                        210,
                        210,
                        210,
                    ),
                )


            elif database_error is not None:

                draw_center_text(
                    draw,
                    "LỖI KHI LƯU DATABASE",
                    FRAME_WIDTH // 2,
                    190,
                    font_success,
                    color=(
                        255,
                        80,
                        80,
                    ),
                )


                draw_center_text(
                    draw,
                    database_error,
                    FRAME_WIDTH // 2,
                    240,
                    font_small,
                    color=(
                        255,
                        255,
                        255,
                    ),
                )


                draw_center_text(
                    draw,
                    (
                        "Ảnh vẫn được giữ "
                        "trong thư mục enrollment"
                    ),
                    FRAME_WIDTH // 2,
                    285,
                    font_small,
                    color=(
                        220,
                        220,
                        220,
                    ),
                )


        # =================================================
        # FEEDBACK
        # =================================================

        if (
            feedback_message
            and
            time.time()
            < feedback_until
        ):

            draw_center_text(
                draw,
                feedback_message,
                FRAME_WIDTH // 2,
                390,
                font_small,
                color=(
                    100,
                    255,
                    120,
                ),
            )


        # =================================================
        # PIL -> OPENCV
        # =================================================

        display_frame = cv2.cvtColor(
            np.array(
                pil_image
            ),
            cv2.COLOR_RGB2BGR,
        )


        cv2.imshow(
            WINDOW_NAME,
            display_frame,
        )


        key = (
            cv2.waitKey(1)
            & 0xFF
        )


        if key == ord("q"):
            break


finally:

    camera.stop()

    cv2.destroyAllWindows()