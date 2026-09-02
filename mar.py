import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time
from collections import deque


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "face_landmarker.task"

# EAR SETTINGS
EAR_THRESHOLD = 0.30

# Additional threshold for fully closed eyes
EYES_CLOSED_THRESHOLD = 0.18


# MAR SETTINGS
# This is a normalized MAR threshold.
# You should calibrate it based on your webcam results.
MAR_THRESHOLD = 0.60


# PERCLOS SETTINGS
PERCLOS_WINDOW = 90.0          # 90 seconds
PERCLOS_THRESHOLD = 80.0       # 80 percent


# CSV SETTINGS
CSV_FILE = "mar_data.csv"

SAVE_INTERVAL = 1.0            # Save every 1 second


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions


options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)


# ============================================================
# CSV INITIALIZATION
# ============================================================

if not os.path.exists(CSV_FILE):

    with open(CSV_FILE, mode="w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "Timestamp",
            "Left_EAR",
            "Right_EAR",
            "Average_EAR",
            "Eye_Status",
            "MAR",
            "Yawn_Status",
            "PERCLOS",
            "Drowsiness_Status"
        ])


# ============================================================
# LANDMARK DEFINITIONS
# ============================================================

# MediaPipe Face Mesh eye landmarks

LEFT_EYE = [
    33,
    160,
    158,
    133,
    153,
    144
]


RIGHT_EYE = [
    362,
    385,
    387,
    263,
    373,
    380
]


# ============================================================
# MOUTH LANDMARKS
# ============================================================

# Outer mouth:
#
# Left corner  = 61
# Right corner = 291
#
# Upper / lower lip pairs are selected
# to measure the actual vertical mouth opening.

MOUTH_LEFT = 61
MOUTH_RIGHT = 291

UPPER_LIP_1 = 13
LOWER_LIP_1 = 14

UPPER_LIP_2 = 82
LOWER_LIP_2 = 87

UPPER_LIP_3 = 312
LOWER_LIP_3 = 317


# ============================================================
# DISTANCE FUNCTION
# ============================================================

def distance(p1, p2):

    return np.linalg.norm(
        np.array(p1) - np.array(p2)
    )


# ============================================================
# EAR CALCULATION
# ============================================================

def calculate_ear(landmarks, eye_points):

    p1 = landmarks[eye_points[0]]
    p2 = landmarks[eye_points[1]]
    p3 = landmarks[eye_points[2]]
    p4 = landmarks[eye_points[3]]
    p5 = landmarks[eye_points[4]]
    p6 = landmarks[eye_points[5]]


    # Standard Eye Aspect Ratio formula
    #
    # EAR =
    # (|p2-p6| + |p3-p5|)
    # --------------------
    #      2|p1-p4|

    vertical_1 = distance(p2, p6)

    vertical_2 = distance(p3, p5)

    horizontal = distance(p1, p4)


    if horizontal == 0:

        return 0.0


    ear = (
        vertical_1 + vertical_2
    ) / (
        2.0 * horizontal
    )


    return ear


# ============================================================
# MAR CALCULATION
# ============================================================

def calculate_mar(landmarks):

    """
    Normalized Mouth Aspect Ratio.

    We measure actual upper-to-lower lip opening
    at multiple locations.

    MAR =
    average vertical mouth opening
    --------------------------------
    mouth width

    This avoids the earlier problem where MAR became
    artificially high even when the mouth was closed.
    """


    # Actual vertical mouth openings

    opening_1 = distance(
        landmarks[UPPER_LIP_1],
        landmarks[LOWER_LIP_1]
    )


    opening_2 = distance(
        landmarks[UPPER_LIP_2],
        landmarks[LOWER_LIP_2]
    )


    opening_3 = distance(
        landmarks[UPPER_LIP_3],
        landmarks[LOWER_LIP_3]
    )


    # Average vertical opening

    average_opening = (
        opening_1 +
        opening_2 +
        opening_3
    ) / 3.0


    # Horizontal mouth width

    mouth_width = distance(
        landmarks[MOUTH_LEFT],
        landmarks[MOUTH_RIGHT]
    )


    if mouth_width == 0:

        return 0.0


    mar = average_opening / mouth_width


    return mar


# ============================================================
# DRAW TEXT FUNCTION
# ============================================================

def draw_text(frame, text, position, color):

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

cap = cv2.VideoCapture(0)


if not cap.isOpened():

    print("ERROR: Cannot access webcam.")

    exit()


print("Webcam started...")
print("Saving data to:", CSV_FILE)


# PERCLOS DATA
#
# Stores:
# (timestamp, eyes_closed_boolean)

perclos_data = deque()


# CSV saving timer

last_save_time = 0


# Video timestamp

start_time = time.time()


# ============================================================
# CREATE FACE LANDMARKER
# ============================================================

with FaceLandmarker.create_from_options(options) as landmarker:


    while True:


        success, frame = cap.read()


        if not success:

            print("ERROR: Cannot read webcam.")

            break


        # ----------------------------------------------------
        # MIRROR FRAME
        # ----------------------------------------------------

        frame = cv2.flip(frame, 1)


        frame_height, frame_width = frame.shape[:2]


        # ----------------------------------------------------
        # CURRENT TIME
        # ----------------------------------------------------

        current_time = time.time()

        elapsed_time = current_time - start_time


        # ----------------------------------------------------
        # CONVERT IMAGE FOR MEDIAPIPE
        # ----------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )


        timestamp_ms = int(
            elapsed_time * 1000
        )


        # ----------------------------------------------------
        # FACE DETECTION
        # ----------------------------------------------------

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )


        # ----------------------------------------------------
        # DEFAULT VALUES
        # ----------------------------------------------------

        left_ear = 0.0

        right_ear = 0.0

        average_ear = 0.0

        mar = 0.0

        perclos = 0.0


        eye_status = "NO FACE"

        yawn_status = "NO YAWN"

        drowsiness_status = "NORMAL"


        eyes_closed = False


        # ====================================================
        # FACE DETECTED
        # ====================================================

        if result.face_landmarks:


            face_landmarks = result.face_landmarks[0]


            # ------------------------------------------------
            # CONVERT NORMALIZED LANDMARKS TO PIXELS
            # ------------------------------------------------

            landmarks = []


            for landmark in face_landmarks:


                x = landmark.x * frame_width

                y = landmark.y * frame_height


                landmarks.append(
                    (x, y)
                )


            # ------------------------------------------------
            # LEFT EAR
            # ------------------------------------------------

            left_ear = calculate_ear(
                landmarks,
                LEFT_EYE
            )


            # ------------------------------------------------
            # RIGHT EAR
            # ------------------------------------------------

            right_ear = calculate_ear(
                landmarks,
                RIGHT_EYE
            )


            # ------------------------------------------------
            # AVERAGE EAR
            # ------------------------------------------------

            average_ear = (
                left_ear +
                right_ear
            ) / 2.0


            # ------------------------------------------------
            # MAR
            # ------------------------------------------------

            mar = calculate_mar(
                landmarks
            )


            # =================================================
            # EYE STATUS
            # =================================================
            #
            # Threshold:
            #
            # EAR >= 0.30
            #       Eyes Open
            #
            # 0.18 <= EAR < 0.30
            #       Partially Closed
            #
            # EAR < 0.18
            #       Eyes Closed
            #
            # =================================================


            if average_ear >= EAR_THRESHOLD:


                eye_status = "EYES OPEN"

                eye_color = (
                    0,
                    255,
                    0
                )


                eyes_closed = False


            elif average_ear >= EYES_CLOSED_THRESHOLD:


                eye_status = "PARTIALLY CLOSED"

                eye_color = (
                    0,
                    165,
                    255
                )


                eyes_closed = False


            else:


                eye_status = "EYES CLOSED"

                eye_color = (
                    0,
                    0,
                    255
                )


                eyes_closed = True


            # =================================================
            # YAWN STATUS
            # =================================================

            if mar >= MAR_THRESHOLD:


                yawn_status = "YAWNING"

                yawn_color = (
                    0,
                    0,
                    255
                )


            else:


                yawn_status = "NO YAWN"

                yawn_color = (
                    0,
                    255,
                    0
                )


            # =================================================
            # DRAW EYE LANDMARKS
            # =================================================

            for point in LEFT_EYE:


                x, y = landmarks[point]


                cv2.circle(
                    frame,
                    (int(x), int(y)),
                    2,
                    eye_color,
                    -1
                )


            for point in RIGHT_EYE:


                x, y = landmarks[point]


                cv2.circle(
                    frame,
                    (int(x), int(y)),
                    2,
                    eye_color,
                    -1
                )


            # =================================================
            # DRAW MOUTH LANDMARKS
            # =================================================

            mouth_points = [

                MOUTH_LEFT,
                MOUTH_RIGHT,

                UPPER_LIP_1,
                LOWER_LIP_1,

                UPPER_LIP_2,
                LOWER_LIP_2,

                UPPER_LIP_3,
                LOWER_LIP_3

            ]


            for point in mouth_points:


                x, y = landmarks[point]


                cv2.circle(
                    frame,
                    (int(x), int(y)),
                    2,
                    yawn_color,
                    -1
                )


        else:


            eye_color = (
                255,
                255,
                255
            )


            yawn_color = (
                255,
                255,
                255
            )


        # ====================================================
        # PERCLOS CALCULATION
        # ====================================================

        perclos_data.append(
            (
                elapsed_time,
                eyes_closed
            )
        )


        # Remove data older than 90 seconds

        while (
            perclos_data
            and
            elapsed_time - perclos_data[0][0]
            > PERCLOS_WINDOW
        ):


            perclos_data.popleft()


        # Calculate PERCLOS

        if len(perclos_data) > 0:


            closed_count = sum(

                1
                for timestamp, closed
                in perclos_data

                if closed

            )


            perclos = (

                closed_count /
                len(perclos_data)

            ) * 100.0


        else:


            perclos = 0.0


        # ====================================================
        # DROWSINESS STATUS
        # ====================================================

        if perclos >= PERCLOS_THRESHOLD:


            drowsiness_status = "DROWSY"

            drowsiness_color = (
                0,
                0,
                255
            )


        elif eyes_closed:


            drowsiness_status = "EYES CLOSED"

            drowsiness_color = (
                0,
                0,
                255
            )


        elif yawn_status == "YAWNING":


            drowsiness_status = "YAWNING DETECTED"

            drowsiness_color = (
                0,
                165,
                255
            )


        else:


            drowsiness_status = "NORMAL"

            drowsiness_color = (
                0,
                255,
                0
            )


        # ====================================================
        # DISPLAY VALUES
        # ====================================================

        draw_text(

            frame,

            f"LEFT EAR: {left_ear:.3f}",

            (20, 35),

            eye_color

        )


        draw_text(

            frame,

            f"RIGHT EAR: {right_ear:.3f}",

            (20, 65),

            eye_color

        )


        draw_text(

            frame,

            f"AVERAGE EAR: {average_ear:.3f}",

            (20, 95),

            eye_color

        )


        draw_text(

            frame,

            f"EYE STATUS: {eye_status}",

            (20, 125),

            eye_color

        )


        draw_text(

            frame,

            f"MAR: {mar:.3f}",

            (20, 170),

            yawn_color

        )


        draw_text(

            frame,

            f"YAWN STATUS: {yawn_status}",

            (20, 200),

            yawn_color

        )


        draw_text(

            frame,

            f"PERCLOS (90s): {perclos:.2f}%",

            (20, 245),

            drowsiness_color

        )


        draw_text(

            frame,

            f"STATUS: {drowsiness_status}",

            (20, 275),

            drowsiness_color

        )


        # ====================================================
        # SAVE DATA EVERY SECOND
        # ====================================================

        if (

            current_time - last_save_time

            >=

            SAVE_INTERVAL

        ):


            timestamp_string = time.strftime(

                "%Y-%m-%d %H:%M:%S"

            )


            with open(

                CSV_FILE,

                mode="a",

                newline=""

            ) as file:


                writer = csv.writer(file)


                writer.writerow([

                    timestamp_string,

                    round(left_ear, 4),

                    round(right_ear, 4),

                    round(average_ear, 4),

                    eye_status,

                    round(mar, 4),

                    yawn_status,

                    round(perclos, 2),

                    drowsiness_status

                ])


            # Update save time

            last_save_time = current_time


        # ====================================================
        # SHOW WINDOW
        # ====================================================

        cv2.imshow(

            "Drowsiness Detection System",

            frame

        )


        # ====================================================
        # EXIT
        # ====================================================

        key = cv2.waitKey(1) & 0xFF


        if key == ord("q"):


            break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()


print("Program stopped.")

print(f"Data saved successfully in: {CSV_FILE}")