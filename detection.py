import cv2
import numpy as np
import mediapipe as mp
import os
import time
import csv
from datetime import datetime

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "face_landmarker.task"

# Eye Aspect Ratio threshold
# EAR below this value = eyes closed
EAR_THRESHOLD = 0.30

# PERCLOS observation window
PERCLOS_WINDOW = 90.0

# PERCLOS drowsiness threshold
PERCLOS_THRESHOLD = 80.0

# CSV output
CSV_FILE = "drowsiness_data.csv"


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    print("ERROR: face_landmarker.task not found.")
    print("Make sure face_landmarker.task is in the same")
    print("folder as detection.py.")

    input("Press Enter to exit...")
    exit()

print("Face Landmarker model found.")


# ============================================================
# MEDIAPIPE FACE LANDMARKER
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.FaceLandmarker.create_from_options(
    options
)

print("MediaPipe initialized.")


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Camera 0 failed.")
    print("Trying camera 1...")

    cap.release()

    cap = cv2.VideoCapture(1)


if not cap.isOpened():

    print("ERROR: Webcam could not be opened.")

    detector.close()

    exit()


print("Webcam started.")
print("Press Q to quit.")


# ============================================================
# EYE LANDMARKS
# ============================================================

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
# DISTANCE FUNCTION
# ============================================================

def distance(p1, p2):

    return np.linalg.norm(
        np.array(p1) - np.array(p2)
    )


# ============================================================
# EAR CALCULATION
# ============================================================

def calculate_ear(eye):

    vertical_1 = distance(
        eye[1],
        eye[5]
    )

    vertical_2 = distance(
        eye[2],
        eye[4]
    )

    horizontal = distance(
        eye[0],
        eye[3]
    )

    if horizontal == 0:

        return 0.0

    ear = (
        vertical_1 + vertical_2
    ) / (
        2 * horizontal
    )

    return ear


# ============================================================
# PERCLOS VARIABLES
# ============================================================

window_start = time.time()

previous_time = time.time()

closed_time = 0.0

observed_time = 0.0


# IMPORTANT:
# Initialize before the loop so there is NO NameError.

drowsiness_status = "CALCULATING"

drowsy_color = (
    0,
    255,
    255
)


# ============================================================
# CSV FILE
# ============================================================

csv_file = open(
    CSV_FILE,
    "w",
    newline=""
)

csv_writer = csv.writer(
    csv_file
)

csv_writer.writerow([
    "Timestamp",
    "EAR",
    "Left_EAR",
    "Right_EAR",
    "Eye_State",
    "PERCLOS",
    "Window_Time",
    "Status"
])


# ============================================================
# MEDIAPIPE TIMESTAMP
# ============================================================

timestamp_ms = 0


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # CAMERA FRAME
    # --------------------------------------------------------

    success, frame = cap.read()

    if not success:

        print("Could not read camera.")

        break


    # Mirror camera

    frame = cv2.flip(
        frame,
        1
    )


    height, width, _ = frame.shape


    # --------------------------------------------------------
    # TIME
    # --------------------------------------------------------

    current_time = time.time()

    delta_time = (
        current_time -
        previous_time
    )

    previous_time = current_time


    # Protect against abnormal time jumps

    if delta_time < 0:

        delta_time = 0

    if delta_time > 1:

        delta_time = 1


    # --------------------------------------------------------
    # CONVERT FRAME
    # --------------------------------------------------------

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )


    # --------------------------------------------------------
    # MEDIAPIPE TIMESTAMP
    # --------------------------------------------------------

    timestamp_ms += max(
        1,
        int(delta_time * 1000)
    )


    # --------------------------------------------------------
    # FACE DETECTION
    # --------------------------------------------------------

    result = detector.detect_for_video(
        image,
        timestamp_ms
    )


    # --------------------------------------------------------
    # DEFAULT VALUES
    # --------------------------------------------------------

    ear = 0.0

    left_ear = 0.0

    right_ear = 0.0

    eye_status = "NO FACE"

    eye_color = (
        0,
        0,
        255
    )


    # --------------------------------------------------------
    # WINDOW TIME
    # --------------------------------------------------------

    elapsed_window = (
        current_time -
        window_start
    )


    # ========================================================
    # FACE DETECTED
    # ========================================================

    if len(result.face_landmarks) > 0:

        face = result.face_landmarks[0]


        # ----------------------------------------------------
        # CONVERT LANDMARKS TO PIXELS
        # ----------------------------------------------------

        landmarks = []

        for landmark in face:

            x = int(
                landmark.x * width
            )

            y = int(
                landmark.y * height
            )

            landmarks.append(
                (x, y)
            )


        # ----------------------------------------------------
        # GET EYES
        # ----------------------------------------------------

        left_eye = [
            landmarks[i]
            for i in LEFT_EYE
        ]

        right_eye = [
            landmarks[i]
            for i in RIGHT_EYE
        ]


        # ----------------------------------------------------
        # CALCULATE EAR
        # ----------------------------------------------------

        left_ear = calculate_ear(
            left_eye
        )

        right_ear = calculate_ear(
            right_eye
        )

        ear = (
            left_ear +
            right_ear
        ) / 2.0


        # ----------------------------------------------------
        # EYE STATE
        # ----------------------------------------------------

        if ear < EAR_THRESHOLD:

            eye_status = "EYES CLOSED"

            eye_color = (
                0,
                0,
                255
            )

            # Add closed-eye time
            closed_time += delta_time

        else:

            eye_status = "EYES OPEN"

            eye_color = (
                0,
                255,
                0
            )


        # ----------------------------------------------------
        # OBSERVED TIME
        # ----------------------------------------------------

        observed_time += delta_time


        # ----------------------------------------------------
        # PERCLOS
        # ----------------------------------------------------

        if observed_time > 0:

            perclos = (
                closed_time /
                observed_time
            ) * 100.0

        else:

            perclos = 0.0


        # ----------------------------------------------------
        # DRAW LEFT EYE LANDMARKS
        # ----------------------------------------------------

        for point in left_eye:

            cv2.circle(
                frame,
                point,
                3,
                (255, 0, 0),
                -1
            )


        # ----------------------------------------------------
        # DRAW RIGHT EYE LANDMARKS
        # ----------------------------------------------------

        for point in right_eye:

            cv2.circle(
                frame,
                point,
                3,
                (255, 0, 0),
                -1
            )


    else:

        perclos = 0.0


    # ========================================================
    # 90-SECOND PERCLOS DECISION
    # ========================================================

    if elapsed_window >= PERCLOS_WINDOW:

        # Calculate final PERCLOS

        if observed_time > 0:

            perclos = (
                closed_time /
                observed_time
            ) * 100.0

        else:

            perclos = 0.0


        # ----------------------------------------------------
        # DROWSINESS DECISION
        # ----------------------------------------------------

        if perclos >= PERCLOS_THRESHOLD:

            drowsiness_status = "DROWSY"

            drowsy_color = (
                0,
                0,
                255
            )

        else:

            drowsiness_status = "ALERT"

            drowsy_color = (
                0,
                255,
                0
            )


        # ----------------------------------------------------
        # RESET 90 SECOND WINDOW
        # ----------------------------------------------------

        window_start = current_time

        closed_time = 0.0

        observed_time = 0.0


    # ========================================================
    # REMAINING WINDOW TIME
    # ========================================================

    remaining_time = max(
        0.0,
        PERCLOS_WINDOW -
        elapsed_window
    )


    # ========================================================
    # DISPLAY EAR
    # ========================================================

    cv2.putText(
        frame,
        f"EAR: {ear:.3f}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # ========================================================
    # LEFT EAR
    # ========================================================

    cv2.putText(
        frame,
        f"Left EAR: {left_ear:.3f}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )


    # ========================================================
    # RIGHT EAR
    # ========================================================

    cv2.putText(
        frame,
        f"Right EAR: {right_ear:.3f}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )


    # ========================================================
    # EYE STATUS
    # ========================================================

    cv2.putText(
        frame,
        eye_status,
        (20, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        eye_color,
        2
    )


    # ========================================================
    # PERCLOS
    # ========================================================

    cv2.putText(
        frame,
        f"PERCLOS: {perclos:.2f}%",
        (20, 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )


    # ========================================================
    # WINDOW
    # ========================================================

    cv2.putText(
        frame,
        f"90s Window: {remaining_time:.1f}s",
        (20, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # ========================================================
    # THRESHOLDS
    # ========================================================

    cv2.putText(
        frame,
        f"EAR Threshold: {EAR_THRESHOLD:.2f}",
        (20, 230),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        f"PERCLOS Threshold: {PERCLOS_THRESHOLD:.0f}%",
        (20, 255),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )


    # ========================================================
    # DROWSINESS STATUS
    # ========================================================

    cv2.putText(
        frame,
        f"STATUS: {drowsiness_status}",
        (20, 300),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        drowsy_color,
        2
    )


    # ========================================================
    # SAVE DATA TO CSV
    # ========================================================

    csv_writer.writerow([
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S.%f"
        ),
        round(ear, 4),
        round(left_ear, 4),
        round(right_ear, 4),
        eye_status,
        round(perclos, 2),
        round(elapsed_window, 2),
        drowsiness_status
    ])

    csv_file.flush()


    # ========================================================
    # SHOW CAMERA
    # ========================================================

    cv2.imshow(
        "Driver Drowsiness Detection",
        frame
    )


    # ========================================================
    # QUIT WITH Q
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

detector.close()

csv_file.close()

print()
print("Program stopped.")
print("Data saved to:", CSV_FILE)