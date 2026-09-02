import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time
from collections import deque


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "face_landmarker.task"

# EAR SETTINGS
EAR_THRESHOLD = 0.30

# Partial closure threshold
PARTIAL_EAR_THRESHOLD = 0.35


# PERCLOS SETTINGS
PERCLOS_WINDOW = 90.0
PERCLOS_THRESHOLD = 80.0


# MAR SETTINGS
# USER-APPROVED THRESHOLD
MAR_THRESHOLD = 20.0


# CSV FILE
CSV_FILE = "mar_data.csv"


# CAMERA
CAMERA_INDEX = 0


# ============================================================
# MEDIAPIPE FACE LANDMARKER SETUP
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


landmarker = FaceLandmarker.create_from_options(options)


# ============================================================
# EYE LANDMARKS
#
# These points are used for EAR calculation.
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
# MOUTH LANDMARKS
#
# IMPORTANT:
#
# These are ACTUAL upper and lower lip pairs.
#
# We are NOT using points that are all on the upper lip.
#
# Each pair represents vertical mouth opening.
# ============================================================

MOUTH_PAIRS = [

    # Outer mouth pairs
    (13, 14),

    # Inner mouth pairs
    (82, 87),
    (81, 88),
    (80, 191),

    # Wider opening measurement
    (312, 317)
]


# ============================================================
# PERCLOS DATA STORAGE
# ============================================================

perclos_data = deque()


# ============================================================
# CSV SETUP
# ============================================================

if not os.path.exists(CSV_FILE):

    with open(
        CSV_FILE,
        mode="w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Timestamp",
            "EAR",
            "Eye_Status",
            "MAR",
            "Mouth_Status",
            "PERCLOS",
            "Drowsiness_Status"
        ])


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

def calculate_ear(landmarks, eye_points, frame_width, frame_height):

    points = []

    for index in eye_points:

        x = landmarks[index].x * frame_width
        y = landmarks[index].y * frame_height

        points.append((x, y))


    # Vertical distances
    A = distance(points[1], points[5])

    B = distance(points[2], points[4])


    # Horizontal distance
    C = distance(points[0], points[3])


    if C == 0:
        return 0


    # EAR FORMULA
    ear = (A + B) / (2.0 * C)


    return ear


# ============================================================
# ================= MAR CALCULATION ==========================
# ============================================================

def calculate_mar(
    landmarks,
    frame_width,
    frame_height
):

    """
    MAR is calculated from actual upper-lip to lower-lip pairs.

    Step 1:
        Calculate vertical pixel distance for each mouth pair.

    Step 2:
        Calculate the average mouth opening.

    Step 3:
        Normalize using mouth width.

    Step 4:
        Scale by 100.

    This creates a practical MAR scale where:

        MAR > 20  --> YAWNING

    IMPORTANT:
    We do NOT simply sum raw pixel distances because
    camera resolution and face distance would make the
    result inconsistent.
    """


    vertical_distances = []


    # --------------------------------------------------------
    # Calculate mouth opening distances
    # --------------------------------------------------------

    for upper_index, lower_index in MOUTH_PAIRS:

        upper_x = (
            landmarks[upper_index].x
            * frame_width
        )

        upper_y = (
            landmarks[upper_index].y
            * frame_height
        )


        lower_x = (
            landmarks[lower_index].x
            * frame_width
        )

        lower_y = (
            landmarks[lower_index].y
            * frame_height
        )


        upper_point = (
            upper_x,
            upper_y
        )

        lower_point = (
            lower_x,
            lower_y
        )


        opening_distance = distance(
            upper_point,
            lower_point
        )


        vertical_distances.append(
            opening_distance
        )


    # --------------------------------------------------------
    # Average vertical mouth opening
    # --------------------------------------------------------

    average_opening = np.mean(
        vertical_distances
    )


    # --------------------------------------------------------
    # MOUTH WIDTH
    #
    # Used for normalization so that MAR does not depend
    # strongly on how close the face is to the camera.
    # --------------------------------------------------------

    left_corner = (
        landmarks[61].x * frame_width,
        landmarks[61].y * frame_height
    )


    right_corner = (
        landmarks[291].x * frame_width,
        landmarks[291].y * frame_height
    )


    mouth_width = distance(
        left_corner,
        right_corner
    )


    if mouth_width == 0:

        return 0


    # --------------------------------------------------------
    # FINAL MAR CALCULATION
    #
    # Normalized mouth opening × 100
    #
    # THIS IS THE MAIN MAR CALCULATION LINE
    # --------------------------------------------------------

    mar = (
        average_opening
        / mouth_width
    ) * 100


    return mar


# ============================================================
# PERCLOS CALCULATION
# ============================================================

def calculate_perclos():

    current_time = time.time()


    # Remove data older than 90 seconds
    while (
        len(perclos_data) > 0
        and
        current_time
        - perclos_data[0][0]
        > PERCLOS_WINDOW
    ):

        perclos_data.popleft()


    if len(perclos_data) == 0:

        return 0


    closed_count = 0


    for timestamp, eye_closed in perclos_data:

        if eye_closed:

            closed_count += 1


    perclos = (
        closed_count
        / len(perclos_data)
    ) * 100


    return perclos


# ============================================================
# DRAW TEXT FUNCTION
#
# No background rectangle behind text.
# ============================================================

def draw_text(
    frame,
    text,
    position,
    color,
    scale=0.7
):

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA
    )


# ============================================================
# CAMERA INITIALIZATION
# ============================================================

cap = cv2.VideoCapture(
    CAMERA_INDEX
)


if not cap.isOpened():

    print(
        "ERROR: Cannot open camera."
    )

    exit()


# ============================================================
# MAIN PROGRAM
# ============================================================

previous_csv_time = 0


print(
    "Drowsiness Detection Started"
)

print(
    "Press Q to quit."
)


while True:


    # --------------------------------------------------------
    # READ CAMERA FRAME
    # --------------------------------------------------------

    success, frame = cap.read()


    if not success:

        print(
            "Failed to read camera."
        )

        break


    # --------------------------------------------------------
    # FLIP FRAME
    # --------------------------------------------------------

    frame = cv2.flip(
        frame,
        1
    )


    frame_height, frame_width = (
        frame.shape[:2]
    )


    # --------------------------------------------------------
    # CONVERT BGR TO RGB
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # CREATE MEDIAPIPE IMAGE
    # --------------------------------------------------------

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # --------------------------------------------------------
    # TIMESTAMP
    # --------------------------------------------------------

    timestamp_ms = int(
        time.time() * 1000
    )


    # --------------------------------------------------------
    # FACE LANDMARK DETECTION
    # --------------------------------------------------------

    detection_result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # Default values
    ear = 0
    mar = 0
    perclos = 0

    eye_status = "NO FACE"
    mouth_status = "NO FACE"

    drowsiness_status = (
        "NO FACE DETECTED"
    )


    # ========================================================
    # IF FACE DETECTED
    # ========================================================

    if detection_result.face_landmarks:


        landmarks = (
            detection_result
            .face_landmarks[0]
        )


        # ====================================================
        # LEFT EAR
        # ====================================================

        left_ear = calculate_ear(
            landmarks,
            LEFT_EYE,
            frame_width,
            frame_height
        )


        # ====================================================
        # RIGHT EAR
        # ====================================================

        right_ear = calculate_ear(
            landmarks,
            RIGHT_EYE,
            frame_width,
            frame_height
        )


        # ====================================================
        # FINAL EAR
        # ====================================================

        ear = (
            left_ear
            + right_ear
        ) / 2


        # ====================================================
        # EYE STATUS
        #
        # EAR >= 0.35
        #       EYES OPEN
        #
        # 0.30 <= EAR < 0.35
        #       PARTIALLY CLOSED
        #
        # EAR < 0.30
        #       EYES CLOSED
        #
        # USER'S MAIN THRESHOLD = 0.30
        # ====================================================

        if ear >= PARTIAL_EAR_THRESHOLD:

            eye_status = (
                "EYES OPEN"
            )

            eye_color = (
                0,
                255,
                0
            )


        elif (
            ear >= EAR_THRESHOLD
            and
            ear < PARTIAL_EAR_THRESHOLD
        ):

            eye_status = (
                "PARTIALLY CLOSED"
            )

            eye_color = (
                0,
                255,
                255
            )


        else:

            eye_status = (
                "EYES CLOSED"
            )

            eye_color = (
                0,
                0,
                255
            )


        # ====================================================
        # ADD DATA FOR PERCLOS
        #
        # Closed = EAR < 0.30
        # ====================================================

        is_eye_closed = (
            ear < EAR_THRESHOLD
        )


        perclos_data.append(
            (
                time.time(),
                is_eye_closed
            )
        )


        # ====================================================
        # CALCULATE PERCLOS
        # ====================================================

        perclos = calculate_perclos()


        # ====================================================
        # CALCULATE MAR
        # ====================================================

        mar = calculate_mar(
            landmarks,
            frame_width,
            frame_height
        )


        # ====================================================
        # MOUTH / YAWN STATUS
        #
        # USER-APPROVED RULE:
        #
        # MAR > 20 --> YAWNING
        # ====================================================

        if mar > MAR_THRESHOLD:

            mouth_status = (
                "YAWNING"
            )

            mouth_color = (
                0,
                0,
                255
            )


        else:

            mouth_status = (
                "NOT YAWNING"
            )

            mouth_color = (
                0,
                255,
                0
            )


        # ====================================================
        # DROWSINESS STATUS
        # ====================================================

        if perclos >= PERCLOS_THRESHOLD:

            drowsiness_status = (
                "DROWSY - HIGH PERCLOS"
            )

            drowsiness_color = (
                0,
                0,
                255
            )


        elif eye_status == "EYES CLOSED":

            drowsiness_status = (
                "EYES CLOSED"
            )

            drowsiness_color = (
                0,
                0,
                255
            )


        elif mouth_status == "YAWNING":

            drowsiness_status = (
                "YAWNING DETECTED"
            )

            drowsiness_color = (
                0,
                165,
                255
            )


        else:

            drowsiness_status = (
                "ALERT"
            )

            drowsiness_color = (
                0,
                255,
                0
            )


        # ====================================================
        # DRAW EYE LANDMARKS
        # ====================================================

        for index in LEFT_EYE:

            x = int(
                landmarks[index].x
                * frame_width
            )

            y = int(
                landmarks[index].y
                * frame_height
            )

            cv2.circle(
                frame,
                (x, y),
                2,
                eye_color,
                -1
            )


        for index in RIGHT_EYE:

            x = int(
                landmarks[index].x
                * frame_width
            )

            y = int(
                landmarks[index].y
                * frame_height
            )

            cv2.circle(
                frame,
                (x, y),
                2,
                eye_color,
                -1
            )


        # ====================================================
        # DRAW MOUTH PAIRS
        # ====================================================

        for upper_index, lower_index in MOUTH_PAIRS:


            x1 = int(
                landmarks[upper_index].x
                * frame_width
            )

            y1 = int(
                landmarks[upper_index].y
                * frame_height
            )


            x2 = int(
                landmarks[lower_index].x
                * frame_width
            )

            y2 = int(
                landmarks[lower_index].y
                * frame_height
            )


            cv2.circle(
                frame,
                (x1, y1),
                2,
                mouth_color,
                -1
            )


            cv2.circle(
                frame,
                (x2, y2),
                2,
                mouth_color,
                -1
            )


            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                mouth_color,
                1
            )


    # ========================================================
    # DISPLAY VALUES
    #
    # NO BACKGROUND BOXES
    # ========================================================

    draw_text(
        frame,
        f"EAR: {ear:.3f}",
        (30, 40),
        (255, 255, 255)
    )


    # Eye status
    if eye_status == "EYES OPEN":

        display_eye_color = (
            0,
            255,
            0
        )


    elif eye_status == "PARTIALLY CLOSED":

        display_eye_color = (
            0,
            255,
            255
        )


    else:

        display_eye_color = (
            0,
            0,
            255
        )


    draw_text(
        frame,
        f"Eye Status: {eye_status}",
        (30, 75),
        display_eye_color
    )


    # MAR
    draw_text(
        frame,
        f"MAR: {mar:.2f}",
        (30, 110),
        (255, 255, 255)
    )


    # Mouth status
    if mouth_status == "YAWNING":

        display_mouth_color = (
            0,
            0,
            255
        )

    else:

        display_mouth_color = (
            0,
            255,
            0
        )


    draw_text(
        frame,
        f"Mouth Status: {mouth_status}",
        (30, 145),
        display_mouth_color
    )


    # PERCLOS
    draw_text(
        frame,
        f"PERCLOS (90s): {perclos:.2f}%",
        (30, 180),
        (255, 255, 255)
    )


    # Drowsiness status
    draw_text(
        frame,
        f"Status: {drowsiness_status}",
        (30, 215),
        drowsiness_color
    )


    # ========================================================
    # SAVE DATA TO CSV
    #
    # Saves approximately once per second
    # ========================================================

    current_time = time.time()


    if (
        current_time
        - previous_csv_time
        >= 1
    ):


        with open(
            CSV_FILE,
            mode="a",
            newline=""
        ) as file:


            writer = csv.writer(
                file
            )


            writer.writerow([
                time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                round(ear, 4),
                eye_status,
                round(mar, 2),
                mouth_status,
                round(perclos, 2),
                drowsiness_status
            ])


        previous_csv_time = (
            current_time
        )


    # ========================================================
    # SHOW FRAME
    # ========================================================

    cv2.imshow(
        "Drowsiness Detection",
        frame
    )


    # ========================================================
    # PRESS Q TO EXIT
    # ========================================================

    if (
        cv2.waitKey(1)
        & 0xFF
        == ord("q")
    ):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

landmarker.close()


print(
    "Program stopped."
)

print(
    f"Data saved in {CSV_FILE}"
)