import cv2
import numpy as np
import mediapipe as mp
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = "face_landmarker.task"


if not os.path.exists(MODEL_PATH):

    print("ERROR: face_landmarker.task not found.")
    print()
    print("Put face_landmarker.task in the same folder")
    print("as ear_test.py")

    exit()


# ============================================================
# MEDIAPIPE
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)


options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)


detector = vision.FaceLandmarker.create_from_options(
    options
)


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

    exit()


print("Camera started.")
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
# DISTANCE
# ============================================================

def distance(p1, p2):

    return np.linalg.norm(
        np.array(p1) -
        np.array(p2)
    )


# ============================================================
# EAR
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

        return 0

    return (
        vertical_1 +
        vertical_2
    ) / (
        2 * horizontal
    )


# ============================================================
# MAIN LOOP
# ============================================================

timestamp = 0


while True:

    ret, frame = cap.read()


    if not ret:

        print("Could not read frame.")

        break


    # Mirror camera
    frame = cv2.flip(
        frame,
        1
    )


    height, width, _ = frame.shape


    # ========================================================
    # CONVERT FRAME
    # ========================================================

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )


    timestamp += 33


    # ========================================================
    # DETECT FACE
    # ========================================================

    result = detector.detect_for_video(
        image,
        timestamp
    )


    # ========================================================
    # FACE FOUND
    # ========================================================

    if len(result.face_landmarks) > 0:

        face = result.face_landmarks[0]


        # ----------------------------------------------------
        # LANDMARK COORDINATES
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
        # EYE POINTS
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
        # EAR
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
        ) / 2


        # ====================================================
        # DISPLAY EAR VALUES
        # ====================================================

        cv2.putText(
            frame,
            f"LEFT EAR  : {left_ear:.4f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            f"RIGHT EAR : {right_ear:.4f}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            f"AVERAGE EAR: {ear:.4f}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )


        # ====================================================
        # DRAW EYE LANDMARKS
        # ====================================================

        for point in left_eye:

            cv2.circle(
                frame,
                point,
                3,
                (255, 0, 0),
                -1
            )


        for point in right_eye:

            cv2.circle(
                frame,
                point,
                3,
                (255, 0, 0),
                -1
            )


        # ====================================================
        # INSTRUCTIONS
        # ====================================================

        cv2.putText(
            frame,
            "Open eyes normally",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            "Close eyes completely",
            (20, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


    else:

        cv2.putText(
            frame,
            "NO FACE DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "EAR Measurement Test",
        frame
    )


    # ========================================================
    # QUIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

detector.close()