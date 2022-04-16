import cv2
import mediapipe as mp
from utils import utils, landmarks, voice

capture = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh

# Parameter to tweak
# --------------------------------------------------------
# minimum detecttion confidence
DETECTION_CONFIDENCE = 0.7
# minimum tracking confidence
TRACKING_CONFIDENCE = 0.6
# minim eye aspect ration
EAR_SAFETY_THRESHOLD = 4.1
# min frames to count with eyes closed
MIN_FRAMES = 5
# --------------------------------------------------------

FRAME_COUNT = 0

# load face model
face_model = face_mesh.FaceMesh(static_image_mode=False,
                                max_num_faces=1,
                                min_detection_confidence=DETECTION_CONFIDENCE,
                                min_tracking_confidence=TRACKING_CONFIDENCE)

while True:

    # get frame from camera
    result, frame = capture.read()

    if not result:
        continue

    # flip frame vertically and convert to RGB
    frame = cv2.flip(frame, 1)

    # get results from face_model
    outputs = face_model.process(frame)

    # draw landmarks
    if outputs.multi_face_landmarks:
        utils.draw_landmarks(frame, outputs, landmarks.RIGHT_EYE_TOP_BOTTOM, utils.GREEN)
        utils.draw_landmarks(frame, outputs, landmarks.RIGHT_EYE_LEFT_RIGHT, utils.GREEN)
        utils.draw_landmarks(frame, outputs, landmarks.LEFT_EYE_TOP_BOTTOM, utils.GREEN)
        utils.draw_landmarks(frame, outputs, landmarks.LEFT_EYE_LEFT_RIGHT, utils.GREEN)

        # calculat EAR for left and right ear
        left_EAR = utils.calculate_EAR(frame, outputs, landmarks.LEFT_EYE_TOP_BOTTOM, landmarks.LEFT_EYE_LEFT_RIGHT)
        right_EAR = utils.calculate_EAR(frame, outputs, landmarks.RIGHT_EYE_TOP_BOTTOM, landmarks.RIGHT_EYE_LEFT_RIGHT)

        # get average EAR
        EAR = (left_EAR + right_EAR) / 2.0

        if EAR >= EAR_SAFETY_THRESHOLD:
            FRAME_COUNT += 1
        else:
            FRAME_COUNT = 0

        # if eyes closed for X number of frames, give a warning
        if FRAME_COUNT >= MIN_FRAMES:
            # print("sleeping")
            voice.warning_voice("warning! sleep detected")

        # display eye aspect ratio in realtime
        utils.show_EAR(frame, EAR)

    # Show window
    cv2.imshow('Gaze Detection', frame)

    # press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the cap object after the loop
capture.release()
cv2.destroyAllWindows()
