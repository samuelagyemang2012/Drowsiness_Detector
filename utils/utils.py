import mediapipe as mp
import cv2
from scipy.spatial import distance as dis

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)


# draw landmarks
def draw_landmarks(image, outputs, land_mark, color):
    height, width = image.shape[:2]

    for face in land_mark:
        point = outputs.multi_face_landmarks[0].landmark[face]
        point_scale = ((int)(point.x * width), (int)(point.y * height))
        cv2.circle(image, point_scale, 1, color, 1)


# display eye aspect ratio
def show_EAR(frame, ear):
    cv2.putText(frame, "EAR: {:.2f}".format(ear), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLUE, 1)


# get euclidean distance between 2 points
def get_euclidean_distance(image, top, bottom):
    height, width = image.shape[0:2]

    point1 = int(top.x * width), int(top.y * height)
    point2 = int(bottom.x * width), int(bottom.y * height)

    distance = dis.euclidean(point1, point2)
    return distance


# calculate eye aspect ratio
def calculate_EAR(image, outputs, top_bottom, left_right):
    landmark = outputs.multi_face_landmarks[0]

    top = landmark.landmark[top_bottom[0]]
    bottom = landmark.landmark[top_bottom[1]]
    top_bottom_dis = get_euclidean_distance(image, top, bottom)

    left = landmark.landmark[left_right[0]]
    right = landmark.landmark[left_right[1]]
    left_right_dis = get_euclidean_distance(image, left, right)

    aspect_ratio = left_right_dis / top_bottom_dis

    return aspect_ratio
