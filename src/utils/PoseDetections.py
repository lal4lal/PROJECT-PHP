import cv2
import mediapipe as mp
import time
from .body_points import *
from .helper_functions import *

class PoseDetector():
    def __init__(self,
               mode=False,
               smooth=True,
               detectionCon=0.5,
               trackCon=0.5):

        self.mode = mode
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            smooth_landmarks=self.smooth,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )

    def detectPose(self, image, draw = True, handBodyOnly = False):
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                if handBodyOnly:
                    self.mpDraw.draw_landmarks(image, self.results.pose_landmarks, HAND_BODY_CONNECTIONS)
                else:
                    self.mpDraw.draw_landmarks(image, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
            
        return image
    
    def getBodyPoints(self, image):
        # return list of body point, left hand coordinate and right hand coordinate
        bodypoint = {}
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, _ = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                bodypoint.update({id: (cx, cy)})

        return bodypoint
    
class HandDetector():
    def __init__(self):
        self.timer = Timer()
        self.wrist_left = None
        self.wrist_right = None

    def detect_hand_inside_body(self, image, bodyPolygon, rightHand, leftHand):
        height, width = image.shape[:2]
        x = int(width * 0.01)
        y = int(height * 0.05)

        xLeft, yLeft = leftHand
        xRight, yRight = rightHand
        is_right_inside = ray_casting(bodyPolygon, xRight, yRight)
        is_left_inside = ray_casting(bodyPolygon, xLeft, yLeft)
        
        inside_time = self.timer.hand_timer(image, is_right_inside, is_left_inside)

        cv2.putText(image, f"INSIDE TIMER: {inside_time:.2f} s", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

    def _calculate_movement(self, previous, current):
        return ((previous[0] - current[0]) ** 2 + (previous[1] - current[1]) ** 2) ** 0.5

    def detect_hand_idle(self, image, wrist):
        height, width = image.shape[:2]
        x = int(width * 0.01)
        y = int(height * 0.1)

        left, right = wrist
        if self.wrist_left is None or self.wrist_right is None:
            self.wrist_left = left
            self.wrist_right = right
            return
        
        left_wrist_movement = self._calculate_movement(self.wrist_left, left)
        right_wrist_movement = self._calculate_movement(self.wrist_right, right)

        left_wrist_idle = True if left_wrist_movement < 5 else False
        right_wrist_idle = True if right_wrist_movement < 5 else False

        idle_time = self.timer.idling_timer(image, left_wrist_idle, right_wrist_idle)
        self.wrist_left = left
        self.wrist_right = right

        cv2.putText(image, f'IDLE TIMER: {idle_time:.2f} s', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
            
class Timer():
    def __init__(self,
                 detection_threshold=3.0):
        self.threshold = detection_threshold
        self.first_inside = True
        self.start_inside = 0.0
        self.durations_inside = 0.0
        self.inside_timer = 0.0
        self.timerInside = 0.0

        self.first_idle = True
        self.start_idle = 0.0
        self.durations_idle = 0.0
        self.idle_timer = 0.0
        self.timerIdle = 0.0

    def hand_timer(self, image, is_right_inside, is_left_inside):
        if is_left_inside and is_right_inside:

            if self.first_inside:
                self.first_inside = False
                self.start_inside = time.perf_counter()
            else:
                current_time = time.perf_counter()
                self.durations_inside = current_time - self.start_inside
                if self.inside_timer < self.threshold:
                    self.inside_timer += self.durations_inside
                else:
                    cv2.circle(image, (image.shape[1] - 30, 30), 30, (0, 0, 255), cv2.FILLED)
                    self.timerInside += self.durations_inside
                self.start_inside = current_time
        else:
            if not self.first_inside:
                self.first_inside = True
                self.durations_inside = 0.0
                self.inside_timer = 0.0
        
        return self.timerInside
    
    def idling_timer(self, image, left_wrist_idle, right_wrist_idle):
        if left_wrist_idle and right_wrist_idle:

            if self.first_idle:
                self.first_idle = False
                self.start_idle = time.perf_counter()
            else:
                current_time = time.perf_counter()
                self.durations_idle = current_time - self.start_idle
                if self.idle_timer < self.threshold:
                    self.idle_timer += self.durations_idle
                else:
                    cv2.circle(image, (image.shape[1] - 90, 30), 30, (255, 0, 0), cv2.FILLED)
                    self.timerIdle += self.durations_idle
                self.start_idle = current_time
        else:
            if not self.first_idle:
                self.first_idle = True
                self.durations_idle = 0.0
                self.idle_timer = 0.0
        
        return self.timerIdle

            

