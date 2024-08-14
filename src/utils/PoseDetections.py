import cv2
import mediapipe as mp
import time
from .handBody_connections import *
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
    
    def getBodyPoints(self,
                    image, 
                    draw=True,
                    return_all=False,
                    return_body=False,
                    return_righthand=False,
                    return_lefthand=False):
        # return list of body point, left hand coordinate and right hand coordinate
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, _ = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))
            if draw:
                cv2.circle(image, (lmList[15][1], lmList[15][2]), 3, (207,255,4), cv2.FILLED)
                cv2.circle(image, (lmList[16][1], lmList[16][2]), 3, (207,255,4), cv2.FILLED)
        
        return lmList, (lmList[15][1], lmList[15][2]), (lmList[16][1], lmList[16][2])
    
class HandDetector():
    def __init__(self):
        self.timer = HandTimer()

    def detect_hand_inside_body(self, image, bodyPolygon, rightHand, leftHand):
        height, width = image.shape[:2]
        x = int(width * 0.01)
        y = int(height * 0.2)

        xLeft, yLeft = leftHand
        xRight, yRight = rightHand
        is_right_inside = ray_casting(bodyPolygon, xRight, yRight)
        is_left_inside = ray_casting(bodyPolygon, xLeft, yLeft)
        
        body_time = self.timer.body_timer(image, is_right_inside, is_left_inside)

        cv2.putText(image, f"TIMER: {body_time:.2f} s", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 0, 0), 2, cv2.LINE_AA)

class HandTimer():
    def __init__(self,
                 hand_detection_threshold=5.0):
        self.threshold = hand_detection_threshold
        self.first_enter = True
        self.start_time = 0.0
        self.durations = 0.0
        self.inside_timer = 0.0
        self.timer = 0.0

    def body_timer(self, image, is_right_inside, is_left_inside):
        if is_left_inside and is_right_inside:
            cv2.circle(image, (image.shape[1] - 30, 30), 30, (0, 0, 255), cv2.FILLED)

            if self.first_enter:
                self.first_enter = False
                self.enter_time = time.perf_counter()
            else:
                current_time = time.perf_counter()
                self.durations = current_time - self.enter_time
                if self.inside_timer < self.threshold:
                    self.inside_timer += self.durations
                else:
                    self.timer += self.durations
                self.enter_time = current_time
        else:
            if not self.first_enter:
                self.first_enter = True
                self.durations = 0.0
                self.inside_timer = 0.0
        
        return self.timer
            

