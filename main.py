import cv2
from src.utils import key_listener, PoseDetections, helper_functions
from src.utils.body_points import *


def main():
    # cap = cv2.VideoCapture("./Presentations/1.mp4")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()
    
    detector = PoseDetections.PoseDetector()
    hand = PoseDetections.HandDetector()
    listener = key_listener.start_listener()

    while True:
        success, image = cap.read()
        if not success:
            break
        
        image = detector.detectPose(image, handBodyOnly=True)
        bodypoint = detector.getBodyPoints(image)
        if bodypoint:
            body = helper_functions.get_body_connections_points(bodypoint)
            if key_listener.enter_pressed:
                wrist = (bodypoint.get(LEFT_WRIST), bodypoint.get(RIGHT_WRIST))

                if bodypoint[LEFT_WRIST] and bodypoint[RIGHT_WRIST]:
                    hand.detect_hand_inside_body(image, body, bodypoint[RIGHT_WRIST], bodypoint[LEFT_WRIST])
                    hand.detect_hand_idle(image, wrist)
                

        cv2.imshow("image", image)
        cv2.waitKey(1)

        if key_listener.esc_pressed:
            break

    cap.release()
    cv2.destroyAllWindows()
    listener.stop()

if __name__ == "__main__":
    main()