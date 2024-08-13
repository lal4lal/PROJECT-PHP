import cv2
from src.utils import key_listener, PoseDetections, helper_functions


def main():
    # cap = cv2.VideoCapture("./Presentations/3.mp4")
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
        lmList = detector.getPositions(image)
        if lmList:
            body = helper_functions.get_body_connections_points(lmList)
            rightHand = (lmList[16][1], lmList[16][2])
            leftHand = (lmList[15][1], lmList[15][2])
            if key_listener.space_pressed:
                hand.detect_hand_inside_body(image, body, rightHand, leftHand)

        cv2.imshow("image", image)
        cv2.waitKey(1)

        if key_listener.esc_pressed:
            break

    cap.release()
    cv2.destroyAllWindows()
    listener.stop()

if __name__ == "__main__":
    main()