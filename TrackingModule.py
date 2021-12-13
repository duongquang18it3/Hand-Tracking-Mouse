import cv2 as cv
import mediapipe as mp
import time
from utils import get_index_fingers

#Định nghĩa lớp tracking bàn tày
class HandTracking:
    #Hàm khởi tạo
    def __init__(
            self,
            mode = False,
            hand_number = 1,
            min_detection_confidence = 0.5,
            min_tracking_confidence = 0.5):
        self.mpHand = mp.solutions.hands
        self.hands = self.mpHand.Hands(mode, hand_number, min_detection_confidence, min_tracking_confidence)
        self.draw_tool = mp.solutions.drawing_utils

    #Hàm tìm bàn tay và trả về danh sách tọa độ các điểm landmarks
    def findLandMark(self, image):
        imgRGB = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        results = self.hands.process(imgRGB)

        landmarksList = results.multi_hand_landmarks
        if landmarksList:
            return list(landmarksList[0].landmark)

        return None

    #Hàm khiểm tra một ngón tay có đang giơ lên hay không
    def isRaiseFinger(self, landmarks, finger):
        point1, point2 = get_index_fingers(finger)

        if (landmarks[point1].y <= landmarks[point2].y):
            return True

        return False






