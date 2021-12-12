import cv2
import mediapipe as mp
import time
import autopy

cap = cv2.VideoCapture(0)
#tao doi tuong mpHands
mpHands = mp.solutions.hands

#tao doi tuong hands, doi tuong nay co bon tham so la:
# static_image_mode=False, \\ neu laf false thi co luc la che do detect, co luc la che do tracking, neu la true thi luon luon detect
# max_num_hands=2, \\ thiet lap so ban tay co the detect hoac tracking
# min_detection_confidence=0.5, \\ phan tram confidence detection, neu duoi 50% se chuyen qua tracking va tuong tu voi min_tracking_confidence
# min_tracking_confidence=0.5
#o day ta de mac dinh
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0


while True:
    success, img = cap.read()

    #chuyen anh ve dang RGB de xu li vi ham process trong object hands chi xu li anh RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #dung ham process cua doi tuong hands de xu ly va gan ket qua vao bien result
    results = hands.process(imgRGB)

    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        #duyet qua tung ban tay duoc detect
        for handLms in results.multi_hand_landmarks:
            h, w, c = img.shape
            #duyet qua tung diem lanmark ta se co toa do va id tuong ung cua tung diem landmark
            list_landmark_points = list(handLms.landmark)
            cx, cy = (list_landmark_points[0].x ), (list_landmark_points[0].y )
            print(cx, cy)
            cx, cy = (list_landmark_points[1].x), (list_landmark_points[1].y)
            print(cx, cy)


            # for id, lm in enumerate(handLms.landmark):
            #     print(id,lm)
            #     h,w,c = img.shape
            #     #chuyen toa do qua toa do pixel
            #     cx,cy = int(lm.x*w), int(lm.y*h)
            #     if id ==2:
            #         cv2.circle(img, (cx,cy), 25, (255,0,0), cv2.FILLED)
            #
            #     if id ==5:
            #         cv2.circle(img, (cx,cy), 25, (255,0,0), cv2.FILLED)
            #
            #     if id ==9:
            #         cv2.circle(img, (cx,cy), 25, (255,0,0), cv2.FILLED)
            #
            #     if id ==13:
            #         cv2.circle(img, (cx,cy), 25, (255,0,0), cv2.FILLED)
            #
            #     if id ==17:
            #         cv2.circle(img, (cx,cy), 25, (255,0,0), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
        print('Het 1 vong')

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,255), 3)


    cv2.imshow("Image", img)
    cv2.waitKey(1)

