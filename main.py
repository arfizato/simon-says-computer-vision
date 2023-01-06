import cv2 as cv
import mediapipe as mp
import time
cap = cv.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

ptime=0
ctime=0


while True: 
    success, img = cap.read()
    rgbImg= cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(rgbImg)
    h,w,c = img.shape

    if results.multi_hand_landmarks:
        # print(results.multi_hand_landmarks)
        for oneHand in results.multi_hand_landmarks:
            lmList=list()
            for id,lm in enumerate(oneHand.landmark):
                cx,cy = int(lm.x*w), int(lm.y*h)
                lmList.append({"id":id,"x":cx,"y":cy,"dx":lm.x,"dy":lm.y})


            if lmList[0]["dy"] < 0.7:
                f1=lmList[4]
                f2=lmList[8]

                cv.circle(img,(f1["x"],f1["y"]), 15,(255,255,0), cv.FILLED)
                cv.putText(img,f"{f1['x']},{f1['y']}",(f1['x']+15,f1['y']), cv.FONT_HERSHEY_DUPLEX ,.5,(0,255,200),1)
                cv.circle(img,(f2["x"],f2["y"]), 15,(255,150,0), cv.FILLED)
                cv.putText(img,f"{f2['x']},{f2['y']}",(f2['x']+15,f2['y']), cv.FONT_HERSHEY_DUPLEX ,.5,(0,255,200),1)
                
                if abs(f1['x']-f2['x'])+ abs(f1['y']-f2['y'])<45:
                    print("SUCCESS")
                
                # if (id== 0 and lm.y <0.5):
                #     cx,cy = int(lm.x*w), int(lm.y*h)
                #     print(img.shape)


            mpDraw.draw_landmarks(img, oneHand, mpHands.HAND_CONNECTIONS)

    ctime=time.time()
    fps= 1/(ctime-ptime)
    ptime= ctime

    cv.putText(img,str(int(fps)),(10,70), cv.FONT_HERSHEY_DUPLEX ,3,(0,255,200),2)

    cv.imshow("Image",img)
    if cv.waitKey(1)>-1:
        break

cv.destroyWindow("Image")