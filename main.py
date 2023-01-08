import cv2 as cv
import mediapipe as mp
import time
import random
from threading import Timer # https://stackoverflow.com/a/3433565/18027442

class HandCam:
    def __init__(self,staticMode=False,maxHands=2,detectionConfidence=0.5,trackingConfidence=0.5):
        self.staticMode= staticMode
        self.maxHands= maxHands
        self.detectionConfidence= detectionConfidence
        self.trackingConfidence= trackingConfidence
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()#self.staticMode, self.maxHands, self.detectionConfidence, self.trackingConfidence)
        self.mpDraw = mp.solutions.drawing_utils
        
        self.gestures=[
            {"name":"noHands",  "img":r"./img/empty.png",   "type":"standard",  "fingersUp":[4,8,12,16,20],     "fingersColliding":[] },
            {"name":"1",        "img":r"./img/1.png",       "type":"standard",  "fingersUp":[8],                "fingersColliding":[4,12,16,20]},
            {"name":"2",        "img":r"./img/2.png",       "type":"standard",  "fingersUp":[8,12],             "fingersColliding":[4,16,20]},
            {"name":"3",        "img":r"./img/3.png",       "type":"standard",  "fingersUp":[8,12,16],          "fingersColliding":[4,20]},
            {"name":"4",        "img":r"./img/4.png",       "type":"standard",  "fingersUp":[8,12,16,20],       "fingersColliding":[4,13]},
            {"name":"5",        "img":r"./img/5.png",       "type":"standard",  "fingersUp":[4,8,12,16,20],     "fingersColliding":[4,5]},
            {"name":"ok",       "img":r"./img/ok.png",      "type":"standard",  "fingersUp":[12,16,20],         "fingersColliding":[4,8]},
            {"name":"rock",     "img":r"./img/rock.png",    "type":"standard",  "fingersUp":[8,20],             "fingersColliding":[4,12,16]},
            {"name":"gangang",  "img":r"./img/gangang.png", "type":"standard",  "fingersUp":[8,12,20],          "fingersColliding":[4,16]}
        ]
        self.selectedGesture= 0
        self.imageIconPath= r"./img/empty.png"
        self.waitUntil=0

    def findHands(self, img, draw = True):
        def callback(value):
            print("fdqkjqsd")
            pass
        rgbImg= cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(rgbImg)

        # cv.createTrackbar("1:on\n0:off", 'Image',0,1, callback)
        if self.results.multi_hand_landmarks:
            # print(results.multi_hand_landmarks)
            for oneHand in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, oneHand, self.mpHands.HAND_CONNECTIONS)
        return img

    def waitToSelectGesture(self):
        self.selectedGesture= random.randint(1,len(self.gestures)-1)
        self.imageIconPath= self.gestures[self.selectedGesture]["img"]

    def verifyGesture(self,img, lmList, draw=True):
        g= self.gestures[self.selectedGesture]
        # ------------------------- fingers standing up right ------------------------ #
        posIsCorrect=True
        for a in g["fingersUp"]:
            xfinger= [b["x"] for b in lmList[a-3:a+1]]
            yfinger= [b["y"] for b in lmList[a-3:a+1]]
            if min(yfinger)!=yfinger[-1] or min(yfinger[:-1])!=yfinger[-2] or min(yfinger[:-2])!=yfinger[-3]:# or max(yfinger[:-1])!=yfinger[-2] 
                for node in lmList[a-3:a+1]:
                    if draw:
                        cv.putText(img,f"{node['y']}",(node['x']+15,node['y']-10), cv.FONT_HERSHEY_DUPLEX ,.3,(255,255,0),1)
                posIsCorrect=False
                break
        # ----------------------------- fingers colliding ---------------------------- #
        collidingfiners=[lmList[a] for a in g["fingersColliding"]]
        fingersAreColliding= True
        for i,finger in enumerate( [yy for yy in collidingfiners]):
            fingerDoesCollide=False
            if (time.time())> self.waitUntil+.5:
                for ii,fingerTip in enumerate(collidingfiners):
                    if abs(finger["x"] - fingerTip["x"])<=15 and abs(finger["y"] - fingerTip["y"])<=15 and i!=ii: 
                        fingerDoesCollide= True
                        break
            if not fingerDoesCollide:
                fingersAreColliding=False
                break
        # ----------------------------- checking booleans ---------------------------- #
        if fingersAreColliding and (time.time())> self.waitUntil+.5:
            if posIsCorrect:    
                print(f"SUCCESS Gesture[{self.selectedGesture}]: {self.gestures[self.selectedGesture]['name']} " )
                self.imageIconPath=r"./img/empty.png"
                # time.sleep(2)
                self.waitUntil= int(time.time())+1
                t= Timer(1,self.waitToSelectGesture)
                t.start()
            else: 
                print(a,min(yfinger),yfinger[-1], yfinger)
        return img

    def findPos(self,img,draw=True):
        h,w,c = img.shape
        if self.results.multi_hand_landmarks:
            oneHand = self.results.multi_hand_landmarks[0]

            lmList=list()
            for id,lm in enumerate(oneHand.landmark):
                cx,cy = int(lm.x*w), int(lm.y*h)
                lmList.append({"id":id,"x":cx,"y":cy,"dx":lm.x,"dy":lm.y})
                if id in [4,8,12,16,20] and draw:
                    cv.circle(img,(cx,cy), 15,(255,id*10,0), cv.FILLED)
                    cv.putText(img,f"{cx},{cy}",(cx+15,cy-10), cv.FONT_HERSHEY_DUPLEX ,.5,(0,0,255),1)
            if draw:
                cv.putText(img,f"Hand {self.results.multi_hand_landmarks.index(oneHand)}",(lmList[0]['x']+15,lmList[0]['y']-10), cv.FONT_HERSHEY_DUPLEX ,.5,(0,255,255),1)
            img = self.verifyGesture(img,lmList,draw)
        return img

                    
    def setImage(self,img):
        s_img = cv.imread(self.imageIconPath, -1)
        x_offset= img.shape[1]-128
        y_offset = 0
        y1, y2 = y_offset, y_offset + s_img.shape[0]
        x1, x2 = x_offset, x_offset + s_img.shape[1]    
        alpha_s = s_img[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s # https://stackoverflow.com/a/14102014/18027442

        for c in range(0, 3):
            img[y1:y2, x1:x2, c] = (alpha_s * s_img[:, :, c] + alpha_l * img[y1:y2, x1:x2, c])

        return img


def main():
    cap = cv.VideoCapture(0)


    ptime=0
    ctime=0
    
    handcam =  HandCam()
    while True: 
        success, img = cap.read()
        img= cv.flip(img,1)
        img= handcam.findHands(img)
        img= handcam.findPos(img)
        img=handcam.setImage(img)
        

        ctime=time.time()
        fps= 1/(ctime-ptime)
        ptime= ctime

        cv.putText(img,str(int(fps)),(10,30), cv.FONT_HERSHEY_DUPLEX ,1,(0,255,255),2)
        cv.putText(img,"Give me a high five to start",(200,20), cv.FONT_HERSHEY_DUPLEX ,.5,(50,50,0),1,bottomLeftOrigin=False)
        cv.putText(img,"Press any key to close",(230,40), cv.FONT_HERSHEY_DUPLEX ,.5,(50,50,0),1,bottomLeftOrigin=False)
        cv.imshow("Image",img)
        if cv.waitKey(1)>-1:
            break
    cv.destroyWindow("Image")

if __name__=="__main__":
    main()

