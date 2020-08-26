import cv2
import numpy as np

def nothing(x):
    pass

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Usamos HSV para saturar uma cor

cap = cv2.VideoCapture(0)  

cv2.namedWindow('trackbars')
cv2.createTrackbar('th','trackbars',3,255,nothing)

while True:

    # ================= trackbars ====================== #
    th = cv2.getTrackbarPos('th','trackbars')

    _, frame = cap.read()

    # ================= tira rosto ======================#
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w + 10,y+h+30),(0,0,0),-1)

    # ===================================================#
    
    blur = cv2.blur(frame,(th,th))

    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array([2,50,50])
    upper = np.array([15,255,255])
    mask = cv2.inRange(hsv, lower, upper)
  
    # Kernel 
    kernel = np.ones((5, 5), np.uint8)  
    dilation = cv2.dilate(mask,kernel,iterations = 2)
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
    filtered = cv2.medianBlur(closing,5)
    ret,thresh = cv2.threshold(filtered,127,255,0)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # União da mascara
    #final = cv2.bitwise_and(frame, frame, mask=filtered)

    if len(contours) == 0:
        continue

    max_area=100
    ci=0	
    for i in range(len(contours)):
        cnt=contours[i]
        area = cv2.contourArea(cnt)
        if(area>max_area):
            max_area=area
            ci=i  

    cnts = contours[ci]

    hull = cv2.convexHull(cnts)

    hull2 = cv2.convexHull(cnts, returnPoints = False)
    defects = cv2.convexityDefects(cnts,hull2)

    FarDefect = []

    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(cnts[s][0])
        end = tuple(cnts[e][0])
        far = tuple(cnts[f][0])
        FarDefect.append(far)
        cv2.line(frame,start,end,[0,255,0],1)
        cv2.circle(frame,far,10,[100,255,255],3)

    # Mostrando
    cv2.drawContours(frame, contours, -1, (122,122,0), 3)
    cv2.imshow("detector", frame)

    key = cv2.waitKey(1)
    
    if key == 27:
        break

cap.realease()
cap.destroyAllWindows()