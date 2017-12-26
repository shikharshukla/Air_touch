from collections import deque  #to give dynamic buffer application 
import numpy as np             #for matric manipulation
import argparse                #to channel command line 
import imutils                 #tweaking display
import cv2                     #main library
import pyautogui               #to give cursor access and control 
pyautogui.FAILSAFE = False     #to override trackpad input

parser = argparse.ArgumentParser()      #object - to get arguments from cmd
parser.add_argument("-v", "--video", help="path to the (optional) video file") #get webcam access and video
parser.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size") #store in buffer
args = vars(parser.parse_args())    #all arguments stored in args variable

Lower = (29, 86, 6)                 #range of color domain set in (B,G,R) specific to hsv
Upper = (64, 255, 255)
pts = deque(maxlen=args["buffer"])  #from DS buffer to our deque 


if not args.get("video", False):    #video feed
    camera = cv2.VideoCapture(0)

else:
    camera = cv2.VideoCapture(args["video"])

while True: #while camera is ON following code runs in loop
    
    (grabbed, frame) = camera.read()  #Camera ON

    if args.get("video") and not grabbed:
        break

    frame = imutils.resize(frame, width=800) #display frame resized from default
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #BGR to hsv format

    
    mask = cv2.inRange(hsv, Lower, Upper)       #masking to detect color and filter noises
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    cv2.imshow('mask',mask)
    
    con = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,#finds contours 
        cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None
    x = 0
    y = 0
    x0 = 0
    y0 = 0
    # only proceed if at least one contour was found
    if len(con) > 0:
        
        c = max(con, key=cv2.contourArea)   # finds max size contour and its min distribution radius
        ((x, y), radius) = cv2.minEnclosingCircle(c)  # CORE STEP ->gives radius(z),x,y for contour
        M = cv2.moments(c)      #finds centroid of distribution
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))   #typecasting

        
        if radius > 10:     #draw radius
            
            cv2.circle(frame, (int(x), int(y)), int(radius),
                (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

    
    pts.appendleft(center)  #to shift data in deque

    # loop over the set of tracked points
    for i in xrange(1, len(pts)):
        
        if pts[i - 1] is None or pts[i] is None:
            continue

        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    # cursor commands according to screen size
    if(radius < 30):
        pyautogui.moveTo(1366-2*x, 2*y) #scaling
    else:
        x0=1366-(2*x)   #to lock coordinates (helps usr understand that he is on verge to click)
        y0=2*y
    
      
    if(radius > 40):
        
        pyautogui.click(x0, y0) #click function
     
    cmd = str(radius)
    cv2.putText(frame,cmd, (0,20), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)#display radius
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()