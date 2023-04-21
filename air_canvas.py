import cv2
import numpy as np


vid = cv2.VideoCapture(0)

#            Red           Green          Blue
colors = [(0, 0, 214), (0, 255, 127), (126, 18, 0)]
color = None

canvas = np.zeros((471, 636, 4), dtype=np.uint8) + 150
canvas = cv2.resize(canvas, None, None, fx=1.5, fy=1.5)

i=1

while True:
    ret, frame = vid.read()
    frame = cv2.resize(frame, None, None, fx=1.5, fy=1.5)
    frame = cv2.flip(frame, 1)
    
    #We have to write this line before creating buttons so that buttons act virtual --> Otherwise they would hide 
    #       our object behind them --> Then we wouldn't be able to access the properties of buttons easily
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    kernel = np.ones((5, 5), dtype=np.uint8)
    
    
    #Color Buttons on Frame
    cv2.rectangle(frame, (20,1), (120,65), (122,122,122), -1)
    cv2.rectangle(frame, (140,1), (240,65), colors[0], -1)
    cv2.rectangle(frame, (260,1), (360,65), colors[1], -1)
    cv2.rectangle(frame, (380,1), (480,65), colors[2], -1)

    cv2.putText(frame, "CLEAR ALL", (30, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "RED", (175, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (285, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (410, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)


    """Working"""
    #Detecting Object --> Color = Blue
    # lower_bound = np.array([90, 80, 50])
    # upper_bound = np.array([130, 255, 255])

    """Working"""
    #Detecting Object --> Color = Red 
    lower_bound = np.array([170, 150, 50])
    upper_bound = np.array([179, 255, 255])
    

    mask = cv2.inRange(hsv, lower_bound, upper_bound)           #Finds object in a color range
    mask = cv2.erode(mask, kernel, iterations=1)                #Removes Extra Noises --> Detects extra if not applied
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)       #Makes the edges more smooth
    mask = cv2.dilate(mask, kernel, iterations=4)               #After erode object size got smaller -> Makes it big again
    
    
    #Finding Contours
    contours, h = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    

    #If Contour present --> Find the largest object(contour)
    if len(contours)>0:
        cmax = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cmax)
        
        #Find the center of the largest contour --> Make a circle around its center coordinates
        M = cv2.moments(cmax)                   # cv2.moments() finds center coordinates
        
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        if i==1:
            previous_point = (cx, cy)
            i+=1
    
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), 2)
        
        if cy < 65 and cx <= 490:
            if cx >= 20 and cx <= 120:
                canvas = np.zeros((471, 636, 4), dtype=np.uint8) + 150
                canvas = cv2.resize(canvas, None, None, fx=1.5, fy=1.5)
                color = None
            
            if cx >= 140 and cx <= 240:
                color = colors[0]
                previous_point = (cx, cy)
            
            if cx >= 260 and cx <= 360:
                color = colors[1]
                previous_point = (cx, cy)
            
            if cx >= 380 and cx <= 480:
                color = colors[2]
                previous_point = (cx, cy)
                
        else:
            if color:
                cv2.line(canvas, previous_point, (cx, cy), color, 5)
                previous_point = (cx, cy)


    cv2.imshow('Canvas', canvas)
    cv2.imshow('Mask', mask)
    cv2.imshow('Camera', frame)
    
    if cv2.waitKey(1) == ord('q'):
        break
    

vid.release()
cv2.destroyAllWindows()