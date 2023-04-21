import cv2
import numpy as np
import mediapipe as mp


vid = cv2.VideoCapture(0)
x1 = 562
y1 = 362
x2 = 610
y2 = 400

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

if not (vid.isOpened()):
    print("Could not open") 

while vid.isOpened():
    # Read video
    ret, frame = vid.read()
    
    #Invert video frames
    frame = cv2.flip(frame, 1)
    
    #Convert to rgb --> mediapipe works with RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    #Detect Hands in rgb frame
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                
                if id == 1:
                    initial_range_x = lm.x
                    initial_range_y = lm.y
                    
                if id == 20:
                    end_range_x = lm.x
                    end_range_y = lm.y
                
                try:
                    # Condition for x 
                    if (end_range_x - initial_range_x) > (end_range_y - initial_range_y):
                        if cx>=300 and cx<370:
                            window = np.zeros((800, 1200), dtype=np.uint8)
                            new_window = cv2.rectangle(window, (x1, y1), (x2, y2), (255, 255, 255), thickness=-5)
                            cv2.imshow('Object Window', new_window)
                            
                        
                        if cx < 300:
                            new_x1 = int(x1 - ((x1-(cx+250))*1.5))
                            new_x2 = int(x2 - ((x1-(cx+250))*1.5))

                            window = np.zeros((800, 1200), dtype=np.uint8)
                            new_window = cv2.rectangle(window, (new_x1, y1), (new_x2, y2), (255, 255, 255), thickness=-5)
                            cv2.imshow('Object Window', new_window)

                        if cx >= 370:
                            new_x1 = int(x1 + ((cx-350)*1.5))
                            new_x2 = int(x2 + ((cx-350)*1.5))

                            window = np.zeros((800, 1200), dtype=np.uint8)
                            new_window = cv2.rectangle(window, (new_x1, y1), (new_x2, y2), (255, 255, 255), thickness=-2)
                            cv2.imshow('Object Window', new_window)
                            
                    # Condition for y 
                    if (end_range_y - initial_range_y) > (end_range_x - initial_range_x):
                        if cy>=240 and cy<290:
                            window = np.zeros((800, 1200), dtype=np.uint8)
                            new_window = cv2.rectangle(window, (x1, y1), (x2, y2), (255, 255, 255), thickness=-5)
                            cv2.imshow('Object Window', new_window)
                            
                        
                        if cy < 240:
                            new_y1 = int(((y1/2) - ((y1-(cy+100))))*2)
                            new_y2 = int(((y2/2) - ((y1-(cy+100))))*2)

                            window = np.zeros((800, 1200), dtype=np.uint8)
                            new_window = cv2.rectangle(window, (x1, new_y1), (x2, new_y2), (255, 255, 255), thickness=-5)
                            cv2.imshow('Object Window', new_window)

                        if cy >= 290:
                            new_y1 = int(y1 + ((cy-250)*1.5))
                            new_y2 = int(y2 + ((cy-250)*1.5))

                            window = np.zeros((800, 1200), dtype=np.uint8)
                            new_window = cv2.rectangle(window, (x1, new_y1), (x2, new_y2), (255, 255, 255), thickness=-2)
                            cv2.imshow('Object Window', new_window)
                            
                except:
                    pass

    else:
        window = np.zeros((800, 1200), dtype=np.uint8)
        new_window = cv2.rectangle(window, (x1, y1), (x2, y2), (255, 255, 255), thickness=-5)
        cv2.imshow('Object Window', new_window)
    
    cv2.imshow('Camera', frame)
    
    if cv2.waitKey(1) == ord('q'):
        break


vid.release()
cv2.destroyAllWindows()
