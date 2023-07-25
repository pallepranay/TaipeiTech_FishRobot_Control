import cv2
import mediapipe as mp
import pandas as pd  
import os
import numpy as np 
import time

fishnumbers=["You_Calling_Fish-1","You_Calling_Fish-2","You_Calling_Fish-3","You_Calling_Fish-4","Call_The_Fish"]

def image_processed(hand_img):

    # Image processing
    # 1. Convert BGR to RGB
    img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)

    # 2. Flip the img in Y-axis
    img_flip = cv2.flip(img_rgb, 1)

    # accessing MediaPipe solutions
    mp_hands = mp.solutions.hands

    # Initialize Hands
    hands = mp_hands.Hands(static_image_mode=True,
    max_num_hands=1, min_detection_confidence=0.7)

    # Results
    output = hands.process(img_flip)

    hands.close()
    try:
        data = output.multi_hand_landmarks[0]
        #print(data)
        data = str(data)

        data = data.strip().split('\n')

        garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']

        without_garbage = []

        for i in data:
            if i not in garbage:
                without_garbage.append(i)
                        
        clean = []

        for i in without_garbage:
            i = i.strip()
            clean.append(i[2:])

        for i in range(0, len(clean)):
            clean[i] = float(clean[i])
        return(clean)
    except:
        return(np.zeros([1,63], dtype=int)[0])

def draw_label(img, text, pos, bg_color, font_scale, thickness):
    font = cv2.FONT_HERSHEY_DUPLEX
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    x, y = pos
    cv2.rectangle(img, (x - 5, y - text_size[1] - 5), (x + text_size[0] + 5, y + 5), bg_color, -1)
    cv2.putText(img, text, pos, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

def put_timer(frame, seconds_left):
    # Define timer properties
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 2
    font_thickness = 3
    text = f"Time Left: {seconds_left}"
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)

    # Calculate timer position
    frame_height, frame_width, _ = frame.shape
    timer_x = int((frame_width - text_size[0]) / 2)
    timer_y = int(frame_height / 2)

    # Draw timer text on frame
    cv2.putText(frame, text, (timer_x, timer_y), font, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)

    # Blink timer by alternating between hiding and showing the text
    if seconds_left % 2 == 0:
        cv2.putText(frame, text, (timer_x, timer_y), font, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)


# Function to create a circular mask
def create_circle_mask(image_shape, center, radius):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    return mask


import pickle
# load model
with open('model.pkl', 'rb') as f:
    svm = pickle.load(f)


import cv2 as cv
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
i = 0    
fishnumber=-1
prev="null"

while True:
       
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # frame = cv.flip(frame,1)
    data = image_processed(frame)
    
    # print(data.shape)
    data = np.array(data)
    y_pred = svm.predict(data.reshape(-1, 63))
    # Delay for  seconds
    # Check if y_pred is changed
    if y_pred != prev:
        start_time = time.time()  # Reset the timer
        prev = y_pred
            
    if(y_pred == "Start_Control"):
        print("Hello! I'm Robotic Fish, You can start giving me command")
        while True:
            rett, framee = cap.read()
            if not rett:
                print("Can't receive framee (stream end?). Exiting ...")
                break
            # framee = cv.flip(frame,1)
            data = image_processed(framee)
            
            # print(data.shape)
            data = np.array(data)
            y_pred = svm.predict(data.reshape(-1, 63))
            # Delay for 2 seconds
            if time.time() - start_time >= 20 and y_pred==prev and y_pred=="A_Gesture_NotGiven":
                y_pred = "Good_Bye"
                prev = y_pred
                print(y_pred)

            elif y_pred!=prev and y_pred=="Call_The_Fish" and prev=="Start_Control" or prev=="Good_Bye":
                print("You are calling all fishes")
                start_time = time.time()
                prev = y_pred
                fishnumber=-2
        
            elif y_pred!=prev and y_pred=="You_Calling_Fish-1" and prev=="Start_Control" or prev=="Good_Bye":
                print("You are calling Fish-1")
                # Check if y_pred is changed
                start_time = time.time()  # Reset the timer
                prev = y_pred
                # Check if y_pred remains unchanged for 15 seconds
                fishnumber=1
            elif y_pred!=prev and y_pred=="You_Calling_Fish-2" and prev=="Start_Control" or prev=="Good_Bye":
                print("You are calling Fish-2")
                start_time = time.time()  # Reset the timer
                prev = y_pred
                # Check if y_pred remains unchanged for 15 seconds
                fishnumber=2
            elif y_pred!=prev and y_pred=="You_Calling_Fish-3" and prev=="Start_Control" or prev=="Good_Bye":
                print("You are c01alling Fish-3")
                start_time = time.time()  # Reset the timer
                prev = y_pred
                # Check if y_pred remains unchanged for 15 seconds
                fishnumber=3
            elif y_pred!=prev and y_pred=="You_Calling_Fish-4" and prev=="Start_Control" or prev=="Good_Bye":
                print("You are calling Fish-4")
                start_time = time.time()  # Reset the timer
                prev = y_pred
                # Check if y_pred remains unchanged for 15 seconds
                fishnumber=4

            elif time.time() - start_time > 3.5 and y_pred!=prev and fishnumber!=-1 and y_pred not in fishnumbers and y_pred!="A_Gesture_NotGiven":
                print(y_pred)
                start_time = time.time()  # Reset the timer
                prev = y_pred
                
Q
            # font
            font = cv2.FONT_HERSHEY_DUPLEX
            
            # org
            org = (50, 100)
            
            # fontScale
            fontScale = 1
            
            # Blue color in BGR
            color = (255, 0, 0)
            
            # Line thickness of 2 px
            thickness = 3

            # Your other code ...

            # Update the text position and background color
            text_position = (50, 100)
            background_color = (0, 0, 255)

            # Load fish images
            fish_off = cv2.imread('closedeye_final.jpg', 1)
            fish_on = cv2.imread('opendeye_final.jpg', 1)

            # Command variable
            command = ''

            # Update the text content and fish image
            if time.time() - start_time < 3.5:
                if y_pred[0] == 'A_Gesture_NotGiven':
                    y_pred[0] = ''
                text_content = str(y_pred[0]) + " wait " + str(time.time() - start_time)[:3] + " " + command
                fish_image = fish_off
            else:
                text_content = " Give Command"
                fish_image = fish_on
                command = y_pred[0]

            # Calculate the fish size to make it fit well on the screen
            fish_size = min(framee.shape[1], framee.shape[0]) // 5

            # Check if the fish_image is not None before resizing
            if fish_image is not None:
                fish_image = cv2.resize(fish_image, (fish_size, fish_size))

            # Define the position to place the fish_image (top-left corner)
            fish_position = (10, 10)

            # Copy the frame to preserve the original
            frame_with_fish = framee.copy()

            # Overlay the fish_image onto the frame in the top-left corner
            if fish_image is not None:
                frame_with_fish[fish_position[1]:fish_position[1] + fish_image.shape[0],
                                fish_position[0]:fish_position[0] + fish_image.shape[1]] = fish_image

            # Customize the text appearance
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            text_color = (255, 255, 255)  # White color (BGR format)

            # Get the size of the text box to center it on the frame
            text_box_size = cv2.getTextSize(text_content, font, font_scale, 2)[0]
            text_x = (frame_with_fish.shape[1] - text_box_size[0]) // 2
            text_y = (frame_with_fish.shape[0] + text_box_size[1]) // 2

            # Draw the text with the background
            cv2.putText(frame_with_fish, text_content, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)

            # Show the frame
            cv2.imshow("Robotic Fish - Prof. Leether Yao", frame_with_fish)
            cv2.resizeWindow("Robotic Fish - Prof. Leether Yao", 640, 480)


            if cv.waitKey(1) == ord('q'):
                break

            if (y_pred == "Good_Bye"):
                print("Good Bye! See you soon...")
                fishnumber = -1
                break

    # font
    font = cv2.FONT_HERSHEY_DUPLEX
    
    # org
    org = (50, 100)
    
    # fontScale
    fontScale = 1
    
    # Blue color in BGR
    color = (255, 0, 0)
    
    # Line thickness of 2 px
    thickness = 3
    
    # Update the text position and background color
    text_position = (50, 100)
    background_color = (0, 0, 255)

    # Draw the text with the background
    draw_label(frame, str(y_pred[0]), text_position, background_color, fontScale, thickness)

    # Show the frame
    cv.imshow('Robotic Fish - Prof. Leether Yao', frame)
    cv.resizeWindow("Robotic Fish - Prof. Leether Yao", 600, 600)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()