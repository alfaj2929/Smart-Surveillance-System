import cv2, numpy as np
import mediapipe as mp
import math, scipy.spatial, time, subprocess
from playsound import playsound
# path to the alarm sound file
alarm_file = "/Users/srushtihirve/Downloads/Alarm-Fast-A1-www.fesliyanstudios.com.mp3"
# keep the value float
confidence_val = 0.7
# True for black Mask Mode
black_mask = False

# True for auto Dependencies install
Dependencies_check = False
if Dependencies_check:
    subprocess.call(['pip', 'install', "mediapipe"])
    subprocess.call(['pip', 'install', "numpy"])
    subprocess.call(['pip', 'install', "scipy"])
    subprocess.call(['pip', 'install', "opencv-python"])
# Drawing Utils
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_holistic = mp.solutions.holistic
mp_holistic.POSE_CONNECTIONS

# Calculate Perpendicular Distance
def shortest_distance(x1, y1, a, b, c):
    d = abs((a * x1 + b * y1 + c)) / (math.sqrt(a * a + b * b))
    return d
# Detect and print which finger touched the line
def break_code(finger):
    # print(finger+" Touched the line\nSystem shutting down")
    print('\x1b[6;30;101m' + finger + ' Touched the line, shutting down the system' + '\x1b[0m')
    # play the alarm sound for 5 seconds
    playsound(alarm_file, block=True)


# For webcam input:
cap = cv2.VideoCapture(0)
frameWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frameHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

# create a black overlay image. You can use any image | multiply with 255 form white image
foreground = np.zeros(shape=[int(frameHeight), int(frameWidth), 3], dtype=np.uint8)
foreground[:, :] = [0, 0, 0]

with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=confidence_val,
        min_tracking_confidence=confidence_val) as hands, mp_holistic.Holistic(
    min_detection_confidence=confidence_val,
    min_tracking_confidence=confidence_val) as holistic:
    while cap.isOpened():
        start = time.time()
        success, image = cap.read()
        _ , frame = cap.read()
        h, w, c = frame.shape
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = hands.process(image)
        results_holistic = holistic.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if black_mask:
            # Select the region in the background where we want to add the image and add the images using cv2.addWeighted()
            added_image = cv2.addWeighted(image[0:int(frameHeight), 0:int(frameWidth), :], 0.,
                                          foreground[0:int(frameHeight), 0:int(frameWidth), :], 1, 0)
            # Change the region with the result
            image[0:int(frameHeight), 0:int(frameWidth)] = added_image

        # Draw Safety Border Line x1,y1----x2,y2-------color--thickness
        line = cv2.line(image, (0, 150), (640, 150), (255, 0, 0), 4)
        # Draw Machine Border Line x1,y1----x2,y2-------color--thickness
        line = cv2.line(image, (0, 100), (640, 100), (0, 0, 255), 4)

        hand_landmarks = results.multi_hand_landmarks
        if results.multi_hand_landmarks:

            # draw box around hand
            for handLMs in hand_landmarks:
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h
                for lm in handLMs.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x > x_max:
                        x_max = x
                    if x < x_min:
                        x_min = x
                    if y > y_max:
                        y_max = y
                    if y < y_min:
                        y_min = y
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(
                    image,  # numpy array on which text is written
                    "Confidence: " + str(confidence_val),  # text
                    (x_min, y_max + 20),  # position at which writing has to start
                    cv2.FONT_HERSHEY_SIMPLEX,  # font family
                    0.6,  # font size
                    (5, 255, 5),  # font color
                    2)  # font stroke

                # Hand 1
            left_hand_landmarks1 = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            left_hand_pixel_coords1 = mp_drawing._normalized_to_pixel_coordinates(left_hand_landmarks1.x,
                                                                                  left_hand_landmarks1.y,
                                                                                  frameWidth,
                                                                                  frameHeight)
            # cv2.circle(image, left_hand_pixel_coords1, 2, (255,0,0), -1)
            try:
                if shortest_distance(left_hand_pixel_coords1[0], left_hand_pixel_coords1[1], 0, 640, -96000) < 10:
                    finger = 'Middle Finger'
                    break_code(finger)
                    break
            except:
                pass

            left_hand_landmarks2 = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            left_hand_pixel_coords2 = mp_drawing._normalized_to_pixel_coordinates(left_hand_landmarks2.x,
                                                                                  left_hand_landmarks2.y,
                                                                                  frameWidth,
                                                                                  frameHeight)
            # cv2.circle(image, left_hand_pixel_coords2, 2, (255,0,0), -1)
            try:
                if shortest_distance(left_hand_pixel_coords2[0], left_hand_pixel_coords2[1], 0, 640, -96000) < 10:
                    finger = 'Ring Finger'
                    break_code(finger)
                    break
            except:
                pass

            left_hand_landmarks3 = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.PINKY_TIP]
            left_hand_pixel_coords3 = mp_drawing._normalized_to_pixel_coordinates(left_hand_landmarks3.x,
                                                                                  left_hand_landmarks3.y,
                                                                                  frameWidth,
                                                                                  frameHeight)
            # cv2.circle(image, left_hand_pixel_coords3, 2, (255,0,0), -1)
            try:
                if shortest_distance(left_hand_pixel_coords3[0], left_hand_pixel_coords3[1], 0, 640, -96000) < 10:
                    finger = 'Pinky Finger'
                    break_code(finger)
                    break
            except:
                pass

            left_hand_landmarks4 = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            left_hand_pixel_coords4 = mp_drawing._normalized_to_pixel_coordinates(left_hand_landmarks4.x,
                                                                                  left_hand_landmarks4.y,
                                                                                  frameWidth,
                                                                                  frameHeight)
            # cv2.circle(image, left_hand_pixel_coords4, 2, (255,0,0), -1)
            try:
                if shortest_distance(left_hand_pixel_coords4[0], left_hand_pixel_coords4[1], 0, 640, -96000) < 10:
                    finger = 'Index Finger'
                    break_code(finger)
                    break
            except:
                pass

            left_hand_landmarks5 = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.THUMB_TIP]
            left_hand_pixel_coords5 = mp_drawing._normalized_to_pixel_coordinates(left_hand_landmarks5.x,
                                                                                  left_hand_landmarks5.y,
                                                                                  frameWidth,
                                                                                  frameHeight)
            # cv2.circle(image, left_hand_pixel_coords5, 2, (255,0,0), -1)
            try:
                if shortest_distance(left_hand_pixel_coords5[0], left_hand_pixel_coords5[1], 0, 640, -96000) < 10:
                    finger = 'Thumb'
                    break_code(finger)
                    break
            except:
                pass

            # Hand 2
            right_hand_landmarks1 = results.multi_hand_landmarks[-1].landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            right_hand_pixel_coords1 = mp_drawing._normalized_to_pixel_coordinates(right_hand_landmarks1.x,
                                                                                   right_hand_landmarks1.y,
                                                                                   frameWidth,
                                                                                   frameHeight)
            # cv2.circle(image, right_hand_pixel_coords1, 2, (255,0,0), -1)
            try:
                if shortest_distance(right_hand_pixel_coords1[0], right_hand_pixel_coords1[1], 0, 640, -96000) < 10:
                    finger = 'Middle Finger'
                    break_code(finger)
            except:
                pass

            right_hand_landmarks2 = results.multi_hand_landmarks[-1].landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            right_hand_pixel_coords2 = mp_drawing._normalized_to_pixel_coordinates(right_hand_landmarks2.x,
                                                                                   right_hand_landmarks2.y,
                                                                                   frameWidth,
                                                                                   frameHeight)
            # cv2.circle(image, right_hand_pixel_coords2, 2, (255,0,0), -1)
            try:
                if shortest_distance(right_hand_pixel_coords2[0], right_hand_pixel_coords2[1], 0, 640, -96000) < 10:
                    finger = 'Ring Finger'
                    break_code(finger)
                    break
            except:
                pass

            right_hand_landmarks3 = results.multi_hand_landmarks[-1].landmark[mp_hands.HandLandmark.PINKY_TIP]
            right_hand_pixel_coords3 = mp_drawing._normalized_to_pixel_coordinates(right_hand_landmarks3.x,
                                                                                   right_hand_landmarks3.y,
                                                                                   frameWidth,
                                                                                   frameHeight)
            # cv2.circle(image, right_hand_pixel_coords3, 2, (255,0,0), -1)
            try:
                if shortest_distance(right_hand_pixel_coords3[0], right_hand_pixel_coords3[1], 0, 640, -96000) < 10:
                    finger = 'Pinky Finger'
                    break_code(finger)
                    break
            except:
                pass

            right_hand_landmarks4 = results.multi_hand_landmarks[-1].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            right_hand_pixel_coords4 = mp_drawing._normalized_to_pixel_coordinates(right_hand_landmarks4.x,
                                                                                   right_hand_landmarks4.y,
                                                                                   frameWidth,
                                                                                   frameHeight)
            # cv2.circle(image, right_hand_pixel_coords4, 2, (255,0,0), -1)
            try:
                if shortest_distance(right_hand_pixel_coords4[0], right_hand_pixel_coords4[1], 0, 640, -96000) < 10:
                    finger = 'Index Finger'
                    break_code(finger)
                    break
            except:
                pass

            right_hand_landmarks5 = results.multi_hand_landmarks[-1].landmark[mp_hands.HandLandmark.THUMB_TIP]
            right_hand_pixel_coords5 = mp_drawing._normalized_to_pixel_coordinates(right_hand_landmarks5.x,
                                                                                   right_hand_landmarks5.y,
                                                                                   frameWidth,
                                                                                   frameHeight)
            # cv2.circle(image, right_hand_pixel_coords5, 2, (255,0,0), -1)
            try:
                if shortest_distance(right_hand_pixel_coords5[0], right_hand_pixel_coords5[1], 0, 640, -96000) < 10:
                    finger = 'Thumb'
                    break_code(finger)
                    break
            except:
                pass

        cv2.putText(
            image,  # numpy array on which text is written
            "Blue Line - Safety Border Line",  # text
            (10, 50),  # position at which writing has to start
            cv2.FONT_HERSHEY_SIMPLEX,  # font family
            0.5,  # font size
            (255, 0, 0),  # font color
            2)  # font stroke

        cv2.putText(
            image,  # numpy array on which text is written
            "Red Line - Machine Danger Line",  # text
            (10, 80),  # position at which writing has to start
            cv2.FONT_HERSHEY_SIMPLEX,  # font family
            0.5,  # font size
            (0, 0, 255),  # font color
            2)  # font stroke

        # Body Skeleton
        mp_drawing.draw_landmarks(
            image,
            results_holistic.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles
            .get_default_pose_landmarks_style())

        # Left Hand Connections
        mp_drawing.draw_landmarks(image,
                                  results_holistic.left_hand_landmarks,
                                  mp_holistic.HAND_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing_styles
                                  .get_default_pose_landmarks_style())

        # Right Hand Connections
        mp_drawing.draw_landmarks(image,
                                  results_holistic.right_hand_landmarks,
                                  mp_holistic.HAND_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing_styles
                                  .get_default_pose_landmarks_style())

        # print FPS
        end = time.time()
        totalTime = end - start
        fps = 1 / totalTime
        cv2.putText(image, f'FPS: {int(fps)}', (420, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('SafetyCAM', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

        # Toggle Mask Mode with Key 'M'
        if cv2.waitKey(33) == ord('m'):
            if not black_mask:
                black_mask = True
            else:
                black_mask = False

cap.release()
cv2.destroyAllWindows()