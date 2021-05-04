import numpy as np
import cv2
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

model = tf.keras.models.load_model(r"asl_93percent_model.h5")

background = None
accumulated_weight = 0.5

ROI_top = 100
ROI_bottom = 300
ROI_right = 150
ROI_left = 350

def cal_accum_avg(frame, accumulated_weight):

    global background

    if background is None:
        background = frame.copy().astype("float")
        return None

    cv2.accumulateWeighted(frame, background, accumulated_weight)

def segment_hand(frame, threshold=25):
    global background

    diff = cv2.absdiff(background.astype("uint8"), frame)


    _ , thresholded = cv2.threshold(diff, threshold, 255,
    cv2.THRESH_BINARY)

     #Fetching contours in the frame (These contours can be of hand or any other object in foreground) …

    contours, hierarchy =cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)

    # If length of contours list = 0, means we didn't get any contours...
    if len(contours) == 0:
        return None
    else:
        # The largest external contour should be the hand
        hand_segment_max_cont = max(contours, key=cv2.contourArea)

        # Returning the hand segment(max contour) and the thresholded image of hand...
        return (thresholded, hand_segment_max_cont)

word_dict = {0: 'a',1: 'b',2: 'c',3: 'd',4: 'e',5: 'f',6: 'g',7: 'h',8: 'i',9: 'j',10: 'k',11: 'l',
12: 'm',13: 'n',14: 'o',15: 'p',16: 'q',17: 'r',18: 's',19: 't',20: 'u',21: 'v',22: 'w',23: 'x',24: 'y',25: 'z'}

import cv2
import streamlit as st

st.title("American Sign Language Recognition")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
FRAME_WINDOW2= st.image([])
cam = cv2.VideoCapture(0)
num_frames =0

res, score = '', 0.0
i = 0
mem = ''
consecutive = 0
sequence = ''

while run:
    ret, frame = cam.read()

    # flipping the frame to prevent inverted image of captured frame...

    frame = cv2.flip(frame, 1)

    frame_copy = frame.copy()

    # ROI from the frame
    roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]

    gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)


    if num_frames < 70:

        cal_accum_avg(gray_frame, accumulated_weight)

        cv2.putText(frame_copy, "FETCHING BACKGROUND...PLEASE WAIT",(80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

    else:
        # segmenting the hand region
        hand = segment_hand(gray_frame)

        # Checking if we are able to detect the hand...
        if hand is not None:

            thresholded, hand_segment = hand

            # Drawing contours around hand segment
            cv2.drawContours(frame_copy, [hand_segment + (ROI_right,ROI_top)], -1, (255, 0, 0),1)

            cv2.imshow("Thesholded Hand Image", thresholded)

            thresholded = cv2.resize(thresholded, (64, 64))
            thresholded = cv2.cvtColor(thresholded,cv2.COLOR_GRAY2RGB)
            thresholded = np.reshape(thresholded,(1,thresholded.shape[0],thresholded.shape[1],3))

            pred = model.predict(thresholded)
            cv2.putText(frame_copy, "Predicted label: "+ word_dict[np.argmax(pred)].upper(),(170, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            if i == 4:
                #res_tmp, score = predict(image_data)
                res = word_dict[np.argmax(pred)].upper()
                i = 0
                if mem == res:
                    consecutive += 1
                else:
                    consecutive = 0
                if consecutive == 2 and res not in ['nothing']:
                    if res == 'S':
                        sequence += ' '
                    elif res == 'D':
                        sequence = sequence[:-1]
                    else:
                        sequence = sequence + res
                    consecutive = 0
            i += 1
            cv2.putText(frame_copy, '%s' % (res.upper()), (100,400), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 4)
            #cv2.putText(frame_copy, '(score = %.5f)' % (float(score)), (100,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
            mem = res
            #cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (255,0,0), 2)
            #cv2.imshow("img", frame_copy)
            img_sequence = np.zeros((200,1200,3), np.uint8)
            cv2.putText(frame_copy, "Predicted sequence: " +'%s' % (sequence.upper()), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            #cv2.imshow('sequence', img_sequence)


    # Draw ROI on frame_copy
    cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right,
    ROI_bottom), (255,128,0), 3)

    # incrementing the number of frames for tracking
    num_frames += 1

    # Display the frame with segmented hand
    cv2.putText(frame_copy, "American Sign recognition",
    (10, 20), cv2.FONT_ITALIC, 0.5, (51,255,51), 1)
    cv2.imshow("Sign Detection", frame_copy)
    #FRAME_WINDOW.image(frame)
    FRAME_WINDOW2.image(frame_copy)
    #pred = model.predict(thresholded)
    #st.write("Predicted label: "+ word_dict[np.argmax(pred)].upper())



    # Close windows with Esc
    k = cv2.waitKey(1) & 0xFF

    if k == 27:
        break

else:
    st.write('Stopped')

# Release the camera and destroy all the windows
#cam.release()
#cv2.destroyAllWindows()
