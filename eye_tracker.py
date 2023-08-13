import cv2 as cv
import mediapipe as mp
import time
import math
import utils
import numpy as np

class EyeTracker:
    # variables
    frame_counter =0
    CEF_COUNTER =0
    TOTAL_BLINKS =0
    # constants
    CLOSED_EYES_FRAME =3
    FONTS =cv.FONT_HERSHEY_COMPLEX

    # face bounder indices
    FACE_OVAL=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]

    # Left eyes indices
    LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]

    # right eyes indices
    RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]

    COUNTER = 0

    EYE_TRACKER_ESTIMATOR_CONSEC_FRAMES = 50


    map_face_mesh = mp.solutions.face_mesh
    
 
    # landmark detection function
    @classmethod
    def landmarksDetection(self,img, results, draw=False):
        img_height, img_width= img.shape[:2]
        # list[(x,y), (x,y)....]
        mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
        if draw :
            [cv.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]

        # returning the list of tuples for each landmarks
        return mesh_coord

    # Euclaidean distance
    @classmethod
    def euclaideanDistance(point, point1):
        x, y = point
        x1, y1 = point1
        distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
        return distance

    # Eyes Extrctor function,
    @classmethod
    def eyesExtractor(self,img, right_eye_coords, left_eye_coords):
        # converting color image to  scale image
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # getting the dimension of image
        dim = gray.shape

        # creating mask from gray scale dim
        mask = np.zeros(dim, dtype=np.uint8)

        # drawing Eyes Shape on mask with white color
        cv.fillPoly(mask, [np.array(right_eye_coords, dtype=np.int32)], 255)
        cv.fillPoly(mask, [np.array(left_eye_coords, dtype=np.int32)], 255)

        # showing the mask
        # cv.imshow('mask', mask)

        # draw eyes image on mask, where white shape is
        eyes = cv.bitwise_and(gray, gray, mask=mask)
        # change black color to gray other than eys
        cv.imshow('eyes draw', eyes)
        eyes[mask==0]=155

        # getting minium and maximum x and y  for right and left eyes
        # For Right Eye
        r_max_x = (max(right_eye_coords, key=lambda item: item[0]))[0]
        r_min_x = (min(right_eye_coords, key=lambda item: item[0]))[0]
        r_max_y = (max(right_eye_coords, key=lambda item : item[1]))[1]
        r_min_y = (min(right_eye_coords, key=lambda item: item[1]))[1]

        # For LEFT Eye
        l_max_x = (max(left_eye_coords, key=lambda item: item[0]))[0]
        l_min_x = (min(left_eye_coords, key=lambda item: item[0]))[0]
        l_max_y = (max(left_eye_coords, key=lambda item : item[1]))[1]
        l_min_y = (min(left_eye_coords, key=lambda item: item[1]))[1]

        # croping the eyes from mask
        cropped_right = eyes[r_min_y: r_max_y, r_min_x: r_max_x]
        cropped_left = eyes[l_min_y: l_max_y, l_min_x: l_max_x]

        # returning the cropped eyes
        return cropped_right, cropped_left

    # Eyes Position Estimator
    @classmethod
    def positionEstimator(cls,cropped_eye):
        # getting height and width of eye
        h, w =cropped_eye.shape

        # remove the noise from images
        gaussain_blur = cv.GaussianBlur(cropped_eye, (9,9),0)
        median_blur = cv.medianBlur(gaussain_blur, 3)

        # applying thrsholding to convert binary_image
        threshed_eye = cv.adaptiveThreshold(median_blur,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv.THRESH_BINARY,11,2)


        # create fixd part for eye with
        piece = int(w/3)

        # slicing the eyes into three parts
        right_piece = threshed_eye[0:h, 0:piece]
        center_piece = threshed_eye[0:h, piece: piece+piece]
        left_piece = threshed_eye[0:h, piece +piece:w]

        # calling pixel counter function
        eye_position, color = cls.pixelCounter(right_piece, center_piece, left_piece)

        return eye_position, color

    # creating pixel counter function
    @classmethod
    def pixelCounter(self,first_piece, second_piece, third_piece):
        # counting black pixel in each part
        right_part = np.sum(first_piece==0)
        center_part = np.sum(second_piece==0)
        left_part = np.sum(third_piece==0)
        # creating list of these values
        eye_parts = [right_part, center_part, left_part]

        # getting the index of max values in the list
        max_index = eye_parts.index(max(eye_parts))
        pos_eye =''
        if max_index==0:
            pos_eye="RIGHT"
            color=[utils.BLACK, utils.GREEN]
        elif max_index==1:
            pos_eye = 'CENTER'
            color = [utils.YELLOW, utils.PINK]
        elif max_index ==2:
            pos_eye = 'LEFT'
            color = [utils.GRAY, utils.YELLOW]
        else:
            pos_eye="Closed"
            color = [utils.GRAY, utils.YELLOW]
        return pos_eye, color


#     with map_face_mesh.FaceMesh(min_detection_confidence =0.5, min_tracking_confidence=0.5) as face_mesh:
    @classmethod
    def eyeTracker(cls,frame,results):
        frame_height, frame_width= frame.shape[:2]
        if results.multi_face_landmarks:
            mesh_coords = cls.landmarksDetection(frame, results)

            # cv.polylines(frame,  [np.array([mesh_coords[p] for p in cls.LEFT_EYE ], dtype=np.int32)], True, (0, 255, 0), 1, cv.LINE_AA)
            # cv.polylines(frame,  [np.array([mesh_coords[p] for p in cls.RIGHT_EYE ], dtype=np.int32)], True, (0, 255, 0), 1, cv.LINE_AA)

            # Blink Detector Counter Completed
            right_coords = [mesh_coords[p] for p in cls.RIGHT_EYE]
            left_coords = [mesh_coords[p] for p in cls.LEFT_EYE]
            crop_right, crop_left = cls.eyesExtractor(frame, right_coords, left_coords)
            # cv.imshow('right', crop_right)
            # cv.imshow('left', crop_left)
            eye_position, color =cls.positionEstimator(crop_right)
    #             cv.putText(frame, eye_position, (20, 20), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            eye_position_left, color = cls.positionEstimator(crop_left)
            if(eye_position != "CENTER" or eye_position_left != "CENTER"):
                cls.COUNTER += 1
                #If no. of frames is greater than threshold frames,
                if (cls.COUNTER >= cls.EYE_TRACKER_ESTIMATOR_CONSEC_FRAMES):
                    return True
            else:
                cls.COUNTER = 0
                return False
