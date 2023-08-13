import cv2
import mediapipe as mp
import numpy as np
import pygame  # For playing sound
import time
from drowsiness_detector import DrowsinessDetector
from eye_tracker import EyeTracker

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

POSE_ESTIMATOR_CONSEC_FRAMES = 100
start = time.time()
COUNTER = 0

# Initialize the alarm
pygame.mixer.init()
pygame.mixer.music.load('assets/audio/alert.mp3')

cap = cv2.VideoCapture(0);

while cap.isOpened():
    success, frame = cap.read()
    start = time.time()
    # Flip the frame horizontally and convert the color space from BGR to RGB
    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

    # To improve performance
    frame.flags.writeable = False

    # Get the result
    results = face_mesh.process(frame)

    # To improve performance
    frame.flags.writeable = True

    # Convert the color space from RGB to BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = frame.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])

            # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])

            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            # See where the user's head tilting
            if y < -10:
                COUNTER += 1
                # If no. of frames is greater than threshold frames,
                if (COUNTER >= POSE_ESTIMATOR_CONSEC_FRAMES):
                    # return True
                    cv2.putText(frame, "Looking: Left", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    pygame.mixer.music.play(-1)
            elif y > 10:
                COUNTER += 1
                # If no. of frames is greater than threshold frames,
                if (COUNTER >= POSE_ESTIMATOR_CONSEC_FRAMES):
                    # return True
                    cv2.putText(frame, "Looking: Right", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    pygame.mixer.music.play(-1)

            elif x < -10:
                COUNTER += 1
                # If no. of frames is greater than threshold frames,
                if (COUNTER >= POSE_ESTIMATOR_CONSEC_FRAMES):
                    # return True
                    cv2.putText(frame, "Looking: Down", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    pygame.mixer.music.play(-1)
            elif x > 10:
                COUNTER += 1
                # If no. of frames is greater than threshold frames,
                if (COUNTER >= POSE_ESTIMATOR_CONSEC_FRAMES):
                    # return True
                    cv2.putText(frame, "Looking: Up", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    pygame.mixer.music.play(-1)
            else:
                # If user is drowsy,
                if (DrowsinessDetector.drowsiness(frame)):
                    # return True
                    cv2.putText(frame, "You are Drowsy", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    pygame.mixer.music.play(-1)

                # If user's eyes are not focused
                elif (EyeTracker.eyeTracker(frame, results)):
                    # return True
                    cv2.putText(frame, "Inattentive", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    pygame.mixer.music.play(-1)

                else:
                    COUNTER = 0
                    # return False
                    pygame.mixer.music.stop()
            # Display the nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix,
                                                             dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

            cv2.line(frame, p1, p2, (255, 0, 0), 3)
    else:
        COUNTER += 1
        if (COUNTER >= POSE_ESTIMATOR_CONSEC_FRAMES):
            cv2.putText(frame, "Inattentive", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            pygame.mixer.music.play(-1)

    end = time.time()
    totalTime = end - start
    fps = 1 / totalTime

    cv2.imshow('Student Engagement Monitoring System', frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

    if cv2.getWindowProperty('Student Engagement Monitoring System', cv2.WND_PROP_VISIBLE) < 1:
        break

cv2.destroyAllWindows()
cap.release()

