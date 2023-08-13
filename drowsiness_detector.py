import cv2
import dlib
from scipy.spatial import distance
from imutils import face_utils

class DrowsinessDetector:
     #Minimum threshold of eye aspect ratio below which alarm is triggerd
    EYE_ASPECT_RATIO_THRESHOLD = 0.3

    #Minimum consecutive frames for which eye ratio is below threshold for alarm to be triggered
    EYE_ASPECT_RATIO_CONSEC_FRAMES = 30

    #COunts no. of consecutuve frames below threshold value
    COUNTER = 0

    #Load face cascade which will be used to draw a rectangle around detected faces.
    face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
    
     #Load face detector and predictor, uses dlib shape predictor file
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    
    #Extract indexes of facial landmarks for the left and right eye
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
    
     #This function calculates and return eye aspect ratio
    @classmethod
    def eye_aspect_ratio(self,eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])

        ear = (A+B) / (2*C)
        return ear
    
    
    
      
    @classmethod
    def drowsiness(cls,frame):
                        #Initialize Pygame and load music
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                #Detect facial points through detector function
                faces = cls.detector(gray, 0)

                #Detect faces through haarcascade_frontalface_default.xml
                face_rectangle = cls.face_cascade.detectMultiScale(gray, 1.3, 5)

                #Draw rectangle around each face detected
                for (x,y,w,h) in face_rectangle:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

                    #Detect facial points
                for face in faces:

                    shape = cls.predictor(gray, face)
                    shape = face_utils.shape_to_np(shape)

                        #Get array of coordinates of leftEye and rightEye
                    leftEye = shape[cls.lStart:cls.lEnd]
                    rightEye = shape[cls.rStart:cls.rEnd]

                        #Calculate aspect ratio of both eyes
                    leftEyeAspectRatio = cls.eye_aspect_ratio(leftEye)
                    rightEyeAspectRatio = cls.eye_aspect_ratio(rightEye)

                    eyeAspectRatio = (leftEyeAspectRatio + rightEyeAspectRatio) / 2

                        #Use hull to remove convex contour discrepencies and draw eye shape around eyes
                    # leftEyeHull = cv2.convexHull(leftEye)
                    # rightEyeHull = cv2.convexHull(rightEye)
                    # cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                    # cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                        #Detect if eye aspect ratio is less than threshold
                    if(eyeAspectRatio < cls.EYE_ASPECT_RATIO_THRESHOLD):
                        cls.COUNTER += 1
                            #If no. of frames is greater than threshold frames,
                        if (cls.COUNTER >= cls.EYE_ASPECT_RATIO_CONSEC_FRAMES):
                            return True   
                    else:
#                             pygame.mixer.music.stop()
                        cls.COUNTER = 0
                        return False
                        