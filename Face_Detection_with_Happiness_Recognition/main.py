# Homework Solution
# No neural networks ,deep learning used
# Importing the libraries
import cv2

# Cascades used to detect smile, eyes and face are used - series of filters applies one after the other for detection
# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Defining a function that will do the detections
# frame = original image ; gray = grayscale image of original image
# (x,y) = coordinates of upper left corner ; w,h = width and height of the rectangle
def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # this method gets us values of x,y,w,h  ; 1.3 = image size reduced to 1.3 times , 5 = atleast 5 neighbour zones should also be accepted ; faces are the faces detected
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # draws a rectangle , takes upper left and lower right points , (255,0,0) = color of rectangle, 2 = thickness of the rectangle
        roi_gray = gray[y:y+h, x:x+w]   # region of interest (cropped - just the face box content)
        roi_color = frame[y:y+h, x:x+w]  #it is the sub zone of frame
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 22) # eyes checked in roi_gray
        for (ex, ey, ew, eh) in eyes:           # same done for eye
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 22)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
    return frame

# Doing some Face Recognition with the webcam
video_capture = cv2.VideoCapture(0)    # 0 if computer's webcam ; 1 if external webcam
while True:
    _, frame = video_capture.read()   # read() gives 2 outputs  ; frame - last frame captured
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # coverion of colored frame to gray scale image
    canvas = detect(gray, frame)
    cv2.imshow('Video', canvas)    # shows video of webcam in other window with detector rectangles
    if cv2.waitKey(1) & 0xFF == ord('q'):  # q pressed => stop face detection
        break
video_capture.release()   # turn off webcam
cv2.destroyAllWindows()    # close window where video was displayed

# sub zone of face is taken to avoid detection of any other thing similar to eyes or smiles in complete photo 