# import required modules
import cv2
# face classification using default haar cascade model
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# capture video
video = cv2.VideoCapture(0)
while True:
    # capture each frames from video
    _, frame = video.read()
    # converting frames into grayscale for face detection
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detecting face using haar cascade model
    face_detect = face_classifier.detectMultiScale(gray_image, 1.1,  4)
    # drawing bounding box within detection region
    for x, y, w, h in face_detect:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 3)
    # showing detected region to user with bounding boxes
    cv2.imshow("Face", frame)
    stop = cv2.waitKey(1)
    # quit screen when user presses 1 button
    if(stop == ord('q')):
        break
video.release()
cv2.destroyAllWindows()



