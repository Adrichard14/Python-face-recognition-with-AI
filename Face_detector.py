import cv2
from random import randrange

#Load some pre-trained data on face frontals opencv
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Choose the image to detect faces in
# img = cv2.imread('rdj.jpeg')
img = cv2.imread('two_people-3.jpg')


webcam = cv2.VideoCapture(0)

# Interate forever over frames
while True:
    #Read the current frame
    successful_frame_read, frame = webcam.read()

    #Convert the image to grayscale
    greyscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect faces
    face_coordinates = trained_face_data.detectMultiScale(greyscaled_img)

    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2)

    cv2.imshow('Face Detector Programming', frame)
    key = cv2.waitKey(1)   
    if key==81 or key==113:
        break

# Release the webcam object
webcam.release()
print('Working')


"""
# Detect faces
face_coordinates = trained_face_data.detectMultiScale(greyscaled_img)
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2)


print(face_coordinates)

cv2.imshow('Face Detector Programming', img)
cv2.waitKey()
"""

print('Working')