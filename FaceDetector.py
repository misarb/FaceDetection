import cv2
from random import randrange


# load some pre-trained Data on face frontals from openCV using haar cascade algorithm
trained_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# image to detect Face in as an example
img = cv2.imread('test.jpg')
## if u want to use webcam
# webcam = cv2.VideoCapture(0)

while True:
    # successful_frame_read , frame = webcam.read()
    # convert img to grayScale
    # gray_img = cv2.cvtColor(frame , cv2.COLOR_GRAY2BGR)
    # # Detecet Face 
    cordinte_face = trained_data.detectMultiScale(img)
    # # Looping into the image and Detecet the Faces 
    for (x,y,w,h) in cordinte_face:
        # trace the rectangle on the face
        cv2.rectangle(img, (x,y), (w+x,h+y), (randrange(256),randrange(256),randrange(256)),2)
    
    #open the image in a windows pop
    cv2.imshow('FaceDetector App', img)

    #make the img wait to not close instantly
    key = cv2.waitKey(1)

    # Press Q to quite out of the programme
    if key==81 or key ==113:
        break

    
print("end")
 
    




