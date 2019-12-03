import os
import numpy as np
import cv2
import keyboard

def find_eyes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresholds = [1, 10, 50, 5, 2, 8, 20, 30, 40]
    for t in thresholds:
        eyes = eye_cascade.detectMultiScale(gray, 1.2, t, cv2.CASCADE_SCALE_IMAGE, (150, 150), (400, 400))
        if (len(eyes) == 2):
            break
    return eyes

if __name__ == "__main__":
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    eye_cascade.load('C:/opencv/opencv-3.4.5/data/haarcascades/haarcascade_eye.xml')

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_cascade.load('C:/opencv/opencv-3.4.5/data/haarcascades/haarcascade_frontalface_default.xml')

    root = './Data/'
    output_root = './Eyes/'
    dirs = os.listdir(root)

    files = []

    for d in dirs:
        sub_dir = root + d + '/'
        fs = os.listdir(sub_dir)
        for f in fs:
            files.append(sub_dir + f)
    
    for file in files:
        if keyboard.is_pressed('esc'):
            break
        if file.split('.')[-1] == 'jpg':
            image = cv2.imread(file)
            image = image[700 : 2700, 1600 : 3600]
            face = cv2.resize(image, (640, 640))
            #cv2.imshow('image', face)
            #cv2.waitKey(1)

            eyes = find_eyes(image[400 : 1600, :])
            if (len(eyes) == 2):
                (x0, y0, w0, h0) = eyes[0]
                (x1, y1, w1, h1) = eyes[1]
                y0 += 400
                y1 += 400
                
                x0 -= int((640 - w0) // 2)
                y0 -= int((640 - h0) // 2)
                w0 = 640
                h0 = 640
                x1 -= int((640 - w1) // 2)
                y1 -= int((640 - h1) // 2)
                w1 = 640
                h1 = 640

                if (x0 >= 0 and x0 + w0 < 2000 and y0 >= 0 and y0 + h0 < 2000 and x1 >= 0 and x1 + w1 < 2000 and y1 >= 0 and y1 + h1 < 2000):
                    eye0 = image[y0 : y0 + h0, x0 : x0 + w0]
                    eye1 = image[y1 : y1 + h1, x1 : x1 + w1]

                    if x0 < x1:
                        image = np.hstack([face, eye0, eye1])
                    else:
                        image = np.hstack([face, eye1, eye0])
                    
                    cv2.imshow('image', image)
                    cv2.waitKey(1)
                    cv2.imwrite(output_root + file.split('/')[-1], image)

    cv2.destroyAllWindows()
