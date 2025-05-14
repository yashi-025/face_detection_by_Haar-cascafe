import cv2

# img = cv2.imread("photos/5 people.jpg")
# img = cv2.imread("photos/26 people.jpg")
img = cv2.imread("photos/lady.jpg")

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_copy = img_gray.copy()
gray_copy2 = img_gray.copy()

''' Haar Cascade is the machine learning based object detection method used to detect objects in images or videos.
It is a pre-trained classifier that can be used to detect faces, eyes, smiles, etc.

minNeighbor defines how many neighbors (detections) each candidate rectangle should have to be retained as a valid detection.
As Haar casade is prone to noise/false prediction/detection, to reduce false prediction increase the minNeighbours value in the detectMultiScale function,
but also miss some object to be detected (i.e. misses true detection with less confindence).

ScaleFactor is the factor by which the image is reduced at each image scale. Increasing it's value leads to:
1) Faster detection: Fewer image scales are processed. The algorithm skips over more possible object sizes.
2) Lower accuracy: May miss objects that fall between skipped scales, especially smaller ones.

So to get the best results, we need to find a balance between the two, i.e , minNeightbours and scaleFactor values.
.
'''

def face_detection(img_gray):
    haar_cascade = cv2.CascadeClassifier("face_detection/cascade/haarcascade_frontalface_default.xml")
    faces_rect = haar_cascade.detectMultiScale(img_gray, scaleFactor = 1.3, minNeighbors = 4)
    print(f"No. of faces in given image: {len(faces_rect)}")
    for x,y,w,h in faces_rect:
        rect = cv2.rectangle(img_gray, (x,y), (x+w, y+h), (255, 255, 255), 2)
        
    return img_gray  
  
def eye_detection(img_gray):
    haar_cascade = cv2.CascadeClassifier("face_detection/cascade/haarcascade_eye.xml")
    eyes_rect = haar_cascade.detectMultiScale(img_gray, scaleFactor = 1.3, minNeighbors = 4)
    print(f"No. of eyes in given image: {len(eyes_rect)}")
    for x,y,w,h in eyes_rect:
        rect = cv2.rectangle(img_gray, (x,y), (x+w, y+h), (255, 255, 255), 2)
        
    return img_gray  

def smile_detection(img_gray):
    haar_cascade = cv2.CascadeClassifier("face_detection/cascade/smile.xml")
    smile_rect = haar_cascade.detectMultiScale(img_gray, scaleFactor = 1.3, minNeighbors = 20)
    print(f"No. of smile in given image: {len(smile_rect)}")
    for x,y,w,h in smile_rect:
        rect = cv2.rectangle(img_gray, (x,y), (x+w, y+h), (255, 255, 255), 2)
        
    return img_gray 
    
face = face_detection(img_gray)
eyes = eye_detection(gray_copy) 
smile = smile_detection(gray_copy2)   

cv2.imshow("Detected faces", face)
cv2.imshow("Detected eyes", eyes)
cv2.imshow("Smile detected", smile)   
cv2.waitKey(0)
cv2.destroyAllWindows()    