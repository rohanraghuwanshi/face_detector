import cv2
import glob

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

images=glob.glob("*.jpg")

for image in images:
    img=cv2.imread(image)
    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces=face_cascade.detectMultiScale(gray_img,
    scaleFactor=1.9,
    minNeighbors=5)

    for x,y,w,h in faces:
        img=cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),15)

    img=cv2.resize(img,(int(img.shape[1]/3),int(img.shape[0]/3)))
    cv2.imshow("Image",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#cv2.imwrite("resized_"+image,re)
