import cv2
import pytesseract
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Rajshree\AppData\Local\Tesseract-OCR\tesseract.exe'

cascade= cv2.CascadeClassifier("haarcascade_russian_plate_number.xml") 
states= {"AN":"Andaman and Nicobar ", "DL":"Dehli", "KA": "Karnataka","MH":"Maharashtra","RJ":"Rajasthan","MP":"Madhya Pradesh"}
img = cv2.imread('img5.jpeg')
#def extract_num(img): 
    #global read
     
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_plate = cascade.detectMultiScale(img_gray,1.1,4)
for (x,y,w,h) in img_plate:
    a,b=(int(0.02*img.shape[0]), int(0.025*img.shape[1]))
    plate = img[y+a:y+h-a, x+b:x+w-b, :]
    
    kernel =np.ones((1,1),np.uint8)
    plate= cv2.dilate(plate,kernel, iterations=1)
    plate= cv2.erode(plate,kernel, iterations=1)
    plate_gray= cv2.cvtColor(plate,cv2.COLOR_BGR2GRAY) 
    (thresh, plate)=cv2.threshold(plate_gray,127,255,cv2.THRESH_BINARY) 

    read= pytesseract.image_to_string(plate, config="--psm 6")
    read= ''.join(e for e in read if e.isalnum())
    stat= read[0:2]
    try:
        print('car belongs to',states[stat])
    except:
        print('state not recognised')
    print(read) 
    cv2.rectangle(img,(x,y),(x+w, y+h),(51,51,255),2)
    cv2.rectangle(img,(x,y-40),(x+w,y),(51,51,255),-1)
    cv2.imshow('plate',plate)

cv2.imshow("Result",img)
cv2.imwrite('result.jpeg',img) 
cv2.waitKey(0)  
cv2.destroyAllWindows() 
 
#extract_num('img1.jpeg')
