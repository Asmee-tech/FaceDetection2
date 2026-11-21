import cv2
import os
import numpy
datassets="Data_Assets"
haar_file="opencv-assets-main/haarcascade_frontalface_default.xml"
(images,labels,names,id)=([],[],{},0)
for(subdirs,dirs,files) in os.walk(datassets):
    for folders in dirs:
        names[id]=folders
        path=os.path.join(datassets,folders)
        for file in os.listdir(path):
            image_path=path+"/"+file
            images.append(cv2.imread(image_path,0))
            label=id
            labels.append(label)
        id+=1
# print(names)
# print(images)
# print(labels)
(images,labels)=[numpy.array(lis)for lis in[images,labels]]
model=cv2.face.LBPHFaceRecognizer_create()
model.train(images,labels)
face_classifier=cv2.CascadeClassifier(haar_file)
video=cv2.VideoCapture(0)
while video.isOpened():
    return_val,img=video.read()
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.2,4)
    # print(faces)
    for (x,y,w,h)in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        face=gray[y:y+h,x:x+w]
        face_resize=cv2.resize(face,(130,130))
        prediction=model.predict(face_resize)
        print(prediction[1],prediction[0])
        if prediction[1]>60:
            cv2.putText(img,f"{names[prediction[0]]}",(x,y-15),cv2.FONT_ITALIC,1,(255,255,255))
        else:
            cv2.putText(img,f"Not recongnised",(x,y-15),cv2.FONT_ITALIC,1,(0,0,255))
    cv2.imshow("Face",img)
    key=cv2.waitKey(10)
    if key==27:
        break