
import tensorflow as tf
import cv2
import sys
import numpy as np
# from tkinter import Tk # GUI Toolkit
# from tkinter.filedialog import askopenfilename  # for file 
# Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
fileName =sys.argv[1]  # show an "Open" dialog box and return the path to the selected file
# print("testing here*****************************",fileName,"**************************************")
if fileName == '':
    print('No file selected')
    print('Program Completed')
    exit()

kernel = np.ones((2,1),np.uint8)
kernel1 = np.ones((1,1),np.uint8)
imag=cv2.imread(fileName,0)
##m=cv2.resize(m,(256,256))
##imag=cv2.imread(os.path.join(path1,j),0)
imag=cv2.resize(imag,(227,227))
##        imag=imag.reshape(1,250,250,1)
##        imag = cv2.Canny(imag,30,20)


ou = cv2.adaptiveThreshold(imag,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,5)
##        ou = cv2.erode(ou, kernel1, iterations=1) 
ou = cv2.morphologyEx(ou, cv2.MORPH_OPEN, kernel)
ou= ou.reshape(227,227)
cv2.imshow('leaf',ou)
cv2.waitKey(1)
ou=255-ou
ou=np.multiply(imag,ou)
cv2.imshow('leaf',ou)
cv2.destroyAllWindows
cv2.waitKey(100)


##The model has a save method, which saves all the details necessary to reconstitute the model. An example from the keras documentation:


model = tf.keras.models.load_model('/home/luminar/Downloads/singletest/my_model.h5')

OUT=model.predict(ou.reshape(-1,227,227,1))

X=np.argmax(OUT,axis=1)
import os
da=os.listdir("/home/luminar/Downloads/singletest/tinydb1")
#da="/home/luminar/Downloads/singletest/tinydb1"
print(str(da[X[0]]),end="")
