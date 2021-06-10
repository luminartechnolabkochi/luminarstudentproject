import os

import cv2  # openCV - image processing
import numpy as np  # numerical python
import tensorflow as tf  # Deep learning
from numpy.random import seed

seed(1)

# from tensorflow import set_random_seed
# set_random_seed(2)
import tensorflow
tensorflow.random.set_seed(2)

# TO get dataet path
path1 = os.getcwd() # Current Working Directory
path = os.path.join(path1,'tinydb1')

import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt	# graph plotting	 

data=os.listdir(path)
kernel = np.ones((2,1),np.uint8)
kernel1 = np.ones((1,1),np.uint8)
out=np.zeros((119,227,227))
label=[]

# creates a HDF5 file 'my_model.h5'


# returns a compiled model
# identical to the previous one

d=0
k=0
m=0
for i in data:
    path1=os.path.join(path,i)
    print(path1)
    class_data=os.listdir(path1)
    print(class_data)
    m=d
    d=d+1
    
    da = [m for x in range(len(class_data))]
    label.extend(da)
    m=m+1
    for j in class_data:

	## PREPROCESSING
        # 1.To read an image to matrix. 0 means read as GrayScale        
        imag = cv2.imread(os.path.join(path1,j),0)
	
	# 2.To resize to a std size
        imag=cv2.resize(imag,(227,227))
##        imag=imag.reshape(1,250,250,1)
##        imag = cv2.Canny(imag,30,20)
        
	#3. Gaussian Edge Detection
        ou = cv2.adaptiveThreshold(imag,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,5)
##        ou = cv2.erode(ou, kernel1, iterations=1) 
        ou = cv2.morphologyEx(ou, cv2.MORPH_OPEN, kernel)

        ou= ou.reshape(227,227)
        cv2.imshow("leaf",ou)
        cv2.waitKey(1)
        ou=255-ou
        ou=np.multiply(imag,ou)
        cv2.imshow('leaf',ou)
        cv2.destroyAllWindows
        cv2.waitKey(1)
        out[k,:,:]=ou # 3d matrix to store images
        
        k=k+1
cv2.destroyAllWindows()

cv2.imshow('leaf1',out[20,:,:])
cv2.waitKey(1)
cv2.destroyAllWindows


from tensorflow.keras import layers, models
##(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
##
##train_images = train_images.reshape((60000, 28, 28, 1))
##test_images = test_images.reshape((10000, 28, 28, 1))
##
### Normalize pixel values to be between 0 and 1
##train_images, test_images = train_images / 255.0, test_images / 255.0
##
##
##

plt.ion()
plt.figure()
plt.imshow(out[0,:,:])
plt.colorbar()
plt.grid(False)
# plt.show()
plt.figure(figsize=(10,10))
for i in range(25):
    a=random.randint(1,100)
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(out[a,:,:], cmap=plt.cm.binary)
    
# plt.show()

# IMG_SIZE = 50
#
# X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
#
# y = np.array(y)
#
# history = model.fit(X, y, batch_size=32, epochs=40, validation_split=0.1)




out=out/255

u=out.reshape(119,227,227,1)
k=np.array(u)
cv2.imshow('leaf1',u[100,:,:,:])
cv2.waitKey(1)
label=np.array(label)
x_train, x_valid, y_train, y_valid = train_test_split(k, label, test_size=0.10, shuffle= True)
model = models.Sequential()
model.add(layers.Conv2D(64, (11, 11), activation='relu', input_shape=(227, 227, 1)))
model.add(layers.MaxPooling2D((2, 2),strides=(2,2)))
model.add(layers.Conv2D(96, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((2, 2),strides=(2,2)))
model.add(layers.Conv2D(256, (4, 4), activation='relu'))
model.add(layers.MaxPooling2D((2, 2),strides=(2,2)))
##model.add(layers.Conv2D(64, (3, 3), activation='relu'))
##model.add(layers.MaxPooling2D((2, 2)))
##
##model.add(layers.Conv2D(64, (3, 3), activation='relu'))
##model.add(layers.MaxPooling2D((2, 2)))
##
model.add(layers.Flatten())
model.add(layers.Dense(50))
model.add(layers.Dense(50))
model.add(layers.Dense(4, activation='softmax')) # TO use CNN as a classifier

model.summary()

model.compile(optimizer='sgd',# Stochastic Gradient Descendent
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# model.compile(
##      loss=tf.keras.losses.sparse_categorical_crossentropy,
##      optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
##      metrics=['accuracy'])
print(type(x_train))
print(type(y_train))
print(x_train.shape)
print((len(y_train)))

# TRAINING THE ALGORITHM
model.fit(x_train, y_train, epochs=30)


# Evaluate the model using the test dataset
loss, acc = model.evaluate(x_valid,y_valid, verbose=2,batch_size=24)
model.save('my_model.h5')
print("accuracy is:",acc)
import math
# def visualize_conv_layer(layer_name):
#
#     layer_output=model.get_layer(layer_name).output
#
#     intermediate_model=tf.keras.models.Model(inputs=model.input,outputs=layer_output)
#
#     intermediate_prediction=intermediate_model.predict(x_train[10].reshape(1,227,227,1))
#
#     row_size=8
#     col_size=8
#
#     img_index=0
#
#     print(np.shape(intermediate_prediction))
#     sh=np.shape(intermediate_prediction)
# ##    fig,ax=plt.subplots(row_size,col_size,figsize=(10,8))
# ##
# ##    for row in range(0,row_size):
# ##        for col in range(0,col_size):
# ##              ax[row][col].imshow(intermediate_prediction[0, :, :, img_index], cmap='gray')
##
##              img_index=img_index+1
# ##    plt.figure(figsize=(10,10))
#     l=math.sqrt(sh[-1])
#     plt.title('inter mediate layer output')
#     for i in range(sh[-1]):
#
#         plt.subplot(l,l,i+1)
#         plt.xticks([])
#         plt.yticks([])
#         plt.grid(False)
#         plt.imshow(intermediate_prediction[0, :, :, i], cmap=plt.cm.binary)
#
#
#     plt.show()
#     return intermediate_prediction
for layer in model.layers:
    print(layer.name)
    
layer_name = 'conv2d_2'

# visualize_conv_layer(layer_name)
