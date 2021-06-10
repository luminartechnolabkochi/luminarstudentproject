import numpy as np
import os
import cv2
from numpy.random import seed

from keras import optimizers
# from tensorflow import set_random_seed
from DLEAF import layer_name

path1=os.getcwd()
path=os.path.join(path1,'tinydb1')
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
kernel = np.ones((2,1),np.uint8)
kernel1 = np.ones((1,1),np.uint8)

data=os.listdir(path)
kernel = np.ones((2,1),np.uint8)
kernel1 = np.ones((1,1),np.uint8)
out=np.zeros((119,224,224,3))
label=[]

d=0
k=0
m=0
plt.ion()


label_name=[]
for i in data:
    path1=os.path.join(path,i)
##    print(path1)
    class_data=os.listdir(path1)
    print(class_data)
    m=d
    d=d+1
    
    da=[m for x in range(len(class_data))]
   
    m=m+1
    label.extend(da)
    for j in class_data:
        print(j)
        a=j.split(".")
        if a[-1] != 'jpg':
            print("continued")
            label.remove(label[-1])


            continue

        
        label_name.append(os.path.join(path1,j))
        
        imag1=cv2.imread(os.path.join(path1,j))
        imag=cv2.imread(os.path.join(path1,j),0)
        imag=cv2.resize(imag,(224,224))
        ou = cv2.adaptiveThreshold(imag,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,5)
##        ou = cv2.erode(ou, kernel1, iterations=1) 
        ou = cv2.morphologyEx(ou, cv2.MORPH_OPEN, kernel)
        print(os.path.join(path1,j))
##        imag=imag.reshape(1,250,250,1)
####        imag = cv2.Canny(imag,30,20)
##        
##      
##        ou = cv2.adaptiveThreshold(imag,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,5)
####        ou = cv2.erode(ou, kernel1, iterations=1) 
##        ou = cv2.morphologyEx(ou, cv2.MORPH_OPEN, kernel)
        
        cv2.imshow('image',ou)
        cv2.waitKey(10)
##        ou=255-ou
##        ou=np.multiply(imag,ou)
        cv2.imshow('image',ou)
        cv2.destroyAllWindows
        cv2.waitKey(1)
        out[k,:,:,0]=ou
        out[k,:,:,1]=ou
        out[k,:,:,2]=ou
        
        k=k+1
cv2.destroyAllWindows()

cv2.imshow('leaf1',out[20,:,:,:])
cv2.waitKey(1000)
cv2.destroyAllWindows
from tensorflow.keras import datasets, layers, models
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


plt.figure()
plt.imshow(np.uint8(out[0,:,:,:]))
plt.colorbar()
plt.grid(False)
plt.show()
plt.figure(figsize=(10,10))
for i in range(25):
    a=random.randint(1,118)
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(np.uint8(out[a,:,:,:]))
    
plt.show()

plt.close('all')

##out=out/255
u=(out.reshape(119,224,224,3))/255
cv2.imshow('leaf1',u[100,:,:,:])
cv2.waitKey(1000)
from keras.layers import Conv2D, MaxPooling2D,Flatten,Dense,Dropout
from keras import applications

# This will load the whole VGG16 network, including the top Dense layers.
# Note: by specifying the shape of top layers, input tensor shape is forced
# to be (224, 224, 3), therefore you can use it only on 224x224 images.
vgg_model = applications.VGG16(weights=None, include_top=True)

# If you are only interested in convolution filters. Note that by not
# specifying the shape of top layers, the input tensor shape is (None, None, 3),
# so you can use them for any size of images.
vgg_model = applications.VGG16(weights='imagenet', include_top=True)
##
## If you want to specify input tensor
from keras.layers import Input
input_tensor = Input(shape=(224, 224, 3))


# To see the models' architecture and layer names, run the following
vgg_model.summary()


vgg_model = applications.VGG16(weights='imagenet',
                               include_top=True,
                               input_shape=(224, 224, 3))

# Creating dictionary that maps layer names to the layers
layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])

# Getting output tensor of the last VGG layer that we want to include
x = layer_dict['block5_pool'].output

# Stacking a new simple convolutional network on top of it    

x = Flatten()(x)
x = Dense(30, activation='relu')(x)
x = Dense(30, activation='relu')(x)


x = Dense(4, activation='softmax')(x)

# Creating new model. Please note that this is NOT a Sequential() model.
from keras.models import Model
custom_model = Model(input=vgg_model.input, output=x)

# Make sure that the pre-trained bottom layers are not trainable
##for layer in custom_model.layers[:-1]:
##    layer.trainable = True
print(custom_model.summary())
# Do not forget to compile it
custom_model.compile(loss='categorical_crossentropy',
                     optimizer=optimizers.Adam(lr=0.0001),
                     metrics=['accuracy'])

x_train, x_valid, y_train, y_valid = train_test_split(u.reshape(119,224,224,3), label, test_size=0.20, shuffle= True)

print(custom_model.summary())
from keras.utils import to_categorical
train_labels = to_categorical(y_train)
custom_model.fit(x_train,train_labels, epochs=5)



loss, acc = custom_model.evaluate(x_valid,to_categorical(y_valid), verbose=2,batch_size=24)
custom_model.save('trainedmodel.h5')

##import math
##def visualize_conv_layer(layer_name):
##  
##    layer_output=custom_model.get_layer(layer_name).output
##
##    intermediate_model=tf.keras.models.Model(inputs=custom_model.input,outputs=layer_output)
##
##    intermediate_prediction=intermediate_model.predict(x_train[10].reshape(1,227,227,1))
##  
##    row_size=8
##    col_size=8
##  
##    img_index=0
##
##    print(np.shape(intermediate_prediction))
##    sh=np.shape(intermediate_prediction)
####    fig,ax=plt.subplots(row_size,col_size,figsize=(10,8))
####
####    for row in range(0,row_size):
####        for col in range(0,col_size):
####              ax[row][col].imshow(intermediate_prediction[0, :, :, img_index], cmap='gray')
####
####              img_index=img_index+1
####    plt.figure(figsize=(10,10))
##    l=math.sqrt(sh[-1])
##    for i in range(sh[-1]):
##        
##        plt.subplot(l,l,i+1)
##        plt.xticks([])
##        plt.yticks([])
##        plt.grid(False)
##        plt.imshow(intermediate_prediction[0, :, :, i], cmap=plt.cm.binary)
##        
##    plt.show()
##    return intermediate_prediction
##for layer in custom_model.layers:
##    print(layer.name)
##    
##layer_name = 'dense_2'

##intermediate_model=tf.keras.models.Model(inputs=model.input,outputs=layer_output)
##
##intermediate_prediction=intermediate_model.predict(x_train.reshape(1,227,227,1))

intermediate_layer_model = Model(inputs=custom_model.input,
                                 outputs=custom_model.get_layer(layer_name).output)

intermediate_output = intermediate_layer_model.predict(u.reshape(119,224,224,3))
import pickle

with open('varrr.pickle', 'wb') as f:
    pickle.dump(intermediate_output, f)

with open('label.pickle', 'wb') as f:
    pickle.dump(label_name, f)








