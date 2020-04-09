#!/usr/bin/env python
# coding: utf-8

# # create some play_around model for fashion MNIST

# In[7]:


# some names of data set
# raw data: train_images, test_images, train_labels, test_labels
# preprocessed data: train_imgs, test_imgs,train_labs, test_labs
# today im playing with CNN, so my yesterday's dense module be turned into full CNN
# and the image will have shape of (batch, height, witdh, channels) (channel last)
# image_size = height = width = 32

image_size = 32 
input_shape = (image_size, image_size, 3)


# In[1]:


from keras.datasets import fashion_mnist
from keras import layers
from keras import models
from keras.utils import to_categorical
from keras import optimizers


# # Module architectures

# In[10]:


# just a quick check whether or not
def output_spatial_size(h_in, w_in, stride, pad, k_h, k_w ):
    return (h_in - k_h +2*pad )/stride + 1, (w_in - k_w +2*pad )/stride + 1


model1 = models.Sequential()

model1.add(layers.Conv2D(filters= 128, kernel_size= image_size, 
                         activation= 'relu', input_shape = input_shape))
model1.add(layers.Flatten())
model1.add(layers.Dense(units= 10, activation= 'softmax'))



# model 2: my made up model, if i had time

#model 3: modified Alex Net
#at first, i thought about upsampling cifar10 to have spatial size of (224, 224) to reserve the whole model but thats a crappy idea due to 2 reasons : waste in computation and  loss in information( 1st conv layer is conv 11*11, stride 4)  while cifar10 images are low-resolution so i try 2 conv layers 5*5, pad 2, 3 conv layers of 3*3, 0 pad  to get my spatial output down to (26, 26) as input to the 2nd pool layer of Alex net. then i copy the rest of the net, cut down 2 conv 3*3 layers in alex net. and all #channels each layer will also be reduced.

#All my later modified model of other famous model will be designed with the same spirit
# In[26]:

def model3():
    model3 = models.Sequential()
    model3.add(layers.Conv2D(filters= 16 ,kernel_size= 5, padding= 'same',
                             activation= 'relu', input_shape = input_shape))
    model3.add(layers.Conv2D(filters= 32, kernel_size= 5, activation= 'relu', padding= 'same'))

    # 3 conv layers 3*3
    model3.add(layers.Conv2D(filters= 32, kernel_size= 3, activation= 'relu'))
    model3.add(layers.Conv2D(filters= 64, kernel_size= 3, activation= 'relu'))
    model3.add(layers.Conv2D(filters= 128, kernel_size= 3, activation= 'relu'))


    # start to copy spatial output design of alex net
    model3.add(layers.MaxPool2D(pool_size= 3, strides= 2))
    model3.add(layers.Conv2D(filters= 128, kernel_size= 3, activation= 'relu', padding= 'same'))
    model3.add(layers.Conv2D(filters= 256, kernel_size= 3, activation= 'relu', padding= 'same'))
    model3.add(layers.MaxPool2D(pool_size= 3, strides= 2))

    # dense layers
    # i will only used 1 dense layer here
    model3.add(layers.Flatten())
    model3.add(layers.Dense(units= 512, activation= 'relu'))
    model3.add(layers.Dense(units= 10, activation= 'softmax'))
    return model3



def compile_model(model, lr):
    'customize model and learning rate'
    model.compile(optimizer= optimizers.RMSprop(lr), 
                   loss= 'categorical_crossentropy',
                   metrics= ['accuracy'])
    
def train_model(model, train_data, validation_data = None, verbose = 0):
    """train data: [train_images, train_labels],
       validation_data = (val_images, val_labels)"""
    if validation_data == None:
        # split train data to train and val data 
        # for the purpose of choosing hyperparams
        model_history = model.fit(x= train_data[0], y= train_data[1],
                   validation_split= 0.1, batch_size= 256,
                   verbose = verbose, epochs = 10)
    else: 
        # in case u wanna use test set for validation just to visual purpose. 
        # this takes more discipline not to accidentally take a peek at test set so i'll randomly use it
        model_history = model.fit(x= train_data[0], y= train_data[1],
                    validation_data= validation_data, batch_size= 256, 
                    verbose = verbose, epochs = 10)
    return model_history


#model3_1
def add_batch_norm(model):
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

def model3_1():
    model3_1 = models.Sequential()
    model3_1.add(layers.Conv2D(filters= 16 ,kernel_size= 5, padding= 'same',
                             activation= 'relu', input_shape = input_shape))
    add_batch_norm(model3_1)
    model3_1.add(layers.Conv2D(filters= 32, kernel_size= 5, padding= 'same'))
    add_batch_norm(model3_1)


    # 3 conv layers 3*3
    model3_1.add(layers.Conv2D(filters= 32, kernel_size= 3))
    add_batch_norm(model3_1)
    model3_1.add(layers.Conv2D(filters= 64, kernel_size= 3))
    add_batch_norm(model3_1)
    model3_1.add(layers.Conv2D(filters= 128, kernel_size= 3))
    add_batch_norm(model3_1)


    # start to copy spatial output design of alex net
    model3_1.add(layers.MaxPool2D(pool_size= 3, strides= 2))
    model3_1.add(layers.Conv2D(filters= 128, kernel_size= 3, padding= 'same'))
    add_batch_norm(model3_1)
    model3_1.add(layers.Conv2D(filters= 256, kernel_size= 3, padding= 'same'))
    add_batch_norm(model3_1)
    model3_1.add(layers.MaxPool2D(pool_size= 3, strides= 2))


    # dense layers
    # i will only used 1 dense layer here
    model3_1.add(layers.Flatten())
    model3_1.add(layers.Dense(units= 512))
    add_batch_norm(model3_1)
    model3_1.add(layers.Dense(units= 10, activation= 'softmax'))
    return model3_1




#model 3_2, truncated of model3_1
#model3_2
# def model3_2():
#     model3_2 = models.Sequential()
#     model3_2.add(layers.Conv2D(filters= 16 ,kernel_size= 5, padding= 'same',
#                              activation= 'relu', input_shape = input_shape))
#     add_batch_norm(model3_2)

#     # 3 conv layers 3*3
#     model3_2.add(layers.Conv2D(filters= 32, kernel_size= 5))
#     add_batch_norm(model3_2)
#     model3_2.add(layers.Conv2D(filters= 64, kernel_size= 3))
#     add_batch_norm(model3_2)

#     # start to copy spatial output design of alex net
#     model3_2.add(layers.MaxPool2D(pool_size= 3, strides= 2))
#     model3_2.add(layers.Conv2D(filters= 128, kernel_size= 3, padding= 'same'))
#     add_batch_norm(model3_2)
#     model3_2.add(layers.MaxPool2D(pool_size= 3, strides= 2))

#     # dense layers
#     # i will only used 1 dense layer here
#     model3_2.add(layers.Flatten())
#     model3_2.add(layers.Dense(units= 512))
#     add_batch_norm(model3_2)
#     model3_2.add(layers.Dense(units= 10, activation= 'softmax'))
#     return model3_2

#model3_2 add dropout
dropout=0.3
def model3_2():
    model3_2 = models.Sequential()
    model3_2.add(layers.Conv2D(filters= 16 ,kernel_size= 5, padding= 'same',
                             activation= 'relu', input_shape = input_shape))
    add_batch_norm(model3_2)
    model3_2.add(layers.Dropout(dropout))


    # 3 conv layers 3*3
    model3_2.add(layers.Conv2D(filters= 32, kernel_size= 5))
    add_batch_norm(model3_2)
    model3_2.add(layers.Dropout(dropout))
    model3_2.add(layers.Conv2D(filters= 32, kernel_size= 3))
    add_batch_norm(model3_2)
    model3_2.add(layers.Dropout(dropout))



    # start to copy spatial output design of alex net
    model3_2.add(layers.MaxPool2D(pool_size= 3, strides= 2))
    model3_2.add(layers.Conv2D(filters= 64, kernel_size= 3, padding= 'same'))
    add_batch_norm(model3_2)
    model3_2.add(layers.Dropout(dropout))
    model3_2.add(layers.MaxPool2D(pool_size= 3, strides= 2))

    # dense layers
    # i will only used 1 dense layer here
    model3_2.add(layers.Flatten())
    model3_2.add(layers.Dense(units= 128))
    add_batch_norm(model3_2)
    model3_2.add(layers.Dropout(dropout))
    model3_2.add(layers.Dense(units= 10, activation= 'softmax'))
    return model3_2