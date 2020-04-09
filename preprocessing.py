#!/usr/bin/env python
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame

from keras import layers
from keras import models
from keras import optimizers
from keras.utils import to_categorical


#take a glance of data.
def some_data_attributes(data):
    """ data is a list of train_images, test_images, train_labels, test_labels"""
    # data shape, dtype
    data_shape = DataFrame(index= ['train_images', 'test_images', 'train_labels', 'test_labels'],
                           data= {'shape': [i.shape for i in data],
                                  'dtype': [i.dtype for i in data]})
    return data_shape


# In[2]:


# reshape data to desire input's shape (batch, input dim), output's shape (batch, # class)
def pre_process(data):
    """to get image dataset the desire type and shape
        again, data is a list of train_images, test_images, train_labels, test_labels"""
    train_imgs = data[0].astype('float')
    test_imgs = data[1].astype('float')
    train_labs = to_categorical(data[2]).astype('float')
    test_labs = to_categorical(data[3]).astype('float')
    
    #display_samples:
    print('onehot train_labels \n {}'.format(train_labs[:3]))
    print('\n train_imgs shape {}'.format(train_imgs.shape)) 
    
    return [train_imgs, test_imgs, train_labs, test_labs]



# In[4]:


def normalize_stats(train_inputs):
    return np.mean(train_inputs, axis= 0), np.std(train_images, axis = 0)

def normalize_data(input_data, stats):
    """normalize data with stats from training set
       input_data: in 2D (None, dim) 
       stats: [mean, std]"""
    return (input_data - mean)/(std + 1e-7)

def rescale_images(images):
    """ im out of names for functions
        rescale each pixel of imput images to [0-1]"""
    return images/np.max(images)


# In[5]:


def count_labels(labels):
    "return a dictionary of label counts"
    labels = labels.reshape(len(labels))
    label_counts = dict()
    for i in labels:
        if i not in label_counts.keys():
            label_counts[i] = 1
        else: label_counts[i] += 1
    return label_counts


# In[6]:


def visualize_10_images(images, labels):
    """ visualilze 10 images for 10 class"""
    fig, (axes)= plt.subplots(3, 4)
    fig.subplots_adjust(hspace= 0.4, wspace = 0.2)
    count = 0
    for i in range(3):
        for j in range(4):
            if count < len(images):
                axes[i, j].imshow(images[count])
                axes[i, j].set_xlabel('label:{}'.format(labels[count]))
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
                count+=1
            else: break
                
    plt.plot()


    ### chossing learning rate for a particular model
def plot_lr(lrs, model):
    """plot loss function to see the effect of different lrs"""
    models = [model_trial(lr) for lr in lrs]
    histories = [model.fit(x = x, y = y, batch_size =256, verbose = 0, epochs = 10) for model in models]

    
    #plot loss
    for i in range(len(lrs)):
        plt.plot(histories[i].history['loss'][2:], label = 'lr: %0.3f'%(lrs[i]))
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.legend()
    plt.show()

def train_val_result(_history):
    ## plot some result
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 5))

    # plot train and val loss
    ax1.plot(_history.history['loss'], label = 'Train_loss')
    ax1.plot(_history.history['val_loss'], label = 'Val_loss')
    ax1.set_title('Train and val loss')
    ax1.set_ylabel('loss')
    ax1.set_label('epochs')
    ax1.legend()

    # plot train and val accuracy
    ax2.plot(_history.history['accuracy'], label = 'Train_acc')
    ax2.plot(_history.history['val_accuracy'], label = 'Val_acc')
    ax2.set_title('Train and val acc')
    ax2.set_ylabel('acc')
    ax2.set_label('epochs')
    ax2.legend()
    plt.show()


