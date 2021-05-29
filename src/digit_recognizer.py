# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Digit Recognizer
# %% [markdown]
# ## Import Libraries

# %%
#import libraries 
import os
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd

# %% [markdown]
# ## Read Data

# %%
train_data=pd.read_csv("train.csv")
train_data.head(10)


# %%
train_data.shape


# %%
test_data=pd.read_csv("test.csv")
test_data.head()


# %%
test_data.shape


# %%
train_data.info()


# %%
X_train=(train_data.iloc[:,1:].values).astype("float32") # all pixel values of train data
y_train=train_data.iloc[:,0].values.astype('int32') # label of train data
X_test=test_data.values.astype('float32')  #all pixel values in test data


# %%
X_train,X_test,y_train

# %% [markdown]
# ## Visualize data

# %%
X_train=X_train.reshape(X_train.shape[0],28,28)    # convert train data set in [num train images,28,28] format
plt.imshow(X_train[24],plt.get_cmap('gray'))
plt.title( "Label of X_train[24] is "+str(y_train[24]))


# %%
#reshape to gray scale image by adding 1 as dimension
X_train=X_train.reshape(X_train.shape[0],28,28,1)
print(X_train.shape)
X_test=X_test.reshape(X_test.shape[0],28,28,1)
print(X_test.shape)

# %% [markdown]
# ## Preprocessing

# %%
# standardization
mean_pixel=X_train.mean().astype(np.float32)
std_pixel=X_train.std().astype(np.float32)
def standardization(x):
    return (x-mean_pixel)/std_pixel
print(y_train.shape)
y_train


# %%
#one hot encoding of labels
from keras.utils.np_utils import to_categorical
y_train=to_categorical(y_train)
print(y_train.shape)
y_train


# %%
num_classes=y_train.shape[1]
num_classes


# %%
#plot an label
plt.plot(y_train[45])
plt.xticks(range(num_classes))
plt.title(y_train[45])


# %%
#image generator
from keras.preprocessing import image
img_gen=image.ImageDataGenerator()


# %%
#cross validation 
from sklearn.model_selection import train_test_split
X=X_train
y=y_train

X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,test_size=0.2,random_state=42)
batches=img_gen.flow(X_train,y_train,batch_size=64)
val_batches=img_gen.flow(X_val,y_val,batch_size=64)

# %% [markdown]
# ## CNN Model with BatchNormalization

# %%
#importing libraries 
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Lambda,BatchNormalization, Convolution2D,MaxPooling2D
from keras.callbacks import EarlyStopping


# %%
#add batch normalization
from keras.layers.normalization import BatchNormalization

def batch_normalization_model():
    model = Sequential([
        Lambda(standardization, input_shape=(28,28,1)),
        Convolution2D(32,(3,3), activation='relu'),
        BatchNormalization(axis=1),
        Convolution2D(32,(3,3), activation='relu'),
        MaxPooling2D(),
        BatchNormalization(axis=1),
        Convolution2D(64,(3,3), activation='relu'),
        BatchNormalization(axis=1),
        Convolution2D(64,(3,3), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        BatchNormalization(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dense(10, activation='softmax')
        ])
    model.compile(optimizer=Adam(lr=0.01),loss='categorical_crossentropy',metrics=['accuracy'])
    return model


# %%
BatchNormalization_Model=batch_normalization_model()
history=BatchNormalization_Model.fit_generator(generator=batches, steps_per_epoch=1000, epochs=1, 
                    validation_data=val_batches, validation_steps=1000)


# %%
#predict for testset
predict=BatchNormalization_Model.predict_classes(X_test,verbose=0)
output=pd.DataFrame({'ImageId':list(range(1,len(predict)+1)),'Label':predict})
output.to_csv('Digit_recognizer_CNN(keras).csv',index=False,header=True)


# %%



