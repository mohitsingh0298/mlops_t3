#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.layers import Convolution2D


# In[2]:


from keras.layers import MaxPooling2D


# In[3]:


from keras.layers import Flatten


# In[4]:


from keras.layers import Dense


# In[5]:


from keras.models import Sequential


# In[6]:


model = Sequential()


# In[ ]:





# In[7]:


def CRP(i):
    model.add(Convolution2D(filters=16*i, 
                            kernel_size=(3,3), 
                            activation='relu',
                       input_shape=(64, 64, 3)
                           ))
    model.add(MaxPooling2D(pool_size=(2, 2)))


# In[ ]:





# In[8]:


CRP(2)


# In[9]:


CRP(4)
CRP(8)

# In[ ]:





# In[ ]:





# In[10]:


model.summary()


# In[11]:


model.add(Flatten())


# In[12]:


model.summary()


# In[13]:


model.add(Dense(units=128, activation='relu'))


# In[14]:


model.summary()


# In[15]:


model.add(Dense(units=1, activation='sigmoid'))


# In[16]:


model.summary()


# In[17]:


model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics = ['accuracy'])


# In[18]:


from keras_preprocessing.image import ImageDataGenerator


# In[ ]:





# In[19]:



train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        '/root/cnn_dataset/training_set/',
        target_size=(64, 64),
        batch_size=100,
        class_mode='binary')
test_set = test_datagen.flow_from_directory(
        '/root/cnn_dataset/test_set/',
        target_size=(64, 64),
        batch_size=100,
        class_mode='binary')
result=model.fit(
        training_set,
        steps_per_epoch=80,
        epochs=5,
        validation_data=test_set,
        validation_steps=800)


# In[27]:


result.history['val_accuracy'][4]


# In[32]:


file1=open("acc.txt","w")
file1.write(str(result.history['val_accuracy'][4]*100))
file1.close()


# In[29]:


model.save('my_cat_dog_model.h5')


# In[ ]:


#from keras.models import load_model


# In[ ]:


#m = load_model('my_cat_dog_model.h5')


# In[ ]:


#from keras.preprocessing import image


# In[ ]:


#test_image = image.load_img('cnn_dataset/single_prediction/cat_or_dog_1.jpg', 
#               target_size=(64,64))


# In[ ]:


#type(test_image)


# In[ ]:


#test_image


# In[ ]:


#test_image = image.img_to_array(test_image)


# In[ ]:


#type(test_image)


# In[ ]:


#test_image.shape


# In[ ]:


#import numpy as np 


# In[ ]:


#test_image = np.expand_dims(test_image, axis=0)


# In[ ]:


#test_image.shape


# In[ ]:





# In[ ]:


#result = model.predict(test_image)


# In[ ]:


#result


# In[ ]:


#if result[0][0] == 1.0:
#    print('dog')
#else:
#    print('cat')


# In[ ]:





# In[ ]:


#r = training_set.class_indices


# In[ ]:


#r


# In[ ]:





# In[ ]:


#test_set = test_datagen.flow_from_directory(
#        'cnn_dataset/test_set/',
#        target_size=(64, 64),
#        class_mode='binary')


# In[ ]:


#my_result = model.predict(test_set)


# In[ ]:





# In[ ]:





# In[ ]:




