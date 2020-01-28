#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
import sys
import math
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import PIL
from PIL import Image
import keras
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPool2D, GlobalMaxPool2D, Dropout
from keras.optimizers import SGD
from keras.models import Model
print("Python", sys.version_info[0],sys.version_info[1],sys.version_info[2])
print("Numpy Version: ", np.__version__)
print("Keras Version: ", keras.__version__)
print("Pandas Version: ", pd.__version__)
print("Seaborn Version: ", sns.__version__)
print("PIL Version: ", PIL.__version__)


#  # Our library versions
#  Python3.7.4 <br />
#  Numpy Version:  1.16.5 <br />
#  Keras Version:  2.3.0 <br />
#  Pandas Version:  0.25.1 <br />
#  Seaborn Version:  0.9.0 <br />

# # Data Directory: This Folder must be in same folder with code.

# In[ ]:


DATA_DIR = "DataSmall"


# In[ ]:



EPOCH = 12
TRAIN_TEST_SPLIT = 0.7
IM_WIDTH = IM_HEIGHT = 200
ID_GENDER_MAP = {0: 'male', 1: 'female'}
GENDER_ID_MAP = dict((g, i) for i, g in ID_GENDER_MAP.items())


# In[ ]:


def parse_filepath(filepath):
    path, filename = os.path.split(filepath)
    filename, ext = os.path.splitext(filename)
    print(filename, path)
    age, gender, _,_ = filename.split("_")
    return int(age), ID_GENDER_MAP[int(gender)]
files = glob.glob(os.path.join(DATA_DIR, "*.jpg"))
attributes = list(map(parse_filepath, files))


# In[ ]:


df = pd.DataFrame(attributes)
df['file'] = files
df.columns = ['age', 'gender', 'file']
df = df[(df['age'] > 10) & (df['age'] < 65)]
df = df.dropna()
df.head()
#df.groupby('age').count()
#df.groupby('gender').count()


# In[ ]:


df.describe()


# In[ ]:


fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
_ = sns.boxplot(data=df, x='gender', y='age', ax=ax1)


# In[ ]:


df.groupby(by=['gender'])['age'].count().plot(kind='bar')


# In[ ]:


df['age'].hist()


# In[ ]:


p = np.random.permutation(len(df))
train_up_to = int(len(df) * TRAIN_TEST_SPLIT)
train_idx = p[:train_up_to]
test_idx = p[train_up_to:]


train_up_to = int(train_up_to * 0.7)
train_idx, valid_idx = train_idx[:train_up_to], train_idx[train_up_to:]

df['gender_id'] = df['gender'].map(lambda gender: GENDER_ID_MAP[gender])

max_age = df['age'].max()
len(train_idx), len(valid_idx), len(test_idx), max_age


# In[ ]:


def get_data_generator(df, indices, for_training, batch_size=16):
    images, ages, genders = [], [], []
    while True:
        for i in indices:
            r = df.iloc[i]
            file, age, gender = r['file'], r['age'], r['gender_id']
            im = Image.open(file)
            im = im.resize((IM_WIDTH, IM_HEIGHT))
            im = np.array(im) / 255.0
            images.append(im)
            ages.append(age / max_age)
            genders.append(to_categorical(gender, 2))
            if len(images) >= batch_size:
                yield np.array(images), [np.array(ages), np.array(genders)]
                images, ages, genders = [], [], []
        if not for_training:
            break


# In[ ]:


def conv_block(inp, filters=32, bn=True, pool=True):
    _ = Conv2D(filters=filters, kernel_size=3, activation='relu')(inp)
    if bn:
        _ = BatchNormalization()(_)
    if pool:
        _ = MaxPool2D()(_)
    return _

input_layer = Input(shape=(IM_HEIGHT, IM_WIDTH, 3))
_ = conv_block(input_layer, filters=32, bn=False, pool=False)
_ = conv_block(_, filters=32*2)
_ = conv_block(_, filters=32*3)
_ = conv_block(_, filters=32*4)
_ = conv_block(_, filters=32*5)
_ = conv_block(_, filters=32*6)
_ = conv_block(_, filters=32*7)
bottleneck = GlobalMaxPool2D()(_)

##### for age calculation ######
_ = Dense(units=64, activation='relu')(bottleneck)
age_output = Dense(units=1, activation='sigmoid', name='age_output')(_)

##### for gender calculation ######
_ = Dense(units=32, activation='relu')(bottleneck)
gender_output = Dense(units=len(GENDER_ID_MAP), activation='softmax', name='gender_output')(_)

model = Model(inputs=input_layer, outputs=[age_output, gender_output])
model.compile(optimizer='rmsprop', 
              loss={'age_output': 'mse', 'gender_output': 'categorical_crossentropy'},
              loss_weights={'age_output': 2., 'gender_output': 1.},
              metrics={'age_output': 'mae', 'gender_output': 'accuracy'})
#model.summary()


# In[ ]:


batch_size = 64
valid_batch_size = 64
train_gen = get_data_generator(df, train_idx, for_training=True, batch_size=batch_size)
valid_gen = get_data_generator(df, valid_idx, for_training=True, batch_size=valid_batch_size)

callbacks = [ModelCheckpoint("./model_checkpoint", monitor='val_loss')]

history = model.fit_generator(train_gen,
                    steps_per_epoch=len(train_idx)//batch_size,
                    epochs=EPOCH,
                    callbacks=callbacks,
                    validation_data=valid_gen,
                    validation_steps=len(valid_idx)//valid_batch_size)


# In[ ]:


def  plot_train_history(history):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    
    axes[0].plot(history.history['gender_output_accuracy'], label='Gender Train accuracy')
    axes[0].plot(history.history['val_gender_output_accuracy'], label='Gener Val accuracy')
    axes[0].set_xlabel('Epochs')
    axes[0].legend()

    axes[1].plot(history.history['age_output_mae'], label='Age Train MAE')
    axes[1].plot(history.history['val_age_output_mae'], label='Age Val MAE')
    axes[1].set_xlabel('Epochs')
    axes[1].legend()  

    axes[2].plot(history.history['loss'], label='Training loss')
    axes[2].plot(history.history['val_loss'], label='Validation loss')
    axes[2].set_xlabel('Epochs')
    axes[2].legend()

plot_train_history(history)


# In[ ]:


test_gen = get_data_generator(df, test_idx, for_training=False, batch_size=128)
dict(zip(model.metrics_names, model.evaluate_generator(test_gen, steps=len(test_idx)//128)))


# In[ ]:


test_gen = get_data_generator(df, test_idx, for_training=False, batch_size=128)
x_test, (age_true, gender_true)= next(test_gen)
age_pred, gender_pred = model.predict_on_batch(x_test)


# In[ ]:


gender_true = gender_true.argmax(axis=-1)
gender_pred = gender_pred.argmax(axis=-1)
age_true = age_true * max_age
age_pred = age_pred * max_age


# In[ ]:


n = 30
random_indices = np.random.permutation(n)
n_cols = 6
n_rows = math.ceil(n / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))
for i, img_idx in enumerate(random_indices):
    ax = axes.flat[i]
    ax.imshow(x_test[img_idx])
    ax.set_title('Predict a:{}, g:{}'.format(int(age_pred[img_idx]), ID_GENDER_MAP[gender_pred[img_idx]]))
    ax.set_xlabel('Real a:{}, g:{}'.format(int(age_true[img_idx]), ID_GENDER_MAP[gender_true[img_idx]]))
    ax.set_xticks([])
    ax.set_yticks([])

