from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Flatten, Dropout, MaxPooling2D, Conv2D
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import numpy as np
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


batch_size = 32
epochs = 15
img_height = 150
img_width = 150

Path = "/home/ragon/Tensorflow_python/venv/project_CNN/FaceShape"

train_dir = os.path.join(Path,'training_set')
validation_dir = os.path.join(Path,'testing_set')

#data preparation

train_image_generator = ImageDataGenerator(rescale = 1.0/255,rotation_range = 45, width_shift_range =.15,height_shift_range = .15,horizontal_flip = True,zoom_range = 0.5,)
validation_image_generator = ImageDataGenerator(rescale = 1.0/255)

train_data = train_image_generator.flow_from_directory(batch_size = batch_size,directory = train_dir,shuffle = True,target_size = (img_height,img_width),class_mode = 'sparse')

validation_data = validation_image_generator.flow_from_directory(batch_size = batch_size,directory = validation_dir,target_size = (img_height,img_width),class_mode = 'sparse')


sample_data,_ = next(train_data)


def plotimages(images_arr):
	fig, axes = plt.subplots(1,5,figsize=(20,20))
	axes = axes.flatten()
	for img,ax in zip(images_arr,axes):
		ax.imshow(img)
		ax.axis('off')
	plt.tight_layout()
	plt.show()

plotimages(sample_data[:5])

#create model

model = models.Sequential([
	Conv2D(16,3,padding = 'same', activation = 'relu' , input_shape = (img_height,img_width,3)),
	MaxPooling2D(),
	Dropout(0.2),
	Conv2D(32,3,padding = 'same', activation = 'relu'),
	MaxPooling2D(),
	Conv2D(64,3,padding = 'same', activation = 'relu'),
	MaxPooling2D(),
	Dropout(0.2),
	Flatten(),
	#Dense(512,activation = 'relu'),
	Dense(512,activation = 'relu'),
	Dense(5,activation = 'softmax')

	])


#compiling

model.compile(optimizer = 'adam',loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])


model.summary()

#traininng


history = model.fit_generator(
	train_data,
	steps_per_epoch = 250,
	epochs = epochs,
	validation_data = validation_data,
	validation_steps = 100
	)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

test_los, test_ac = model.evaluate_generator(validation_data, verbose=2)
print(test_ac)
model.save("model_data_trial.h5")







