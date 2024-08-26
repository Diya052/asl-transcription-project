import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import tensorflow as tf


train_dataset_path = 'data/training'
test_dataset_path = 'data/test'

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(train_dataset_path, color_mode = 'rgb', class_mode = 'categorical', shuffle = 'True')

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)

test_generator = test_datagen.flow_from_directory(test_dataset_path, color_mode = 'rgb', class_mode = 'categorical', shuffle = 'True')

labels = {value: key for key, value in train_generator.class_indices.items()}
print("Labels for all classes\n")
for key, value in labels.items():
    print(f"{key} : {value}")



