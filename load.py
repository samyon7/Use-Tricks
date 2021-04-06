import tensorflow as tf

IMG_SIZE = 350

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=36,
    width_shift_range=0.4,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
  '/content/OOP_Model/files/training',             
  target_size=(IMG_SIZE,IMG_SIZE),  
  batch_size=32,
  class_mode='categorical'
)

validation_generator = train_datagen.flow_from_directory(
  '/content/OOP_Model/files/validation',
  target_size=(IMG_SIZE,IMG_SIZE),
  batch_size=32,
  class_mode='categorical'
)