import tensorflow as tf
from load import *

model = tf.keras.models.Sequential([ 
    tf.keras.layers.Conv2D(32, (3,3), activation='elu', input_shape=(IMG_SIZE, IMG_SIZE, 3) ),
    tf.keras.layers.MaxPooling2D(2,2), 
    tf.keras.layers.Conv2D(64, (3,3), activation='selu', kernel_initializer='he_uniform'), 
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='selu', padding='same'), 
    tf.keras.layers.MaxPooling2D(2,2), 
    tf.keras.layers.Conv2D(128, (3,3), activation='elu', kernel_initializer='lecun_normal'), 
    tf.keras.layers.MaxPooling2D(2,2), 
    tf.keras.layers.Conv2D(128, (3,3), activation='exponential'), 
    tf.keras.layers.MaxPooling2D(2,2), 
    tf.keras.layers.Dropout(0.6),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform'), 
    tf.keras.layers.Dense(4, activation='softmax'),
]) 

model.compile(optimizer=tf.keras.optimizers.Nadam(amsgrad=True,learning_rate=0.00001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy') > 0.95):
      print("\nAccuracy better than target training!")
      self.model.stop_training = True

callbacks = myCallback()

history = model.fit(
    train_generator,
    epochs = 100,
    validation_data = validation_generator,
    callbacks=[callbacks]
)