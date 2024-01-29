from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense, Dropout, Flatten
from keras.optimizers import Adam
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_dir = "data/train"
val_dir = "data/test"

train_data = train_datagen.flow_from_directory(directory=train_dir,
                                               batch_size=32,
                                               target_size=(48,48),
                                               color_mode="grayscale",
                                               class_mode='categorical')
val_data = val_datagen.flow_from_directory(directory=val_dir,
                                               batch_size=32,
                                               target_size=(48,48),
                                               color_mode="grayscale",
                                               class_mode='categorical')

model = Sequential([

    Conv2D(32,3,activation='relu',input_shape=(48,48,1)),
    Conv2D(64,3,activation='relu'),
    MaxPooling2D(2),
    Dropout(0.25), # Overfitting

    Conv2D(128, 3, activation='relu'),
    MaxPooling2D(2),
    Conv2D(128, 3, activation='relu'),
    MaxPooling2D(2),
    Dropout(0.25),

    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(7,activation='softmax')

])

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

model_history = model.fit(train_data,
                          epochs=50,
                          steps_per_epoch=len(train_data),
                          validation_data=val_data,
                          validation_steps=len(val_data))

model_json = model.to_json()
with open("model.json","w") as json_file:
  json_file.write(model_json)

model.save_weights('model.h5')