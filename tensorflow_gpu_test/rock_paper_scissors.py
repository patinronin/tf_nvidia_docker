import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

def create_train_validator(TRAINING_DIR):
    training_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    train_generator = training_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=(150, 150),
        class_mode='categorical',
        batch_size=126
    )
    return train_generator

def create_test_validator(VALIDATION_DIR):
    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=(150, 150),
        class_mode='categorical',
        batch_size=126
    )
    return validation_generator

def create_model():
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

def train_model(model, train_generator, validation_generator):
    history = model.fit(
        train_generator,
        epochs=25,
        steps_per_epoch=20,
        validation_data=validation_generator,
        verbose=1,
        validation_steps=3
    )
    model.save("rps.h5")


def predict(path, model):
    img = image.load_img(path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    print(classes)

def full_process():
    container_route = os.getcwd() + "/tensorflow_gpu_test"
    rock_dir = os.path.join(container_route + '/data/rps/rock')
    paper_dir = os.path.join(container_route + '/data/rps/paper')
    scissors_dir = os.path.join(container_route + '/data/rps/scissors')

    print('total training rock images:', len(os.listdir(rock_dir)))
    print('total training paper images:', len(os.listdir(paper_dir)))
    print('total training scissors images:', len(os.listdir(scissors_dir)))

    TRAINING_DIR = container_route + "/data/rps/"
    VALIDATION_DIR = container_route + "/data/rps-test-set/"
    path = container_route + "/data/rps-test-set/paper/testpaper01-00.png"
    train_generator = create_train_validator(TRAINING_DIR)
    validation_generator = create_test_validator(VALIDATION_DIR)
    model = create_model()
    train_model(model, train_generator, validation_generator)
    predict(path, model)

