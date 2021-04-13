import os
import sys
from matplotlib import pyplot
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Hide tensorflow messages


# VGG16 Transfer learning
def define_model():
    model = VGG16(include_top=False, input_shape=(200, 200, 3))  # Load VGG16 pretrained model
    for layer in model.layers:
        layer.trainable = False  # Deactivate the training for all layers
    flat1 = Flatten()(model.layers[-1].output)  # Flatten the colvolutional base output
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)  # Add new layer
    output = Dense(3, activation='softmax')(class1)  # Add new layer
    model = Model(inputs=model.inputs, outputs=output)  # Define the Classifier
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Plot Loss and Accuracy
def summarize_diagnostics(history):
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(filename + '_plot.png')
    pyplot.close()


def console():
    model = define_model()
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_it = datagen.flow_from_directory('Dataset/Train/',
                                           class_mode='categorical', batch_size=12, target_size=(200, 200))
    test_it = datagen.flow_from_directory('Dataset/Test/',
                                          class_mode='categorical', batch_size=12, target_size=(200, 200))

    history = model.fit(train_it, steps_per_epoch=len(train_it),
                        validation_data=test_it, validation_steps=len(test_it), epochs=10, verbose=1)
    _, acc = model.evaluate(test_it, steps=len(test_it), verbose=1)
    print('> %.3f' % (acc * 100.0))
    summarize_diagnostics(history)
    model.save('model.h5')


console()
