from glob import glob
import sys
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator


pathTrain = '/content/drive/My Drive/Imagenes_DL/train'
pathTest  = '/content/drive/My Drive/Imagenes_DL/test'
filesTrain = glob(pathTrain + "/*")
print(filesTrain)
filesTest = glob(pathTest + "/*")
print(filesTrain)


# model generation
InceptionV3_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in InceptionV3_model.layers[:-15]:
    layer.trainable = False

x = InceptionV3_model.output
x = GlobalAveragePooling2D()(x)
x = Flatten()(x)

#Adding the Hidden layer
x = Dense(units=512, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(units=512, activation='relu')(x)
x = Dropout(0.3)(x)

#Adding the Output Layer
output = Dense(units=5, activation='softmax')(x)

model = Model(InceptionV3_model.input, output)

model.summary()

# model compilation
loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=loss, metrics= ['accuracy'])

dataGenerator = ImageDataGenerator(rescale = 1./255)
train_set = dataGenerator.flow_from_directory(pathTrain,
                                              target_size = (224, 224),
                                              batch_size = 32,
                                              class_mode = 'categorical')

test_set = dataGenerator.flow_from_directory(pathTest,
                                             target_size = (224, 224),
                                             batch_size = 32,
                                             class_mode = 'categorical')


# Model training
print(len(train_set))
print(len(test_set))
iTime = time.time()
trainModel = model.fit(train_set,
                       validation_data=test_set,
                       epochs=20,
                       steps_per_epoch=len(train_set),
                       validation_steps=len(test_set))
dTime = (time.time() - iTime)/60
print()
print('Model Fitness time (min): ')
print(dTime)

# plot the loss
plt.figure()
plt.plot(trainModel.history['loss'], 'o-b', label='train_loss')
plt.plot(trainModel.history['val_loss'], 'o-r', label='val_loss')
plt.grid()
plt.legend()

# plot the accuracy
plt.figure()
plt.plot(trainModel.history['accuracy'], 'o-b', label='train_acc')
plt.plot(trainModel.history['val_accuracy'], 'o-r', label='val_acc')
plt.grid()
plt.legend()

plt.show()

# prediction with models
val_pred = model.predict(test_set)
val_pred = np.argmax(val_pred, axis=1)
print(val_pred)


# Evaluate model

states = ["Apoptosis", "Confluency", "Growing", "Semi-Apoptosis", "Semi-Confluency"]

val_true = np.array(test_set.classes)
cf_matrix = confusion_matrix(val_pred, val_true)

ax = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues')

ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Real Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(np.arange(1, 6))
ax.yaxis.set_ticklabels(np.arange(1, 6))

## Display the visualization of the Confusion Matrix.
plt.show()

# predicted with images and classes
def image_predicted(img):
    fig = plt.figure(figsize=(40, 8))
    ax1 = fig.add_subplot(121)
    print()
    print(
        '-----------------------------------------------------------------------------------------------------------------')
    print(
        '--------------------------------->>>>>    Cell State Detection    <<<<<------------------------------------------')
    print(
        '-----------------------------------------------------------------------------------------------------------------')
    print()
    ax1.imshow(img)
    ima = image.img_to_array(img)
    ima = ima / 255
    ima = np.expand_dims(ima, axis=0)
    print("State predicted: ", states[np.argmax(model.predict(ima))])
    print()

    ax2 = fig.add_subplot(122)
    ax2.barh(states, model.predict(ima)[0], align='center')
    print("Probability prediction vector: ", model.predict(ima)[0])
    print()
    print(
        '----------------------------------------------------------------------------------------------------------------')
    print()