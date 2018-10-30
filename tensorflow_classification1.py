# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

## Functions below copied from TF website, used to graph confidence that prediction
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist #this is a data set with 70,000 grayscale images of clothing, imported from tensorflow
### above treturns four numpy arrays 
## images for training, labels for training, images for testing, labels for testing 
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
### labels are an array of integers from 0 to 9 (0 - Tshirt 1- Trouser 2- Pullover etc.)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#train_images.shape ## shows there are 60,000 images with each image representing 28 x 28 pixels 
print(len(train_labels)) # 60,000


## Now lets preprocess the data 
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False) 

#lets scale down the pixel values from 0-255 to range between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0 

## display first 25 images of training set, with class names
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


## BUILD the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
## flatten - transforms images into 1d array 
## Dense- 128 nodes in first layer, 10 node softmax later (10 prob. scores that sum to 1)
## each node in 10 node layer returns prob that image belongs to one of 10 classes

## Compile model with settings 
model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

## optimizer - how model is updated based on data it sees
## loss function - how accurate model is during training. minimize this function
## metrics - monitors training and testing steps, 'accuracy' bases it off fraction of images that are correctly classified 

## TRAINING THE MODEL
model.fit(train_images, train_labels, epochs=5, verbose = 0) ## epoch is time step incremented
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc, 'Test Loss: ', test_loss)
train_loss, train_acc = model.evaluate(train_images, train_labels)
print('Train accuracy:', test_acc, 'Train Loss: ', test_loss)
## Why are they the same? 
## accuracy is about 88%, which is supposedly less than train images


## Make predictions 
predictions = model.predict(test_images) ## model predicts label for each image in test_images
print(str(predictions[0])) ## returns array of 10 nums describing the "confidence" of the model that corresponds to each of 10 classes for image 0
print(str(np.argmax(predictions[0])))
print(str(test_labels[0]))

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)

plt.show()

