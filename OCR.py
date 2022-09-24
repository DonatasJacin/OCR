import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.mnist
(trainImages, trainLabels), (testImages, testLabels) = data.load_data()
trainImages = trainImages / 255.0
testImages = testImages / 255.0

classNames = ['0','1','2','3','4','5','6','7','8','9']
network = keras.Sequential([
	keras.layers.Flatten(input_shape = (28,28)),
	keras.layers.Dense(256, activation = 'sigmoid'),
	keras.layers.Dense(10, activation = 'softmax') #Softmax used as it ensures the values of the 10 output neurons adds up to 1, which is exactly whats needed as the program
	])											   #should output to what extent it thinks the image is each number, and if the probabilities are assigned correctly, the total 
												   #would be 1. Also, there are more than 2 classes, so an activation which works well with 2 categories of target labels such as the sigmoid
												   #is not appropriate for the output layer.

network.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]) #categorical entropy because multiple classes are used, binary is insufficient
network.fit(trainImages, trainLabels, epochs = 20)

prediction = network.predict(testImages)	

correct = 10000
incorrectLabel = []
incorrectImage = []
incorrectActual = []
for i in range(10000):
	if classNames[np.argmax(prediction[i])] != classNames[testLabels[i]]:
		correct = correct - 1 
		incorrectLabel.append(classNames[np.argmax(prediction[i])])
		incorrectImage.append(testImages[i])
		incorrectActual.append(classNames[testLabels[i]])
print(correct)

for i in range(10): #Displays 10 incorrect predictions
	plt.grid(False)
	plt.imshow(incorrectImage[i], cmap = plt.cm.binary)
	plt.title("Prediction: " + incorrectLabel[i])
	plt.xlabel("Actual: " + incorrectActual[i])
	plt.show()