# BrutDL
A Keras-like simple deep learning library using only numpy and the python standard library.

This library is inspired by the coursera deepleaning.ai Deep Learning Specialization.


## Simple Demo
A simple use of the library to classify written digits:

First, import the required modules.
from the Keras module import the mnist written digits dataset, and from the BrutDL library import:

The 'Model' class - to create a neural network.

The 'FcLayer' class - the fully connected layer class to add to the model.

The 'Data' class - holds your dataset.
```python
from keras.datasets import mnist
from BrutDL import Model, FcLayer, Data
```

Then, get the mnist dataset (as numpy arrays) and store these arrays in a 'Data' object.
The data is formatted to the right shape and the labels are converted to one-hot arrays.
```python
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 784) / 255.
X_test = X_test.reshape(X_test.shape[0], 784) / 255.

data = Data(train_data=(X_train, Y_train), val_data=(X_test, Y_test),
            start_dim=784, end_dim=10, to_one_hot=True, val_is_test=True)
print(data)
```
output:
```
Data:
	Train: [(784, 60000), (10, 60000)]
	Test: [(784, 10000), (10, 10000)]
	Valid: [(784, 10000), (10, 10000)]
```

Now, create your model and add as its layers, it can be done by passing them when creating the model, or later with the add() function.
```python
model = Model(input_shape=784, loss='mse',
              layers=[
                  FcLayer(100, activation='relu'),
                  FcLayer(50, activation='relu'),
                  FcLayer(50, activation='relu'),
                  FcLayer(10, activation='softmax')
              ])
print(model)
```
output:
```
Model:
	Fully Connected Layer: [784 -> 100]
	RELU Activation Layer
	Fully Connected Layer: [100 -> 50]
	RELU Activation Layer
	Fully Connected Layer: [50 -> 50]
	RELU Activation Layer
	Fully Connected Layer: [50 -> 10]
	SOFTMAX Activation Layer
```

Lastly, train your model using the data object, get the train and test accuracies, and plot the results.
```python
model.train(data, lr=0.2, batch_size=64, epochs=10, lr_descent=True)

model.classifier_accuracy(data.train_set, type_='Train')
model.classifier_accuracy(data.test_set, type_='Test')

model.plot_cost()
```
output:
```
epoch 0 [==============================>] Cost: 0.005715, Lr: 0.000002, Val Cost: 0.007325
epoch 1 [==============================>] Cost: 0.005043, Lr: 0.000002, Val Cost: 0.005679
epoch 2 [==============================>] Cost: 0.004246, Lr: 0.000002, Val Cost: 0.004991
epoch 3 [==============================>] Cost: 0.005170, Lr: 0.000002, Val Cost: 0.004587
epoch 4 [==============================>] Cost: 0.006866, Lr: 0.000002, Val Cost: 0.004308
epoch 5 [==============================>] Cost: 0.006463, Lr: 0.000002, Val Cost: 0.004141
epoch 6 [==============================>] Cost: 0.001534, Lr: 0.000002, Val Cost: 0.003941

DONE

pred: [5 0 4 1 9 2 1 3 1 4]
true: [5 0 4 1 9 2 1 3 1 4]
Train Accuracy: 0.9476500000000001

pred: [7 2 1 0 4 1 4 9 6 9]
true: [7 2 1 0 4 1 4 9 5 9]
Test Accuracy: 0.9479000000000003
```
![Output Of model.plot_cost()](/imgs/mnist_plot_1.png)
