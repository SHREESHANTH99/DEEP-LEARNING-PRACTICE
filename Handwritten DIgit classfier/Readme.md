# Handwritten Digit Classifier
This project is a simple handwritten digit classifier using a neural network built with TensorFlow and Keras. The model is trained on the MNIST dataset, which consists of 60,000 training images and 10,000 testing images of handwritten digits (0-9).
## Dataset
The MNIST dataset is a collection of grayscale images of handwritten digits. Each image is 28x28 pixels in size. The dataset is divided into a training set of 60,000 images and a test set of 10,000 images. The labels for the images are the corresponding digits (0-9).
## Preprocessing
Before training the model, the pixel values of the images are normalized to a range of 0 to 1 by dividing by 255. This helps the model converge faster during training.
```python
X_train = X_train / 255.0
X_test = X_test / 255.0
```
## Model Architecture
The neural network model consists of the following layers:
1. **Flatten Layer**: This layer flattens the 28x28 input images into a 1D array of 784 pixels.
2. **Dense Layer**: A fully connected layer with 128 neurons and ReLU activation function.
3. **Dense Layer**: Another fully connected layer with 64 neurons and ReLU activation function.
4. **Output Layer**: A fully connected layer with 10 neurons (one for each digit) and a softmax activation function to output probabilities for each class.
```python
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
```
## Model Compilation
The model is compiled using the Adam optimizer and sparse categorical cross-entropy loss function. The accuracy metric is used to evaluate the model's performance.
```python
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```
### Model Training
The model is trained for 10 epochs with a validation split of 20% of the training data. The training history is recorded for later analysis.
```python
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)
``` 
### Model Evaluation
After training, the model is evaluated on the test set to determine its accuracy. Predictions are made on the test set, and the accuracy score is calculated using the true labels and predicted labels.
```python
y_prob = model.predict(X_test)
y_pred = y_prob.argmax(axis=1)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
```
### Results
The model achieves a accuracy of 97.4% on the test set, which indicates how well it can classify handwritten digits. The loss and validation loss during training can be visualized using the following code:
```python
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
