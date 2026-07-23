# Credit Card Customer Churn Prediction
This project focuses on predicting customer churn for a credit card company using a deep learning model. The goal is to identify customers who are likely to leave the service, allowing the company to take proactive measures to retain them.
## Dataset
The dataset used for this project contains information about credit card customers, including demographic details, account information, and transaction history. The target variable is a binary indicator of whether a customer has churned (1) or not (0).
## Data Preprocessing
1. **Loading the Data**: The dataset was loaded into a Pandas DataFrame for analysis.
2. **Handling Missing Values**: Any missing values in the dataset were identified and handled appropriately, either by imputing or removing them.
3. **Encoding Categorical Variables**: Categorical features were encoded using one-hot encoding to convert them into a numerical format suitable for model training.
4. **Feature Scaling**: The features were scaled using standardization to ensure that they have a mean of 0 and a standard deviation of 1, which helps improve the performance of the neural network.
## Model Building
A deep learning model was built using the Keras library. The architecture of the model is as follows:
- **Input Layer**: The input layer consists of 11 neurons corresponding to the 11 features in the dataset.
- **Hidden Layers**: Two hidden layers were added, each with 11 neurons and ReLU activation functions. These layers help the model learn complex patterns in the data.
- **Output Layer**: The output layer has a single neuron with a sigmoid activation function, which outputs a probability value between 0 and 1, indicating the likelihood of churn.
## Model Compilation
The model was compiled using the binary cross-entropy loss function, which is suitable for binary classification problems. The Adam optimizer was used for training, and accuracy was chosen as the evaluation metric.
## Model Training
The model was trained on the training dataset for 100 epochs, with a validation split of 20%. The training process involved feeding the scaled features and corresponding labels into the model, allowing it to learn the relationships between the input features and the target variable.
## Model Evaluation
After training, the model's performance was evaluated on the test dataset. Predictions were made using the trained model, and the predicted probabilities were converted into binary labels using a threshold of 0.5. The accuracy of the model was calculated by comparing the predicted labels with the actual labels in the test set.
## Results
The model achieved an accuracy of approximately 86% on the test dataset.
