# Deep Learning Practice using Tensorflow

Here I'm gonna write all the important tensorflow features needed for solving Deep Learning problems

**tf.expand_dims(X, axis=-1)**: This code add an extra dimension to X on the last axis.

**tf.constant(X)**: This is used to convert numpy arrays into tensors.

**shift+ctrl+space** : This is the keyboard shortcut in google colab to get a short description of a function.

**ctrl+space** : This is a keyboard shortcut in google colab to get multiple options which can be filled in as attributes

**tf.cast(X, dtype=tf.float32)** : This can be used to change the datatype of X.

## Steps in modelling with Tensorflow

1. **Creating a model** - define the input and output layers, as well as the hidden layers of a deep learning model
2. **Compiling a model** - define the loss function (in other words, the function which tells our model how wrong it is compared to the truth labels) and the optimizer (tells how your model should update its internal patterns to better its prediction) and evaluation metrics (human interpretable values to know how well the model is doing)
3. **Fitting a model** - letting the model try to find patterns between X & y (features and labels). Epochs (how many times the model will go through all of the training examples)
4. **Evaluate the model**: We need to evaluate the model on the test data (To know how reliable are our model predictions)

### Simple example code

```
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

X = np.array([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])
y = np.array([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])

# Set random seed 
tf.random.set_seed(42)

# 1. Create a model using the Sequential API
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])

# 2. Compile the model
model.compile(loss=tf.keras.losses.mae, # mean absolute error
              optimizer=tf.keras.optimizers.SGD(), # stochastic gradient descent
              metrics=["mae"])

# 3. Fit the model
model.fit(tf.expand_dims(X, axis=-1), y, epochs=5)
```

## Loss functions

Common loss functions used in deep learning models are:

1. **Mean Squared Error (MSE)**:
    * Measures the average squared difference between the predicted and actual values.
    * Suitable for regression problems.
    * MSE = mean(square(y_true - y_pred), axis=-1)

2. **Mean Absolute Error (MAE)**:
    * Measures the average absolute difference between the predicted and actual values.
    * Particulary useful when dealing with outliers, as it is less sensitive to extreme values.
    * MAE = mean(abs(y_true - y_pred), axis=-1)

3. **Binary Cross-Entropy (Log Loss)**:
    * Measures the binary classification loss by comparing predicted probabilities to actual labels
    * Suitable for binary classification problems

4. **Categorical Cross-Entropy**:
    * Measures the multiclass classification loss by comparing predicted class probabilities to one-hot encoded labels.
    * Suitable for multiclass classification problems

## Optimizers

Optimizers are algorithms used to adjust the weights of a neural network to minimize the loss function. Commonly used optimizers in deep learning are: 

1. Stochastic Gradient Descent (SGD)
2. Momentum Stochastic Gradient Descent
3. RMSprop
4. Adam (Adaptive Moment Estimation)

Each optimizer has its own strengths and is chosen based on the specific needs of the model and the nature of the data. Adam is widely used due to its adaptive learning rate and momentum properties, making it suitable for most deep learning tasks

**NOTE**: We don't need to get into the mathematical details of each. We can just make use of these as they are pre-defined in frameworks like tensorflow and pytorch.

## Metrices

In deep learning, different metrics are used to evaluate the performance of models depending on the type of problem (regression, classification, etc.). Here are some commonly used metrics along with brief explanations:

### Classification Metrics

1. **Accuracy**:
    * Measures the proportion of correctly classified instances out of the total instances.
    * Suitable for balanced datasets.
    * Accuracy = (Number of correct predictions) / (Total number of predictions)

2. **Precision**:
    * Measures the proportion of true positive out of all positive predictions.
    * When the cost of false positive prediction is high (ex - predicting whether a person has a disease or not) we can choose precision as the metric. The model with high precision i.e low false positives will be selected.
    * Works well with both balanced and unbalanced datasets.
    * Precision = (True Positive) / (Total positive predictions) or (True positive + False positive)

3. **Recall** (Sensitivity or True positive rate):
    * Measures the proportion of true positive predictions out of all actual positive instances.
    * Works well for both balanced and unbalanced datasets.
    * It is used to generate ROC-AUC curves (which is another metric to compare models)
    * Recall = (True Positives) / (Actual Total positives (not predictions)) or (True positive + False Negative)

4. **F1 Score**:
    * Harmonic mean of precision and recall.
    * Provides a single metric that balances precision and recall
    * Most used (except for cases with specific requirements)
    * F1 Score = 2 * ((precision * recall) / (precision + recall))

5. **ROC-AUC**:
    * ROC-AUC stands for Receiver Operation Characteristic - Area Under the Curve
    * Used for binary classification (different specialised versions can be used for multiclass classification)
    * We draw ROC curve as True positive rate (sensitivity or recall) vs False positive rate.
    * False positive rate measures the proportion of false positive predictions out of all actual negative instances.
    * False positive rate = (False positive) / (False positive + True Negative)
    * The curve has True positive rate as y-axis and False positive rate as x-axis
    * For different threshold value points are plotted onto the graph.
    * In the context of ROC-AUC, a threshold is a value used to determine the cutoff point at which the model classifies an instance as belonging to the positive class or negative class.
    * Recall and AUC are in direct corespondence, higher the Recall: higher the AUC, lower the Recall: lower the AUC.
    * Higher recall generally means a better model.
    * The highest value for AUC is 1 (when model predict all correct value for any threshold i.e recall has a value of 1 for every threshold)
    * Higher AUC means that the model has higher recall value for different threshold. So, we can say that in general a model with higher AUC is better (but it can be case dependent)

6. **Classification Report**:
    * It is a table that summarizes the performance of a classification model.
    * It contains different metrices:
        * Accuracy
        * Precision
        * Recall
        * F1 Score

7. **Confusion Matrix**:
    * A confusion matrix is a performance measurement tool for classification problems.
    * It shows the number of correct and incorrect predictions made by a classifier, broken down by each class.
    * The matrix helps visualize the performance of a classification model and can be used to compute various evaluation metrics like accuracy, precision, recall, and F1 score.

NOTE: 'Precision', 'Recall', 'F1 Score' and 'ROC-AUC' are all metrices used for binary classification. To make them work for multiclass classification we will have to use averaging methods like macro-averaged or micro-averaged. 'Accuracy' and 'Confusion Matrix' can be used for both binary and multiclass classification.

### Regression Metrics

1. **Mean Absolute Error (MAE)**
2. **Mean Squared Error (MSE)**
3. **Root Mean Squared Error (RMSE)**