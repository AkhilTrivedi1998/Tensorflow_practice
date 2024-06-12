# Neural Network Regression

Regression is a statistical method used to understand the relationship between one dependent variable and one or more independent variables. It is a fundamental tool in data analysis and predictive modeling, allowing us to make predictions about the dependent variable based on the values of the independent variables.

The input to the regression model should be numerical. So, we need to convert to categorical data to numerical format before passing it to the model. 
**One-hot encoding** is a technique used in machine learning and data preprocessing to represent categorical variables as binary vectors.

Example:

    Suppose you have a categorical variable "Color" with three categories: Red, Green, and Blue.
    After one-hot encoding, each category is represented as follows:
        Red: [1, 0, 0]
        Green: [0, 1, 0]
        Blue: [0, 0, 1]

**Important**: When applying one-hot encoding to testing data, it's crucial to ensure consistency with the encoding applied to the training data. This can be done by saving the encoding schemas which can be achieved using pd.get_dummies() in pandas or OneHotEncoder in scikit-learn (it is pre-defined feature of these functions)

