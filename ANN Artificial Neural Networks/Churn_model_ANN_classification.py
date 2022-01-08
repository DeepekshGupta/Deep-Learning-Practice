# Importing Libraries
import numpy as np
import pandas as pd
import tensorflow as tf

# ----------------------------------------------Data Preprocessing---------------------------------------------------------------------------

dataset = pd.read_csv('Datasets\Churn_Modelling_ANN.csv')
X = dataset.iloc[:, 3:-1].values #(Matrix of fearues) |  the first three columns of our dataset are irrelevant to our model as row no, customerID and surname wont affect our prediction 
y = dataset.iloc[:, -1].values # (dependent variables)


# Encodeing categorical data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

# One hot encodeing the 'Geography' column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# ----------------------------------------------Building the ANN---------------------------------------------------------------------------

# initializing the ANN - object to build the ANN as a sequence of layers 
ann = tf.keras.models.Sequential()

# we use the add method to add the dense class which adds a fully connected layer to our ANN (units = number of hidden neurons, activation = "activation function")  
# the number of neurons cant be determined, we have to get it through trial and error, this is one of the HYPER PARAMETERS
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
# similarly we can add another hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Output Layer
# units = 1 because we only want 1 output value, and sigmoid function because we want a binary output with probabilities, if our output was categorical, we'd use softmax instead of sigmoid
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# ----------------------------------------------Training the ANN---------------------------------------------------------------------------

# now we complile the ANN
# adam optimizer is basically stochastic Gradient decent optimizer : 'adam'
# loss function is binary crossentropy because our output is binary, if we had a categorical classification output, we'd use "categorical_crossentropy" a
# metric is taken as accuracy, however we could have multiple metrics at the same time
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# for Training the ANN, we use fit method
# batch learning is always better thast why batch size, default value is 32
# more the epochs more the accuracy
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)



# ----------------------------------------------Making the predictions and Model evaluation---------------------------------------------------------------------------

# single prediction
# print(ann.predict(sc.transform([[France(1,0,0), creditScore(600), Male(1), Age(40), tenure(3), Balance(60k), Products(2), creditCard?(1), ActiveMember?(1), Estimated Salary(50k)]])) > 0.5)
print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)# return 1 if value is greater than 0.5

# predictig on test set
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5) # return 1 if value is greater than 0.5
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))