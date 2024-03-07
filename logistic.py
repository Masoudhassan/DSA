from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
iris = load_iris()  # load iris dataset
X = iris.data  # feature matrix
y = iris.target  # target vector

# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4)

# create a logistic regression model
logreg = LogisticRegression()

# train the model on the training data
logreg.fit(X_train, y_train)

# make predictions on the test data
y_pred = logreg.predict(X_test)

# calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)