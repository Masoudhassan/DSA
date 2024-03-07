from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.cluster import KMeans
iris = load_iris()  # load iris dataset
X = iris.data  # feature matrix
y = iris.target  # target vector

# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4)
# create a logistic regression model
model = LogisticRegression()

# train the model using the training data
model.fit(X_train, y_train)

# make predictions on the test data
y_pred = model.predict(X_test)

# calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

# print the accuracy
print("Accuracy:", accuracy)
# create a support vector machine model
svm_model = SVC()

# train the model using the training data
svm_model.fit(X_train, y_train)

# make predictions on the test data
svm_y_pred = svm_model.predict(X_test)

# calculate the accuracy of the model
svm_accuracy = accuracy_score(y_test, svm_y_pred)

# print the accuracy
print("SVM Accuracy:", svm_accuracy)
# create a k-means model
kmeans_model = KMeans(n_clusters=3)

# train the model using the feature matrix
kmeans_model.fit(X)

# make predictions on the data
kmeans_labels = kmeans_model.predict(X)

# print the predicted labels
print("K-Means Labels:", kmeans_labels)