# 2. Use pd_speech_features.csv
#     a. Perform Scaling
#     b. Apply PCA (k=3)
#     c. Use SVM to report performance

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
import warnings

warnings.filterwarnings('ignore')

# pd_speech_dataset is a dataframe where we load the csv data.
pd_speech_dataset = pd.read_csv("C:/Users/Kiran Kumar Kongari/PycharmProjects/ML-Assignment-5/Datasets/"
                                "pd_speech_features.csv")
print("\nThe Dataframe is : \n", pd_speech_dataset)

x = pd_speech_dataset.drop('class', axis=1).values
y = pd_speech_dataset['class'].values

# a. Performing Scaling
scaler = StandardScaler()
X_Scale = scaler.fit_transform(x)

# b. Applying PCA (k=3)
pca2 = PCA(n_components=3)
principalComponents = pca2.fit_transform(X_Scale)
principalDf = pd.DataFrame(data=principalComponents,
                           columns=['principal component 1', 'principal component 2', 'principal component 3'])
finalDf = pd.concat([principalDf, pd_speech_dataset[['class']]], axis=1)
print("\nThe Dataframe after applying PCA : \n", finalDf)

x = finalDf.drop('class', axis=1).values
y = finalDf['class'].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Classification using Linear Support Vector Machine's
print("\n-----------------------------------------------------------------------------------------\n"
      "Classification using Linear Support Vector Machine : \n")

classifier = LinearSVC(verbose=0)

# Calculating training data accuracy score
y_pred = classifier.fit(X_train, y_train).predict(X_train)
train_accuracy = accuracy_score(y_pred, y_train)*100
print('Accuracy for our Training dataset with PCA is: %.4f %%' % train_accuracy)

# Calculating testing data accuracy score
y_pred = classifier.fit(X_train, y_train).predict(X_test)
test_accuracy = accuracy_score(y_pred, y_test)*100
print('Accuracy for our Testing dataset with Tuning is: %.4f %%' % test_accuracy)

