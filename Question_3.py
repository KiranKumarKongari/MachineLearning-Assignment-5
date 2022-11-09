# 3. Apply Linear Discriminant Analysis (LDA) on Iris.csv dataset to reduce dimensionality of data to k=2.

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

# Iris_dataset is a dataframe where we load the csv data.
iris_dataset = pd.read_csv("C:/Users/Kiran Kumar Kongari/PycharmProjects/ML-Assignment-5/Datasets/Iris.csv")
print("\nThe Original Iris Dataframe is : \n", iris_dataset)

# Dropping the 'Id' column as it is not required in the analysis.
iris_dataset.drop(['Id'], axis=1, inplace=True)

x = iris_dataset.iloc[:, [1, 2, 3]].values
y = iris_dataset.iloc[:, -1].values

# a. Performing Scaling
scaler = StandardScaler()
X_Scale = scaler.fit_transform(x)

# Implementing Linear Discriminant Analysis (LDA) to reduce dimensionality of data to k=2.
lda = LinearDiscriminantAnalysis(n_components=2)
X_train_lda = lda.fit_transform(X_Scale, y)
finalDf = pd.DataFrame(X_train_lda)
finalDf['class'] = y
finalDf.columns = ["LD1", "LD2", "class"]
print("\nThe Dataframe after applying LDA : \n", finalDf)

# Plotting the graph
markers = ['s', 'x', 'o']
colors = ['r', 'b', 'g']
sns.lmplot(x="LD1", y="LD2", data=finalDf, hue='class', markers=markers, fit_reg=False, legend=False)
plt.legend(loc='upper center')
plt.show()

