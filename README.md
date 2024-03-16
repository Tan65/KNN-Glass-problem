# KNN-Glass-problem
repare a model for glass classification using KNN  Data Description:  RI : refractive index  Na: Sodium (unit measurement: weight percent in corresponding oxide, as are attributes 4-10)  Mg: Magnesium  AI: Aluminum  Si: Silicon  K:Potassium  Ca: Calcium  Ba: Barium  Fe: Iron
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Read the CSV file into a DataFrame
data = pd.read_csv('glass.csv')

# Step 2: Split the data into features and target
X = data.drop('Type', axis=1)  # Features
y = data['Type']  # Target variable

# Step 3: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Create a KNN classifier
k = 3  # Number of neighbors
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Step 6: Predict the target values for the test data
y_pred = knn.predict(X_test)

# Step 7: Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Exploratory Data Analysis

# Univariate Analysis - Histograms
plt.figure(figsize=(12, 8))
for i, col in enumerate(X.columns):
    plt.subplot(3, 3, i + 1)
    sns.histplot(data[col], kde=True)
    plt.title(f'Histogram of {col}')
plt.tight_layout()
plt.show()

# Bivariate Analysis - Pairplot
plt.figure(figsize=(12, 8))
sns.pairplot(data, hue='Type', diag_kind='kde')
plt.show()

# Multivariate Analysis - Correlation Heatmap
plt.figure(figsize=(10, 8))
corr = data.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()
