# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Convert to DataFrame for better visualization
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = y
print("Dataset Overview:")
print(df.head())
print("\nDataset Shape:", df.shape)
print("\nSpecies: 0=Setosa, 1=Versicolor, 2=Virginica")

# ========================================
# 1. LINEAR REGRESSION
# ========================================
# Predict petal length from petal width

print("\n" + "="*50)
print("LINEAR REGRESSION")
print("="*50)

# Prepare data for linear regression
X_lin = df[['petal width (cm)']].values  # Feature
y_lin = df['petal length (cm)'].values   # Target

# Split the data
X_train_lin, X_test_lin, y_train_lin, y_test_lin = train_test_split(
    X_lin, y_lin, test_size=0.2, random_state=42
)

# Create and train the model
lin_model = LinearRegression()
lin_model.fit(X_train_lin, y_train_lin)

# Make predictions
y_pred_lin = lin_model.predict(X_test_lin)

# Evaluate the model
mse = mean_squared_error(y_test_lin, y_pred_lin)
r2 = r2_score(y_test_lin, y_pred_lin)

print(f"\nModel Coefficient (Slope): {lin_model.coef_[0]:.4f}")
print(f"Model Intercept: {lin_model.intercept_:.4f}")
print(f"\nMean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# Visualize the results
plt.figure(figsize=(10, 5))
plt.scatter(X_test_lin, y_test_lin, color='blue', label='Actual', alpha=0.6)
plt.scatter(X_test_lin, y_pred_lin, color='red', label='Predicted', alpha=0.6)
plt.plot(X_test_lin, y_pred_lin, color='green', linewidth=2, label='Regression Line')
plt.xlabel('Petal Width (cm)')
plt.ylabel('Petal Length (cm)')
plt.title('Linear Regression: Petal Length vs Petal Width')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ========================================
# 2. LOGISTIC REGRESSION
# ========================================
# Classify Iris species

print("\n" + "="*50)
print("LOGISTIC REGRESSION")
print("="*50)

# Prepare data for logistic regression
X_log = df.drop('species', axis=1).values  # All features
y_log = df['species'].values               # Target (species)

# Split the data
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(
    X_log, y_log, test_size=0.2, random_state=42
)

# Create and train the model
log_model = LogisticRegression(max_iter=200)
log_model.fit(X_train_log, y_train_log)

# Make predictions
y_pred_log = log_model.predict(X_test_log)

# Evaluate the model
accuracy = accuracy_score(y_test_log, y_pred_log)

print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test_log, y_pred_log, 
                          target_names=iris.target_names))

# Confusion Matrix
cm = confusion_matrix(y_test_log, y_pred_log)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Logistic Regression')
plt.show()

# Show some predictions
print("\nSample Predictions:")
print("Actual vs Predicted:")
for i in range(min(10, len(y_test_log))):
    print(f"  {iris.target_names[y_test_log[i]]:12} -> {iris.target_names[y_pred_log[i]]:12}")




# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from mpl_toolkits.mplot3d import Axes3D

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print("Dataset Overview:")
print(df.head())

# ========================================
# MULTIPLE LINEAR REGRESSION
# Using TWO variables to predict ONE
# ========================================

print("\n" + "="*50)
print("MULTIPLE LINEAR REGRESSION")
print("Predicting Petal Length using Petal Width AND Sepal Length")
print("="*50)

# Prepare data - Using 2 features to predict 1 target
X_multi = df[['petal width (cm)', 'sepal length (cm)']].values  # Two features
y_multi = df['petal length (cm)'].values                         # Target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42
)

# Create and train the model
multi_model = LinearRegression()
multi_model.fit(X_train, y_train)

# Make predictions
y_pred = multi_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Coefficients:")
print(f"  Petal Width coefficient: {multi_model.coef_[0]:.4f}")
print(f"  Sepal Length coefficient: {multi_model.coef_[1]:.4f}")
print(f"  Intercept: {multi_model.intercept_:.4f}")
print(f"\nEquation: Petal Length = {multi_model.coef_[0]:.4f} × Petal Width + {multi_model.coef_[1]:.4f} × Sepal Length + {multi_model.intercept_:.4f}")
print(f"\nMean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# ========================================
# VISUALIZATION 1: 3D Scatter Plot
# ========================================
fig = plt.figure(figsize=(12, 5))

# 3D plot
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X_test[:, 0], X_test[:, 1], y_test, 
           color='blue', label='Actual', s=50, alpha=0.6)
ax1.scatter(X_test[:, 0], X_test[:, 1], y_pred, 
           color='red', label='Predicted', s=50, alpha=0.6)
ax1.set_xlabel('Petal Width (cm)')
ax1.set_ylabel('Sepal Length (cm)')
ax1.set_zlabel('Petal Length (cm)')
ax1.set_title('3D View: Multiple Linear Regression')
ax1.legend()

# ========================================
# VISUALIZATION 2: Actual vs Predicted
# ========================================
ax2 = fig.add_subplot(122)
ax2.scatter(y_test, y_pred, alpha=0.6, color='purple')
ax2.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
ax2.set_xlabel('Actual Petal Length (cm)')
ax2.set_ylabel('Predicted Petal Length (cm)')
ax2.set_title('Actual vs Predicted Values')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ========================================
# Show predictions comparison
# ========================================
print("\n" + "="*50)
print("Sample Predictions Comparison:")
print("="*50)
print(f"{'Petal Width':<12} {'Sepal Length':<14} {'Actual':<10} {'Predicted':<10} {'Error':<10}")
print("-" * 60)
for i in range(min(10, len(y_test))):
    error = abs(y_test[i] - y_pred[i])
    print(f"{X_test[i, 0]:<12.2f} {X_test[i, 1]:<14.2f} {y_test[i]:<10.2f} {y_pred[i]:<10.2f} {error:<10.4f}")

# ========================================
# BONUS: Using ALL features
# ========================================
print("\n" + "="*50)
print("BONUS: Using ALL 3 features to predict Petal Length")
print("="*50)

X_all = df[['sepal length (cm)', 'sepal width (cm)', 'petal width (cm)']].values
y_all = df['petal length (cm)'].values

X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42
)

model_all = LinearRegression()
model_all.fit(X_train_all, y_train_all)
y_pred_all = model_all.predict(X_test_all)

mse_all = mean_squared_error(y_test_all, y_pred_all)
r2_all = r2_score(y_test_all, y_pred_all)

print(f"\nUsing 3 features:")
print(f"  R² Score: {r2_all:.4f}")
print(f"  MSE: {mse_all:.4f}")
print(f"\nUsing 2 features:")
print(f"  R² Score: {r2:.4f}")
print(f"  MSE: {mse:.4f}")
print(f"\nImprovement: {((r2_all - r2) / r2 * 100):.2f}% better R² score with more features!")



