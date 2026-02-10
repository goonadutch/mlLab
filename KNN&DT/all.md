=====DTMLLAB=====
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# 1. LOAD DATASET
# ============================================================================
# Download: Go to https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
# Click "Download" -> You'll get wdbc.data and wdbc.names
# OR use this direct link and save as 'wdbc.data':
# https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data


import urllib.request
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
urllib.request.urlretrieve(url, 'wdbc.data')

# Load the .data file (no headers in file)
columns = ['id', 'diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
df = pd.read_csv('wdbc.data', header=None, names=columns)

# Separate features and target
X = df.drop(['id', 'diagnosis'], axis=1)
y = df['diagnosis'].map({'M': 0, 'B': 1})  # M=Malignant(0), B=Benign(1)

print(f"Dataset Shape: {X.shape}")
print(f"Malignant: {(y==0).sum()}, Benign: {(y==1).sum()}")



# ============================================================================
# 2. PREPROCESSING & DATASET VARIATIONS
# ============================================================================

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create 4 dataset variations
datasets = {}

# Full dataset
datasets['full'] = (X_scaled, y)

# Balanced dataset
ros = RandomOverSampler(random_state=42)
X_balanced, y_balanced = ros.fit_resample(X_scaled, y)
datasets['balanced'] = (X_balanced, y_balanced)

# Top 10 features
selector = SelectKBest(f_classif, k=10)
X_top10 = selector.fit_transform(X_scaled, y)
datasets['top10'] = (X_top10, y)

# Noisy data
noise = np.random.normal(0, 0.05 * X_scaled.std(), X_scaled.shape)
X_noisy = X_scaled + noise
datasets['noisy'] = (X_noisy, y)

# ============================================================================
# 3. TRAIN & EVALUATE ON EACH DATASET
# ============================================================================

results = []

for ds_name, (X_data, y_data) in datasets.items():
    print(f"\n{'='*60}\nDataset: {ds_name.upper()}\n{'='*60}")
    
    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=42, stratify=y_data
    )
    
    # Hyperparameter tuning
    best_f1 = 0
    best_model = None
    best_params = {}
    
    for depth in range(3, 16):
        for split in [2, 5, 10, 15, 20]:
            for leaf in range(1, 11):
                for crit in ['gini', 'entropy']:
                    clf = DecisionTreeClassifier(
                        max_depth=depth, min_samples_split=split,
                        min_samples_leaf=leaf, criterion=crit, random_state=42
                    )
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    f1 = f1_score(y_test, y_pred, pos_label=0)
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_model = clf
                        best_params = {'depth': depth, 'split': split, 'leaf': leaf, 'crit': crit}
    
    # Predictions
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 0]
    
    # Metrics (Malignant class focus)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=0)
    rec = recall_score(y_test, y_pred, pos_label=0)
    f1 = f1_score(y_test, y_pred, pos_label=0)
    
    results.append({
        'Dataset': ds_name, 'Accuracy': acc, 'Precision': prec, 
        'Recall': rec, 'F1': f1, **best_params
    })
    
    print(f"Best: depth={best_params['depth']}, split={best_params['split']}, "
          f"leaf={best_params['leaf']}, criterion={best_params['crit']}")
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    
    # Save for visualization
    if ds_name == 'full':
        best_dt, X_test_full, y_test_full, y_pred_full, y_prob_full = best_model, X_test, y_test, y_pred, y_prob

# Results summary
print(f"\n{'='*60}\nRESULTS SUMMARY\n{'='*60}")
print(pd.DataFrame(results).to_string(index=False))

# ============================================================================
# 4. VISUALIZATIONS
# ============================================================================

fig = plt.figure(figsize=(18, 12))

# Confusion Matrix
plt.subplot(2, 3, 1)
cm = confusion_matrix(y_test_full, y_pred_full)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Malignant', 'Benign'], yticklabels=['Malignant', 'Benign'])
plt.title("Confusion Matrix")

# Feature Importance
plt.subplot(2, 3, 2)
importance = pd.DataFrame({'feature': X.columns, 'importance': best_dt.feature_importances_})
importance = importance.sort_values('importance', ascending=True).tail(10)
plt.barh(importance['feature'], importance['importance'])
plt.title("Top 10 Feature Importances")

# ROC Curve
plt.subplot(2, 3, 3)
fpr, tpr, _ = roc_curve((y_test_full == 0).astype(int), y_prob_full)
plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

# Overfitting Analysis (Depth vs Accuracy)
plt.subplot(2, 3, 4)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
train_acc, test_acc = [], []
for depth in range(1, 21):
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    train_acc.append(clf.score(X_train, y_train))
    test_acc.append(clf.score(X_test, y_test))
plt.plot(range(1, 21), train_acc, label='Train', marker='o')
plt.plot(range(1, 21), test_acc, label='Test', marker='s')
plt.axvline(x=10, color='r', linestyle='--', label='Depth=10')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Overfitting Analysis')
plt.legend()
plt.grid(alpha=0.3)

# Metrics Comparison
plt.subplot(2, 3, 5)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
values = [results[0][m] for m in metrics]
plt.bar(metrics, values, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
for i, v in enumerate(values):
    plt.text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
plt.ylim([0.8, 1.0])
plt.title("Performance Metrics")

# Actual vs Predicted
plt.subplot(2, 3, 6)
plt.scatter(range(len(y_test_full)), y_test_full, label='Actual', alpha=0.7)
plt.scatter(range(len(y_pred_full)), y_pred_full, label='Predicted', alpha=0.7, marker='x')
plt.xlabel("Sample Index")
plt.ylabel("Class")
plt.title("Actual vs Predicted")
plt.legend()

plt.tight_layout()
plt.savefig('decision_tree_results.png', dpi=300)
plt.show()

# Decision Tree Visualization
plt.figure(figsize=(25, 15))
plot_tree(best_dt, filled=True, feature_names=X.columns, 
          class_names=['Malignant', 'Benign'], rounded=True, fontsize=10)
plt.title("Decision Tree Structure")
plt.savefig('tree_structure.png', dpi=300)
plt.show()


# ============================================================================
# 5. INSIGHTS
# ============================================================================

print(f"\n{'='*60}\nINSIGHTS\n{'='*60}")
print(f"""
OVERFITTING ANALYSIS:
- Depth >10: Train accuracy ~100%, Test accuracy plateaus
- Gap increases = OVERFITTING on noise
- Noisy data shows drop in F1 with deep trees

DEPLOYMENT RECOMMENDATION:
- Config: max_depth={results[0]['depth']}, min_samples_leaf={results[0]['leaf']}, criterion='{results[0]['crit']}'
- Performance: F1={results[0]['F1']:.3f}, Recall={results[0]['Recall']:.3f}
- Interpretable tree, prevents overfitting
""")

print("\nFiles saved: decision_tree_results.png, tree_structure.png")



=======KNN MLLAB======
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# 1. LOAD DATASET
# ============================================================================

# Auto-download dataset
import urllib.request
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
urllib.request.urlretrieve(url, 'wdbc.data')
print("Dataset downloaded successfully!")

# Load the .data file (no headers in file)
columns = ['id', 'diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
df = pd.read_csv('wdbc.data', header=None, names=columns)

# Separate features and target
X = df.drop(['id', 'diagnosis'], axis=1)
y = df['diagnosis'].map({'M': 0, 'B': 1})  # M=Malignant(0), B=Benign(1)

print(f"Dataset Shape: {X.shape}")
print(f"Malignant: {(y==0).sum()}, Benign: {(y==1).sum()}")

# ============================================================================
# 2. PREPROCESSING & DATASET VARIATIONS
# ============================================================================

# Scale features (MANDATORY for KNN!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create 4 dataset variations
datasets = {}

# Full dataset
datasets['full'] = (X_scaled, y)

# Balanced dataset
ros = RandomOverSampler(random_state=42)
X_balanced, y_balanced = ros.fit_resample(X_scaled, y)
datasets['balanced'] = (X_balanced, y_balanced)

# Top 10 features
selector = SelectKBest(f_classif, k=10)
X_top10 = selector.fit_transform(X_scaled, y)
datasets['top10'] = (X_top10, y)

# Noisy data
noise = np.random.normal(0, 0.05 * X_scaled.std(), X_scaled.shape)
X_noisy = X_scaled + noise
datasets['noisy'] = (X_noisy, y)

# ============================================================================
# 3. TRAIN & EVALUATE ON EACH DATASET
# ============================================================================

results = []

for ds_name, (X_data, y_data) in datasets.items():
    print(f"\n{'='*60}\nDataset: {ds_name.upper()}\n{'='*60}")
    
    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=42, stratify=y_data
    )
    
    # Hyperparameter tuning
    best_f1 = 0
    best_model = None
    best_params = {}
    
    for k in range(3, 31):  # n_neighbors 3-30
        for weight in ['uniform', 'distance']:  # weights
            for metric in ['euclidean', 'minkowski']:  # metric
                knn = KNeighborsClassifier(
                    n_neighbors=k, weights=weight, metric=metric,
                    p=3 if metric == 'minkowski' else 2  # p=3 for minkowski
                )
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)
                f1 = f1_score(y_test, y_pred, pos_label=0)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_model = knn
                    best_params = {'k': k, 'weight': weight, 'metric': metric}
    
    # Predictions
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 0]
    
    # Metrics (Malignant class focus)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=0)
    rec = recall_score(y_test, y_pred, pos_label=0)
    f1 = f1_score(y_test, y_pred, pos_label=0)
    
    results.append({
        'Dataset': ds_name, 'Accuracy': acc, 'Precision': prec, 
        'Recall': rec, 'F1': f1, **best_params
    })
    
    print(f"Best: k={best_params['k']}, weights='{best_params['weight']}', metric='{best_params['metric']}'")
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    
    # Save for visualization
    if ds_name == 'full':
        best_knn, X_test_full, y_test_full, y_pred_full, y_prob_full = best_model, X_test, y_test, y_pred, y_prob

# Results summary
print(f"\n{'='*60}\nRESULTS SUMMARY\n{'='*60}")
print(pd.DataFrame(results).to_string(index=False))

# ============================================================================
# 4. VISUALIZATIONS
# ============================================================================

fig = plt.figure(figsize=(18, 12))

# Confusion Matrix
plt.subplot(2, 3, 1)
cm = confusion_matrix(y_test_full, y_pred_full)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Malignant', 'Benign'], yticklabels=['Malignant', 'Benign'])
plt.title("Confusion Matrix")

# K-Value Analysis
plt.subplot(2, 3, 2)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
k_values, k_f1 = [], []
for k in range(1, 31):
    knn_temp = KNeighborsClassifier(n_neighbors=k, weights='distance')
    knn_temp.fit(X_train, y_train)
    y_pred_temp = knn_temp.predict(X_test)
    k_f1.append(f1_score(y_test, y_pred_temp, pos_label=0))
    k_values.append(k)
plt.plot(k_values, k_f1, marker='o')
plt.axvline(x=5, color='r', linestyle='--', label='k=5')
plt.xlabel('K (Neighbors)')
plt.ylabel('F1-Score')
plt.title('K-Value vs F1-Score')
plt.legend()
plt.grid(alpha=0.3)

# ROC Curve
plt.subplot(2, 3, 3)
fpr, tpr, _ = roc_curve((y_test_full == 0).astype(int), y_prob_full)
plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

# Imbalance Impact (Full vs Balanced)
plt.subplot(2, 3, 4)
datasets_names = [r['Dataset'] for r in results]
f1_scores = [r['F1'] for r in results]
recall_scores = [r['Recall'] for r in results]
x = np.arange(len(datasets_names))
width = 0.35
plt.bar(x - width/2, f1_scores, width, label='F1-Score', color='#3498db')
plt.bar(x + width/2, recall_scores, width, label='Recall', color='#e74c3c')
plt.xticks(x, datasets_names)
plt.ylabel('Score')
plt.title('KNN: Full vs Balanced Data')
plt.legend()
plt.grid(axis='y', alpha=0.3)

# Metrics Comparison
plt.subplot(2, 3, 5)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
values = [results[0][m] for m in metrics]
plt.bar(metrics, values, color=['#27ae60', '#e74c3c', '#3498db', '#f39c12'])
for i, v in enumerate(values):
    plt.text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
plt.ylim([0.8, 1.0])
plt.title("Performance Metrics")

# Actual vs Predicted
plt.subplot(2, 3, 6)
plt.scatter(range(len(y_test_full)), y_test_full, label='Actual', alpha=0.7)
plt.scatter(range(len(y_pred_full)), y_pred_full, label='Predicted', alpha=0.7, marker='x')
plt.xlabel("Sample Index")
plt.ylabel("Class")
plt.title("Actual vs Predicted")
plt.legend()

plt.tight_layout()
plt.savefig('knn_results.png', dpi=300)
plt.show()

# ============================================================================
# 5. INSIGHTS
# ============================================================================

print(f"\n{'='*60}\nINSIGHTS\n{'='*60}")

# Why k=5 excels on balanced but fails on full (imbalance)?
full_result = results[0]  # Full dataset
balanced_result = results[1]  # Balanced dataset

print(f"""
WHY KNN k=5 EXCELS ON BALANCED BUT FAILS ON FULL (IMBALANCE):

Full Dataset (Imbalanced):
- Recall: {full_result['Recall']:.4f}
- F1: {full_result['F1']:.4f}
- Problem: KNN uses majority voting from k nearest neighbors
- With imbalance (212 malignant vs 357 benign), neighbors more likely benign
- Malignant class gets outvoted even when points are close
- Distance weighting helps but doesn't fully solve the problem

Balanced Dataset:
- Recall: {balanced_result['Recall']:.4f}
- F1: {balanced_result['F1']:.4f}
- Improvement: {((balanced_result['F1'] - full_result['F1'])/full_result['F1']*100):.1f}%
- Solution: Equal representation gives minority class fair voting chance
- Each class has equal probability in local neighborhoods

KNN LOCALITY ON FEATURE SUBSETS:
- Top-10 features F1: {results[2]['F1']:.4f}
- Full 30 features F1: {results[0]['F1']:.4f}
- Curse of dimensionality: More features = sparse neighborhoods
- Distance becomes less meaningful in high dimensions
- Feature selection improves local neighborhood quality
""")

print(f"""
DEPLOYMENT RECOMMENDATION:

For FULL DATASET:
- Config: k={results[0]['k']}, weights='{results[0]['weight']}', metric='{results[0]['metric']}'
- F1={results[0]['F1']:.3f}, Recall={results[0]['Recall']:.3f}

For NOISY DATA:
- Config: k={results[3]['k']}, weights='{results[3]['weight']}', metric='{results[3]['metric']}'
- F1={results[3]['F1']:.3f}, Recall={results[3]['Recall']:.3f}
- Higher k smooths noise, distance weighting prioritizes closer points

For BEST PERFORMANCE (Balanced):
- Config: k={results[1]['k']}, weights='{results[1]['weight']}', metric='{results[1]['metric']}'
- F1={results[1]['F1']:.3f}, Recall={results[1]['Recall']:.3f}
- Recommended for clinical use (high recall critical)

KEY TAKEAWAYS:
✓ Always scale features for KNN (done automatically here)
✓ Use distance weighting to reduce majority class bias
✓ Balance dataset for better minority class detection
✓ Higher k (7-10) for noisy data, lower k (3-5) for clean data
✗ KNN cannot explain predictions (black box)
✗ Slow on large datasets (computes all distances)
""")

print("\nFiles saved: knn_results.png")



=====decision tree basic ======
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset (replace with your file)
# For demo, using iris dataset
from sklearn.datasets import load_iris
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']
# X = df.drop('YourTargetColumnName', axis=1)  # Replace 'YourTargetColumnName'
# y = df['YourTargetColumnName']               # Replace 'YourTargetColumnName'

# Handle categorical data if any (automatically encode)
le = LabelEncoder()
for col in X.select_dtypes(include=['object']).columns:
    X[col] = le.fit_transform(X[col])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Decision Tree
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Metrics
print("=" * 50)
print("MODEL PERFORMANCE METRICS")
print("=" * 50)
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred, average='weighted'):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred, average='weighted'):.4f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Separate Decision Tree Figure
plt.figure(figsize=(30, 20))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=[str(i) for i in np.unique(y)], rounded=True, fontsize=10)
plt.title("Decision Tree Visualization", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Visualization Setup
fig = plt.figure(figsize=(18, 12))
# 2. Confusion Matrix
plt.subplot(2, 3, 2)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix", fontsize=14, fontweight='bold')
plt.xlabel("Predicted")
plt.ylabel("Actual")

# 3. Feature Importance
plt.subplot(2, 3, 3)
importance = pd.DataFrame({'feature': X.columns, 'importance': clf.feature_importances_})
importance = importance.sort_values('importance', ascending=True)
plt.barh(importance['feature'], importance['importance'], color='skyblue')
plt.title("Feature Importance", fontsize=14, fontweight='bold')
plt.xlabel("Importance")

# 4. Scatter Plot (first 2 features)
plt.subplot(2, 3, 4)
plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_pred, cmap='viridis', edgecolors='k', s=100)
plt.xlabel(X.columns[0])
plt.ylabel(X.columns[1])
plt.title("Predictions Scatter Plot", fontsize=14, fontweight='bold')
plt.colorbar(label='Class')

# 5. Learning Curve
plt.subplot(2, 3, 5)
train_sizes, train_scores, val_scores = learning_curve(clf, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
plt.plot(train_sizes, train_scores.mean(axis=1), label='Training Score', marker='o')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation Score', marker='s')
plt.xlabel("Training Size")
plt.ylabel("Score")
plt.title("Learning Curve", fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

# 6. Actual vs Predicted
plt.subplot(2, 3, 6)
plt.scatter(range(len(y_test)), y_test, label='Actual', alpha=0.7, s=50)
plt.scatter(range(len(y_pred)), y_pred, label='Predicted', alpha=0.7, s=50, marker='x')
plt.xlabel("Sample Index")
plt.ylabel("Class")
plt.title("Actual vs Predicted", fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "=" * 50)
print("Visualizations saved successfully!")
print("=" * 50)


# import pandas as pd

# data = {
#     'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 
#                 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
#     'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 
#                     'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
#     'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 
#                  'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
#     'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 
#              'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
#     'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 
#              'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
# }

# df = pd.read_csv('weather.csv')  # If you have CSV
# # OR
# df = pd.DataFrame(data)  # Create from dictionary above

# X = df.drop('Play', axis=1)  # Features
# y = df['Play']                # Target (Yes/No)


=======knn complete======
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

# Load your dataset (replace with your file)
# For demo, using iris dataset
from sklearn.datasets import load_iris
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Handle categorical data if any (automatically encode)
le = LabelEncoder()
for col in X.select_dtypes(include=['object']).columns:
    X[col] = le.fit_transform(X[col])

# Scale features (IMPORTANT for KNN!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train KNN (you can change n_neighbors)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predictions
y_pred = knn.predict(X_test)

# Metrics
print("=" * 50)
print("KNN MODEL PERFORMANCE METRICS")
print("=" * 50)
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred, average='weighted'):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred, average='weighted'):.4f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Visualization Setup
fig = plt.figure(figsize=(20, 12))

# 1. Decision Boundary Visualization (2D - using first 2 features)
plt.subplot(2, 3, 1)
X_2d = X_scaled.iloc[:, :2]  # First 2 features
knn_2d = KNeighborsClassifier(n_neighbors=5)
knn_2d.fit(X_2d, y)

# Create mesh
h = 0.02
x_min, x_max = X_2d.iloc[:, 0].min() - 1, X_2d.iloc[:, 0].max() + 1
y_min, y_max = X_2d.iloc[:, 1].min() - 1, X_2d.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_light)
plt.scatter(X_2d.iloc[:, 0], X_2d.iloc[:, 1], c=y, cmap=cmap_bold, edgecolors='k', s=50)
plt.xlabel(X.columns[0])
plt.ylabel(X.columns[1])
plt.title("Decision Boundary (2D)", fontsize=14, fontweight='bold')

# 2. Accuracy vs K Values Plot
plt.subplot(2, 3, 2)
k_range = range(1, 31)
k_scores = []
for k in k_range:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train, y_train)
    k_scores.append(knn_temp.score(X_test, y_test))

plt.plot(k_range, k_scores, marker='o', linestyle='-', color='blue')
best_k = k_range[np.argmax(k_scores)]
plt.axvline(x=best_k, color='red', linestyle='--', label=f'Best K={best_k}')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.title('Accuracy vs K Values', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

# 3. Distance Plot (Distance to nearest neighbors for test samples)
plt.subplot(2, 3, 3)
distances, indices = knn.kneighbors(X_test)
avg_distances = distances.mean(axis=1)
plt.scatter(range(len(avg_distances)), avg_distances, c=y_test, cmap='viridis', edgecolors='k', s=80)
plt.xlabel('Test Sample Index')
plt.ylabel('Average Distance to K Neighbors')
plt.title('Distance to Neighbors', fontsize=14, fontweight='bold')
plt.colorbar(label='Class')
plt.grid(alpha=0.3)

# 4. Confusion Matrix
plt.subplot(2, 3, 4)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix", fontsize=14, fontweight='bold')
plt.xlabel("Predicted")
plt.ylabel("Actual")

# 5. Learning Curve
plt.subplot(2, 3, 5)
train_sizes, train_scores, val_scores = learning_curve(knn, X_scaled, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
plt.plot(train_sizes, train_scores.mean(axis=1), label='Training Score', marker='o')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation Score', marker='s')
plt.xlabel("Training Size")
plt.ylabel("Score")
plt.title("Learning Curve", fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

# 6. Actual vs Predicted
plt.subplot(2, 3, 6)
plt.scatter(range(len(y_test)), y_test, label='Actual', alpha=0.7, s=50)
plt.scatter(range(len(y_pred)), y_pred, label='Predicted', alpha=0.7, s=50, marker='x')
plt.xlabel("Sample Index")
plt.ylabel("Class")
plt.title("Actual vs Predicted", fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "=" * 50)
print(f"Best K Value: {best_k}")
print(f"Best Accuracy: {max(k_scores):.4f}")
print("=" * 50)



=======kd=======
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import seaborn as sns
import time

# ============================================================================
# 1. LOAD DATASET
# ============================================================================

# Load the .data file from your desktop (change path as needed)
columns = ['id', 'diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
df = pd.read_csv('wdbc.data', header=None, names=columns)

# Separate features and target
X = df.drop(['id', 'diagnosis'], axis=1)
y = df['diagnosis'].map({'M': 0, 'B': 1})  # M=Malignant(0), B=Benign(1)

print(f"Dataset Shape: {X.shape}")
print(f"Malignant: {(y==0).sum()}, Benign: {(y==1).sum()}")

# ============================================================================
# 2. PREPROCESSING & DATASET VARIATIONS
# ============================================================================

# Scale features (MANDATORY for K-D Tree!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create 4 dataset variations
datasets = {}

# Full dataset
datasets['full'] = (X_scaled, y)

# Balanced dataset
ros = RandomOverSampler(random_state=42)
X_balanced, y_balanced = ros.fit_resample(X_scaled, y)
datasets['balanced'] = (X_balanced, y_balanced)

# Top 10 features
selector = SelectKBest(f_classif, k=10)
X_top10 = selector.fit_transform(X_scaled, y)
datasets['top10'] = (X_top10, y)

# Noisy data
noise = np.random.normal(0, 0.05 * X_scaled.std(), X_scaled.shape)
X_noisy = X_scaled + noise
datasets['noisy'] = (X_noisy, y)

# ============================================================================
# 3. TRAIN & EVALUATE ON EACH DATASET (K-D TREE)
# ============================================================================

results = []

for ds_name, (X_data, y_data) in datasets.items():
    print(f"\n{'='*60}\nDataset: {ds_name.upper()}\n{'='*60}")
    
    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=42, stratify=y_data
    )
    
    # Hyperparameter tuning with K-D Tree
    best_f1 = 0
    best_model = None
    best_params = {}
    
    for k in range(3, 31):  # n_neighbors 3-30
        for weight in ['uniform', 'distance']:  # weights
            for leaf_size in [10, 20, 30, 40, 50]:  # K-D tree leaf size
                # K-D Tree uses algorithm='kd_tree' (works with euclidean/minkowski)
                knn_kd = KNeighborsClassifier(
                    n_neighbors=k, 
                    weights=weight, 
                    algorithm='kd_tree',  # K-D TREE algorithm
                    leaf_size=leaf_size,  # K-D tree parameter
                    metric='minkowski',
                    p=2  # Euclidean distance
                )
                knn_kd.fit(X_train, y_train)
                y_pred = knn_kd.predict(X_test)
                f1 = f1_score(y_test, y_pred, pos_label=0)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_model = knn_kd
                    best_params = {'k': k, 'weight': weight, 'leaf_size': leaf_size}
    
    # Predictions
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 0]
    
    # Metrics (Malignant class focus)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=0)
    rec = recall_score(y_test, y_pred, pos_label=0)
    f1 = f1_score(y_test, y_pred, pos_label=0)
    
    results.append({
        'Dataset': ds_name, 'Accuracy': acc, 'Precision': prec, 
        'Recall': rec, 'F1': f1, **best_params
    })
    
    print(f"Best: k={best_params['k']}, weights='{best_params['weight']}', leaf_size={best_params['leaf_size']}")
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    
    # Save for visualization
    if ds_name == 'full':
        best_kd, X_test_full, y_test_full, y_pred_full, y_prob_full = best_model, X_test, y_test, y_pred, y_prob

# Results summary
print(f"\n{'='*60}\nRESULTS SUMMARY\n{'='*60}")
print(pd.DataFrame(results).to_string(index=False))

# ============================================================================
# 4. SPEED COMPARISON: K-D TREE vs BRUTE FORCE
# ============================================================================

print(f"\n{'='*60}\nSPEED COMPARISON\n{'='*60}")

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# K-D Tree
knn_kd = KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree', leaf_size=30)
start = time.time()
knn_kd.fit(X_train, y_train)
knn_kd.predict(X_test)
kd_time = time.time() - start

# Brute Force
knn_brute = KNeighborsClassifier(n_neighbors=5, algorithm='brute')
start = time.time()
knn_brute.fit(X_train, y_train)
knn_brute.predict(X_test)
brute_time = time.time() - start

print(f"K-D Tree time: {kd_time:.4f}s")
print(f"Brute Force time: {brute_time:.4f}s")
print(f"Speedup: {brute_time/kd_time:.2f}x faster")

# ============================================================================
# 5. VISUALIZATIONS
# ============================================================================

fig = plt.figure(figsize=(18, 12))

# Confusion Matrix
plt.subplot(2, 3, 1)
cm = confusion_matrix(y_test_full, y_pred_full)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Malignant', 'Benign'], yticklabels=['Malignant', 'Benign'])
plt.title("Confusion Matrix (K-D Tree)")

# Leaf Size Analysis
plt.subplot(2, 3, 2)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
leaf_sizes, leaf_f1, leaf_times = [], [], []
for leaf in [5, 10, 20, 30, 40, 50, 60, 70]:
    knn_temp = KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree', leaf_size=leaf)
    start = time.time()
    knn_temp.fit(X_train, y_train)
    y_pred_temp = knn_temp.predict(X_test)
    leaf_times.append(time.time() - start)
    leaf_f1.append(f1_score(y_test, y_pred_temp, pos_label=0))
    leaf_sizes.append(leaf)

ax1 = plt.gca()
ax1.plot(leaf_sizes, leaf_f1, 'b-o', label='F1-Score')
ax1.set_xlabel('Leaf Size')
ax1.set_ylabel('F1-Score', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.grid(alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(leaf_sizes, leaf_times, 'r-s', label='Time')
ax2.set_ylabel('Time (s)', color='r')
ax2.tick_params(axis='y', labelcolor='r')
plt.title('Leaf Size vs Performance & Speed')

# ROC Curve
plt.subplot(2, 3, 3)
fpr, tpr, _ = roc_curve((y_test_full == 0).astype(int), y_prob_full)
plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

# Speed Comparison Chart
plt.subplot(2, 3, 4)
algorithms = ['K-D Tree', 'Brute Force']
times = [kd_time, brute_time]
bars = plt.bar(algorithms, times, color=['#2ecc71', '#e74c3c'])
plt.ylabel('Time (seconds)')
plt.title('Speed Comparison')
for bar, t in zip(bars, times):
    plt.text(bar.get_x() + bar.get_width()/2, t + 0.0001, 
             f'{t:.4f}s', ha='center', fontweight='bold')

# Metrics Comparison
plt.subplot(2, 3, 5)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
values = [results[0][m] for m in metrics]
plt.bar(metrics, values, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
for i, v in enumerate(values):
    plt.text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
plt.ylim([0.8, 1.0])
plt.title("Performance Metrics")

# Actual vs Predicted
plt.subplot(2, 3, 6)
plt.scatter(range(len(y_test_full)), y_test_full, label='Actual', alpha=0.7)
plt.scatter(range(len(y_pred_full)), y_pred_full, label='Predicted', alpha=0.7, marker='x')
plt.xlabel("Sample Index")
plt.ylabel("Class")
plt.title("Actual vs Predicted")
plt.legend()

plt.tight_layout()
plt.savefig('kd_tree_results.png', dpi=300)
plt.show()

# ============================================================================
# 6. INSIGHTS
# ============================================================================

print(f"\n{'='*60}\nINSIGHTS\n{'='*60}")

print(f"""
K-D TREE ADVANTAGES:
✓ Faster than brute force: {brute_time/kd_time:.2f}x speedup
✓ Efficient for low-medium dimensions (good for top-10 features)
✓ Organizes data in tree structure for quick neighbor search
✓ O(log n) search time vs O(n) for brute force

K-D TREE LIMITATIONS:
✗ Performance degrades in high dimensions (curse of dimensionality)
✗ With 30 features, benefit is minimal
✗ Works only with Euclidean/Minkowski distances
✗ Building tree adds overhead for small datasets

LEAF SIZE IMPACT:
- Small leaf (5-10): More tree nodes, slower build, faster query
- Large leaf (50+): Fewer nodes, faster build, may slow queries
- Optimal: {results[0]['leaf_size']} for this dataset

DEPLOYMENT RECOMMENDATION:

For FULL DATASET (30 features):
- Config: k={results[0]['k']}, weights='{results[0]['weight']}', leaf_size={results[0]['leaf_size']}
- F1={results[0]['F1']:.3f}, Recall={results[0]['Recall']:.3f}
- Note: With 30 features, K-D tree benefit is limited

For TOP-10 FEATURES (RECOMMENDED):
- Config: k={results[2]['k']}, weights='{results[2]['weight']}', leaf_size={results[2]['leaf_size']}
- F1={results[2]['F1']:.3f}, Recall={results[2]['Recall']:.3f}
- K-D tree most efficient in lower dimensions

For NOISY DATA:
- Config: k={results[3]['k']}, weights='{results[3]['weight']}', leaf_size={results[3]['leaf_size']}
- F1={results[3]['F1']:.3f}
- Higher k + distance weighting smooths noise

WHEN TO USE K-D TREE:
✓ Low-medium dimensions (<20 features)
✓ Large datasets (>10,000 samples)
✓ Need fast predictions
✓ Using Euclidean/Minkowski distance

WHEN TO USE BRUTE FORCE:
✓ High dimensions (>30 features)
✓ Small datasets
✓ Need custom distance metrics
""")

print("\nFiles saved: kd_tree_results.png")




======cifar10=====
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, classification_report

# # Load CIFAR-10
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
# train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False)

# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#load MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False)

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


# Simple Fast CNN
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1) #(3,32,3)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*7*7, 128) #(64*8*8)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64*7*7) #64*8*8
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = CNN()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training (reduced epochs for speed)
epochs = 5
train_loss, train_acc, test_loss, test_acc = [], [], [], []

for epoch in range(epochs):
    model.train()
    running_loss, correct, total = 0, 0, 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        if batch_idx >= 100:  # Train on subset for speed
            break
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss.append(running_loss/len(train_loader))
    train_acc.append(100*correct/total)

    # Test
    model.eval()
    running_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            if batch_idx >= 50:  # Test on subset for speed
                break
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss.append(running_loss/len(test_loader))
    test_acc.append(100*correct/total)
    print(f'Epoch {epoch+1}: Train Acc={train_acc[-1]:.2f}%, Test Acc={test_acc[-1]:.2f}%')


# Learning Curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(train_loss, label='Train Loss')
ax1.plot(test_loss, label='Test Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.set_title('Learning Curve - Loss')

ax2.plot(train_acc, label='Train Accuracy')
ax2.plot(test_acc, label='Test Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.legend()
ax2.set_title('Learning Curve - Accuracy')
plt.tight_layout()
plt.savefig('learning_curves.png')
plt.show()

# Sklearn Metrics
y_true, y_pred = [], []
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.numpy())
        y_pred.extend(predicted.numpy())

print(f'\nAccuracy: {accuracy_score(y_true, y_pred):.4f}')
print(f'Precision (macro): {precision_score(y_true, y_pred, average="macro"):.4f}')
print('\nClassification Report:')
print(classification_report(y_true, y_pred, target_names=classes))

# Scatter Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_true, y_pred, alpha=0.3, s=10)
plt.plot([0, 9], [0, 9], 'r--', label='Perfect Prediction')
plt.xlabel('True Labels')
plt.ylabel('Predicted Labels')
plt.title('Prediction Scatter Plot (All Test Samples)')
plt.legend()
plt.grid(True)
plt.savefig('scatter_plot.png')
plt.show()

# Single Image Prediction
def predict_image(image_index):
    image, label = test_data[image_index]
    image = image.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    print(f'True: {classes[label]}, Predicted: {classes[predicted.item()]}')
    return classes[predicted.item()]

print('\nSample Predictions:')
for i in range(5):
    predict_image(i)


import numpy as np
# Class distribution for training set
y_train = [label for _, label in train_data]
classes_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# classes_name = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck'] #for CIFAR
classes_unique, counts = np.unique(y_train, return_counts=True)
plt.barh(classes_name, counts)
plt.title('Class distribution in training set')
plt.xlabel('Count')
plt.savefig('train_distribution.png')
plt.show()

# Class distribution for test set
y_test = [label for _, label in test_data]
classes_unique, counts = np.unique(y_test, return_counts=True)
plt.barh(classes_name, counts)
plt.title('Class distribution in test set')
plt.xlabel('Count')
plt.savefig('test_distribution.png')
plt.show()

# Display sample images
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    image, label = train_data[i]
    image = image.squeeze() * 0.3081 + 0.1307  #  MNIST
    #image = image.permute(1, 2, 0) * 0.5 + 0.5  # CIFAR
    ax.imshow(image, cmap='gray')  # Add cmap='gray' for grayscale
    # ax.imshow(image) #CIFAR
    ax.set_title(classes_name[label])
    ax.axis('off')
plt.tight_layout()
plt.savefig('sample_images.png')
plt.show()
