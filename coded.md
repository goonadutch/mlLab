---
jupyter:
  colab:
  kernelspec:
    display_name: Python 3
    name: python3
  language_info:
    name: python
  nbformat: 4
  nbformat_minor: 0
---

::: {.cell .markdown id="r9SaF9E_IOYV"}
# **Perceptron**
:::

::: {.cell .code collapsed="true" id="Nanh2x1qILRL"}
``` python
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

class Perceptron:
    def __init__(self, lr=0.1, epochs=10):
        self.lr, self.epochs = lr, epochs

    def fit(self, X, y):
        self.w, self.b = np.zeros(X.shape[1]), 0
        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                pred = 1 if np.dot(xi, self.w) + self.b >= 0 else 0
                self.w += self.lr * (yi - pred) * xi
                self.b += self.lr * (yi - pred)

    def predict(self, X):
        return [1 if np.dot(x, self.w) + self.b >= 0 else 0 for x in X]

X = np.array([[0,0],[0,1],[1,0],[1,1]])
datasets = {
    "AND": [0,0,0,1],
    "OR":  [0,1,1,1],
    "XOR": [0,1,1,0]
}

a, b = 1, 1   # change inputs here

for name, y in datasets.items():
    y = np.array(y)
    model = Perceptron()
    model.fit(X, y)
    preds = model.predict(X)

    print(f"\n{name} Gate")
    print("Accuracy:", accuracy_score(y, preds))
    print(classification_report(y, preds, zero_division=0))
    print(f"Test ({a}, {b}) = {model.predict(np.array([[a, b]]))[0]}")
```
:::

::: {.cell .markdown id="6VSUklXTIwTk"}
# **MLP - INBUILT**
:::

::: {.cell .code collapsed="true" id="4MyHxnURIXoe"}
``` python
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

# Data
X = np.array([[0,0],[0,1],[1,0],[1,1]])
datasets = {
    "AND": [0,0,0,1],
    "OR":  [0,1,1,1],
    "XOR": [0,1,1,0]
}

# Train & evaluate
for name, y in datasets.items():
    y = np.array(y)

    model = MLPClassifier(hidden_layer_sizes=(2,), max_iter=1000, random_state=0)
    model.fit(X, y)
    preds = model.predict(X)

    print(f"\n{name} Gate")
    print("Accuracy:", accuracy_score(y, preds))
    print(classification_report(y, preds, zero_division=0))
```
:::

::: {.cell .markdown id="fGyBF7b9JAe3"}
# **MLP - IMPLEMENTATION**
:::

::: {.cell .code collapsed="true" id="3ep10yFFI9Ox"}
``` python
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

sig = lambda x: 1 / (1 + np.exp(-x))

class MLP:
    def __init__(self, lr=0.1, epochs=10):
        self.lr, self.epochs = lr, epochs

    def fit(self, X, y):
        np.random.seed(0)
        self.W1 = np.random.randn(2, 2)   # 2 inputs → 2 hidden
        self.W2 = np.random.randn(2, 1)   # 2 hidden → 1 output
        self.b1 = np.zeros(2)
        self.b2 = np.zeros(1)

        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                # forward
                h   = sig(xi @ self.W1 + self.b1)
                out = sig(h  @ self.W2 + self.b2)

                # backprop
                d2 = (yi - out) * out * (1 - out)
                d1 = (d2 @ self.W2.T) * h * (1 - h)

                # update
                self.W2 += self.lr * h.reshape(-1,1)  * d2
                self.b2 += self.lr * d2
                self.W1 += self.lr * xi.reshape(-1,1) * d1
                self.b1 += self.lr * d1

    def predict(self, X):
        return [1 if sig(sig(x @ self.W1 + self.b1) @ self.W2 + self.b2) >= 0.5 else 0 for x in X]

X = np.array([[0,0],[0,1],[1,0],[1,1]])
datasets = {
    "AND": [0,0,0,1],
    "OR":  [0,1,1,1],
    "XOR": [0,1,1,0]
}

a, b = 1, 1   # change inputs here

for name, y in datasets.items():
    y = np.array(y)
    model = MLP()
    model.fit(X, y)
    preds = model.predict(X)

    print(f"\n{name} Gate")
    print("Accuracy:", accuracy_score(y, preds))
    print(classification_report(y, preds, zero_division=0))
    print(f"Test ({a}, {b}) = {model.predict(np.array([[a, b]]))[0]}")
```
:::

::: {.cell .markdown id="WWOmuE9bJ8Or"}
# **MLP - WITH IRIS**
:::

::: {.cell .code execution_count="2" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":1000}" collapsed="true" id="Atbxju9bJWoi" outputId="b9f90cb9-95d7-4d8e-df0c-674da2a6e27e"}
``` python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load data
X, y = load_iris(return_X_y=True)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Model
model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=0)

# Train
model.fit(X_train, y_train)

# Test prediction
preds = model.predict(X_test)

# Metrics
print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

# Learning curve
train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 5)
)

# Plot
plt.plot(train_sizes, train_scores.mean(axis=1), label="Train")
plt.plot(train_sizes, test_scores.mean(axis=1), label="Validation")
plt.xlabel("Training Size")
plt.ylabel("Score")
plt.legend()
plt.title("Learning Curve (MLP - Iris)")
plt.show()
```

::: {.output .stream .stderr}
    /usr/local/lib/python3.12/dist-packages/sklearn/neural_network/_multilayer_perceptron.py:691: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (1000) reached and the optimization hasn't converged yet.
      warnings.warn(
:::

::: {.output .stream .stdout}
    Accuracy: 1.0
                  precision    recall  f1-score   support

               0       1.00      1.00      1.00        11
               1       1.00      1.00      1.00        13
               2       1.00      1.00      1.00         6

        accuracy                           1.00        30
       macro avg       1.00      1.00      1.00        30
    weighted avg       1.00      1.00      1.00        30
:::

::: {.output .stream .stderr}
    /usr/local/lib/python3.12/dist-packages/sklearn/neural_network/_multilayer_perceptron.py:691: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (1000) reached and the optimization hasn't converged yet.
      warnings.warn(
    /usr/local/lib/python3.12/dist-packages/sklearn/neural_network/_multilayer_perceptron.py:691: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (1000) reached and the optimization hasn't converged yet.
      warnings.warn(
    /usr/local/lib/python3.12/dist-packages/sklearn/neural_network/_multilayer_perceptron.py:691: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (1000) reached and the optimization hasn't converged yet.
      warnings.warn(
    /usr/local/lib/python3.12/dist-packages/sklearn/neural_network/_multilayer_perceptron.py:691: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (1000) reached and the optimization hasn't converged yet.
      warnings.warn(
    /usr/local/lib/python3.12/dist-packages/sklearn/neural_network/_multilayer_perceptron.py:691: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (1000) reached and the optimization hasn't converged yet.
      warnings.warn(
    /usr/local/lib/python3.12/dist-packages/sklearn/neural_network/_multilayer_perceptron.py:691: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (1000) reached and the optimization hasn't converged yet.
      warnings.warn(
    /usr/local/lib/python3.12/dist-packages/sklearn/neural_network/_multilayer_perceptron.py:691: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (1000) reached and the optimization hasn't converged yet.
      warnings.warn(
    /usr/local/lib/python3.12/dist-packages/sklearn/neural_network/_multilayer_perceptron.py:691: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (1000) reached and the optimization hasn't converged yet.
      warnings.warn(
    /usr/local/lib/python3.12/dist-packages/sklearn/neural_network/_multilayer_perceptron.py:691: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (1000) reached and the optimization hasn't converged yet.
      warnings.warn(
    /usr/local/lib/python3.12/dist-packages/sklearn/neural_network/_multilayer_perceptron.py:691: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (1000) reached and the optimization hasn't converged yet.
      warnings.warn(
    /usr/local/lib/python3.12/dist-packages/sklearn/neural_network/_multilayer_perceptron.py:691: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (1000) reached and the optimization hasn't converged yet.
      warnings.warn(
    /usr/local/lib/python3.12/dist-packages/sklearn/neural_network/_multilayer_perceptron.py:691: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (1000) reached and the optimization hasn't converged yet.
      warnings.warn(
    /usr/local/lib/python3.12/dist-packages/sklearn/neural_network/_multilayer_perceptron.py:691: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (1000) reached and the optimization hasn't converged yet.
      warnings.warn(
    /usr/local/lib/python3.12/dist-packages/sklearn/neural_network/_multilayer_perceptron.py:691: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (1000) reached and the optimization hasn't converged yet.
      warnings.warn(
:::

::: {.output .display_data}
![](vertopal_89a58630a7204172bb0624b277e751ae/264b8aab5ca372d44c7f630137c3e05ad865abb8.png)
:::
:::

::: {.cell .markdown id="Lakt7CqZKcbN"}
# **MLP - SYNTHETIC DATASET**
:::

::: {.cell .code collapsed="true" id="GQQhj8PkKGQF"}
``` python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Generate synthetic data (FIXED properly)
X, y = make_classification(n_samples=150, n_features=4, n_classes=3,
                           n_informative=3, n_redundant=0, random_state=0)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Model
model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=0)

# Train
model.fit(X_train, y_train)

# Test prediction
preds = model.predict(X_test)

# Metrics
print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

# Learning curve
train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 5)
)

# Plot
plt.plot(train_sizes, train_scores.mean(axis=1), label="Train")
plt.plot(train_sizes, test_scores.mean(axis=1), label="Validation")
plt.xlabel("Training Size")
plt.ylabel("Score")
plt.legend()
plt.title("Learning Curve (MLP - Synthetic Data)")
plt.show()
```
:::

::: {.cell .markdown id="uwW4Puy-L7DY"}
# **LEARNING CURVE FROM SKLEARN**
:::

::: {.cell .code collapsed="true" id="kUHwMnZjKheu"}
``` python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import LearningCurveDisplay

# Load data
X, y = load_iris(return_X_y=True)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Model
model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=0)

# Train
model.fit(X_train, y_train)

# Predict
preds = model.predict(X_test)

# Metrics
print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

# Learning Curve (sklearn style)
LearningCurveDisplay.from_estimator(
    model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 5)
)

plt.title("Learning Curve (MLP - Iris)")
plt.show()
```
:::

::: {.cell .markdown id="JWwqhlBbN70v"}
# **REGULARISATION - L1, L2 and Earlystopping**
:::

::: {.cell .code collapsed="true" id="Nozbk8J6MAs9"}
``` python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, LearningCurveDisplay
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load data
X, y = load_iris(return_X_y=True)

# Scaling (important for SGD)
       #X = StandardScaler().fit_transform(X)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# =========================================================
# 1. BASELINE (MLP)
# =========================================================
model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=500, random_state=0)

model.fit(X_train, y_train)
preds = model.predict(X_test)

print("Baseline Accuracy:", accuracy_score(y_test, preds))

LearningCurveDisplay.from_estimator(model, X, y, cv=5)
plt.title("Baseline (MLP)")
plt.show()


# =========================================================
# 2. L2 REGULARIZATION (MLP)
# =========================================================
model_l2 = MLPClassifier(hidden_layer_sizes=(10,), max_iter=500, random_state=0,
                         alpha=0.01)

model_l2.fit(X_train, y_train)
preds_l2 = model_l2.predict(X_test)

print("L2 Accuracy:", accuracy_score(y_test, preds_l2))

LearningCurveDisplay.from_estimator(model_l2, X, y, cv=5)
plt.title("L2 Regularization (MLP)")
plt.show()


# =========================================================
# 3. EARLY STOPPING (MLP)
# =========================================================
model_es = MLPClassifier(hidden_layer_sizes=(10,), max_iter=500, random_state=0,
                         early_stopping=True, validation_fraction=0.2)

model_es.fit(X_train, y_train)
preds_es = model_es.predict(X_test)

print("Early Stopping Accuracy:", accuracy_score(y_test, preds_es))

LearningCurveDisplay.from_estimator(model_es, X, y, cv=5)
plt.title("Early Stopping (MLP)")
plt.show()


# =========================================================
# 4. L1 REGULARIZATION (SGD CLASSIFIER)
# =========================================================
model_l1 = SGDClassifier(loss='log_loss', penalty='l1', max_iter=1000, random_state=0)

model_l1.fit(X_train, y_train)
preds_l1 = model_l1.predict(X_test)

print("L1 Accuracy (SGD):", accuracy_score(y_test, preds_l1))

LearningCurveDisplay.from_estimator(model_l1, X, y, cv=5)
plt.title("L1 Regularization (SGDClassifier)")
plt.show()
```
:::

::: {.cell .markdown id="1-TJlKJCQrjZ"}
# **CNN - DROPOUT**
:::

::: {.cell .code execution_count="12" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="xJGbg26qN-0m" outputId="d5c8cb7a-9790-405e-c36a-c11735b02e3e"}
``` python
import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
X, y = load_iris(return_X_y=True)

# Scale + reshape for CNN (fake 2D)
X = StandardScaler().fit_transform(X)
X = X.reshape(-1, 2, 2, 1)   # 4 features → 2x2 image

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# =========================================================
# 1. BASELINE CNN
# =========================================================
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(8, (2,2), activation='relu', input_shape=(2,2,1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=30, verbose=0)

print("Baseline Accuracy:", model.evaluate(X_test, y_test, verbose=0)[1])


# =========================================================
# 2. CNN WITH DROPOUT
# =========================================================
model_do = tf.keras.Sequential([
    tf.keras.layers.Conv2D(8, (2,2), activation='relu', input_shape=(2,2,1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.3),   # 👈 Dropout
    tf.keras.layers.Dense(3, activation='softmax')
])

model_do.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_do.fit(X_train, y_train, epochs=30, verbose=0)

print("Dropout Accuracy:", model_do.evaluate(X_test, y_test, verbose=0)[1])
```

::: {.output .stream .stderr}
    /usr/local/lib/python3.12/dist-packages/keras/src/layers/convolutional/base_conv.py:113: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
      super().__init__(activity_regularizer=activity_regularizer, **kwargs)
:::

::: {.output .stream .stdout}
    Baseline Accuracy: 0.6000000238418579
    Dropout Accuracy: 0.5666666626930237
:::
:::

::: {.cell .markdown id="gD22XfT0RAeG"}
# **CNN - DROPOUT WITH LEARNING CURVE**
:::

::: {.cell .code collapsed="true" id="XgroUqe2QviH"}
``` python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load + preprocess
X, y = load_iris(return_X_y=True)
X = StandardScaler().fit_transform(X)
X = X.reshape(-1, 2, 2, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# =========================================================
# 1. BASELINE CNN
# =========================================================
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(8, (2,2), activation='relu', input_shape=(2,2,1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(3, activation='softmax')
])


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=30, validation_split=0.2, verbose=0)

print("Baseline Accuracy:", model.evaluate(X_test, y_test, verbose=0)[1]) # the one not needed

# Plot learning curve
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title("Baseline CNN Learning Curve")
plt.legend()
plt.show()

y_pred = np.argmax(model.predict(X_test), axis=1)
y_prob = model.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, f1_score, log_loss

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
print("Log Loss:", log_loss(y_test, y_prob))


# =========================================================
# 2. CNN WITH DROPOUT
# =========================================================
model_do = tf.keras.Sequential([
    tf.keras.layers.Conv2D(8, (2,2), activation='relu', input_shape=(2,2,1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(3, activation='softmax')
])

model_do.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history_do = model_do.fit(X_train, y_train, epochs=30, validation_split=0.2, verbose=0)

print("Dropout Accuracy:", model_do.evaluate(X_test, y_test, verbose=0)[1])

# Plot learning curve
plt.plot(history_do.history['accuracy'], label='Train')
plt.plot(history_do.history['val_accuracy'], label='Validation')
plt.title("CNN + Dropout Learning Curve")
plt.legend()
plt.show()
```
:::

::: {.cell .markdown id="hlM_oqNVSle3"}
# **CNN - CLASSIFICATION WITH CIFAR10**
:::

::: {.cell .code collapsed="true" id="ErBbn3bmRF5u"}
``` python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing import image

# Load
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
X_train, X_test = X_train/255.0, X_test/255.0

# Model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(32,32,3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32,activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10,activation='softmax')
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

history = model.fit(X_train,y_train,epochs=5,validation_split=0.2,verbose=0)

# Accuracy
print("Accuracy:", model.evaluate(X_test,y_test,verbose=0)[1])

# Classification report
y_pred = np.argmax(model.predict(X_test),axis=1)
print(classification_report(y_test,y_pred))

# Learning curve
plt.plot(history.history['accuracy'],label='Train')
plt.plot(history.history['val_accuracy'],label='Val')
plt.legend(); plt.title("CIFAR Learning Curve"); plt.show()

# Prediction
img = image.load_img("test.jpg",target_size=(32,32))
img = image.img_to_array(img)/255.0
img = np.expand_dims(img,0)

classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
print("Prediction:", classes[np.argmax(model.predict(img))])
```
:::

::: {.cell .markdown id="Gy504ZWhSvW3"}
# **CNN - CLASSIFICATION WITH MNIST**
:::

::: {.cell .code id="LpAg21NvSqb0"}
``` python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing import image

# Load
(X_train,y_train),(X_test,y_test)=tf.keras.datasets.mnist.load_data()
X_train,X_test=X_train/255.0,X_test/255.0
X_train=X_train.reshape(-1,28,28,1)
X_test=X_test.reshape(-1,28,28,1)

# Model
model=tf.keras.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32,activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10,activation='softmax')
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

history=model.fit(X_train,y_train,epochs=5,validation_split=0.2,verbose=0)

print("Accuracy:",model.evaluate(X_test,y_test,verbose=0)[1])

y_pred=np.argmax(model.predict(X_test),axis=1)
print(classification_report(y_test,y_pred))

plt.plot(history.history['accuracy'],label='Train')
plt.plot(history.history['val_accuracy'],label='Val')
plt.legend(); plt.title("MNIST Learning Curve"); plt.show()

# Prediction
img=image.load_img("digit.jpg",target_size=(28,28),color_mode="grayscale")
img=image.img_to_array(img)/255.0
img=img.reshape(1,28,28,1)

print("Prediction:",np.argmax(model.predict(img)))
```
:::

::: {.cell .markdown id="eMzIAS8oS5LR"}
# **CNN - CLASSIFICATION WTIH CUSTOM DATASET**
:::

::: {.cell .code id="fjdjmxxNS9vW"}
``` python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing import image

# Load
train_data=tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/train",image_size=(224,224),batch_size=32)

test_data=tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/test",image_size=(224,224),batch_size=32)

class_names=train_data.class_names

# Model
model=tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255,input_shape=(224,224,3)),
    tf.keras.layers.Conv2D(16,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32,activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(class_names),activation='softmax')
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

history=model.fit(train_data,epochs=5,validation_data=test_data)

print("Accuracy:",model.evaluate(test_data)[1])

# Classification report
y_true=np.concatenate([y for x,y in test_data],axis=0)
y_pred=np.argmax(model.predict(test_data),axis=1)
print(classification_report(y_true,y_pred))

# Learning curve
plt.plot(history.history['accuracy'],label='Train')
plt.plot(history.history['val_accuracy'],label='Val')
plt.legend(); plt.title("Folder Learning Curve"); plt.show()

# Prediction
img=image.load_img("test.jpg",target_size=(224,224))
img=image.img_to_array(img)/255.0
img=np.expand_dims(img,0)

print("Prediction:",class_names[np.argmax(model.predict(img))])
```
:::

::: {.cell .code id="sJrhxGHEQbnf"}
``` python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing import image

#  Load from ONE directory and auto-split 80% train / 20% test
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset",                  # single root folder
    image_size=(224, 224),
    batch_size=32,
    validation_split=0.2,       # 20% goes to test
    subset="training",          # this portion is train
    seed=42                     # same seed = consistent split
)

test_data = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset",                  #  same folder
    image_size=(224, 224),
    batch_size=32,
    validation_split=0.2,       #  must match above
    subset="validation",        #  this portion is test
    seed=42                     #  must match above
)

class_names = train_data.class_names
print("Classes:", class_names)  # ['cats', 'dogs']

# Model (unchanged)
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, epochs=5, validation_data=test_data)

# Evaluate
print("Accuracy:", model.evaluate(test_data)[1])

# Classification report
y_true = np.concatenate([y for x, y in test_data], axis=0)
y_pred = np.argmax(model.predict(test_data), axis=1)
print(classification_report(y_true, y_pred, target_names=class_names))

# Learning curve
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.legend(); plt.title("Learning Curve"); plt.show()

# Single image prediction
img = image.load_img("test.jpg", target_size=(224, 224))
img = image.img_to_array(img) / 255.0
img = np.expand_dims(img, 0)
print("Prediction:", class_names[np.argmax(model.predict(img))])
```
:::

::: {.cell .markdown id="mG9EvRqXY2nm"}
# **SENTIMENT ANALYSIS**
:::

::: {.cell .code id="RLVVMd5bY7iU"}
``` python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, Embedding, Conv1D, MaxPooling1D, Bidirectional
from tensorflow.keras.preprocessing import sequence

# Load data
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

max_len = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_len)
X_test = sequence.pad_sequences(X_test, maxlen=max_len)

# =========================================================
# 1. CNN
# =========================================================
model = Sequential([
    Embedding(top_words, 32, input_length=max_len),
    Conv1D(32, 3, activation='relu'),
    MaxPooling1D(2),
    LSTM(50),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=2, batch_size=64, validation_split=0.2)

print("CNN Accuracy:", model.evaluate(X_test, y_test)[1])

y_pred = (model.predict(X_test) > 0.5).astype(int)
print(classification_report(y_test, y_pred))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("CNN Learning Curve")
plt.show()


# =========================================================
# 2. SIMPLE RNN
# =========================================================
model = Sequential([
    Embedding(top_words, 32, input_length=max_len),
    SimpleRNN(50),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=2, batch_size=64, validation_split=0.2)

print("RNN Accuracy:", model.evaluate(X_test, y_test)[1])

y_pred = (model.predict(X_test) > 0.5).astype(int)
print(classification_report(y_test, y_pred))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("RNN Learning Curve")
plt.show()


# =========================================================
# 3. LSTM
# =========================================================
model = Sequential([
    Embedding(top_words, 32, input_length=max_len),
    LSTM(50),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=2, batch_size=64, validation_split=0.2)

print("LSTM Accuracy:", model.evaluate(X_test, y_test)[1])

y_pred = (model.predict(X_test) > 0.5).astype(int)
print(classification_report(y_test, y_pred))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("LSTM Learning Curve")
plt.show()


# =========================================================
# 4. BIDIRECTIONAL LSTM
# =========================================================
model = Sequential([
    Embedding(top_words, 32, input_length=max_len),
    Bidirectional(LSTM(50)),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=2, batch_size=64, validation_split=0.2)

print("BiLSTM Accuracy:", model.evaluate(X_test, y_test)[1])

y_pred = (model.predict(X_test) > 0.5).astype(int)
print(classification_report(y_test, y_pred))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("BiLSTM Learning Curve")
plt.show()


# =========================================================
# 5. CNN + LSTM (HYBRID)
# =========================================================
model = Sequential([
    Embedding(top_words, 32, input_length=max_len),
    Conv1D(32, 3, activation='relu'),
    MaxPooling1D(2),
    LSTM(50),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=2, batch_size=64, validation_split=0.2)

print("CNN + LSTM Accuracy:", model.evaluate(X_test, y_test)[1])

y_pred = (model.predict(X_test) > 0.5).astype(int)
print(classification_report(y_test, y_pred))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("CNN + LSTM Learning Curve")
plt.show()
```
:::

::: {.cell .code id="lwQHAfB6Y_El"}
``` python
from tensorflow.keras.preprocessing.text import one_hot

# Example text
text = "this movie was amazing and very good"

# Encode text
encoded = one_hot(text, top_words)

# Pad
padded = sequence.pad_sequences([encoded], maxlen=max_len)

# Predict (use last trained model, e.g. BiLSTM)
pred = model.predict(padded)

print("Prediction:", "Positive" if pred > 0.5 else "Negative")
```
:::

::: {.cell .markdown id="XG6DQN6zZzPo"}
# **TEXT GENERATION**
:::

::: {.cell .code id="tKK2930fZ7Lj"}
``` python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN, LSTM, Bidirectional, Conv1D, GlobalMaxPooling1D

# =========================================================
# PREPROCESSING
# =========================================================
text = "deep learning is fun and deep learning is powerful"

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
seq = tokenizer.texts_to_sequences([text])[0]

input_seq = []
for i in range(1, len(seq)):
    input_seq.append(seq[:i+1])

input_seq = pad_sequences(input_seq)
X, y = input_seq[:,:-1], input_seq[:,-1]
y = to_categorical(y)

vocab_size = len(tokenizer.word_index) + 1
max_len = X.shape[1]

# =========================================================
# 1. RNN
# =========================================================
model = Sequential([
    Embedding(vocab_size, 10, input_length=max_len),
    SimpleRNN(50),
    Dense(vocab_size, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X, y, epochs=200, verbose=0)

print("\nRNN Accuracy:", model.evaluate(X, y)[1])
y_pred = np.argmax(model.predict(X), axis=1)
print(classification_report(np.argmax(y,axis=1), y_pred))

plt.plot(history.history['accuracy'])
plt.title("RNN Learning Curve")
plt.show()

# =========================================================
# 2. LSTM
# =========================================================
model = Sequential([
    Embedding(vocab_size, 10, input_length=max_len),
    LSTM(50),
    Dense(vocab_size, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X, y, epochs=200, verbose=0)

print("\nLSTM Accuracy:", model.evaluate(X, y)[1])
y_pred = np.argmax(model.predict(X), axis=1)
print(classification_report(np.argmax(y,axis=1), y_pred))

plt.plot(history.history['accuracy'])
plt.title("LSTM Learning Curve")
plt.show()

# =========================================================
# 3. BiLSTM
# =========================================================
model = Sequential([
    Embedding(vocab_size, 10, input_length=max_len),
    Bidirectional(LSTM(50)),
    Dense(vocab_size, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X, y, epochs=200, verbose=0)

print("\nBiLSTM Accuracy:", model.evaluate(X, y)[1])
y_pred = np.argmax(model.predict(X), axis=1)
print(classification_report(np.argmax(y,axis=1), y_pred))

plt.plot(history.history['accuracy'])
plt.title("BiLSTM Learning Curve")
plt.show()

# =========================================================
# 4. CNN
# =========================================================
model = Sequential([
    Embedding(vocab_size, 10, input_length=max_len),
    Conv1D(32, 3, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(vocab_size, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X, y, epochs=200, verbose=0)

print("\nCNN Accuracy:", model.evaluate(X, y)[1])
y_pred = np.argmax(model.predict(X), axis=1)
print(classification_report(np.argmax(y,axis=1), y_pred))

plt.plot(history.history['accuracy'])
plt.title("CNN Learning Curve")
plt.show()

# =========================================================
# 5. CNN + LSTM
# =========================================================
model = Sequential([
    Embedding(vocab_size, 10, input_length=max_len),
    Conv1D(32, 3, activation='relu'),
    LSTM(50),
    Dense(vocab_size, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X, y, epochs=200, verbose=0)

print("\nCNN+LSTM Accuracy:", model.evaluate(X, y)[1])
y_pred = np.argmax(model.predict(X), axis=1)
print(classification_report(np.argmax(y,axis=1), y_pred))

plt.plot(history.history['accuracy'])
plt.title("CNN+LSTM Learning Curve")
plt.show()

# =========================================================
# USER INPUT TEXT GENERATION (uses last model)
# =========================================================
seed = "deep learning"

for _ in range(3):
    encoded = tokenizer.texts_to_sequences([seed])[0]
    encoded = pad_sequences([encoded], maxlen=max_len)

    pred = np.argmax(model.predict(encoded))

    for word, index in tokenizer.word_index.items():
        if index == pred:
            seed += " " + word
            break

print("\nGenerated Text:", seed)
```
:::

::: {.cell .markdown id="vZag0_Mik0BV"}
# **TRANSFER LEARNING**
:::

::: {.cell .code id="SFpxmNxCk29B"}
``` python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# =========================================================
# 1. AS-IT-IS MODE
# =========================================================
img = image.load_img("test.jpg", target_size=(224,224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

model = MobileNetV2(weights='imagenet')

pred = model.predict(x)
print("\nAs-it-is Prediction:", decode_predictions(pred, top=1)[0][0])


# =========================================================
# LOAD DATA (COMMON FOR 2 & 3)
# =========================================================
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

X_train = tf.image.resize(X_train, (224,224)) / 255.0
X_test = tf.image.resize(X_test, (224,224)) / 255.0


# =========================================================
# 2. FEATURE EXTRACTION (INDEPENDENT)
# =========================================================
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base.trainable = False

model = tf.keras.Sequential([
    base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=3, validation_split=0.2, verbose=0)

print("\nFeature Extraction Accuracy:", model.evaluate(X_test, y_test, verbose=0)[1])

y_pred = np.argmax(model.predict(X_test), axis=1)
print(classification_report(y_test, y_pred))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Feature Extraction Learning Curve")
plt.show()


# =========================================================
# 3. FINE-TUNING (INDEPENDENT)
# =========================================================
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))

# Freeze most layers
for layer in base.layers[:-20]:
    layer.trainable = False

model = tf.keras.Sequential([
    base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=3, validation_split=0.2, verbose=0)

print("\nFine-tuning Accuracy:", model.evaluate(X_test, y_test, verbose=0)[1])

y_pred = np.argmax(model.predict(X_test), axis=1)
print(classification_report(y_test, y_pred))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Fine-tuning Learning Curve")
plt.show()


# =========================================================
# 4. USER IMAGE (CUSTOM MODEL PREDICTION)
# =========================================================
img = image.load_img("test.jpg", target_size=(224,224))
x = image.img_to_array(img)/255.0
x = np.expand_dims(x, axis=0)

pred = np.argmax(model.predict(x))
print("\nCustom Model Prediction (class index):", pred)
```
:::

::: {.cell .code id="0rKh-9VGljiz"}
``` python
```
:::

::: {.cell .code id="33jGbuSgljYG"}
``` python
import matplotlib.pyplot as plt

# Labels (x-axis)
models = ['CNN', 'RNN', 'LSTM', 'BiLSTM']

# Values (y-axis)
accuracy = [0.82, 0.78, 0.85, 0.88]

# Plot
plt.bar(models, accuracy)

# Labels
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Model Comparison")

plt.show()
```
:::

::: {.cell .markdown id="KSuFG2gRNyGh"}
# **IMAGE TO ONE LAYER OF CNN**
:::

::: {.cell .code id="iSnXG7rDOvRe"}
``` python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# =========================================================
# 1. LOAD IMAGE
# =========================================================
img = image.load_img("myimage.jpg", target_size=(32,32))
img_array = image.img_to_array(img) / 255.0
img_array = img_array.reshape(1, 32, 32, 3)

# =========================================================
# 2. CNN LAYER (edge-like extraction)
# =========================================================
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(1, (3,3), activation='relu', input_shape=(32,32,3))
])

# =========================================================
# 3. GET OUTPUT
# =========================================================
output = model.predict(img_array)

print("Output shape:", output.shape)  # (1, 30, 30, 1)

# =========================================================
# 4. SHOW AS GRAYSCALE (SKELETON-LIKE)
# =========================================================
plt.imshow(output[0,:,:,0], cmap='gray')
plt.title("Feature Map (Skeleton-like)")
plt.axis('off')
plt.show()
```
:::
