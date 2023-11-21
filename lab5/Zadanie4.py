import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import normalize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.math import confusion_matrix as keras_confusion_matrix
from tensorflow.keras.metrics import CategoricalAccuracy
from ann_visualizer.visualize import ann_viz

df = pd.read_csv('diabetes.csv')
df['class'] = df['class'].replace({'tested_negative': 0, 'tested_positive': 1})

df = df.sample(frac=1, random_state=2)
train_size = int(0.7 * len(df))
train_set, test_set = df.iloc[:train_size], df.iloc[train_size:]

train_inputs = train_set.iloc[:, :-1].values
train_labels = train_set.iloc[:, -1].values
test_inputs = test_set.iloc[:, :-1].values
test_labels = test_set.iloc[:, -1].values

train_inputs = normalize(train_inputs, axis=0)
test_inputs = normalize(test_inputs, axis=0)

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model = Sequential()
model.add(Dense(6, activation='relu', input_dim=8))
model.add(Dense(3, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[CategoricalAccuracy()])

# Train the model
history = model.fit(train_inputs, train_labels, epochs=100, validation_split=0.3)

plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Test')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

pred_probs = model.predict(test_inputs)
predictions = np.argmax(pred_probs, axis=1)

print("Macierz błędu:")
print(keras_confusion_matrix(np.argmax(test_labels, axis=1), predictions))

# Calculate and display accuracy using Keras CategoricalAccuracy
accuracy = CategoricalAccuracy()(test_labels, pred_probs).numpy()
print("Dokładność procentowa: {:.2%}".format(accuracy))

ann_viz(model, title="Sieć neuronowa w formie grafu")