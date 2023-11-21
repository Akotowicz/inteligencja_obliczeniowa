import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics

df = pd.read_csv("diabetes.csv")

df['class'] = df['class'].replace({'tested_negative': 0, 'tested_positive': 1})
predictors = list(set(list(df.columns))-set(['class']))
df[predictors] = df[predictors]/df[predictors].max()

train_inputs, train_classes, test_inputs, test_classes = train_test_split(df[predictors].values, df['class'].values, test_size=0.30)

mlp = MLPClassifier(hidden_layer_sizes=(6,3), activation='relu', solver='adam', max_iter=500)
mlp.fit(train_inputs, test_inputs)

predict_train = mlp.predict(train_inputs)
predict_test = mlp.predict(train_classes)

print("Macierz błędu na zbiorze testowym:")
print(confusion_matrix(test_classes, predict_test))

accuracy = metrics.accuracy_score(test_classes, predict_test)
print("\nDokładność procentowa na zbiorze testowym: {:.2%}".format(accuracy))

