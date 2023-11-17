from sklearn import tree
import pandas as pd
from sklearn.model_selection import train_test_split
# matrix
from sklearn.metrics import confusion_matrix
# plot
import seaborn as sn
import matplotlib.pyplot as plt

df = pd.read_csv("iris.csv")
(train_set, test_set) = train_test_split(df.values, train_size=0.7,
                                         random_state=12)

train_inputs = train_set[:, 0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_inputs, train_classes)

r = tree.export_text(clf, feature_names=list(df.columns.values[0:4]))
print(r)

good_predictions = 0
len = test_set.shape[0]

for i in range(len):
    if clf.predict([test_inputs[i]]) == test_classes[i]:
        good_predictions = good_predictions + 1
    else:
        print("Błąd: ", test_set[i])

def macierz_bledu(clf, test_inputs, test_classes, pd, matrices):
    y_pred = clf.predict(test_inputs)
    print(confusion_matrix(test_classes, y_pred))

    df_cm = pd.DataFrame(confusion_matrix(test_classes, y_pred),
                            index=matrices,
                            columns=matrices)
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)
    plt.show()

print("Ilość dobrych odpowiedzi: ", good_predictions)
print(good_predictions / len * 100, "%")

matrices = ["Setosa", "Versicolor", "Virginica"]
print(macierz_bledu(clf,test_inputs, test_classes, pd, matrices))