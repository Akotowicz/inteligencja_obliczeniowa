from sklearn import tree
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
# matrix
from sklearn.metrics import confusion_matrix
# plot
import matplotlib.pyplot as plt

df = pd.read_csv("iris.csv")
(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=12)

train_inputs = train_set[:, 0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]

def decisionTree(train_inputs, train_classes, test_inputs):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_inputs, train_classes)
    return clf.predict(test_inputs)

def knn(n, train_inputs, train_classes, test_inputs):
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(train_inputs, train_classes)
    return knn.predict(test_inputs)

def naive_bayes(train_inputs, train_classes, test_inputs):
    gnb = GaussianNB()
    gnb.fit(train_inputs, train_classes)
    return gnb.predict(test_inputs)

# ################## procentowa dokładność i macierz błędu ################################

len = test_set.shape[0]
matrices = ['Virginica', 'Versicolor', 'Setosa']

def good_predictions(classificator_answers, test_classes):
    good_predictions = 0
    for i in range(len):
        if(classificator_answers[i] == test_classes[i]):
            good_predictions = good_predictions + 1
    return good_predictions / len * 100

def macierz_bledu(classificator_answers, test_classes, pd, matrices):
    print(confusion_matrix(test_classes, classificator_answers))
    # df_cm = pd.DataFrame(confusion_matrix(test_classes, classificator_answers),
    #                     index=matrices,
    #                     columns=matrices)
    # plt.figure(figsize=(10, 7))
    # sn.heatmap(df_cm, annot=True)
    # plt.show()

# ##################### wypisanie ###########################################################

dd = decisionTree(train_inputs, train_classes, test_inputs)
knn_3 = knn(3, train_inputs, train_classes, test_inputs)
knn_5 = knn(5, train_inputs, train_classes, test_inputs)
knn_11 = knn(11, train_inputs, train_classes, test_inputs)
nb = naive_bayes(train_inputs, train_classes, test_inputs)

print("Ilość procentowa dobrych odpowiedzi dla klasyfikatora DD:           ", good_predictions(dd, test_classes), "%")
print("Ilość procentowa dobrych odpowiedzi dla klasyfikatora KNN3:         ", good_predictions(knn_3, test_classes), "%")
print("Ilość procentowa dobrych odpowiedzi dla klasyfikatora KNN5:         ", good_predictions(knn_5, test_classes), "%")
print("Ilość procentowa dobrych odpowiedzi dla klasyfikatora KNN11:        ", good_predictions(knn_11, test_classes), "%")
print("Ilość procentowa dobrych odpowiedzi dla klasyfikatora Native Bayes: ", good_predictions(nb, test_classes), "%")

print("\nMacierz błędu dla dd:")
macierz_bledu(dd, test_classes, pd, matrices)
print("\nMacierz błędu dla KNN3:")
macierz_bledu(knn_3, test_classes, pd, matrices)
print("\nMacierz błędu dla KNN5:")
macierz_bledu(knn_5, test_classes, pd, matrices)
print("\nMacierz błędu dla KNN11:")
macierz_bledu(knn_11, test_classes, pd, matrices)
print("\nMacierz błędu dla Native Bayes:")
macierz_bledu(nb, test_classes, pd, matrices)

# ##################### wykres słupkowy ###########################################################

x = [good_predictions(dd, test_classes), good_predictions(knn_3, test_classes),good_predictions(knn_5, test_classes),good_predictions(knn_11, test_classes), good_predictions(nb, test_classes)]
bar_labels = ['dd','knn 3','knn 5', 'knn 11', 'naive bayes']
bar_colors = ['lightblue', 'purple', 'darkred',  'darkblue', 'black']

plt.bar(bar_labels, x, color=bar_colors)
plt.show()