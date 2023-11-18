import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

data = pd.read_csv("../lab4/iris.csv")
df_norm = data[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

target = data[['variety']].replace(['Setosa','Versicolor','Virginica'],[0,1,2])
df = pd.concat([df_norm, target], axis=1)

train, test = train_test_split(df, test_size = 0.3)
trainX = train[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]
trainY=train.variety
testX= test[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]
testY =test.variety

def check(hidden):
    clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=hidden)
    clf.fit(trainX, trainY)
    prediction = clf.predict(testX)
    print('The accuracy of the Multi-layer Perceptron is:',metrics.accuracy_score(prediction,testY))

check((2))
check((3))
check((3,3))