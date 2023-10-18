import math
import random
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# ################ Zadanie 1a ###########################
def prime(n):
    if n == 1:
        return False;
    elif n > 1:
        for i in range(2, n):
            if (n % i) == 0:
                return False
    return True;

print("Wynik:", prime(1));
print("Wynik:", prime(27));
print("Wynik:", prime(11));

# ################ Zadanie 1b ###########################
def select_primes(x):
    foundPrimes = []
    for i in range(0, len(x)):
        if (prime(x[i]) == True):
            foundPrimes.append(x[i])
    return foundPrimes

print("\nWynik:", select_primes([3, 6, 11, 25, 19]));

# ################ Zadanie 2a ###########################

w1 = np.array([3, 8, 9, 10, 12])
w2 = np.array([8, 7, 7, 5, 6])
wAdd = w1 + w2
print("\nSuma wektorow", wAdd)
wMultiply = w1 * w2
print("Iloczyn wektorow", wMultiply)

# ################ Zadanie 2b ###########################
print("\nIloczyn skalarny", sum(wMultiply))

# ################ Zadanie 2c ###########################
def dlugoscEuklidesowa(v):
    dl = 0
    for i in range(0, len(v)):
        dl += v[i] ** 2
    return math.sqrt(dl)

print("\nDługość euklidesowa w1", dlugoscEuklidesowa(w1))
print("Długość euklidesowa w2", dlugoscEuklidesowa(w2))

# ################ Zadanie 2d ###########################
def createRandVector(ile, od, do):
    randV = []
    for i in range(0, ile):
        randV.append(random.randint(od, do))
    return randV

randV = createRandVector(50, 1, 100)
print("Randomowy wektor", randV)

# ################ Zadanie 2e ###########################

print("Średnia", np.mean(randV))
print("Min", np.min(randV))
print("Max", np.max(randV))
print("Odchylenie std", np.std(randV))

# ################ Zadanie 2f ###########################

def normalizacjaV(v):
    newV = []
    min = np.min(v)
    max = np.max(v)
    for i in range(0, len(v)):
        newV.append((v[i] - min) / (max - min))
    return newV

normRandV = normalizacjaV(randV)
print("\nNormalizacja", normRandV)
indexMaxRandV = randV.index(max(randV))
print("Index max randV", indexMaxRandV)
print("Co stoi na tym miejscu po normalizacji?", normRandV[indexMaxRandV])

# ################ Zadanie 2g ###########################

def standaryzacjaV(v):
    newV = []
    mean = np.mean(v)
    std = np.std(v)
    for i in range(0, len(v)):
        newV.append((v[i] - mean) / std)
    return newV

WZ = standaryzacjaV(randV)
print("\nStandaryzacja", WZ)
print("Średnia", np.mean(WZ))
print("Std", np.std(WZ))

# ################ Zadanie 3 a ###########################

miasta = pd.read_csv(r"miasta.csv")
print("\nMiasta:\n", miasta)
print("\nMiasta Values\n", miasta.values)

# ################ Zadanie 3 b ###########################

miasta.loc[len(miasta.index)] = [2010, 460, 555, 405]
print("\nMiasta z rokiem 2010:\n", miasta)

# ################ Zadanie 3 c ###########################

plt.plot(miasta['Rok'], miasta['Gdansk'], 'r.-' )
plt.ylabel('Liczba ludności [w tys.]')
plt.xlabel('Lata')
plt.suptitle('Ludność W mieście Gdańsk')
plt.show()

# ################ Zadanie 3 d ###########################

y = miasta.drop(columns=['Rok'])
plt.plot(miasta['Rok'], y, '.-' )
plt.ylabel('Liczba ludności [w tys.]')
plt.xlabel('Lata')
plt.suptitle('Zmiany ludności miastach Polski')
plt.legend(['Gdansk', 'Poznan', 'Szczecin'])
plt.show()
