import numpy
import numpy as np

def akt(x):
    return 1 / (1 + numpy.exp(-x))

def forwardPass(wiek, waga, wzrost):
    hidden1 = np.sum(np.array([-0.46122, 0.97314, -0.39203, 0.80109]) * np.array([wiek, waga, wzrost, 1]))
    hidden1_po_aktywacji = akt(hidden1)
    hidden2 = np.sum(np.array([0.78548, 2.10584, -0.57847, 0.43529]) * np.array([wiek, waga, wzrost, 1]))

    hidden2_po_aktywacji = akt(hidden2)
    output = np.sum(
        np.array([-0.81546, 1.03775, -0.2368])
        * np.array([hidden1_po_aktywacji, hidden2_po_aktywacji, 1])
    )
    return output

print(forwardPass(23, 75, 176))
