from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time #Pour attendre / déboggage


# Lecture des données
donnees = pd.read_csv('/home/etud/Téléchargements/mobile_train.csv')

# Préparation des données
X = donnees.drop('price_range', axis=1)
Y = donnees['price_range']

X = (X - X.min()) / (X.max() - X.min()) * 1000

# Séparation des données en ensembles d'entraînement et de validation
X_apprentissage, X_validation, Y_apprentissage, Y_validation = train_test_split(X, Y, test_size=0.2, random_state=990)

# Liste des activations à tester
activations = ['identity', 'logistic', 'tanh', 'relu']

# Liste des nombres de couches cachées à tester
couches_cachees = range(1, 110)

# Initialisation de la liste des précisions pour chaque combinaison d'activation et de nombre de couches cachées
precisions = np.zeros((len(activations), len(couches_cachees)))

# Entraînement du modèle pour chaque combinaison d'activation et de nombre de couches cachées
for i, activation in enumerate(activations):
    for j, nb_couches_cachees in enumerate(couches_cachees):
        clf = MLPClassifier(hidden_layer_sizes=(nb_couches_cachees,), activation=activation, max_iter=1000)
        clf.fit(X_apprentissage, Y_apprentissage)
        Y_prediction = clf.predict(X_validation)
        precision = accuracy_score(Y_validation, Y_prediction)
        precisions[i, j] = precision
        
# Tracé de la courbe de précision en fonction des activations et des nombres de couches cachées
for i, activation in enumerate(activations):
    plt.plot(couches_cachees, precisions[i], label=activation)
plt.xlabel('Nombre de couches cachées')
plt.ylabel('Précision')
plt.legend()
plt.show()
