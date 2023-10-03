import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time#Pour attendre / déboggage



# Fonction de calcul des distances entre deux vecteurs
def euclidean_distance(x1, x2):
    return np.sum((x1 - x2)*(x1 - x2))

# Fonction de recherche des k voisins les plus proches
def get_neighbors(X_train, Y_train, x_test, k):
    distances = []
    for i in range(X_train.shape[0]):
        distance = euclidean_distance(X_train.iloc[i], x_test)
        distances.append((distance, Y_train.iloc[i]))
    distances.sort(key=lambda x: x[0])
    neighbors = [distances[i][1] for i in range(k)]
    return neighbors

# Fonction de prédiction de la classe d'un nouvel exemple
def predict(X_train, Y_train, x_test, k):
    neighbors = get_neighbors(X_train, Y_train, x_test, k)
    prediction = max(set(neighbors), key=neighbors.count)
    return prediction

# Fonction d'évaluation de la précision du modèle
def evaluate(X_train, Y_train, X_valid, Y_valid, k):
    correct = 0
    for i in range(X_valid.shape[0]):
        print(" i : ", i)
       
        y_pred = predict(X_train, Y_train, X_valid.iloc[i], k)
        if y_pred == Y_valid.iloc[i]:
            correct += 1
    accuracy = correct / X_valid.shape[0]
    return accuracy



# Lecture des données
data = pd.read_csv('/home/etud/Téléchargements/mobile_train.csv')

# Préparation des données
X = data.drop('price_range', axis=1)
Y = data['price_range']

# Normalisation des données
X = (X - X.min()) / (X.max() - X.min()) * 1000


# Séparation des données en ensembles d'entraînement et de validation
X_train = X.sample(frac=0.8, random_state=990)
X_valid = X.drop(X_train.index)
Y_train = Y.loc[X_train.index]
Y_valid = Y.loc[X_valid.index]


def evaluate_with_coefficients(X_train, Y_train, X_valid, Y_valid, coefficients):
    X = X_train * coefficients
    X_valid = X_valid * coefficients
    
    k = 50
    return evaluate(X, Y_train, X_valid, Y_valid, k)


# Définition de la fonction de recherche des coefficients optimaux (bruptes forces) aléatoires
def find_optimal_coefficients(X_train, Y_train, X_valid, Y_valid):
    np.random.seed(42)
    best_coefficients = np.zeros(X_train.shape[1])
    best_accuracy = 0
    for i in range(100000):
        print("Fois numéro : ", i)
        coefficients = np.random.randint(0, 201, size=X_train.shape[1])
        accuracy = evaluate_with_coefficients(X_train, Y_train, X_valid, Y_valid, coefficients)
        if accuracy > best_accuracy:
            print("Meilleurs trucs : ", accuracy, " Coefficients : ", coefficients)
            best_accuracy = accuracy
            best_coefficients = coefficients
    return best_coefficients, best_accuracy




#On regarde quels valeurs sont importantes
coefficients = np.array([
    0.08,  # battery_power
    0.01,  # blue
    0.02,  # clock_speed
    0.02,  # dual_sim
    0.04,  # fc
    0.02,  # four_g
    0.05,  # int_memory
    0.02,  # m_dep
    0.03,  # mobile_wt
    0.07,  # n_cores
    0.04,  # pc
    0.05,  # px_height
    0.05,  # px_width
    0.25,   # ram
    0.04,  # sc_h
    0.04,  # sc_w
    0.03,  # talk_time
    0.02,  # three_g
    0.01,  # touch_screen
    0.01   # wifi
])

X_train = X_train * coefficients;
X_valid = X_valid * coefficients;


#Entre 35 et 40 c'est le mieux

def afficherMeilleursK(X_train, Y_train, X_valid, Y_valid):
    # Recherche du meilleur k
    accuracies = []
    for k in range(1, 400):
         accuracy = evaluate(X_train, Y_train, X_valid, Y_valid, k)
         accuracies.append(accuracy)
         print('k =', k, ': accuracy =', accuracy)

     # Affichage de la courbe d'évolution de la précision en fonction de k
    plt.plot(range(1, 400), accuracies)
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs k')
    plt.show()




k = 40
#print('k =', k, ': accuracy =', evaluate(X_train, Y_train, X_valid, Y_valid, k))

# Recherche du meilleur k
#afficherMeilleursK(X_train, Y_train, X_valid, Y_valid)




#Test avec le valid


X_test = pd.read_csv('/home/etud/Bureau/Python/mobile_test_data.csv')
# Normalisation des données
X_test = (X_test - X_test.min()) / (X_test.max() - X_test.min()) * 1000

X_test*=coefficients


print(X_test)


Y_test = []

for i in range(0, X_test.shape[0]):
    print(" i : ", i, " / ", X_test.shape[0])
    Y_test.append(predict(X_train, Y_train, X_test.iloc[i], k))


print(Y_test)

#Sauvegarde du fichier
np.savetxt('/home/etud/Bureau/Python/mobile.txt', Y_test, fmt='%d')
#Ecriture du résultat

