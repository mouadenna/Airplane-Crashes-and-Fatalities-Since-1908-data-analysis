import numpy as np
from IPython.display import clear_output


class ReseauNeural(object):

    def __init__(self, tailles):
        self.num_layers = len(tailles)
        self.tailles = tailles
        self.biais = [np.random.randn(y, 1) for y in tailles[1:]]
        self.poids = [np.random.randn(y, x) for x, y in zip(tailles[:-1], tailles[1:])]

    def propagation_directe(self, a,f):
        for b, w in zip(self.biais, self.poids):
            a = f(np.dot(w, a)+b)
        return a
    
    def Descente_gradient1(self, donnees_entrainement, taille_mini_lot, eta,_return,f, f_prime, donnees_test=None):
        L = []
        n_test = len(donnees_test)
        L.append([0, self.calc_loss(donnees_test, f)])
        n = len(donnees_entrainement)
        mini_lots = [donnees_entrainement[k:k + taille_mini_lot] for k in range(0, n, taille_mini_lot)]
        total_loss = 0  # Initialize total_loss for calculating average
        total_examples_processed = 0  
        clear_output(wait=True)
        
        for i, mini_lot in enumerate(mini_lots, 1):
            eta0=eta
            self.mettre_a_jour(mini_lot, eta, f, f_prime)
            total_examples_processed += len(mini_lot)  # Increment total examples processed
            total_loss += self.calc_loss(mini_lot, f)
            
            if total_examples_processed >= _return:
                avg_loss = self.calc_loss(donnees_test, f)
                print(f"Examples {i * taille_mini_lot} - Average Test Loss: {avg_loss}")

                if avg_loss>=L[-1][-1]:
                    L.append([i * taille_mini_lot, avg_loss])
                total_examples_processed = 0  # Reset total examples processed
                
        return L
        
    def Descente_gradient(self, donnees_entrainement, epochs, taille_mini_lot, eta,check_lr, f, f_prime, donnees_test=None):
        L = []
        n_test = len(donnees_test)
        L.append([0, self.calc_loss(donnees_test, f)])
        n = len(donnees_entrainement)
        mini_lots = [donnees_entrainement[k:k + taille_mini_lot] for k in range(0, n, taille_mini_lot)]
        total_loss = 0  # Initialize total_loss for calculating average
        total_examples_processed = 0  
        exemples=0
        clear_output(wait=True)
        
        for j in range(1, epochs + 1):
            eta0=eta
            #clear_output(wait=True)
            print(f"epoch : {j}")
            np.random.shuffle(donnees_entrainement)
            mini_lots = [donnees_entrainement[k:k + taille_mini_lot] for k in range(0, n, taille_mini_lot)]
            for i, mini_lot in enumerate(mini_lots, 1):
                self.mettre_a_jour(mini_lot, eta, f, f_prime)
                total_examples_processed += len(mini_lot)  # Increment total examples processed
                exemples+=len(mini_lot)
                total_loss += self.calc_loss(mini_lot, f)

                if exemples>=50000:
                    avg_loss = self.calc_loss(donnees_test, f)
                    print(f"[Examples {i * taille_mini_lot}] - [Average Test Loss: {avg_loss}]")
                    exemples=0
                    
                if total_examples_processed >= check_lr: #checker le lr chaque check_lr exemples
                    if avg_loss>=L[-1][-1]:
                        eta=eta/2
                        print(eta)
                        total_examples_processed = 0  # Reset total examples processed
            L.append([j , i * taille_mini_lot, avg_loss])
        return L

    def calc_loss(self, donnees_test, f):
        total_loss = 0
        for x, y in donnees_test:
            predicted = self.propagation_directe(x, f)
            loss = self.loss(predicted, y)
            total_loss += loss
        avg_loss = total_loss / len(donnees_test)
        return avg_loss[0][0]
    
    
    def loss(self,predicted, actual):# MSE/2 
        return ((predicted - actual)**2)/2
        

    def mettre_a_jour(self, mini_lot, eta,f,f_prime):
        nabla_b = [np.zeros(b.shape) for b in self.biais]
        nabla_w = [np.zeros(w.shape) for w in self.poids]
        for x, y in mini_lot:
            delta_nabla_b, delta_nabla_w = self.retroprop(x, y,f,f_prime)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.poids = [w-(eta/len(mini_lot))*nw for w, nw in zip(self.poids, nabla_w)]
        self.biais = [b-(eta/len(mini_lot))*nb for b, nb in zip(self.biais, nabla_b)]


    def retroprop(self, x, y,f,f_prime):
        nabla_b = [np.zeros(b.shape) for b in self.biais]
        nabla_w = [np.zeros(w.shape) for w in self.poids]
        activation = x
        activations = [x] # liste pour stocker toutes les activations, couche par couche
        zs = [] # liste pour stocker tous les vecteurs z, couche par couche
        #passage en avant (calculer les activations correspondant à x)
        for b, w in zip(self.biais, self.poids):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = f(z)
            activations.append(activation)
        # passage en arrière
        delta = self.derivee_cout(activations[-1], y) * f_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = f_prime(z)
            delta = np.dot(self.poids[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def derivee_cout(self, activation_sortie, y):
        return (activation_sortie-y)

    
#fonction d'activation avec une sortie entre -1 et 1
def sigmoid(z):
    return 2.0 / (1.0 + np.exp(-z)) - 1.0
def sigmoid_prime(z):
    return 2.0 * np.exp(-z) / ((1.0 + np.exp(-z)) ** 2)
def tanh(z):
    return np.tanh(z)
def tanh_prime(z):
    return 1-tanh(z)**2
def arctan(z):
    return np.arctan(z)/(np.pi/2)
def arctan_prime(z):
    return (1/(1+z**2))/(np.pi/2)
def relu(z):
    return np.maximum(0, z)
def relu_prime(z):
    return np.where(z > 0, 1, 0) 
