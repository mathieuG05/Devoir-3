import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.optimize import fsolve
from sympy import solve, integrate, symbols, sqrt, sin, cos

#A
#Trouver la valeur de a
def équation_a(x_0):#Équation
    return round(x_0, 3)
print('La valeur de a est :', équation_a(x_0=5))#Résultat

#Trouver la valeur de mu
def équation_mu(m, ga, k):#Équation
    return round(sqrt((k/m) - ((ga**2)/(4*m**2))), 3)
print('la valeur de mu est :',équation_mu(m=10, ga=10, k=200))#Résultat

#Trouver la valeur de alpha
def équation_alpha(m, ga):#Équation
    return round(ga/(2*m), 3)
print('La valeur de alpha est :', équation_alpha(m=10, ga=10))#Résultat

#Trouver la valeur de b
def équation_b(v_x0, x_0, al, mu):#Équation
   return round((v_x0 + al*x_0)/mu, 3)
print('La valeur de b est :', équation_b(v_x0=0, x_0=5, al=0.5, mu=4.444))#Résultat

#Trouver la valeur de T
def équation_T(mu):#Équation
    return round((2*np.pi)/mu, 3)
print('La valeur de T est:', équation_T(mu=4.444))#Résultat

#B
#Tracer le graphique
#Valeurs des constantes
mu = 4.444
a = 5
al = 0.628
b = 0.563
#Définir l'équation
def x(t):
    return np.exp(-al*t)*(a*np.cos(mu*t)+b*np.sin(mu*t))#Équation

t = np.linspace(0, 10, 500)

x = x(t)

#Paramètre du graphique
plt.figure(figsize=(10, 6))
plt.plot(t, x)
plt.xlabel('Temps (s)')
plt.ylabel('Position (m)')
plt.title('Position en x en fonction du temps')
plt.grid(True)
plt.xlim(0, 3*1.414)
plt.show()

#C
#Valeur des variables
m = 10
ga = 10
k = 200
mu = 4.444
T = 2*np.pi/mu
x0 = 5.0
v0 = 0.0
N = 500
dt = 3*T / N

# Initialisation des listes de données
data_t = [0]
data_x = [x0]
data_vx = [v0]

# Méthode d'Euler pour l'analyse numérique
for i in range(1, N):  # Correction d'indentation ici
    # Calcul de la nouvelle vitesse
    v = (-ga * data_vx[i-1] - k * data_x[i-1]) / m
    data_vx.append(data_vx[i-1] + dt * v)
    # Calcul de la nouvelle position
    data_x.append(data_x[i-1] + dt * data_vx[i])
    # Ajout du nouveau temps
    data_t.append(data_t[i-1] + dt)

#D
mu = 4.444
a = 5
al = 0.628
b = 0.563

# Définition de l'équation donnée
def x(t):
    return np.exp(-al*t) * (a*np.cos(mu*t) + b*np.sin(mu*t))

# Paramètres du graphique
t = np.linspace(0, 10, 500)

# Création du graphique
plt.figure(figsize=(10, 6))

# Ajout du graphique de l'analyse numérique
plt.plot(data_t, data_x, label='Analyse numérique', color='red')

# Ajout du graphique de l'équation donnée
plt.plot(t, x(t), label='Équation donnée', linestyle='--', color='blue')

# Paramètres du graphique
plt.xlabel('Temps (s)')
plt.ylabel('Position (m)')
plt.title('Comparaison de l\'analyse numérique avec l\'équation donnée')
plt.grid(True)
plt.legend()
plt.xlim(0, 3 * np.sqrt(2))  # Limite en x jusqu'à 3 * sqrt(2)
plt.show()

#E