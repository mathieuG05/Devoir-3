import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.optimize import fsolve
from sympy import solve, integrate, symbols, sqrt, sin, cos

# A
# Trouver la valeur de a
def équation_a(x_0):  # Équation
    return round(x_0, 3)

print('La valeur de a est :', équation_a(x_0=5))  # Résultat

# Trouver la valeur de mu
def équation_mu(m, ga, k):  # Équation
    return round(sqrt((k/m) - ((ga**2)/(4*m**2))), 3)

print('la valeur de mu est :', équation_mu(m=10, ga=10, k=200))  # Résultat

# Trouver la valeur de alpha
def équation_alpha(m, ga):  # Équation
    return round(ga/(2*m), 3)

print('La valeur de alpha est :', équation_alpha(m=10, ga=10))  # Résultat

# Trouver la valeur de b
def équation_b(v_x0, x_0, al, mu):  # Équation
   return round((v_x0 + al*x_0)/mu, 3)

print('La valeur de b est :', équation_b(v_x0=0, x_0=5, al=0.5, mu=4.444))  # Résultat

# Trouver la valeur de T
def équation_T(mu):  # Équation
    return round((2*np.pi)/mu, 3)

print('La valeur de T est:', équation_T(mu=4.444))  # Résultat

# B
# Tracer le graphique
# Valeurs des constantes
mu = 4.444
a = 5
al = 0.5
b = 0.563

# Définir l'équation
def x_(t):
    return np.exp(-al*t)*(a*np.cos(mu*t)+b*np.sin(mu*t))  # Équation

t = np.linspace(0, 10, 500)
x_ = x_(t)

# Paramètre du graphique
plt.figure(figsize=(10, 6))
plt.plot(t, x_)
plt.xlabel('Temps (s)')
plt.ylabel('Position (m)')
plt.title('Position en x en fonction du temps')
plt.grid(True)
plt.xlim(0, 3*1.414)
plt.show()

# C
# Valeur des variables
m = 10
ga = 10
k = 200
mu = 4.444
T = 2*np.pi/mu
N = 500
h = 3*T / N  

# Initialisation des listes de données
data_t = [0]
data_x = [5]
data_vx = [0]

# Définir les fonctions
def f1(vx):
    return vx

def f2(x, vx):
    return (-k * x - ga * vx) / m  

# Méthode d'Euler pour l'analyse numérique
for i in range(N):
    data_vx_n = data_vx[i] + h * f2(data_x[i], data_vx[i])
    data_x_n = data_x[i] + h * f1(data_vx[i])
    data_t_n = data_t[i] + h

    data_vx.append(data_vx_n)
    data_x.append(data_x_n)
    data_t.append(data_t_n)

#D
# Tracer les résultats de l'analyse numérique et l'équation donnée sur le même graphique
plt.figure(figsize=(10, 6))
plt.plot(t, x_, label='Équation donnée', color='green')  # Tracer l'équation donnée
plt.plot(data_t, data_x, label='Analyse numérique', color='blue')  # Tracer l'analyse numérique
plt.xlabel('Temps (s)')
plt.ylabel('Position (m)')
plt.title('Comparaison entre l\'analyse numérique et l\'équation donnée pour N = 500')
plt.grid(True)
plt.legend()
plt.xlim(0 ,3*2*np.pi/mu)
plt.show()

# E
# Valeurs des constantes
mu = 4.444
a = 5
al = 0.5
b = 0.563

# Définir l'équation
def x_(t):
    return np.exp(-al*t)*(a*np.cos(mu*t)+b*np.sin(mu*t))

def euler(N):
    T = 2*np.pi/mu
    h = 3*T / N
    data_t = [0]
    data_x = [5]
    data_vx = [0]

    for i in range(N):
        data_vx_n = data_vx[i] + h * f2(data_x[i], data_vx[i])
        data_x_n = data_x[i] + h * f1(data_vx[i])
        data_t_n = data_t[i] + h
        data_vx.append(data_vx_n)
        data_x.append(data_x_n)
        data_t.append(data_t_n)
    
    # Calculer la différence entre les deux courbes à 3T
    t_3T = 3 * T
    index_3T = int(t_3T / (3*T/N))
    diff = abs(data_x[index_3T] - x_(t_3T))
    tolerance = 0.05 * x_(t_3T)
    return diff <= tolerance

# Recherche de la valeur de N
N = 500
while not euler(N):
    N += 100
print("La valeur requise de N pour que la différence soit de moins de 5 % à 3T est :", N)
