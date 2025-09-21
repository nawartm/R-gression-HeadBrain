# 📊 Projet de Régression : Prédire le Poids du Cerveau & les Ventes Publicitaires

> 🧠 *Estimer le poids du cerveau à partir du volume du crâne*  
> 📺 *Prédire les ventes en fonction des budgets publicitaires (TV, radio, journal)*

Ce notebook Jupyter (`Régression.ipynb`) présente **deux études de cas complètes de régression en Machine Learning** :
1. **Régression simple** : prédire le poids du cerveau humain à partir du volume du crâne.
2. **Régression multiple** : prédire les ventes d’un produit en fonction des dépenses publicitaires sur différents médias.
3. **Comparaison de modèles** : régression linéaire, arbres de décision, forêts aléatoires.

---

## 🎯 Objectifs du Projet

### 1. 🧠 Régression Simple — HeadBrain Dataset
> *Peut-on estimer le poids du cerveau d’une personne simplement en mesurant le volume de son crâne ?*

- ✅ Nettoyer les données (valeurs manquantes, doublons, aberrations).
- ✅ Visualiser la relation entre le volume du crâne et le poids du cerveau.
- ✅ Construire un modèle de régression linéaire simple.
- ✅ Évaluer la performance du modèle avec des métriques clés (MAE, MSE, R²…).
- ✅ Comparer les résultats avec/sans données aberrantes.

### 2. 📺 Régression Multiple — Advertising Dataset
> *Quel impact ont les budgets publicitaires (TV, radio, journal) sur les ventes ?*

- ✅ Explorer les corrélations entre les variables.
- ✅ Construire deux modèles :
  - Modèle complet : `Ventes = f(TV, radio, journal)`
  - Modèle réduit : `Ventes = f(TV, radio)`
- ✅ Comparer les performances pour déterminer si la variable "journal" apporte de la valeur.
- ✅ Tester des modèles plus avancés : **Arbres de décision** et **Forêts aléatoires**.

---

## 📚 Pour qui est ce projet ?

| Public | Ce qu’il y trouvera |
|--------|----------------------|
| 👩‍🎓 **Étudiants en data / stats / ML** | Un tutoriel complet, étape par étape, avec du code exécutable, des graphiques et des explications claires. Parfait pour apprendre ou réviser. |
| 👨‍🏫 **Enseignants / Formateurs** | Un support pédagogique prêt à l’emploi pour illustrer la régression linéaire, le nettoyage de données, l’évaluation de modèles. |
| 👩‍💻 **Data Scientists juniors** | Un exemple concret de pipeline de modélisation : de l’exploration à la comparaison de modèles. |
| 👔 **Non-techniciens (managers, curieux)** | Des explications simples, des visualisations parlantes, et des résultats concrets pour comprendre comment le Machine Learning peut répondre à des questions business ou scientifiques. |

---

## ⚙️ Étapes Techniques Réalisées

### 🔍 Exploration & Nettoyage des Données
- Lecture avec `pandas`
- Détection des doublons et valeurs manquantes
- Visualisation des outliers avec `boxplots` (méthode IQR)
- Suppression des valeurs aberrantes pour améliorer la qualité du modèle

### 📈 Analyse Visuelle & Statistique
- Nuages de points (`scatterplot`) pour observer les relations
- Matrices de corrélation avec `seaborn.heatmap`
- Pairplots pour visualiser les distributions et relations entre toutes les variables

### 🤖 Modélisation
#### Régression Linéaire Simple & Multiple
- Utilisation de `sklearn.linear_model.LinearRegression`
- Division des données : `train_test_split`
- Calcul des coefficients (a, b) et interprétation
- Prédiction sur l’ensemble de test

#### Modèles Avancés
- **Arbre de décision** : `DecisionTreeRegressor`
- **Forêt aléatoire** : `RandomForestRegressor`

### 📊 Évaluation des Modèles
Métriques calculées pour chaque modèle :
- **MAE** (Mean Absolute Error) → Erreur moyenne absolue
- **MSE** (Mean Squared Error) → Pénalise les grandes erreurs
- **RMSE** (Root Mean Squared Error) → Interprétable dans l’unité de la cible
- **R²** (Coefficient de détermination) → % de variance expliquée (0 à 1, 1 = parfait)
- **R** (Racine de R²) → Corrélation prédictive

---

## 🧩 Technologies & Bibliothèques Utilisées

```python
import pandas as pd        # Manipulation des données
import numpy as np         # Calculs numériques
import seaborn as sns      # Visualisations statistiques
import matplotlib.pyplot as plt  # Graphiques

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
