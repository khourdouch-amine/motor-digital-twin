# ============================================================
# ETAPE 6 : PREPARATION DES DONNEES POUR L'INTELLIGENCE ARTIFICIELLE
# ============================================================

# Importer pandas pour manipuler le tableau de données.
import pandas as pd

# Importer numpy pour effectuer des calculs numériques.
import numpy as np

# Importer matplotlib pour afficher les graphiques.
import matplotlib.pyplot as plt

# Importer Path pour construire les chemins des fichiers.
from pathlib import Path

# ------------------------------------------------------------
# 1. CHARGER LES DONNEES SIMULEES
# ------------------------------------------------------------

# Identifier le dossier contenant ce script.
dossier_projet = Path(__file__).resolve().parent

# Construire le chemin du fichier CSV généré à l'étape 4.
chemin_csv = dossier_projet / "motor_sensor_data.csv"

# Vérifier que le fichier CSV existe.
if not chemin_csv.exists():
    raise FileNotFoundError(
        f"Fichier introuvable : {chemin_csv}\n"
        "Vérifie que le script de l'étape 4 a été exécuté."
    )

# Charger les données dans un DataFrame pandas.
data = pd.read_csv(chemin_csv)

# Afficher la taille du jeu de données initial.
print("\n========== DONNEES INITIALES ==========")
print(f"Nombre de lignes : {len(data)}")
print(f"Nombre de colonnes : {len(data.columns)}")

# ------------------------------------------------------------
# 2. DEFINIR LES VARIABLES D'ENTREE
# ------------------------------------------------------------

# Choisir les mesures qui seront fournies au modèle d'IA.
features = [
    "speed_sensor_rpm",
    "temperature_sensor_c",
    "current_sensor_a",
    "vibration_sensor_m_s2"
]

# Vérifier que toutes les variables d'entrée existent.
colonnes_absentes = [
    colonne for colonne in features
    if colonne not in data.columns
]

# Arrêter le programme si une variable est absente.
if colonnes_absentes:
    raise ValueError(
        f"Variables d'entrée absentes : {colonnes_absentes}"
    )

# Extraire les mesures qui serviront d'entrées au modèle.
X = data[features].copy()

# ------------------------------------------------------------
# 3. CREER LA CIBLE DE CLASSIFICATION
# ------------------------------------------------------------

# Convertir l'indicateur de surcharge en booléen.
surcharge = data["overload_active"].astype(bool)

# Convertir l'indicateur de défaut de roulement en booléen.
defaut_roulement = data["bearing_fault_active"].astype(bool)

# Créer une cible égale à 1 si au moins un défaut est actif.
y = (surcharge | defaut_roulement).astype(int)

# Ajouter la cible au tableau de travail.
data["fault_label"] = y

# ------------------------------------------------------------
# 4. CONTROLER LA REPARTITION DES CLASSES
# ------------------------------------------------------------

# Compter les exemples de chaque classe.
repartition = y.value_counts().sort_index()

# Calculer le pourcentage de chaque classe.
pourcentages = y.value_counts(normalize=True).sort_index() * 100

# Afficher les effectifs.
print("\n========== REPARTITION DES CLASSES ==========")
print("0 = aucun défaut simulé actif")
print("1 = au moins un défaut simulé actif")
print(repartition.rename(index={0: "Normal", 1: "Anomalie"}))

# Afficher les pourcentages.
print("\nPourcentages par classe :")
print(pourcentages.rename(index={0: "Normal", 1: "Anomalie"}).round(2))

# ------------------------------------------------------------
# 5. VERIFIER LA QUALITE DU JEU D'ENTRAINEMENT
# ------------------------------------------------------------

# Vérifier les valeurs manquantes dans les variables d'entrée.
manquants_X = int(X.isna().sum().sum())

# Vérifier les valeurs infinies dans les variables d'entrée.
infinis_X = int(np.isinf(X.to_numpy()).sum())

# Vérifier les valeurs manquantes dans la cible.
manquants_y = int(y.isna().sum())

# Afficher les résultats.
print("\n========== QUALITE DES DONNEES ==========")
print(f"Valeurs manquantes dans X : {manquants_X}")
print(f"Valeurs infinies dans X : {infinis_X}")
print(f"Valeurs manquantes dans y : {manquants_y}")

# Vérifier que les données peuvent être utilisées.
if manquants_X > 0 or infinis_X > 0 or manquants_y > 0:
    raise ValueError(
        "Le jeu de données contient des valeurs invalides."
    )

# ------------------------------------------------------------
# 6. CONSTRUIRE LE JEU DE DONNEES PREPARE
# ------------------------------------------------------------

# Regrouper les variables d'entrée et la cible dans un tableau.
dataset_prepare = X.copy()

# Ajouter la cible de classification.
dataset_prepare["fault_label"] = y

# Ajouter le temps pour conserver la traçabilité des mesures.
dataset_prepare["time_s"] = data["time_s"]

# Ajouter les deux indicateurs d'origine pour l'analyse ultérieure.
dataset_prepare["overload_active"] = data["overload_active"]

# Ajouter l'indicateur du défaut de roulement.
dataset_prepare["bearing_fault_active"] = data["bearing_fault_active"]

# Définir le chemin du fichier préparé.
chemin_sortie = dossier_projet / "motor_ai_dataset.csv"

# Enregistrer le jeu de données préparé sans index supplémentaire.
dataset_prepare.to_csv(chemin_sortie, index=False)

# Afficher le chemin du fichier généré.
print("\n========== EXPORT ==========")
print(f"Fichier préparé : {chemin_sortie}")

# ------------------------------------------------------------
# 7. AFFICHER UN RESUME DU JEU PREPARE
# ------------------------------------------------------------

# Afficher les premières lignes du jeu préparé.
print("\nPremières lignes du jeu préparé :")
print(dataset_prepare.head())

# Afficher les statistiques des quatre variables d'entrée.
print("\nStatistiques des variables d'entrée :")
print(dataset_prepare[features].describe().round(3))

# ------------------------------------------------------------
# 8. VISUALISER LES CLASSES ET LES CAPTEURS
# ------------------------------------------------------------

# Créer une seule fenêtre avec trois graphiques.
fig, axes = plt.subplots(
    nrows=3,
    ncols=1,
    figsize=(13, 12)
)

# Graphique 1 : nombre d'exemples dans chaque classe.
axes[0].bar(
    ["Normal", "Anomalie"],
    [int((y == 0).sum()), int((y == 1).sum())]
)

# Ajouter le titre et les axes au premier graphique.
axes[0].set_title("Répartition des classes")
axes[0].set_ylabel("Nombre d'échantillons")
axes[0].grid(axis="y")

# Graphique 2 : courant mesuré et périodes de défaut.
axes[1].plot(
    data["time_s"],
    data["current_sensor_a"],
    label="Courant simulé",
    linewidth=0.8
)

# Ajouter une ligne verticale au début de la surcharge.
axes[1].axvline(
    x=200,
    linestyle="--",
    label="Début de la surcharge"
)

# Ajouter une ligne verticale à la fin de la surcharge.
axes[1].axvline(
    x=350,
    linestyle="--",
    label="Fin de la surcharge"
)

# Ajouter le titre, les unités et la grille.
axes[1].set_title("Courant et période de surcharge")
axes[1].set_xlabel("Temps (s)")
axes[1].set_ylabel("Courant (A)")
axes[1].grid(True)
axes[1].legend()

# Graphique 3 : vibrations et apparition du défaut de roulement.
axes[2].plot(
    data["time_s"],
    data["vibration_sensor_m_s2"],
    label="Vibration simulée",
    linewidth=0.8
)

# Marquer le début du défaut de roulement.
axes[2].axvline(
    x=400,
    linestyle="--",
    label="Début du défaut de roulement"
)

# Ajouter les informations au troisième graphique.
axes[2].set_title("Vibrations et défaut de roulement")
axes[2].set_xlabel("Temps (s)")
axes[2].set_ylabel("Accélération (m/s²)")
axes[2].grid(True)
axes[2].legend()

# Ajouter un titre général à la fenêtre.
fig.suptitle(
    "Préparation des données pour la maintenance prédictive",
    fontsize=14
)

# Ajuster les espaces entre les graphiques.
fig.tight_layout()

# Afficher les trois graphiques dans une seule fenêtre.
plt.show()