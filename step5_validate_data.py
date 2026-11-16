
# ============================================================
# ETAPE 5 : VALIDATION DES DONNEES DU JUMEAU NUMERIQUE
# ============================================================

# Importer pandas pour lire et analyser le fichier CSV.
import pandas as pd

# Importer numpy pour rechercher les valeurs infinies.
import numpy as np

# Importer matplotlib pour tracer les graphiques.
import matplotlib.pyplot as plt

# Importer Path pour gérer le chemin du fichier.
from pathlib import Path


# ------------------------------------------------------------
# 1. LOCALISER LE FICHIER CSV
# ------------------------------------------------------------

# Chercher le CSV dans le dossier où se trouve ce script.
dossier_projet = Path(__file__).resolve().parent

# Construire le chemin complet du fichier généré à l'étape 4.
chemin_csv = dossier_projet / "motor_sensor_data.csv"

# Arrêter le programme si le fichier est absent.
if not chemin_csv.exists():
    raise FileNotFoundError(
        f"Fichier introuvable : {chemin_csv}\n"
        "Exécute d'abord le script de l'étape 4."
    )

# Lire les données du moteur depuis le fichier CSV.
data = pd.read_csv(chemin_csv)


# ------------------------------------------------------------
# 2. VERIFIER LA STRUCTURE DES DONNEES
# ------------------------------------------------------------

# Afficher le nombre de lignes et de colonnes.
print("\n========== STRUCTURE DU FICHIER ==========")
print(f"Nombre de lignes : {data.shape[0]}")
print(f"Nombre de colonnes : {data.shape[1]}")

# Afficher les noms de toutes les colonnes.
print("\nColonnes disponibles :")
print(data.columns.tolist())

# Définir les colonnes essentielles attendues.
colonnes_requises = [
    "time_s",
    "speed_sensor_rpm",
    "temperature_sensor_c",
    "current_sensor_a",
    "vibration_sensor_m_s2",
    "overload_active",
    "bearing_fault_active"
]

# Rechercher les colonnes qui manquent.
colonnes_absentes = [
    colonne
    for colonne in colonnes_requises
    if colonne not in data.columns
]

# Arrêter le programme si une colonne essentielle manque.
if colonnes_absentes:
    raise ValueError(
        f"Colonnes manquantes : {colonnes_absentes}"
    )

# Afficher les premières lignes pour contrôler le contenu.
print("\nPremières lignes du fichier :")
print(data.head())


# ------------------------------------------------------------
# 3. RECHERCHER LES DONNEES MANQUANTES
# ------------------------------------------------------------

# Compter les cellules vides pour chaque colonne.
valeurs_manquantes = data.isna().sum()

# Calculer le nombre total de cellules vides.
total_manquant = int(valeurs_manquantes.sum())

# Afficher le résultat du contrôle.
print("\n========== VALEURS MANQUANTES ==========")
print(valeurs_manquantes)
print(f"Total des valeurs manquantes : {total_manquant}")


# ------------------------------------------------------------
# 4. RECHERCHER LES VALEURS INFINIES
# ------------------------------------------------------------

# Sélectionner uniquement les colonnes numériques.
donnees_numeriques = data.select_dtypes(include=np.number)

# Compter les valeurs infinies dans les données numériques.
total_infini = int(
    np.isinf(donnees_numeriques.to_numpy()).sum()
)

# Afficher le résultat du contrôle.
print("\n========== VALEURS INFINIES ==========")
print(f"Nombre de valeurs infinies : {total_infini}")


# ------------------------------------------------------------
# 5. VERIFIER LE TEMPS ET LES CAPTEURS
# ------------------------------------------------------------

# Vérifier que le temps augmente strictement à chaque ligne.
temps_croissant = data["time_s"].is_monotonic_increasing

# Vérifier qu'il n'y a pas de temps dupliqués.
temps_uniques = data["time_s"].is_unique

# Afficher les résultats liés à l'axe temporel.
print("\n========== CONTROLE TEMPOREL ==========")
print(f"Temps croissant : {temps_croissant}")
print(f"Temps tous uniques : {temps_uniques}")

# Définir les colonnes des capteurs à examiner.
colonnes_capteurs = [
    "speed_sensor_rpm",
    "temperature_sensor_c",
    "current_sensor_a",
    "vibration_sensor_m_s2"
]

# Afficher les statistiques descriptives des capteurs.
print("\n========== STATISTIQUES DES CAPTEURS ==========")
print(data[colonnes_capteurs].describe().round(3))


# ------------------------------------------------------------
# 6. VERIFIER LES DEFAUTS SIMULES
# ------------------------------------------------------------

# Convertir les indicateurs de défaut en booléens.
surcharge = data["overload_active"].astype(bool)

# Convertir l'indicateur du défaut de roulement en booléen.
defaut_roulement = data["bearing_fault_active"].astype(bool)

# Compter le nombre de lignes avec surcharge active.
nombre_surcharge = int(surcharge.sum())

# Compter le nombre de lignes avec défaut de roulement actif.
nombre_roulement = int(defaut_roulement.sum())

# Afficher le nombre d'échantillons associés à chaque défaut.
print("\n========== CONTROLE DES DEFAUTS ==========")
print(f"Echantillons en surcharge : {nombre_surcharge}")
print(f"Echantillons avec défaut de roulement : {nombre_roulement}")

# Vérifier les intervalles de temps associés à la surcharge.
if surcharge.any():
    temps_surcharge = data.loc[surcharge, "time_s"]
    print(
        "Intervalle de surcharge : "
        f"{temps_surcharge.min():.2f} à "
        f"{temps_surcharge.max():.2f} s"
    )
else:
    print("ATTENTION : aucune surcharge détectée.")

# Vérifier le début du défaut de roulement.
if defaut_roulement.any():
    temps_roulement = data.loc[defaut_roulement, "time_s"]
    print(
        "Défaut de roulement à partir de : "
        f"{temps_roulement.min():.2f} s"
    )
else:
    print("ATTENTION : aucun défaut de roulement détecté.")


# ------------------------------------------------------------
# 7. CALCULER LES INDICATEURS DE QUALITE
# ------------------------------------------------------------

# Calculer le pourcentage de valeurs manquantes.
pourcentage_manquant = (
    total_manquant / data.size * 100
    if data.size > 0 else 0
)

# Vérifier si le jeu de données est non vide et complet.
donnees_completes = (
    not data.empty
    and total_manquant == 0
    and total_infini == 0
)

# Afficher un bilan synthétique.
print("\n========== BILAN DE VALIDATION ==========")
print(f"Pourcentage de valeurs manquantes : {pourcentage_manquant:.4f} %")
print(f"Données complètes : {donnees_completes}")

# Signaler les contrôles temporels qui échouent.
if not temps_croissant or not temps_uniques:
    print("ATTENTION : vérifier l'ordre ou les doublons temporels.")

# Rappeler que la présence des défauts doit être vérifiée.
if nombre_surcharge == 0 or nombre_roulement == 0:
    print("ATTENTION : vérifier les étiquettes de défaut.")


# ------------------------------------------------------------
# 8. TRACER LES RESULTATS DANS UNE SEULE FENETRE
# ------------------------------------------------------------

# Créer une fenêtre contenant quatre graphiques superposés.
fig, axes = plt.subplots(
    nrows=4,
    ncols=1,
    figsize=(13, 14),
    sharex=True
)

# Récupérer l'axe temporel en secondes.
temps = data["time_s"]

# Graphique 1 : vitesse mesurée par le capteur.
axes[0].plot(
    temps,
    data["speed_sensor_rpm"],
    label="Vitesse mesurée",
    linewidth=0.8
)
axes[0].set_title("Vitesse du moteur")
axes[0].set_ylabel("Vitesse (tr/min)")
axes[0].grid(True)
axes[0].legend()

# Graphique 2 : température mesurée.
axes[1].plot(
    temps,
    data["temperature_sensor_c"],
    label="Température mesurée",
    linewidth=0.8
)
axes[1].set_title("Température du moteur")
axes[1].set_ylabel("Température (°C)")
axes[1].grid(True)
axes[1].legend()

# Graphique 3 : courant mesuré.
axes[2].plot(
    temps,
    data["current_sensor_a"],
    label="Courant estimé du capteur",
    linewidth=0.8
)
axes[2].set_title("Courant électrique simulé")
axes[2].set_ylabel("Courant (A)")
axes[2].grid(True)
axes[2].legend()

# Graphique 4 : vibration mesurée.
axes[3].plot(
    temps,
    data["vibration_sensor_m_s2"],
    label="Vibration mesurée",
    linewidth=0.8
)

# Marquer le début du défaut de roulement.
axes[3].axvline(
    x=400,
    linestyle="--",
    label="Début du défaut à 400 s"
)

# Ajouter les titres et la grille au dernier graphique.
axes[3].set_title("Vibrations et défaut de roulement")
axes[3].set_xlabel("Temps (s)")
axes[3].set_ylabel("Accélération (m/s²)")
axes[3].grid(True)
axes[3].legend()

# Ajouter un titre général à la fenêtre.
fig.suptitle(
    "Validation des données du jumeau numérique du moteur",
    fontsize=14
)

# Ajuster les espaces pour éviter le chevauchement des textes.
fig.tight_layout()

# Afficher tous les graphiques dans une seule fenêtre.
plt.show()
