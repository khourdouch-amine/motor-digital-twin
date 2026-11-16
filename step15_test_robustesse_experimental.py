
"""
Étape 15 — Test de robustesse au bruit
Projet : Jumeau numérique d'un moteur électrique

Le modèle et les données de référence sont conservés.
Le bruit est ajouté uniquement aux variables du jeu de test.
Les données utilisées étant synthétiques, les résultats ne prouvent
pas la robustesse du modèle sur un moteur industriel réel.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# ============================================================
# 1. CHEMINS ET CONFIGURATION
# ============================================================

DOSSIER_PROJET = Path(__file__).resolve().parent

DOSSIER_RESULTATS = (
    DOSSIER_PROJET / "resultats_step14_experimental"
)

CHEMIN_MODELE = (
    DOSSIER_RESULTATS
    / "modele_detection_roulement_experimental.joblib"
)

CHEMIN_TEST = DOSSIER_RESULTATS / "dataset_test.csv"

DOSSIER_SORTIE = (
    DOSSIER_PROJET / "resultats_step15_experimental"
)

DOSSIER_SORTIE.mkdir(parents=True, exist_ok=True)

VARIABLES = [
    "rms_m_s2",
    "ecart_type_m_s2",
    "facteur_crete",
]

CIBLE = "target_bearing_fault"

# Bruit gaussien exprimé en multiples de l'écart-type
# observé pour chaque variable dans le jeu de test.
NIVEAUX_BRUIT = [0.0, 0.05, 0.10, 0.20, 0.30]

GRAINE_ALEATOIRE = 42


# ============================================================
# 2. CHARGEMENT ET VALIDATION
# ============================================================

print("=" * 65)
print("ÉTAPE 15 — TEST DE ROBUSTESSE AU BRUIT")
print("=" * 65)

for chemin in [CHEMIN_MODELE, CHEMIN_TEST]:
    if not chemin.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {chemin}"
        )

modele = joblib.load(CHEMIN_MODELE)
donnees_test = pd.read_csv(CHEMIN_TEST)

colonnes_requises = VARIABLES + [CIBLE]
colonnes_manquantes = [
    colonne
    for colonne in colonnes_requises
    if colonne not in donnees_test.columns
]

if colonnes_manquantes:
    raise ValueError(
        f"Colonnes manquantes : {colonnes_manquantes}"
    )

if donnees_test[colonnes_requises].isna().any().any():
    raise ValueError(
        "Le jeu de test contient des valeurs manquantes."
    )

X_test = donnees_test[VARIABLES].astype(float)
y_test = donnees_test[CIBLE].astype(int)

if not set(y_test.unique()).issubset({0, 1}):
    raise ValueError("La cible doit contenir uniquement 0 et 1.")

if len(y_test.unique()) != 2:
    raise ValueError(
        "Le jeu de test doit contenir les deux classes."
    )

if (X_test < 0).any().any():
    raise ValueError(
        "Une variable contient une valeur négative avant perturbation."
    )

print(f"\nFenêtres de test : {len(X_test)}")
print(f"Variables testées : {VARIABLES}")
print(f"Niveaux de bruit : {NIVEAUX_BRUIT}")


# ============================================================
# 3. TEST À DIFFÉRENTS NIVEAUX DE BRUIT
# ============================================================

generateur = np.random.default_rng(GRAINE_ALEATOIRE)
resultats = []

# Échelle fixe calculée sur le jeu de test original.
# Elle reste identique à tous les niveaux de bruit.
ecarts_types = X_test.std(ddof=0).to_numpy()

if np.any(ecarts_types <= 0):
    raise ValueError(
        "Au moins une variable possède un écart-type nul."
    )

for niveau in NIVEAUX_BRUIT:

    X_perturbe = X_test.copy()

    if niveau > 0:
        bruit = generateur.normal(
            loc=0.0,
            scale=niveau * ecarts_types,
            size=X_perturbe.shape,
        )

        X_perturbe.loc[:, VARIABLES] = (
            X_perturbe.to_numpy() + bruit
        )

        # Les trois indicateurs physiques sont non négatifs.
        X_perturbe = X_perturbe.clip(lower=0)

    predictions = modele.predict(X_perturbe)

    probabilites = modele.predict_proba(X_perturbe)[:, 1]

    metriques = {
        "niveau_bruit": niveau,
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            y_test, predictions
        ),
        "precision_defaut": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall_defaut": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1_defaut": f1_score(
            y_test, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_test, probabilites),
        "fausses_alertes": int(
            ((y_test == 0) & (predictions == 1)).sum()
        ),
        "defauts_manques": int(
            ((y_test == 1) & (predictions == 0)).sum()
        ),
    }

    resultats.append(metriques)

    print(f"\n--- Bruit : {niveau:.0%} de l'écart-type ---")
    for nom, valeur in metriques.items():
        if nom != "niveau_bruit":
            print(f"{nom:24s}: {valeur:.4f}"
                  if isinstance(valeur, (float, np.floating))
                  else f"{nom:24s}: {valeur}")


# ============================================================
# 4. EXPORT DES RÉSULTATS
# ============================================================

tableau_resultats = pd.DataFrame(resultats)

chemin_csv = DOSSIER_SORTIE / "robustesse_bruit_experimental.csv"

tableau_resultats.to_csv(
    chemin_csv,
    index=False,
    encoding="utf-8-sig",
)

print("\n" + "=" * 65)
print("TEST DE ROBUSTESSE TERMINÉ")
print("=" * 65)
print(f"\nRésultats enregistrés dans : {chemin_csv}")
print("\nAttention : le test utilise des données synthétiques.")
print(
    "Chaque niveau de bruit correspond à une perturbation "
    "aléatoire ; les résultats ne représentent pas une "
    "validation industrielle."
)
