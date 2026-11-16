
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
)


# ============================================================
# 1. Configuration
# ============================================================

dossier = Path("resultats_step14")

modele = joblib.load(
    dossier / "modele_detection_roulement.joblib"
)

test = pd.read_csv(
    dossier / "dataset_test.csv"
)

variables = [
    "rms_m_s2",
    "ecart_type_m_s2",
    "facteur_crete",
]

cible = "target_bearing_fault"

X_test = test[variables].copy()
y_test = test[cible].astype(int)

# Niveaux de bruit relatif appliqués aux indicateurs.
niveaux_bruit = [0.00, 0.01, 0.03, 0.05, 0.10]

# Graine fixe pour rendre les résultats reproductibles.
generateur = np.random.default_rng(42)


# ============================================================
# 2. Évaluation avec différents niveaux de bruit
# ============================================================

resultats = []

for niveau in niveaux_bruit:

    X_bruite = X_test.copy()

    # Chaque indicateur reçoit un bruit gaussien indépendant.
    # L'écart-type du bruit est proportionnel à l'écart-type
    # de l'indicateur calculé sur le jeu de test.
    for variable in variables:

        ecart_type = X_test[variable].std()

        bruit = generateur.normal(
            loc=0.0,
            scale=niveau * ecart_type,
            size=len(X_test),
        )

        X_bruite[variable] = X_test[variable].to_numpy() + bruit

    # Les indicateurs RMS et écart-type ne peuvent pas être négatifs.
    X_bruite["rms_m_s2"] = X_bruite["rms_m_s2"].clip(lower=0)
    X_bruite["ecart_type_m_s2"] = (
        X_bruite["ecart_type_m_s2"].clip(lower=0)
    )

    predictions = modele.predict(X_bruite)

    resultats.append({
        "bruit_relatif": niveau,
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
    })


# ============================================================
# 3. Affichage et sauvegarde
# ============================================================

tableau = pd.DataFrame(resultats)

print("=== TEST DE ROBUSTESSE AU BRUIT ===")
print(tableau.round(4).to_string(index=False))

fichier_sortie = dossier / "robustesse_bruit.csv"

tableau.to_csv(fichier_sortie, index=False)

print(f"\nRésultats enregistrés : {fichier_sortie}")

print(
    "\nAttention : le bruit ajouté est artificiel. "
    "Ce test évalue la sensibilité numérique du modèle, "
    "pas sa robustesse à toutes les conditions d'un moteur réel."
)
