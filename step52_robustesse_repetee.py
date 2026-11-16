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

# Définir les chemins du projet
DOSSIER_PROJET = Path(__file__).resolve().parent
DOSSIER_RESULTATS = DOSSIER_PROJET / "resultats_step14_experimental"
CHEMIN_MODELE = DOSSIER_RESULTATS / "modele_detection_roulement_experimental.joblib"
CHEMIN_TEST = DOSSIER_RESULTATS / "dataset_test.csv"
DOSSIER_SORTIE = DOSSIER_PROJET / "resultats_step52_experimental"
DOSSIER_SORTIE.mkdir(parents=True, exist_ok=True)

# Définir les paramètres de l'expérience
VARIABLES = ["rms_m_s2", "ecart_type_m_s2", "facteur_crete"]
CIBLE = "target_bearing_fault"
NIVEAUX_BRUIT = [0.0, 0.05, 0.10, 0.20, 0.30]
N_REPETITIONS = 30
GRAINE_INITIALE = 2026

# Vérifier la présence des fichiers
for chemin in [CHEMIN_MODELE, CHEMIN_TEST]:
    if not chemin.exists():
        raise FileNotFoundError(f"Fichier introuvable : {chemin}")

# Charger le modèle et les données
modele = joblib.load(CHEMIN_MODELE)
donnees = pd.read_csv(CHEMIN_TEST)

colonnes = VARIABLES + [CIBLE]
manquantes = [c for c in colonnes if c not in donnees.columns]
if manquantes:
    raise ValueError(f"Colonnes manquantes : {manquantes}")

if donnees[colonnes].isna().any().any():
    raise ValueError("Des valeurs manquantes ont été détectées.")

X_test = donnees[VARIABLES].astype(float)
y_test = donnees[CIBLE].astype(int)

if set(y_test.unique()) != {0, 1}:
    raise ValueError("Le jeu de test doit contenir les deux classes 0 et 1.")

if (X_test < 0).any().any():
    raise ValueError("Les caractéristiques initiales doivent être non négatives.")

# Utiliser une échelle de bruit fixe calculée sur le test original
ecarts_types = X_test.std(ddof=0).to_numpy()
if np.any(ecarts_types <= 0):
    raise ValueError("Une caractéristique possède un écart-type nul.")

print("=" * 76)
print("ETAPE 52 — ROBUSTESSE REPETEE AU BRUIT")
print("=" * 76)
print(f"Fenêtres de test : {len(X_test)}")
print(f"Répétitions par niveau : {N_REPETITIONS}")
print(f"Niveaux de bruit : {NIVEAUX_BRUIT}")

resultats = []

# Répéter les perturbations indépendamment pour chaque niveau
for niveau in NIVEAUX_BRUIT:
    for repetition in range(N_REPETITIONS):
        generateur = np.random.default_rng(
            GRAINE_INITIALE + repetition + int(niveau * 10000)
        )

        X_perturbe = X_test.copy()

        if niveau > 0:
            bruit = generateur.normal(
                loc=0.0,
                scale=niveau * ecarts_types,
                size=X_test.shape,
            )
            valeurs = X_test.to_numpy() + bruit
            X_perturbe.loc[:, VARIABLES] = valeurs
            X_perturbe = X_perturbe.clip(lower=0)

        predictions = modele.predict(X_perturbe)
        probabilites = modele.predict_proba(X_perturbe)[:, 1]

        resultats.append({
            "niveau_bruit": niveau,
            "repetition": repetition + 1,
            "accuracy": accuracy_score(y_test, predictions),
            "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
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
        })

# Exporter les résultats détaillés et leur synthèse
detail = pd.DataFrame(resultats)
detail.to_csv(
    DOSSIER_SORTIE / "robustesse_repetitions_detaillees.csv",
    index=False,
    encoding="utf-8-sig",
)

metriques = [
    "accuracy",
    "balanced_accuracy",
    "precision_defaut",
    "recall_defaut",
    "f1_defaut",
    "roc_auc",
    "fausses_alertes",
    "defauts_manques",
]

resume = detail.groupby("niveau_bruit")[metriques].agg(["mean", "std"])
resume.columns = [f"{metrique}_{statistique}" for metrique, statistique in resume.columns]
resume = resume.reset_index()
resume.to_csv(
    DOSSIER_SORTIE / "robustesse_repetitions_resume.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\nRésumé : moyenne et écart-type sur les répétitions")
print(resume.round(4).to_string(index=False))
print(f"\nRésultats enregistrés dans : {DOSSIER_SORTIE.resolve()}")
print("Les données utilisées restent entièrement synthétiques.")
