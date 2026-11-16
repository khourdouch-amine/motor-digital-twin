
"""
Étape 19 — Diagnostic des probabilités du classifieur
Projet : Jumeau numérique d'un moteur électrique

Ce script vérifie la séparation des probabilités sur la validation.
Il ne modifie aucun modèle ni dataset existant.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix, roc_auc_score


# ============================================================
# 1. PARAMÈTRES
# ============================================================

DOSSIER = Path(__file__).resolve().parent

FICHIER_DATASET = (
    DOSSIER
    / "resultats_step14_experimental"
    / "dataset_train.csv"
)

SORTIE = (
    DOSSIER
    / "resultats_step19_experimental"
)

SORTIE.mkdir(parents=True, exist_ok=True)

CARACTERISTIQUES = [
    "rms_m_s2",
    "ecart_type_m_s2",
    "facteur_crete",
]

CIBLE = "target_bearing_fault"
GRAINE = 42


# ============================================================
# 2. CHARGEMENT
# ============================================================

print("=" * 70)
print("ÉTAPE 19 — DIAGNOSTIC DES PROBABILITÉS")
print("=" * 70)

if not FICHIER_DATASET.exists():
    raise FileNotFoundError(
        f"Dataset introuvable : {FICHIER_DATASET}"
    )

df = pd.read_csv(FICHIER_DATASET)

colonnes = CARACTERISTIQUES + [CIBLE, "scenario_id"]

if not set(colonnes).issubset(df.columns):
    raise ValueError(
        "Une ou plusieurs colonnes nécessaires sont absentes."
    )

if df[colonnes].isna().any().any():
    raise ValueError(
        "Des valeurs manquantes sont présentes."
    )

X = df[CARACTERISTIQUES]
y = df[CIBLE].astype(int)
groupes = df["scenario_id"]


# ============================================================
# 3. REPRODUCTION DE LA SÉPARATION DE VALIDATION
# ============================================================

separateur = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=GRAINE,
)

idx_train, idx_validation = next(
    separateur.split(X, y, groups=groupes)
)

X_train = X.iloc[idx_train]
y_train = y.iloc[idx_train]

X_validation = X.iloc[idx_validation]
y_validation = y.iloc[idx_validation]

groupes_train = set(
    groupes.iloc[idx_train].unique()
)

groupes_validation = set(
    groupes.iloc[idx_validation].unique()
)

if groupes_train.intersection(groupes_validation):
    raise RuntimeError(
        "Chevauchement des scénarios entre les deux ensembles."
    )

print("\nRépartition des classes en validation :")
print(y_validation.value_counts().sort_index())

print("\nScénarios de validation :")
print(
    sorted(
        groupes.iloc[idx_validation].unique().tolist()
    )
)


# ============================================================
# 4. ENTRAÎNEMENT ET PROBABILITÉS
# ============================================================

modele = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=GRAINE,
    n_jobs=-1,
)

modele.fit(X_train, y_train)

probabilites = modele.predict_proba(X_validation)[:, 1]

diagnostic = pd.DataFrame({
    "scenario_id": groupes.iloc[idx_validation].to_numpy(),
    "cible_reelle": y_validation.to_numpy(),
    "probabilite_defaut": probabilites,
})

diagnostic["prediction_050"] = (
    diagnostic["probabilite_defaut"] >= 0.50
).astype(int)

diagnostic.to_csv(
    SORTIE / "probabilites_validation.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 5. DISTRIBUTION PAR CLASSE
# ============================================================

print("\n" + "=" * 70)
print("PROBABILITÉS PAR CLASSE RÉELLE")
print("=" * 70)

for classe in [0, 1]:
    valeurs = probabilites[
        y_validation.to_numpy() == classe
    ]

    print(f"\nClasse réelle {classe} :")

    if len(valeurs) == 0:
        print("Aucune observation.")
        continue

    print(f"Nombre : {len(valeurs)}")
    print(f"Minimum : {valeurs.min():.6f}")
    print(f"1er percentile : {np.quantile(valeurs, 0.01):.6f}")
    print(f"Médiane : {np.median(valeurs):.6f}")
    print(f"99e percentile : {np.quantile(valeurs, 0.99):.6f}")
    print(f"Maximum : {valeurs.max():.6f}")


# ============================================================
# 6. SEUILS ET MATRICES DE CONFUSION
# ============================================================

print("\n" + "=" * 70)
print("VÉRIFICATION DES SEUILS")
print("=" * 70)

for seuil in [0.16, 0.30, 0.50, 0.70, 0.90, 0.94]:
    predictions = (
        probabilites >= seuil
    ).astype(int)

    matrice = confusion_matrix(
        y_validation,
        predictions,
        labels=[0, 1],
    )

    print(f"\nSeuil : {seuil:.2f}")
    print(matrice)

print(
    "\nROC-AUC :",
    f"{roc_auc_score(y_validation, probabilites):.6f}",
)

print(
    "\nFichier détaillé :",
    SORTIE / "probabilites_validation.csv",
)

print(
    "\nAttention : diagnostic fondé sur des données synthétiques."
)
