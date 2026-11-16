
"""
Étape 16 — Comparaison de modèles de détection des défauts
Projet : Jumeau numérique d'un moteur électrique

Modèles comparés :
1. Random Forest
2. SVM avec noyau RBF
3. Régression logistique

Les ensembles d'entraînement et de test sont ceux de l'étape 14.
Les fichiers des étapes précédentes ne sont pas modifiés.
Les résultats concernent exclusivement des données synthétiques.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

DOSSIER_PROJET = Path(__file__).resolve().parent

DOSSIER_DONNEES = (
    DOSSIER_PROJET / "resultats_step14_experimental"
)

DOSSIER_SORTIE = (
    DOSSIER_PROJET / "resultats_step16_experimental"
)

FICHIER_TRAIN = DOSSIER_DONNEES / "dataset_train.csv"
FICHIER_TEST = DOSSIER_DONNEES / "dataset_test.csv"

DOSSIER_SORTIE.mkdir(parents=True, exist_ok=True)

VARIABLES = [
    "rms_m_s2",
    "ecart_type_m_s2",
    "facteur_crete",
]

CIBLE = "target_bearing_fault"

GRAINE = 42


# ============================================================
# 2. CHARGEMENT ET VALIDATION
# ============================================================

print("=" * 70)
print("ÉTAPE 16 — COMPARAISON DES ALGORITHMES")
print("=" * 70)

for chemin in [FICHIER_TRAIN, FICHIER_TEST]:
    if not chemin.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {chemin}"
        )

train = pd.read_csv(FICHIER_TRAIN)
test = pd.read_csv(FICHIER_TEST)

colonnes_requises = VARIABLES + [CIBLE, "scenario_id"]

for nom, donnees in [
    ("entraînement", train),
    ("test", test),
]:
    manquantes = [
        col for col in colonnes_requises
        if col not in donnees.columns
    ]

    if manquantes:
        raise ValueError(
            f"Colonnes manquantes dans {nom} : {manquantes}"
        )

    if donnees[colonnes_requises].isna().any().any():
        raise ValueError(
            f"Valeurs manquantes dans les données de {nom}."
        )

    if set(donnees[CIBLE].unique()) != {0, 1}:
        raise ValueError(
            f"Les deux classes 0 et 1 doivent être présentes dans {nom}."
        )

ids_train = set(train["scenario_id"].unique())
ids_test = set(test["scenario_id"].unique())

chevauchement = ids_train.intersection(ids_test)

if chevauchement:
    raise ValueError(
        f"Chevauchement des scénarios : {sorted(chevauchement)}"
    )

X_train = train[VARIABLES].astype(float)
y_train = train[CIBLE].astype(int)

X_test = test[VARIABLES].astype(float)
y_test = test[CIBLE].astype(int)

print(f"\nFenêtres d'entraînement : {len(train)}")
print(f"Fenêtres de test         : {len(test)}")
print(f"Scénarios train          : {len(ids_train)}")
print(f"Scénarios test           : {len(ids_test)}")
print(f"Chevauchement            : {len(chevauchement)}")


# ============================================================
# 3. DÉFINITION DES MODÈLES
# ============================================================

modeles = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=GRAINE,
        n_jobs=-1,
    ),

    "SVM RBF": Pipeline([
        ("normalisation", StandardScaler()),
        ("classificateur", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=GRAINE,
        )),
    ]),

    "Régression logistique": Pipeline([
        ("normalisation", StandardScaler()),
        ("classificateur", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=GRAINE,
        )),
    ]),
}


# ============================================================
# 4. ENTRAÎNEMENT ET ÉVALUATION
# ============================================================

resultats = []
rapports = []

for nom, modele in modeles.items():

    print("\n" + "-" * 70)
    print(f"MODÈLE : {nom}")
    print("-" * 70)

    modele.fit(X_train, y_train)

    predictions = modele.predict(X_test)

    probabilites = modele.predict_proba(X_test)[:, 1]

    matrice = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = matrice.ravel()

    metriques = {
        "modele": nom,
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
        "vrais_negatifs": int(tn),
        "fausses_alertes": int(fp),
        "defauts_manques": int(fn),
        "vrais_positifs": int(tp),
    }

    resultats.append(metriques)

    rapport = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Sans défaut", "Défaut de roulement"],
        output_dict=True,
        zero_division=0,
    )

    for classe, valeurs in rapport.items():
        if isinstance(valeurs, dict):
            rapports.append({
                "modele": nom,
                "classe": classe,
                **valeurs,
            })

    print(f"Accuracy              : {metriques['accuracy']:.4f}")
    print(
        "Balanced accuracy     : "
        f"{metriques['balanced_accuracy']:.4f}"
    )
    print(
        "Précision défaut      : "
        f"{metriques['precision_defaut']:.4f}"
    )
    print(
        "Rappel défaut         : "
        f"{metriques['recall_defaut']:.4f}"
    )
    print(f"F1-score défaut        : {metriques['f1_defaut']:.4f}")
    print(f"ROC-AUC                : {metriques['roc_auc']:.4f}")
    print(f"Fausses alertes        : {fp}")
    print(f"Défauts manqués        : {fn}")

    print("\nMatrice de confusion :")
    print(matrice)

    # Sauvegarde de chaque modèle dans un fichier distinct.
    nom_fichier = (
        nom.lower()
        .replace(" ", "_")
        .replace("é", "e")
        .replace("è", "e")
    )

    chemin_modele = (
        DOSSIER_SORTIE / f"modele_{nom_fichier}.joblib"
    )

    joblib.dump(modele, chemin_modele)

    print(f"Modèle sauvegardé : {chemin_modele}")


# ============================================================
# 5. EXPORT DES TABLEAUX
# ============================================================

tableau = pd.DataFrame(resultats)

tableau = tableau.sort_values(
    by="f1_defaut",
    ascending=False,
).reset_index(drop=True)

chemin_metriques = DOSSIER_SORTIE / "comparaison_metriques.csv"

tableau.to_csv(
    chemin_metriques,
    index=False,
    encoding="utf-8-sig",
)

chemin_rapports = (
    DOSSIER_SORTIE / "rapports_classification.csv"
)

pd.DataFrame(rapports).to_csv(
    chemin_rapports,
    index=False,
    encoding="utf-8-sig",
)

print("\n" + "=" * 70)
print("RÉCAPITULATIF — CLASSEMENT PAR F1-SCORE")
print("=" * 70)

colonnes_affichage = [
    "modele",
    "accuracy",
    "balanced_accuracy",
    "precision_defaut",
    "recall_defaut",
    "f1_defaut",
    "roc_auc",
    "fausses_alertes",
    "defauts_manques",
]

print(
    tableau[colonnes_affichage].to_string(
        index=False,
        formatters={
            col: "{:.4f}".format
            for col in [
                "accuracy",
                "balanced_accuracy",
                "precision_defaut",
                "recall_defaut",
                "f1_defaut",
                "roc_auc",
            ]
        },
    )
)

print(f"\nMétriques : {chemin_metriques}")
print(f"Rapports  : {chemin_rapports}")

print(
    "\nAttention : la comparaison porte sur des données synthétiques. "
    "Les résultats ne constituent pas une validation industrielle."
)
