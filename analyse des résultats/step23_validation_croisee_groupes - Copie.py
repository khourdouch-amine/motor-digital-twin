
"""
Étape 23 — Validation croisée groupée par scénario
Projet : Jumeau numérique d'un moteur électrique

La validation utilise uniquement le jeu d'entraînement de l'étape 14.
Les fenêtres d'un même scénario restent dans le même pli.
Le jeu de test final n'est ni utilisé ni modifié.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

DOSSIER_PROJET = Path(__file__).resolve().parent
DOSSIER_DONNEES = DOSSIER_PROJET / "resultats_step14_experimental"
DOSSIER_SORTIE = DOSSIER_PROJET / "resultats_step23_experimental"

FICHIER_TRAIN = DOSSIER_DONNEES / "dataset_train.csv"

VARIABLES = [
    "rms_m_s2",
    "ecart_type_m_s2",
    "facteur_crete",
]

CIBLE = "target_bearing_fault"
GROUPE = "scenario_id"
GRAINE = 42
N_PLIS = 4


# ============================================================
# 2. CHARGEMENT ET VALIDATION
# ============================================================

print("=" * 72)
print("ÉTAPE 23 — VALIDATION CROISÉE GROUPÉE PAR SCÉNARIO")
print("=" * 72)

if not FICHIER_TRAIN.exists():
    raise FileNotFoundError(
        f"Fichier d'entraînement introuvable : {FICHIER_TRAIN}"
    )

train = pd.read_csv(FICHIER_TRAIN)

colonnes_requises = VARIABLES + [CIBLE, GROUPE, "scenario_type"]
manquantes = [
    colonne for colonne in colonnes_requises
    if colonne not in train.columns
]

if manquantes:
    raise ValueError(f"Colonnes manquantes : {manquantes}")

if train[colonnes_requises].isna().any().any():
    raise ValueError("Des valeurs manquantes ont été détectées.")

if set(train[CIBLE].unique()) != {0, 1}:
    raise ValueError("Les deux classes 0 et 1 doivent être présentes.")

X = train[VARIABLES].astype(float)
y = train[CIBLE].astype(int)
groupes = train[GROUPE]

nombre_scenarios = groupes.nunique()

if nombre_scenarios < N_PLIS:
    raise ValueError(
        f"Il faut au moins {N_PLIS} scénarios distincts."
    )

print(f"\nNombre de fenêtres : {len(train)}")
print(f"Nombre de scénarios : {nombre_scenarios}")
print(f"Nombre de plis : {N_PLIS}")
print("\nScénarios et types :")

print(
    train.groupby([GROUPE, "scenario_type"])
    .size()
    .to_string()
)


# ============================================================
# 3. DÉFINITION DES MODÈLES
# ============================================================

def creer_modeles():
    """Créer des modèles neufs pour chaque pli."""

    return {
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
# 4. VALIDATION CROISÉE GROUPÉE
# ============================================================

validation = GroupKFold(n_splits=N_PLIS)

resultats = []
repartition_plis = []

for numero_pli, (indices_train, indices_val) in enumerate(
    validation.split(X, y, groups=groupes),
    start=1,
):
    X_train = X.iloc[indices_train]
    y_train = y.iloc[indices_train]

    X_val = X.iloc[indices_val]
    y_val = y.iloc[indices_val]

    groupes_train = set(groupes.iloc[indices_train].unique())
    groupes_val = set(groupes.iloc[indices_val].unique())

    chevauchement = groupes_train & groupes_val

    if chevauchement:
        raise RuntimeError(
            f"Fuite de groupes au pli {numero_pli} : {chevauchement}"
        )

    if y_train.nunique() < 2 or y_val.nunique() < 2:
        raise ValueError(
            f"Les deux classes doivent être présentes dans "
            f"l'entraînement et la validation du pli {numero_pli}. "
            "Il faut revoir le nombre ou la répartition des plis."
        )

    print("\n" + "-" * 72)
    print(f"PLI {numero_pli}/{N_PLIS}")
    print("-" * 72)
    print("Scénarios d'entraînement :", sorted(groupes_train))
    print("Scénarios de validation :", sorted(groupes_val))
    print("Fenêtres d'entraînement :", len(indices_train))
    print("Fenêtres de validation :", len(indices_val))

    repartition_plis.append({
        "pli": numero_pli,
        "scenarios_train": ",".join(map(str, sorted(groupes_train))),
        "scenarios_validation": ",".join(map(str, sorted(groupes_val))),
        "fenetres_train": len(indices_train),
        "fenetres_validation": len(indices_val),
        "classes_train": y_train.value_counts().to_dict().__str__(),
        "classes_validation": y_val.value_counts().to_dict().__str__(),
    })

    for nom_modele, modele in creer_modeles().items():

        modele.fit(X_train, y_train)

        predictions = modele.predict(X_val)
        probabilites = modele.predict_proba(X_val)[:, 1]

        matrice = confusion_matrix(
            y_val,
            predictions,
            labels=[0, 1],
        )

        tn, fp, fn, tp = matrice.ravel()

        resultat = {
            "pli": numero_pli,
            "modele": nom_modele,
            "accuracy": accuracy_score(y_val, predictions),
            "balanced_accuracy": balanced_accuracy_score(
                y_val, predictions
            ),
            "precision_defaut": precision_score(
                y_val, predictions, zero_division=0
            ),
            "recall_defaut": recall_score(
                y_val, predictions, zero_division=0
            ),
            "f1_defaut": f1_score(
                y_val, predictions, zero_division=0
            ),
            "roc_auc": roc_auc_score(y_val, probabilites),
            "vrais_negatifs": int(tn),
            "fausses_alertes": int(fp),
            "defauts_manques": int(fn),
            "vrais_positifs": int(tp),
            "fenetres_validation": len(y_val),
            "scenarios_validation": ",".join(
                map(str, sorted(groupes_val))
            ),
        }

        resultats.append(resultat)

        print(f"\nModèle : {nom_modele}")
        print(f"Accuracy : {resultat['accuracy']:.4f}")
        print(
            "Balanced accuracy : "
            f"{resultat['balanced_accuracy']:.4f}"
        )
        print(f"Précision défaut : {resultat['precision_defaut']:.4f}")
        print(f"Rappel défaut : {resultat['recall_defaut']:.4f}")
        print(f"F1 défaut : {resultat['f1_defaut']:.4f}")
        print(f"ROC-AUC : {resultat['roc_auc']:.4f}")
        print(f"Fausses alertes : {fp}")
        print(f"Défauts manqués : {fn}")


# ============================================================
# 5. EXPORT DES RÉSULTATS
# ============================================================

DOSSIER_SORTIE.mkdir(parents=True, exist_ok=True)

df_resultats = pd.DataFrame(resultats)
df_plis = pd.DataFrame(repartition_plis)

df_resultats.to_csv(
    DOSSIER_SORTIE / "metriques_par_pli.csv",
    index=False,
    encoding="utf-8-sig",
)

df_plis.to_csv(
    DOSSIER_SORTIE / "repartition_scenarios_par_pli.csv",
    index=False,
    encoding="utf-8-sig",
)

colonnes_metriques = [
    "accuracy",
    "balanced_accuracy",
    "precision_defaut",
    "recall_defaut",
    "f1_defaut",
    "roc_auc",
    "fausses_alertes",
    "defauts_manques",
]

resume = (
    df_resultats
    .groupby("modele")[colonnes_metriques]
    .agg(["mean", "std"])
)

resume.columns = [
    f"{metrique}_{statistique}"
    for metrique, statistique in resume.columns
]

resume = resume.reset_index()

resume.to_csv(
    DOSSIER_SORTIE / "resume_validation_croisee.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\n" + "=" * 72)
print("RÉSUMÉ — MOYENNE ET ÉCART-TYPE ENTRE LES PLIS")
print("=" * 72)

colonnes_resume = [
    "modele",
    "f1_defaut_mean",
    "f1_defaut_std",
    "recall_defaut_mean",
    "recall_defaut_std",
    "fausses_alertes_mean",
    "defauts_manques_mean",
]

print(
    resume[colonnes_resume].to_string(
        index=False,
        formatters={
            colonne: "{:.4f}".format
            for colonne in colonnes_resume
            if colonne != "modele"
        },
    )
)

print("\nFichiers enregistrés :")
for chemin in [
    DOSSIER_SORTIE / "metriques_par_pli.csv",
    DOSSIER_SORTIE / "repartition_scenarios_par_pli.csv",
    DOSSIER_SORTIE / "resume_validation_croisee.csv",
]:
    print("-", chemin.resolve())

print(
    "\nAttention : les résultats concernent des données synthétiques. "
    "La validation groupée évalue la généralisation entre scénarios "
    "simulés, pas la fiabilité industrielle."
)
