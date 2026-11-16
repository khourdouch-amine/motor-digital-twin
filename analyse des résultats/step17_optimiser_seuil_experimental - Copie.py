
"""
Étape 17 — Optimisation du seuil de détection
Projet : Jumeau numérique d'un moteur électrique

Le jeu de test final n'est pas utilisé pour choisir le seuil.
Les scénarios d'entraînement sont divisés en apprentissage et validation.
Les données sont synthétiques : les résultats ne prouvent pas une
performance industrielle réelle.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
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

RACINE = Path(__file__).resolve().parent

DOSSIER_DONNEES = RACINE / "resultats_step14_experimental"
DOSSIER_SORTIE = RACINE / "resultats_step17_experimental"

FICHIER_TRAIN = DOSSIER_DONNEES / "dataset_train.csv"
FICHIER_TEST = DOSSIER_DONNEES / "dataset_test.csv"

DOSSIER_SORTIE.mkdir(parents=True, exist_ok=True)

VARIABLES = [
    "rms_m_s2",
    "ecart_type_m_s2",
    "facteur_crete",
]

CIBLE = "target_bearing_fault"
GROUPE = "scenario_id"

GRAINE = 42
PROPORTION_VALIDATION = 0.25

# Exemple de contrainte opérationnelle :
# viser un rappel d'au moins 98 % sur la validation.
# Ce choix doit être considéré comme un objectif exploratoire,
# pas comme une norme industrielle.
RAPPEL_CIBLE = 0.98

SEUILS = np.round(np.arange(0.05, 0.951, 0.01), 2)


# ============================================================
# 2. CHARGEMENT ET VALIDATION
# ============================================================

print("=" * 70)
print("ÉTAPE 17 — OPTIMISATION DU SEUIL")
print("=" * 70)

for chemin in [FICHIER_TRAIN, FICHIER_TEST]:
    if not chemin.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {chemin}"
        )

train_complet = pd.read_csv(FICHIER_TRAIN)
test_final = pd.read_csv(FICHIER_TEST)

colonnes = VARIABLES + [CIBLE, GROUPE]

for nom, df in [
    ("entraînement", train_complet),
    ("test final", test_final),
]:
    manquantes = [c for c in colonnes if c not in df.columns]

    if manquantes:
        raise ValueError(
            f"Colonnes manquantes dans {nom} : {manquantes}"
        )

    if df[colonnes].isna().any().any():
        raise ValueError(
            f"Valeurs manquantes dans {nom}."
        )

    if set(df[CIBLE].unique()) != {0, 1}:
        raise ValueError(
            f"Les deux classes doivent être présentes dans {nom}."
        )

ids_train_complet = set(train_complet[GROUPE].unique())
ids_test_final = set(test_final[GROUPE].unique())

if ids_train_complet.intersection(ids_test_final):
    raise ValueError(
        "Les scénarios d'entraînement et de test se chevauchent."
    )

# Séparation par groupes : aucune fenêtre d'un scénario ne doit
# apparaître simultanément en apprentissage et en validation.
separateur = GroupShuffleSplit(
    n_splits=1,
    test_size=PROPORTION_VALIDATION,
    random_state=GRAINE,
)

indices_apprentissage, indices_validation = next(
    separateur.split(
        train_complet[VARIABLES],
        train_complet[CIBLE],
        groups=train_complet[GROUPE],
    )
)

train = train_complet.iloc[indices_apprentissage].copy()
validation = train_complet.iloc[indices_validation].copy()

ids_apprentissage = set(train[GROUPE].unique())
ids_validation = set(validation[GROUPE].unique())

if ids_apprentissage.intersection(ids_validation):
    raise ValueError(
        "Chevauchement de scénarios entre apprentissage et validation."
    )

if train[CIBLE].nunique() != 2 or validation[CIBLE].nunique() != 2:
    raise ValueError(
        "Une classe manque dans l'un des sous-ensembles. "
        "Il faut revoir la stratégie de séparation par groupes."
    )

print(f"\nApprentissage : {len(train)} fenêtres")
print(f"Validation    : {len(validation)} fenêtres")
print(f"Test final    : {len(test_final)} fenêtres")
print(f"Scénarios apprentissage : {sorted(ids_apprentissage)}")
print(f"Scénarios validation    : {sorted(ids_validation)}")
print(f"Scénarios test final    : {sorted(ids_test_final)}")


# ============================================================
# 3. ENTRAÎNEMENT SUR LA PARTIE APPRENTISSAGE
# ============================================================

X_train = train[VARIABLES].astype(float)
y_train = train[CIBLE].astype(int)

X_val = validation[VARIABLES].astype(float)
y_val = validation[CIBLE].astype(int)

modele = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=GRAINE,
    n_jobs=-1,
)

modele.fit(X_train, y_train)

probabilites_val = modele.predict_proba(X_val)[:, 1]


# ============================================================
# 4. CHOIX DU SEUIL SUR LA VALIDATION UNIQUEMENT
# ============================================================

resultats_validation = []

for seuil in SEUILS:
    predictions = (probabilites_val >= seuil).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        predictions,
        labels=[0, 1],
    ).ravel()

    rappel = recall_score(
        y_val, predictions, zero_division=0
    )

    precision = precision_score(
        y_val, predictions, zero_division=0
    )

    resultats_validation.append({
        "seuil": float(seuil),
        "accuracy": accuracy_score(y_val, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            y_val, predictions
        ),
        "precision_defaut": precision,
        "recall_defaut": rappel,
        "f1_defaut": f1_score(
            y_val, predictions, zero_division=0
        ),
        "fausses_alertes": int(fp),
        "defauts_manques": int(fn),
        "vrais_negatifs": int(tn),
        "vrais_positifs": int(tp),
    })

tableau_validation = pd.DataFrame(resultats_validation)

chemin_validation = (
    DOSSIER_SORTIE / "seuils_validation.csv"
)

tableau_validation.to_csv(
    chemin_validation,
    index=False,
    encoding="utf-8-sig",
)

# Règle de sélection :
# 1. privilégier les seuils atteignant le rappel cible ;
# 2. parmi eux, choisir le meilleur F1-score ;
# 3. en cas d'égalité, privilégier la précision.
candidats = tableau_validation[
    tableau_validation["recall_defaut"] >= RAPPEL_CIBLE
].copy()

if not candidats.empty:
    candidats = candidats.sort_values(
        by=["f1_defaut", "precision_defaut", "seuil"],
        ascending=[False, False, False],
    )
    seuil_choisi = float(candidats.iloc[0]["seuil"])
    statut_seuil = (
        f"Seuil choisi parmi les seuils ayant un rappel "
        f">= {RAPPEL_CIBLE:.0%} sur la validation."
    )
else:
    # Si aucun seuil n'atteint l'objectif, retenir celui qui
    # maximise le rappel, puis le F1-score.
    candidats = tableau_validation.sort_values(
        by=["recall_defaut", "f1_defaut", "precision_defaut"],
        ascending=[False, False, False],
    )
    seuil_choisi = float(candidats.iloc[0]["seuil"])
    statut_seuil = (
        "Aucun seuil n'a atteint le rappel cible sur la validation. "
        "Le seuil retenu maximise d'abord le rappel."
    )

print("\n=== SÉLECTION DU SEUIL ===")
print(statut_seuil)
print(f"Seuil retenu : {seuil_choisi:.2f}")

ligne_choisie = tableau_validation[
    tableau_validation["seuil"] == seuil_choisi
].iloc[0]

print(
    f"Rappel validation    : "
    f"{ligne_choisie['recall_defaut']:.4f}"
)
print(
    f"Précision validation : "
    f"{ligne_choisie['precision_defaut']:.4f}"
)
print(
    f"F1-score validation  : "
    f"{ligne_choisie['f1_defaut']:.4f}"
)


# ============================================================
# 5. RÉENTRAÎNEMENT SUR TOUTES LES DONNÉES D'ENTRAÎNEMENT
# ============================================================

# Le seuil est figé avant l'évaluation finale.
# Le test final ne sert ni à choisir le seuil ni à ajuster le modèle.
modele_final = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=GRAINE,
    n_jobs=-1,
)

modele_final.fit(
    train_complet[VARIABLES].astype(float),
    train_complet[CIBLE].astype(int),
)

# Sauvegarde du modèle final et de son seuil associé.
chemin_modele = (
    DOSSIER_SORTIE / "modele_random_forest_seuil_optimise.joblib"
)

joblib.dump(
    {
        "modele": modele_final,
        "variables": VARIABLES,
        "seuil": seuil_choisi,
        "rappel_cible_validation": RAPPEL_CIBLE,
        "graine": GRAINE,
    },
    chemin_modele,
)


# ============================================================
# 6. ÉVALUATION FINALE — UNE SEULE FOIS
# ============================================================

X_test = test_final[VARIABLES].astype(float)
y_test = test_final[CIBLE].astype(int)

probabilites_test = modele_final.predict_proba(X_test)[:, 1]

predictions_seuil_standard = (
    probabilites_test >= 0.50
).astype(int)

predictions_seuil_optimise = (
    probabilites_test >= seuil_choisi
).astype(int)


def calculer_metriques(y_reel, predictions, probabilites):
    tn, fp, fn, tp = confusion_matrix(
        y_reel,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "accuracy": accuracy_score(y_reel, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            y_reel, predictions
        ),
        "precision_defaut": precision_score(
            y_reel, predictions, zero_division=0
        ),
        "recall_defaut": recall_score(
            y_reel, predictions, zero_division=0
        ),
        "f1_defaut": f1_score(
            y_reel, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_reel, probabilites),
        "fausses_alertes": int(fp),
        "defauts_manques": int(fn),
        "vrais_negatifs": int(tn),
        "vrais_positifs": int(tp),
    }


metriques_standard = calculer_metriques(
    y_test,
    predictions_seuil_standard,
    probabilites_test,
)

metriques_optimise = calculer_metriques(
    y_test,
    predictions_seuil_optimise,
    probabilites_test,
)

comparaison = pd.DataFrame([
    {
        "strategie": "Seuil standard 0.50",
        "seuil": 0.50,
        **metriques_standard,
    },
    {
        "strategie": "Seuil optimisé sur validation",
        "seuil": seuil_choisi,
        **metriques_optimise,
    },
])

chemin_comparaison = (
    DOSSIER_SORTIE / "comparaison_seuils_test_final.csv"
)

comparaison.to_csv(
    chemin_comparaison,
    index=False,
    encoding="utf-8-sig",
)

print("\n" + "=" * 70)
print("ÉVALUATION FINALE SUR LE TEST")
print("=" * 70)

for _, ligne in comparaison.iterrows():
    print(f"\n--- {ligne['strategie']} ---")
    print(f"Seuil                : {ligne['seuil']:.2f}")
    print(f"Accuracy             : {ligne['accuracy']:.4f}")
    print(
        f"Balanced accuracy    : "
        f"{ligne['balanced_accuracy']:.4f}"
    )
    print(
        f"Précision défaut     : "
        f"{ligne['precision_defaut']:.4f}"
    )
    print(
        f"Rappel défaut        : "
        f"{ligne['recall_defaut']:.4f}"
    )
    print(f"F1-score défaut      : {ligne['f1_defaut']:.4f}")
    print(f"ROC-AUC              : {ligne['roc_auc']:.4f}")
    print(f"Fausses alertes      : {int(ligne['fausses_alertes'])}")
    print(f"Défauts manqués      : {int(ligne['defauts_manques'])}")

print("\n=== MATRICES DE CONFUSION ===")
print("Seuil standard :")
print(confusion_matrix(
    y_test,
    predictions_seuil_standard,
    labels=[0, 1],
))

print("Seuil optimisé :")
print(confusion_matrix(
    y_test,
    predictions_seuil_optimise,
    labels=[0, 1],
))

print("\n=== FICHIERS ENREGISTRÉS ===")
print(f"Seuils de validation : {chemin_validation}")
print(f"Comparaison finale   : {chemin_comparaison}")
print(f"Modèle et seuil      : {chemin_modele}")

print(
    "\nAttention : données synthétiques. "
    "L'évaluation ne constitue pas une validation industrielle."
)
