
"""
Étape 18 — Analyse du seuil de décision
Projet : Jumeau numérique d'un moteur électrique

Objectifs :
- Reconstituer la séparation apprentissage/validation de l'étape 17.
- Comparer les seuils uniquement sur la validation.
- Examiner le compromis entre fausses alertes et défauts manqués.
- Exporter les métriques et une courbe de comparaison.

Attention :
Les données sont synthétiques. Les résultats ne constituent pas
une validation industrielle.
"""

from pathlib import Path

import matplotlib.pyplot as plt
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
    confusion_matrix,
    roc_auc_score,
)


# ============================================================
# 1. CHEMINS ET PARAMÈTRES
# ============================================================

DOSSIER_PROJET = Path(__file__).resolve().parent

FICHIER_DATASET = (
    DOSSIER_PROJET
    / "resultats_step14_experimental"
    / "dataset_train.csv"
)

DOSSIER_SORTIE = (
    DOSSIER_PROJET
    / "resultats_step18_experimental"
)

DOSSIER_SORTIE.mkdir(parents=True, exist_ok=True)

GRAINE_ALEATOIRE = 42
TAILLE_VALIDATION = 0.25

CARACTERISTIQUES = [
    "rms_m_s2",
    "ecart_type_m_s2",
    "facteur_crete",
]

CIBLE = "target_bearing_fault"


# ============================================================
# 2. CHARGEMENT ET VALIDATION DES DONNÉES
# ============================================================

print("=" * 70)
print("ÉTAPE 18 — ANALYSE DES SEUILS DE DÉCISION")
print("=" * 70)

if not FICHIER_DATASET.exists():
    raise FileNotFoundError(
        f"Dataset introuvable : {FICHIER_DATASET}"
    )

df = pd.read_csv(FICHIER_DATASET)

colonnes_requises = (
    CARACTERISTIQUES
    + [CIBLE, "scenario_id"]
)

colonnes_manquantes = [
    colonne
    for colonne in colonnes_requises
    if colonne not in df.columns
]

if colonnes_manquantes:
    raise ValueError(
        f"Colonnes manquantes : {colonnes_manquantes}"
    )

if df[colonnes_requises].isna().any().any():
    raise ValueError(
        "Des valeurs manquantes ont été détectées."
    )

if set(df[CIBLE].unique()) != {0, 1}:
    raise ValueError(
        "La cible doit contenir les deux classes 0 et 1."
    )

X = df[CARACTERISTIQUES]
y = df[CIBLE].astype(int)
groupes = df["scenario_id"]


# ============================================================
# 3. SÉPARATION PAR SCÉNARIO
# ============================================================

separateur = GroupShuffleSplit(
    n_splits=1,
    test_size=TAILLE_VALIDATION,
    random_state=GRAINE_ALEATOIRE,
)

indices_apprentissage, indices_validation = next(
    separateur.split(X, y, groups=groupes)
)

X_apprentissage = X.iloc[indices_apprentissage]
y_apprentissage = y.iloc[indices_apprentissage]

X_validation = X.iloc[indices_validation]
y_validation = y.iloc[indices_validation]

groupes_apprentissage = set(
    groupes.iloc[indices_apprentissage].unique()
)

groupes_validation = set(
    groupes.iloc[indices_validation].unique()
)

if groupes_apprentissage.intersection(groupes_validation):
    raise RuntimeError(
        "Fuite de scénarios entre apprentissage et validation."
    )

if y_apprentissage.nunique() < 2 or y_validation.nunique() < 2:
    raise ValueError(
        "Les deux classes doivent être présentes "
        "dans l'apprentissage et la validation."
    )

print(f"\nFenêtres d'apprentissage : {len(X_apprentissage)}")
print(f"Fenêtres de validation   : {len(X_validation)}")

print(
    "Scénarios d'apprentissage :",
    sorted(groupes_apprentissage),
)

print(
    "Scénarios de validation   :",
    sorted(groupes_validation),
)


# ============================================================
# 4. ENTRAÎNEMENT DU MODÈLE SUR L'APPRENTISSAGE
# ============================================================

modele = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=GRAINE_ALEATOIRE,
    n_jobs=-1,
)

modele.fit(X_apprentissage, y_apprentissage)

probabilites = modele.predict_proba(
    X_validation
)[:, 1]


# ============================================================
# 5. ÉVALUATION DE PLUSIEURS SEUILS
# ============================================================

seuils = np.round(
    np.arange(0.05, 0.951, 0.01),
    2,
)

resultats = []

for seuil in seuils:
    predictions = (
        probabilites >= seuil
    ).astype(int)

    matrice = confusion_matrix(
        y_validation,
        predictions,
        labels=[0, 1],
    )

    vrais_negatifs, fausses_alertes, defauts_manques, vrais_positifs = (
        matrice.ravel()
    )

    resultats.append({
        "seuil": seuil,
        "accuracy": accuracy_score(
            y_validation, predictions
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_validation, predictions
        ),
        "precision_defaut": precision_score(
            y_validation,
            predictions,
            zero_division=0,
        ),
        "rappel_defaut": recall_score(
            y_validation,
            predictions,
            zero_division=0,
        ),
        "f1_defaut": f1_score(
            y_validation,
            predictions,
            zero_division=0,
        ),
        "fausses_alertes": int(fausses_alertes),
        "defauts_manques": int(defauts_manques),
        "vrais_positifs": int(vrais_positifs),
        "vrais_negatifs": int(vrais_negatifs),
    })

resultats_df = pd.DataFrame(resultats)

fichier_csv = (
    DOSSIER_SORTIE
    / "analyse_seuils_validation.csv"
)

resultats_df.to_csv(
    fichier_csv,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 6. RÉSUMÉ DES SEUILS IMPORTANTS
# ============================================================

print("\n" + "=" * 70)
print("COMPARAISON DES SEUILS SUR LA VALIDATION")
print("=" * 70)

seuils_importants = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.94]

for seuil in seuils_importants:
    ligne = resultats_df.loc[
        np.isclose(resultats_df["seuil"], seuil)
    ]

    if ligne.empty:
        continue

    ligne = ligne.iloc[0]

    print(
        f"Seuil {seuil:.2f} | "
        f"Précision {ligne['precision_defaut']:.3f} | "
        f"Rappel {ligne['rappel_defaut']:.3f} | "
        f"F1 {ligne['f1_defaut']:.3f} | "
        f"Fausses alertes {ligne['fausses_alertes']} | "
        f"Défauts manqués {ligne['defauts_manques']}"
    )


# ============================================================
# 7. MEILLEURS SEUILS SELON DIFFÉRENTS CRITÈRES
# ============================================================

meilleur_f1 = resultats_df.loc[
    resultats_df["f1_defaut"].idxmax()
]

candidats_rappel = resultats_df[
    resultats_df["rappel_defaut"] >= 0.98
]

print("\n" + "=" * 70)
print("SEUILS À EXAMINER")
print("=" * 70)

print(
    f"Meilleur seuil selon F1 : {meilleur_f1['seuil']:.2f} "
    f"(F1 = {meilleur_f1['f1_defaut']:.4f})"
)

if not candidats_rappel.empty:
    seuil_prudent = candidats_rappel.sort_values(
        by=["precision_defaut", "f1_defaut"],
        ascending=False,
    ).iloc[0]

    print(
        "Meilleur seuil parmi ceux avec rappel >= 98 % : "
        f"{seuil_prudent['seuil']:.2f}"
    )

    print(
        f"Rappel : {seuil_prudent['rappel_defaut']:.4f} | "
        f"Précision : {seuil_prudent['precision_defaut']:.4f} | "
        f"F1 : {seuil_prudent['f1_defaut']:.4f}"
    )
else:
    print(
        "Aucun seuil n'atteint un rappel de 98 % "
        "sur cette validation."
    )


# ============================================================
# 8. GRAPHIQUE DES MÉTRIQUES
# ============================================================

figure, axes = plt.subplots(
    2,
    1,
    figsize=(11, 9),
    sharex=True,
)

axes[0].plot(
    resultats_df["seuil"],
    resultats_df["precision_defaut"],
    label="Précision",
)

axes[0].plot(
    resultats_df["seuil"],
    resultats_df["rappel_defaut"],
    label="Rappel",
)

axes[0].plot(
    resultats_df["seuil"],
    resultats_df["f1_defaut"],
    label="F1-score",
)

axes[0].set_ylabel("Score")
axes[0].set_ylim(0, 1.05)
axes[0].set_title(
    "Évolution des performances selon le seuil"
)
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(
    resultats_df["seuil"],
    resultats_df["fausses_alertes"],
    label="Fausses alertes",
)

axes[1].plot(
    resultats_df["seuil"],
    resultats_df["defauts_manques"],
    label="Défauts manqués",
)

axes[1].set_xlabel("Seuil de décision")
axes[1].set_ylabel("Nombre de fenêtres")
axes[1].set_title(
    "Compromis entre fausses alertes et défauts manqués"
)
axes[1].grid(True, alpha=0.3)
axes[1].legend()

figure.tight_layout()

fichier_graphique = (
    DOSSIER_SORTIE
    / "courbes_analyse_seuils.png"
)

figure.savefig(
    fichier_graphique,
    dpi=200,
    bbox_inches="tight",
)

plt.close(figure)


# ============================================================
# 9. RÉSUMÉ FINAL
# ============================================================

print("\n" + "=" * 70)
print("FICHIERS ENREGISTRÉS")
print("=" * 70)

print(f"Tableau des seuils : {fichier_csv}")
print(f"Graphique          : {fichier_graphique}")

print(
    "\nAnalyse réalisée uniquement sur la validation. "
    "Le test final n'a pas été utilisé pour sélectionner un seuil."
)

print(
    "Attention : données synthétiques, sans validation industrielle."
)
