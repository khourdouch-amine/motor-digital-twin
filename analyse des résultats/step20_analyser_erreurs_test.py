
"""
Étape 20 — Analyse des erreurs sur le test final
Projet : Jumeau numérique d'un moteur électrique

Le modèle est entraîné uniquement sur le dataset d'apprentissage.
Le test final sert uniquement à l'évaluation et à l'analyse des erreurs.
Aucun fichier des étapes précédentes n'est modifié.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report


# ============================================================
# 1. CHEMINS ET PARAMÈTRES
# ============================================================

DOSSIER = Path(__file__).resolve().parent

FICHIER_TRAIN = (
    DOSSIER
    / "resultats_step14_experimental"
    / "dataset_train.csv"
)

FICHIER_TEST = (
    DOSSIER
    / "resultats_step14_experimental"
    / "dataset_test.csv"
)

DOSSIER_SORTIE = (
    DOSSIER
    / "resultats_step20_experimental"
)

DOSSIER_SORTIE.mkdir(parents=True, exist_ok=True)

CARACTERISTIQUES = [
    "rms_m_s2",
    "ecart_type_m_s2",
    "facteur_crete",
]

CIBLE = "target_bearing_fault"
SEUIL = 0.50
GRAINE = 42


# ============================================================
# 2. CHARGEMENT ET VALIDATION
# ============================================================

print("=" * 70)
print("ÉTAPE 20 — ANALYSE DES ERREURS DU TEST FINAL")
print("=" * 70)

for chemin in [FICHIER_TRAIN, FICHIER_TEST]:
    if not chemin.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {chemin}"
        )

train = pd.read_csv(FICHIER_TRAIN)
test = pd.read_csv(FICHIER_TEST)

colonnes_requises = (
    CARACTERISTIQUES
    + [CIBLE, "scenario_id"]
)

for nom, df in [("train", train), ("test", test)]:
    manquantes = [
        colonne
        for colonne in colonnes_requises
        if colonne not in df.columns
    ]

    if manquantes:
        raise ValueError(
            f"Colonnes absentes dans {nom} : {manquantes}"
        )

    if df[colonnes_requises].isna().any().any():
        raise ValueError(
            f"Valeurs manquantes dans le fichier {nom}."
        )

scenarios_train = set(train["scenario_id"].unique())
scenarios_test = set(test["scenario_id"].unique())

if scenarios_train.intersection(scenarios_test):
    raise RuntimeError(
        "Les scénarios d'apprentissage et de test se chevauchent."
    )

X_train = train[CARACTERISTIQUES]
y_train = train[CIBLE].astype(int)

X_test = test[CARACTERISTIQUES]
y_test = test[CIBLE].astype(int)

print(f"\nFenêtres d'apprentissage : {len(train)}")
print(f"Fenêtres de test         : {len(test)}")
print(f"Scénarios train          : {sorted(scenarios_train)}")
print(f"Scénarios test           : {sorted(scenarios_test)}")


# ============================================================
# 3. ENTRAÎNEMENT DU MODÈLE DE RÉFÉRENCE
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

probabilites = modele.predict_proba(X_test)[:, 1]
predictions = (probabilites >= SEUIL).astype(int)


# ============================================================
# 4. CONSTRUCTION DU RAPPORT D'ERREURS
# ============================================================

rapport = test.copy()

rapport["probabilite_defaut"] = probabilites
rapport["prediction"] = predictions

rapport["type_resultat"] = np.select(
    [
        (rapport[CIBLE] == 0) & (rapport["prediction"] == 0),
        (rapport[CIBLE] == 0) & (rapport["prediction"] == 1),
        (rapport[CIBLE] == 1) & (rapport["prediction"] == 0),
        (rapport[CIBLE] == 1) & (rapport["prediction"] == 1),
    ],
    [
        "vrai_negatif",
        "fausse_alerte",
        "defaut_manque",
        "vrai_positif",
    ],
    default="inconnu",
)

fichier_complet = (
    DOSSIER_SORTIE
    / "predictions_completes_test.csv"
)

rapport.to_csv(
    fichier_complet,
    index=False,
    encoding="utf-8-sig",
)

erreurs = rapport[
    rapport["type_resultat"].isin(
        ["fausse_alerte", "defaut_manque"]
    )
].copy()

fichier_erreurs = (
    DOSSIER_SORTIE
    / "erreurs_test.csv"
)

erreurs.to_csv(
    fichier_erreurs,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 5. RÉSULTATS GÉNÉRAUX
# ============================================================

matrice = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1],
)

print("\n" + "=" * 70)
print("MATRICE DE CONFUSION")
print("=" * 70)
print(matrice)

print("\nRapport de classification :")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["Sans défaut", "Défaut"],
        zero_division=0,
        digits=4,
    )
)

print("\nNombre d'erreurs :")
print(erreurs["type_resultat"].value_counts())


# ============================================================
# 6. ANALYSE DES ERREURS PAR SCÉNARIO
# ============================================================

print("\n" + "=" * 70)
print("ERREURS PAR SCÉNARIO")
print("=" * 70)

if erreurs.empty:
    print("Aucune erreur détectée sur le test.")
else:
    colonnes_affichage = [
        "scenario_id",
        "scenario_type",
        "temps_debut_s",
        CIBLE,
        "prediction",
        "probabilite_defaut",
        "type_resultat",
        "rms_m_s2",
        "ecart_type_m_s2",
        "facteur_crete",
    ]

    colonnes_affichage = [
        colonne
        for colonne in colonnes_affichage
        if colonne in erreurs.columns
    ]

    print(
        erreurs[colonnes_affichage].to_string(
            index=False,
            float_format=lambda valeur: f"{valeur:.4f}",
        )
    )


# ============================================================
# 7. STATISTIQUES DES INDICATEURS
# ============================================================

print("\n" + "=" * 70)
print("INDICATEURS PAR TYPE DE RÉSULTAT")
print("=" * 70)

colonnes_statistiques = [
    "rms_m_s2",
    "ecart_type_m_s2",
    "facteur_crete",
    "probabilite_defaut",
]

statistiques = rapport.groupby(
    "type_resultat"
)[colonnes_statistiques].agg(
    ["count", "mean", "std", "min", "max"]
)

print(statistiques.round(4).to_string())

statistiques.to_csv(
    DOSSIER_SORTIE / "statistiques_par_resultat.csv",
    encoding="utf-8-sig",
)


# ============================================================
# 8. EXPORTATION DES ERREURS AGRÉGÉES PAR SCÉNARIO
# ============================================================

erreurs_par_scenario = rapport.groupby(
    ["scenario_id", "scenario_type"],
    dropna=False,
).agg(
    nombre_fenetres=(CIBLE, "size"),
    nombre_defauts_reels=(CIBLE, "sum"),
    nombre_alertes=("prediction", "sum"),
    nombre_erreurs=(
        "type_resultat",
        lambda serie: int(
            serie.isin(
                ["fausse_alerte", "defaut_manque"]
            ).sum()
        ),
    ),
    probabilite_moyenne=(
        "probabilite_defaut",
        "mean",
    ),
).reset_index()

erreurs_par_scenario.to_csv(
    DOSSIER_SORTIE / "bilan_par_scenario.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 9. RÉSUMÉ FINAL
# ============================================================

print("\n" + "=" * 70)
print("FICHIERS ENREGISTRÉS")
print("=" * 70)

print(f"Prédictions complètes : {fichier_complet}")
print(f"Erreurs uniquement    : {fichier_erreurs}")
print(
    "Statistiques         :",
    DOSSIER_SORTIE / "statistiques_par_resultat.csv",
)
print(
    "Bilan par scénario   :",
    DOSSIER_SORTIE / "bilan_par_scenario.csv",
)

print(
    "\nAttention : les données sont synthétiques. "
    "Ces résultats ne constituent pas une validation industrielle."
)
