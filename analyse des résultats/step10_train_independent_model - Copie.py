
"""
ÉTAPE 10 — ENTRAÎNEMENT ET ÉVALUATION SUR DES SIMULATIONS INDÉPENDANTES
======================================================================

Projet :
Conception et simulation d'un jumeau numérique d'un moteur électrique.

Objectifs :
1. Charger les scénarios générés à l'étape 9.
2. Séparer les simulations complètes en entraînement et en test.
3. Entraîner un modèle Random Forest.
4. Évaluer les performances sur des simulations inconnues.
5. Enregistrer le modèle et les résultats.

Important :
Les données sont synthétiques. Les performances obtenues ne garantissent
pas les mêmes résultats sur un moteur industriel réel.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight

import joblib


# ============================================================
# 1. CONFIGURATION
# ============================================================

DOSSIER_PROJET = Path(__file__).resolve().parent

DOSSIER_DONNEES = (
    DOSSIER_PROJET
    / "scenarios_step9"
)

FICHIER_DONNEES = (
    DOSSIER_DONNEES
    / "dataset_scenarios_complet.csv"
)

DOSSIER_RESULTATS = (
    DOSSIER_PROJET
    / "resultats_step10"
)

DOSSIER_RESULTATS.mkdir(
    parents=True,
    exist_ok=True,
)

# Variables disponibles pour le modèle.
# Il s'agit uniquement de mesures simulées de capteurs.
VARIABLES_ENTREE = [
    "speed_sensor_rpm",
    "temperature_sensor_c",
    "current_sensor_a",
    "vibration_sensor_m_s2",
]

# Cible : classe de fonctionnement du moteur
VARIABLE_CIBLE = "fault_class"

# Proportion des simulations réservées au test
PROPORTION_TEST = 0.30

# Reproductibilité
GRAINE_ALEATOIRE = 42

# Noms des classes
NOMS_CLASSES = [
    "Normal",
    "Surcharge",
    "Défaut de roulement",
    "Défauts combinés",
]


# ============================================================
# 2. CHARGEMENT ET VALIDATION DES DONNÉES
# ============================================================

def charger_donnees():
    """Charge le dataset et vérifie sa structure."""

    print("=" * 75)
    print("ÉTAPE 10 — ENTRAÎNEMENT SUR DES SIMULATIONS INDÉPENDANTES")
    print("=" * 75)

    if not FICHIER_DONNEES.exists():
        raise FileNotFoundError(
            "Dataset introuvable :\n"
            f"{FICHIER_DONNEES}\n"
            "Exécute d'abord step9_generate_scenarios.py."
        )

    donnees = pd.read_csv(FICHIER_DONNEES)

    print("\nChargement réussi.")
    print(f"Nombre d'observations : {len(donnees):,}")
    print(
        "Nombre de simulations : "
        f"{donnees['scenario_id'].nunique()}"
    )

    colonnes_requises = (
        VARIABLES_ENTREE
        + [VARIABLE_CIBLE, "scenario_id"]
    )

    colonnes_manquantes = [
        colonne
        for colonne in colonnes_requises
        if colonne not in donnees.columns
    ]

    if colonnes_manquantes:
        raise ValueError(
            "Colonnes manquantes : "
            f"{colonnes_manquantes}"
        )

    if donnees[colonnes_requises].isnull().any().any():
        raise ValueError(
            "Des valeurs manquantes ont été détectées."
        )

    if not np.isfinite(
        donnees[VARIABLES_ENTREE].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Des valeurs infinies ou non finies ont été détectées."
        )

    print("\nRépartition des classes :")

    print(
        donnees[VARIABLE_CIBLE]
        .value_counts()
        .sort_index()
        .to_string()
    )

    return donnees


# ============================================================
# 3. SÉPARATION PAR SIMULATION COMPLÈTE
# ============================================================

def separer_simulations(donnees):
    """
    Sépare les simulations complètes.

    Aucune simulation ne doit apparaître à la fois dans
    l'ensemble d'entraînement et dans l'ensemble de test.
    """

    identifiants = donnees["scenario_id"].to_numpy()

    separateur = GroupShuffleSplit(
        n_splits=100,
        test_size=PROPORTION_TEST,
        random_state=GRAINE_ALEATOIRE,
    )

    meilleure_separation = None
    meilleur_score = -1

    # Chercher une séparation où les classes sont présentes
    # dans les ensembles d'entraînement et de test.
    classes_attendues = set(
        donnees[VARIABLE_CIBLE].unique()
    )

    for indices_train, indices_test in separateur.split(
        donnees,
        donnees[VARIABLE_CIBLE],
        groups=identifiants,
    ):

        classes_train = set(
            donnees.iloc[indices_train][VARIABLE_CIBLE].unique()
        )

        classes_test = set(
            donnees.iloc[indices_test][VARIABLE_CIBLE].unique()
        )

        # Chaque classe doit être présente des deux côtés.
        if (
            classes_train == classes_attendues
            and classes_test == classes_attendues
        ):
            score = len(indices_test)

            if score > meilleur_score:
                meilleur_score = score
                meilleure_separation = (
                    indices_train,
                    indices_test,
                )

    if meilleure_separation is None:
        raise ValueError(
            "Impossible de trouver une séparation contenant "
            "toutes les classes dans les deux ensembles. "
            "Il faudra vérifier les scénarios générés."
        )

    indices_train, indices_test = meilleure_separation

    train = donnees.iloc[indices_train].copy()
    test = donnees.iloc[indices_test].copy()

    groupes_train = set(train["scenario_id"].unique())
    groupes_test = set(test["scenario_id"].unique())

    # Vérification de l'absence de chevauchement
    intersection = groupes_train.intersection(groupes_test)

    if intersection:
        raise RuntimeError(
            "Erreur : certaines simulations sont présentes "
            "dans les deux ensembles."
        )

    print("\nSéparation des simulations :")

    print(
        "Simulations d'entraînement : "
        f"{sorted(groupes_train)}"
    )

    print(
        "Simulations de test : "
        f"{sorted(groupes_test)}"
    )

    print(f"\nObservations d'entraînement : {len(train):,}")
    print(f"Observations de test : {len(test):,}")

    print("\nClasses dans l'entraînement :")
    print(
        train[VARIABLE_CIBLE]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nClasses dans le test :")
    print(
        test[VARIABLE_CIBLE]
        .value_counts()
        .sort_index()
        .to_string()
    )

    return train, test


# ============================================================
# 4. ENTRAÎNEMENT DU MODÈLE
# ============================================================

def entrainer_modele(train):
    """Entraîne une forêt aléatoire multiclasse."""

    X_train = train[VARIABLES_ENTREE]
    y_train = train[VARIABLE_CIBLE]

    modele = RandomForestClassifier(
        n_estimators=150,
        max_depth=18,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=GRAINE_ALEATOIRE,
        n_jobs=-1,
    )

    print("\nEntraînement du modèle Random Forest...")

    modele.fit(X_train, y_train)

    print("Entraînement terminé.")

    return modele


# ============================================================
# 5. ÉVALUATION
# ============================================================

def evaluer_modele(modele, test):
    """Évalue le modèle sur les simulations réservées au test."""

    X_test = test[VARIABLES_ENTREE]
    y_test = test[VARIABLE_CIBLE]

    predictions = modele.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        predictions,
    )

    f1_macro = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    classes_presentes = sorted(
        set(y_test.unique()).union(set(predictions))
    )

    matrice = confusion_matrix(
        y_test,
        predictions,
        labels=classes_presentes,
    )

    noms_presentes = [
        NOMS_CLASSES[int(classe)]
        for classe in classes_presentes
    ]

    rapport = classification_report(
        y_test,
        predictions,
        labels=classes_presentes,
        target_names=noms_presentes,
        zero_division=0,
    )

    print("\n" + "=" * 75)
    print("RÉSULTATS DE L'ÉVALUATION")
    print("=" * 75)

    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"Balanced accuracy : {balanced_accuracy:.4f}")
    print(f"F1 macro : {f1_macro:.4f}")

    print("\nRapport de classification :")
    print(rapport)

    print("Matrice de confusion :")
    print(matrice)

    # Sauvegarde des prédictions pour analyse ultérieure
    resultats = test[
        [
            "scenario_id",
            "scenario_type",
            "time_s",
            VARIABLE_CIBLE,
        ]
    ].copy()

    resultats["prediction"] = predictions
    resultats["prediction_correcte"] = (
        resultats[VARIABLE_CIBLE]
        == resultats["prediction"]
    )

    chemin_predictions = (
        DOSSIER_RESULTATS
        / "predictions_test.csv"
    )

    resultats.to_csv(
        chemin_predictions,
        index=False,
    )

    # Rapport textuel
    chemin_rapport = (
        DOSSIER_RESULTATS
        / "rapport_classification.txt"
    )

    with open(
        chemin_rapport,
        "w",
        encoding="utf-8",
    ) as fichier:

        fichier.write("ÉVALUATION DU MODÈLE\n")
        fichier.write("====================\n\n")

        fichier.write(f"Accuracy : {accuracy:.4f}\n")
        fichier.write(
            f"Balanced accuracy : {balanced_accuracy:.4f}\n"
        )
        fichier.write(f"F1 macro : {f1_macro:.4f}\n\n")
        fichier.write(rapport)

    # Visualisation de la matrice de confusion
    fig, ax = plt.subplots(figsize=(8, 6))

    affichage = ConfusionMatrixDisplay(
        confusion_matrix=matrice,
        display_labels=noms_presentes,
    )

    affichage.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )

    ax.set_title(
        "Étape 10 — Matrice de confusion\n"
        "Test sur des simulations indépendantes"
    )

    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    chemin_figure = (
        DOSSIER_RESULTATS
        / "step10_matrice_confusion.png"
    )

    plt.savefig(
        chemin_figure,
        dpi=150,
        bbox_inches="tight",
    )

    plt.show()

    # Retourner les métriques utiles
    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "f1_macro": float(f1_macro),
        "rapport": rapport,
        "matrice_confusion": matrice.tolist(),
        "classes": classes_presentes,
    }


# ============================================================
# 6. IMPORTANCE DES VARIABLES
# ============================================================

def analyser_importance(modele):
    """Affiche et sauvegarde l'importance des capteurs."""

    importances = pd.DataFrame({
        "capteur": VARIABLES_ENTREE,
        "importance": modele.feature_importances_,
    })

    importances = importances.sort_values(
        "importance",
        ascending=False,
    )

    print("\nImportance des variables :")
    print(importances.to_string(index=False))

    importances.to_csv(
        DOSSIER_RESULTATS / "importance_capteurs.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.barh(
        importances["capteur"],
        importances["importance"],
    )

    ax.invert_yaxis()
    ax.set_xlabel("Importance relative")
    ax.set_title("Importance des capteurs pour le modèle IA")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        DOSSIER_RESULTATS / "step10_importance_capteurs.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.show()


# ============================================================
# 7. SAUVEGARDE DU MODÈLE
# ============================================================

def sauvegarder_modele(modele, metriques):
    """Sauvegarde le modèle et les métriques au bon format."""

    # Chemin du modèle entraîné
    chemin_modele = (
        DOSSIER_RESULTATS
        / "modele_random_forest.joblib"
    )

    # Sauvegarder le modèle
    joblib.dump(modele, chemin_modele)

    # Chemin du fichier de métriques
    chemin_metriques = (
        DOSSIER_RESULTATS
        / "metriques_evaluation.json"
    )

    # Convertir les types NumPy en types Python natifs.
    # Cela évite l'erreur de sérialisation JSON.
    def convertir_types_numpy(objet):

        if isinstance(objet, dict):
            return {
                str(cle): convertir_types_numpy(valeur)
                for cle, valeur in objet.items()
            }

        if isinstance(objet, list):
            return [
                convertir_types_numpy(valeur)
                for valeur in objet
            ]

        if isinstance(objet, tuple):
            return [
                convertir_types_numpy(valeur)
                for valeur in objet
            ]

        if isinstance(objet, np.ndarray):
            return convertir_types_numpy(objet.tolist())

        if isinstance(objet, np.integer):
            return int(objet)

        if isinstance(objet, np.floating):
            return float(objet)

        if isinstance(objet, np.bool_):
            return bool(objet)

        return objet

    # Conversion de toutes les métriques
    metriques_json = convertir_types_numpy(metriques)

    # Écriture du fichier JSON
    with open(
        chemin_metriques,
        "w",
        encoding="utf-8",
    ) as fichier:

        json.dump(
            metriques_json,
            fichier,
            ensure_ascii=False,
            indent=4,
        )

    print("\nModèle sauvegardé :")
    print(chemin_modele)

    print("\nMétriques sauvegardées :")
    print(chemin_metriques)

    print("\nSauvegarde terminée avec succès.")



# ============================================================
# 8. PROGRAMME PRINCIPAL
# ============================================================

def main():

    donnees = charger_donnees()

    train, test = separer_simulations(donnees)

    modele = entrainer_modele(train)

    metriques = evaluer_modele(modele, test)

    analyser_importance(modele)

    sauvegarder_modele(modele, metriques)

    print("\n" + "=" * 75)
    print("ÉTAPE 10 TERMINÉE")
    print("=" * 75)

    print(
        "\nAttention : les résultats mesurent les performances "
        "sur des simulations synthétiques. Une validation sur "
        "des données réelles sera nécessaire avant toute "
        "utilisation industrielle."
    )


if __name__ == "__main__":
    main()
