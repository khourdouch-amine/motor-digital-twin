
from pathlib import Path

import pandas as pd


# ============================================================
# ETAPE 66.11 - COMPARAISON DES JEUX DE DONNEES
# ============================================================

DOSSIER_PROJET = Path(__file__).resolve().parent

FICHIERS = {
    "Version originale": (
        DOSSIER_PROJET
        / "scenarios_step9"
        / "dataset_scenarios_complet.csv"
    ),
    "Version experimentale": (
        DOSSIER_PROJET
        / "scenarios_step9_experimental"
        / "dataset_scenarios_complet.csv"
    ),
}

COLONNES_NUMERIQUES = [
    "speed_sensor_rpm",
    "temperature_sensor_c",
    "current_sensor_a",
    "vibration_sensor_m_s2",
]

donnees_chargees = {}

print("=" * 65)
print("ETAPE 66.11 - COMPARAISON DES JEUX DE DONNEES")
print("=" * 65)

# Chargement et verification des deux fichiers
for nom, chemin in FICHIERS.items():
    print(f"\n--- {nom} ---")
    print(f"Fichier : {chemin}")

    if not chemin.exists():
        print("ERREUR : fichier introuvable.")
        continue

    donnees = pd.read_csv(chemin)
    donnees_chargees[nom] = donnees

    print(f"Nombre de lignes : {len(donnees)}")
    print(
        "Nombre de scenarios : "
        f"{donnees['scenario_id'].nunique()}"
    )

    print("\nNombre de lignes par type de scenario :")
    print(
        donnees.groupby("scenario_type")
        .size()
        .to_string()
    )

    print("\nStatistiques des signaux :")
    print(
        donnees[COLONNES_NUMERIQUES]
        .agg(["mean", "std", "min", "max"])
        .round(5)
        .to_string()
    )

# Comparaison directe
if len(donnees_chargees) == 2:
    originale = donnees_chargees["Version originale"]
    experimentale = donnees_chargees["Version experimentale"]

    print("\n" + "=" * 65)
    print("COMPARAISON DIRECTE")
    print("=" * 65)

    memes_dimensions = originale.shape == experimentale.shape
    memes_colonnes = list(originale.columns) == list(
        experimentale.columns
    )

    print(f"Meme nombre de lignes et colonnes : {memes_dimensions}")
    print(f"Meme ordre de colonnes : {memes_colonnes}")

    if memes_dimensions and memes_colonnes:
        identiques = originale.equals(experimentale)
        print(f"Fichiers strictement identiques : {identiques}")

        if not identiques:
            print("\nDifferences des moyennes des signaux :")

            for colonne in COLONNES_NUMERIQUES:
                moyenne_originale = originale[colonne].mean()
                moyenne_experimentale = experimentale[colonne].mean()

                difference = (
                    moyenne_experimentale - moyenne_originale
                )

                print(
                    f"{colonne} : "
                    f"originale={moyenne_originale:.5f}, "
                    f"experimentale={moyenne_experimentale:.5f}, "
                    f"difference={difference:.5f}"
                )
    else:
        print(
            "Les fichiers ont des dimensions ou des colonnes "
            "differentes. La comparaison ligne par ligne "
            "n'est pas effectuee."
        )

print("\nVerification terminee. Aucun fichier source modifie.")
