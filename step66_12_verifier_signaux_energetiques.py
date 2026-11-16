
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# ETAPE 66.12 - VERIFICATION DES SIGNAUX ENERGETIQUES
# ============================================================

DOSSIER_PROJET = Path(__file__).resolve().parent

FICHIER_ORIGINAL = (
    DOSSIER_PROJET
    / "scenarios_step9"
    / "dataset_scenarios_complet.csv"
)

FICHIER_EXPERIMENTAL = (
    DOSSIER_PROJET
    / "scenarios_step9_experimental"
    / "dataset_scenarios_complet.csv"
)

COLONNES_A_VERIFIER = [
    "scenario_id",
    "scenario_type",
    "time_s",
    "speed_sensor_rpm",
    "current_sensor_a",
    "load_factor_simulated",
    "overload_active",
    "bearing_fault_active",
]

print("=" * 65)
print("ETAPE 66.12 - VERIFICATION DES SIGNAUX ENERGETIQUES")
print("=" * 65)

for fichier in [FICHIER_ORIGINAL, FICHIER_EXPERIMENTAL]:
    if not fichier.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {fichier}"
        )

original = pd.read_csv(FICHIER_ORIGINAL)
experimental = pd.read_csv(FICHIER_EXPERIMENTAL)

print(f"\nLignes version originale : {len(original)}")
print(f"Lignes version experimentale : {len(experimental)}")

if len(original) != len(experimental):
    raise ValueError(
        "Les deux fichiers n'ont pas le meme nombre de lignes."
    )

print("\nComparaison des colonnes utilisees pour l'energie :")

for colonne in COLONNES_A_VERIFIER:
    if colonne not in original.columns or colonne not in experimental.columns:
        print(f"{colonne} : colonne absente")
        continue

    serie_originale = original[colonne]
    serie_experimentale = experimental[colonne]

    identiques = serie_originale.equals(serie_experimentale)

    if pd.api.types.is_numeric_dtype(serie_originale):
        difference_max = np.max(
            np.abs(
                serie_originale.to_numpy(dtype=float)
                - serie_experimentale.to_numpy(dtype=float)
            )
        )
        print(
            f"{colonne} : identiques={identiques}, "
            f"ecart_absolu_max={difference_max:.12g}"
        )
    else:
        print(f"{colonne} : identiques={identiques}")

print("\nVerification terminee. Aucun fichier source modifie.")
