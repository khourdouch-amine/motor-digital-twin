
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# ETAPE 66.14 - ANALYSE DES PERIODES DE SURCHARGE
# ============================================================

DOSSIER_PROJET = Path(__file__).resolve().parent

FICHIER_SCENARIOS = (
    DOSSIER_PROJET
    / "scenarios_step9"
    / "dataset_scenarios_complet.csv"
)

if not FICHIER_SCENARIOS.exists():
    raise FileNotFoundError(
        f"Fichier introuvable : {FICHIER_SCENARIOS}"
    )

donnees = pd.read_csv(FICHIER_SCENARIOS)

colonnes_requises = [
    "scenario_id",
    "scenario_type",
    "time_s",
    "speed_sensor_rpm",
    "current_sensor_a",
    "load_factor_simulated",
    "overload_active",
]

colonnes_manquantes = [
    colonne
    for colonne in colonnes_requises
    if colonne not in donnees.columns
]

if colonnes_manquantes:
    raise ValueError(
        "Colonnes manquantes : "
        + ", ".join(colonnes_manquantes)
    )

print("=" * 70)
print("ETAPE 66.14 - ANALYSE DES PERIODES DE SURCHARGE")
print("=" * 70)

print("\nResume par scenario :")

for scenario_id, groupe in donnees.groupby("scenario_id"):
    actif = groupe[groupe["overload_active"] == 1]
    inactif = groupe[groupe["overload_active"] == 0]

    if actif.empty:
        continue

    print(
        f"\nScenario {scenario_id} "
        f"({groupe['scenario_type'].iloc[0]})"
    )
    print(f"Nombre d'echantillons en surcharge : {len(actif)}")
    print(
        "Facteur de charge moyen pendant la surcharge : "
        f"{actif['load_factor_simulated'].mean():.3f}"
    )
    print(
        "Courant moyen pendant la surcharge : "
        f"{actif['current_sensor_a'].mean():.3f} A"
    )
    print(
        "Courant moyen hors surcharge : "
        f"{inactif['current_sensor_a'].mean():.3f} A"
        if not inactif.empty
        else "Aucun echantillon hors surcharge"
    )
    print(
        "Vitesse moyenne pendant la surcharge : "
        f"{actif['speed_sensor_rpm'].mean():.2f} tr/min"
    )
    print(
        "Vitesse moyenne hors surcharge : "
        f"{inactif['speed_sensor_rpm'].mean():.2f} tr/min"
        if not inactif.empty
        else "Aucun echantillon hors surcharge"
    )

print("\nVerification globale :")

masque = donnees["overload_active"] == 1

print(f"Echantillons en surcharge : {int(masque.sum())}")
print(
    "Facteur de charge moyen en surcharge : "
    f"{donnees.loc[masque, 'load_factor_simulated'].mean():.3f}"
)
print(
    "Facteur de charge moyen hors surcharge : "
    f"{donnees.loc[~masque, 'load_factor_simulated'].mean():.3f}"
)

print(
    "\nAttention : les signaux et les surcharges sont synthetiques. "
    "Cette analyse controle leur coherence interne."
)
