
from pathlib import Path

import pandas as pd


# ============================================================
# ETAPE 66.15 - COMPARAISON DES PERTES EN SURCHARGE
# ============================================================

DOSSIER_PROJET = Path(__file__).resolve().parent

FICHIER_DETAIL = (
    DOSSIER_PROJET
    / "resultats_step12"
    / "donnees_energetiques_detaillees.csv"
)

if not FICHIER_DETAIL.exists():
    raise FileNotFoundError(
        f"Fichier introuvable : {FICHIER_DETAIL}"
    )

donnees = pd.read_csv(FICHIER_DETAIL)

print("=" * 70)
print("ETAPE 66.15 - COMPARAISON DES PERTES EN SURCHARGE")
print("=" * 70)

print("\nColonnes disponibles :")
print(donnees.columns.tolist())

colonnes_possibles = [
    "scenario_id",
    "scenario_type",
    "time_s",
    "overload_active",
    "puissance_electrique_w",
    "puissance_mecanique_w",
    "pertes_totales_w",
    "pertes_w",
    "pertes_constantes_w",
    "pertes_variables_w",
]

colonnes_presentes = [
    colonne
    for colonne in colonnes_possibles
    if colonne in donnees.columns
]

print("\nColonnes pertinentes disponibles :")
print(colonnes_presentes)

print(
    "\nCette premiere verification sert a identifier les noms "
    "exactement utilises dans le fichier detaille."
)
print("Aucune donnee n'a ete modifiee.")
