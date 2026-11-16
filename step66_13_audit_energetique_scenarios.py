
from pathlib import Path

import pandas as pd


# ============================================================
# ETAPE 66.13 - AUDIT ENERGETIQUE PAR TYPE DE SCENARIO
# ============================================================

DOSSIER_PROJET = Path(__file__).resolve().parent

FICHIER_BILAN = (
    DOSSIER_PROJET
    / "resultats_step12"
    / "bilan_energetique_par_scenario.csv"
)

if not FICHIER_BILAN.exists():
    raise FileNotFoundError(
        f"Fichier de bilan introuvable : {FICHIER_BILAN}"
    )

bilan = pd.read_csv(FICHIER_BILAN)

colonnes_requises = [
    "scenario_id",
    "scenario_type",
    "puissance_electrique_moyenne_w",
    "puissance_mecanique_moyenne_w",
    "pertes_moyennes_w",
    "rendement_moyen_pct",
]

colonnes_manquantes = [
    colonne
    for colonne in colonnes_requises
    if colonne not in bilan.columns
]

if colonnes_manquantes:
    raise ValueError(
        "Colonnes manquantes : "
        + ", ".join(colonnes_manquantes)
    )

print("=" * 70)
print("ETAPE 66.13 - AUDIT ENERGETIQUE PAR TYPE DE SCENARIO")
print("=" * 70)

print(f"\nNombre de scenarios : {len(bilan)}")

print("\nBilan individuel :")
print(
    bilan[colonnes_requises]
    .sort_values("scenario_id")
    .round(3)
    .to_string(index=False)
)

print("\nSynthese par type de scenario :")

synthese = (
    bilan.groupby("scenario_type")
    .agg(
        nombre_scenarios=("scenario_id", "count"),
        puissance_electrique_moyenne_w=(
            "puissance_electrique_moyenne_w", "mean"
        ),
        puissance_mecanique_moyenne_w=(
            "puissance_mecanique_moyenne_w", "mean"
        ),
        pertes_moyennes_w=("pertes_moyennes_w", "mean"),
        rendement_moyen_pct=("rendement_moyen_pct", "mean"),
    )
    .round(3)
)

print(synthese.to_string())

print("\nVerification du bilan de puissance :")

bilan["residu_recalcule_w"] = (
    bilan["puissance_electrique_moyenne_w"]
    - bilan["puissance_mecanique_moyenne_w"]
    - bilan["pertes_moyennes_w"]
)

print(
    "Residus absolus maximaux : "
    f"{bilan['residu_recalcule_w'].abs().max():.6f} W"
)

print(
    "\nAttention : les donnees sont synthetiques. "
    "Cette verification controle la coherence interne, "
    "pas la validite experimentale du modele."
)
