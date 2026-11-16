
from pathlib import Path

import pandas as pd


# ============================================================
# ETAPE 66.22 - CORRELATIONS ENERGETIQUES
# ============================================================

DOSSIER = Path(__file__).resolve().parent

FICHIER = (
    DOSSIER
    / "resultats_step12"
    / "donnees_energetiques_detaillees.csv"
)

donnees = pd.read_csv(FICHIER)

correlation_charge = donnees[
    "load_factor_simulated"
].corr(
    donnees["pertes_estimees_w"]
)

correlation_puissance = donnees[
    "puissance_mecanique_estimee_w"
].corr(
    donnees["pertes_estimees_w"]
)

print("=" * 60)
print("ETAPE 66.22 - CORRELATIONS ENERGETIQUES")
print("=" * 60)

print(
    f"Correlation charge / pertes : "
    f"{correlation_charge:.4f}"
)

print(
    f"Correlation puissance mecanique / pertes : "
    f"{correlation_puissance:.4f}"
)

print(
    "\nAttention : une correlation indique une association "
    "statistique, pas une relation causale."
)
