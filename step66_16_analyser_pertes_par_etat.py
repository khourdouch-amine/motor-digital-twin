
from pathlib import Path

import pandas as pd


# ============================================================
# ETAPE 66.16 - ANALYSE DES PERTES PAR ETAT DE CHARGE
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

colonnes_requises = [
    "scenario_id",
    "scenario_type",
    "time_s",
    "overload_active",
    "load_factor_simulated",
    "puissance_mecanique_estimee_w",
    "puissance_electrique_estimee_w",
    "pertes_constantes_estimees_w",
    "pertes_variables_estimees_w",
    "pertes_estimees_w",
    "rendement_estime_pct",
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

# Separation des periodes selon l'indicateur de surcharge
donnees["etat_charge"] = donnees["overload_active"].map(
    {0: "Hors surcharge", 1: "En surcharge"}
)

print("=" * 75)
print("ETAPE 66.16 - ANALYSE DES PERTES PAR ETAT DE CHARGE")
print("=" * 75)

# Moyennes globales pour chaque etat
colonnes_analyse = [
    "load_factor_simulated",
    "puissance_mecanique_estimee_w",
    "puissance_electrique_estimee_w",
    "pertes_constantes_estimees_w",
    "pertes_variables_estimees_w",
    "pertes_estimees_w",
    "rendement_estime_pct",
]

print("\n1. Comparaison globale :")
print(
    donnees.groupby("etat_charge")[colonnes_analyse]
    .mean()
    .round(3)
    .to_string()
)

# Analyse par scenario et par etat
print("\n2. Comparaison par scenario :")

synthese = (
    donnees.groupby(
        ["scenario_id", "scenario_type", "etat_charge"]
    )[colonnes_analyse]
    .mean()
    .round(3)
)

print(synthese.to_string())

# Verification de coherence du bilan de puissance
residu = (
    donnees["puissance_electrique_estimee_w"]
    - donnees["puissance_mecanique_estimee_w"]
    - donnees["pertes_estimees_w"]
)

print("\n3. Controle du bilan de puissance :")
print(
    "Residuel absolu maximal : "
    f"{residu.abs().max():.12g} W"
)

print(
    "\nAttention : les resultats sont issus d'un modele "
    "energetique applique a des donnees synthetiques. "
    "Ils ne constituent pas des mesures industrielles."
)
