
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# ETAPE 66.18 - VERIFICATION DU DISCRIMINANT ENERGETIQUE
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


from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# ETAPE 66.18 - VERIFICATION DU DISCRIMINANT ENERGETIQUE
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

# Reconstitution independante des constantes nominales.
PUISSANCE_MECANIQUE_NOMINALE_W = 5500.0
RENDEMENT_NOMINAL = 0.88
TENSION_TRIPHASEE_V = 400.0
FACTEUR_PUISSANCE = 0.82

FRACTION_PERTES_CONSTANTES = 0.30
FRACTION_PERTES_VARIABLES = 0.70

PUISSANCE_ELECTRIQUE_NOMINALE_W = (
    PUISSANCE_MECANIQUE_NOMINALE_W
    / RENDEMENT_NOMINAL
)

PERTES_NOMINALES_W = (
    PUISSANCE_ELECTRIQUE_NOMINALE_W
    - PUISSANCE_MECANIQUE_NOMINALE_W
)

PERTES_CONSTANTES_W = (
    FRACTION_PERTES_CONSTANTES
    * PERTES_NOMINALES_W
)

PERTES_VARIABLES_NOMINALES_W = (
    FRACTION_PERTES_VARIABLES
    * PERTES_NOMINALES_W
)

COURANT_NOMINAL_A = (
    PUISSANCE_ELECTRIQUE_NOMINALE_W
    / (
        np.sqrt(3.0)
        * TENSION_TRIPHASEE_V
        * FACTEUR_PUISSANCE
    )
)

COEFFICIENT_PUISSANCE_W_A = (
    np.sqrt(3.0)
    * TENSION_TRIPHASEE_V
    * FACTEUR_PUISSANCE
)

COEFFICIENT_PERTES_W_A2 = (
    PERTES_VARIABLES_NOMINALES_W
    / COURANT_NOMINAL_A**2
)

# Chargement des puissances mecaniques deja calculees.
donnees = pd.read_csv(FICHIER_DETAIL)

puissance_mecanique = donnees[
    "puissance_mecanique_estimee_w"
].to_numpy(dtype=float)

# Calcul du discriminant pour chaque echantillon.
a = COEFFICIENT_PUISSANCE_W_A
k = COEFFICIENT_PERTES_W_A2

discriminant = (
    a**2
    - 4.0
    * k
    * (puissance_mecanique + PERTES_CONSTANTES_W)
)

print("=" * 65)
print("ETAPE 66.18 - VERIFICATION DU DISCRIMINANT")
print("=" * 65)

print(f"Nombre d'echantillons : {len(discriminant)}")
print(f"Discriminant minimal : {discriminant.min():.6f}")
print(f"Discriminant maximal : {discriminant.max():.6f}")
print(
    "Discriminants negatifs : "
    f"{np.sum(discriminant < 0)}"
)
print(
    "Discriminants nuls : "
    f"{np.sum(discriminant == 0)}"
)
print(
    "Discriminants positifs : "
    f"{np.sum(discriminant > 0)}"
)

print(
    "\nPuissance mecanique maximale : "
    f"{puissance_mecanique.max():.3f} W"
)

print(
    "\nAttention : les constantes nominales sont "
    "reconstituees a partir des hypotheses du modele. "
    "Cette verification ne constitue pas une validation "
    "experimentale du moteur."
)

