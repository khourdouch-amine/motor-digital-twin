
import pandas as pd
from pathlib import Path

# Charger le dataset binaire
fichier = Path(
    "resultats_step13/dataset_detection_binaire.csv"
)
df = pd.read_csv(fichier)

# Résumer chaque scénario
resume = (
    df.groupby(["scenario_id", "scenario_type"])
    .agg(
        nombre_fenetres=("target_bearing_fault", "size"),
        fenetres_sans_defaut=(
            "target_bearing_fault",
            lambda x: (x == 0).sum()
        ),
        fenetres_avec_defaut=(
            "target_bearing_fault",
            lambda x: (x == 1).sum()
        )
    )
    .reset_index()
)

print("=== DIAGNOSTIC DES SCÉNARIOS ===")
print(resume.to_string(index=False))

print("\n=== GROUPES DE SCÉNARIOS AVEC DÉFAUT ACTIF ===")
print(
    resume[resume["fenetres_avec_defaut"] > 0]
    .to_string(index=False)
)

print("\n=== BILAN ===")
print(
    "Scénarios avec défaut actif :",
    (resume["fenetres_avec_defaut"] > 0).sum()
)
print(
    "Scénarios sans défaut actif :",
    (resume["fenetres_avec_defaut"] == 0).sum()
)
