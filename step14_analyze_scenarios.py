
import pandas as pd
from pathlib import Path

# Charger le fichier corrigé
fichier = Path(
    "resultats_step13/indicateurs_vibratoires_5s_corriges.csv"
)
df = pd.read_csv(fichier)

print("=== SCÉNARIOS DISPONIBLES PAR CLASSE ===")

# Nombre de scénarios distincts pour chaque classe de défaut
resume_classes = (
    df.groupby("fault_class")
    .agg(
        nombre_fenetres=("scenario_id", "size"),
        nombre_scenarios=("scenario_id", "nunique")
    )
)

print("\nRépartition des classes de défaut :")
print(resume_classes.to_string())

# Nombre de scénarios par sévérité, parmi les fenêtres avec défaut actif
defauts = df[df["bearing_fault_active"] == 1]

resume_severites = (
    defauts.groupby("bearing_fault_severity_window")
    .agg(
        nombre_fenetres=("scenario_id", "size"),
        nombre_scenarios=("scenario_id", "nunique")
    )
)

print("\nRépartition des sévérités :")
print(resume_severites.to_string())

# Détail des scénarios et des sévérités qu'ils contiennent
print("\nDétail par scénario :")
print(
    df.groupby(
        ["scenario_id", "scenario_type", "bearing_fault_severity_window"]
    ).size().to_string()
)

print("\nAnalyse terminée.")
