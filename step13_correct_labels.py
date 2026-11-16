
import pandas as pd
from pathlib import Path

# Charger les indicateurs vibratoires
fichier = Path("resultats_step13/indicateurs_vibratoires_5s.csv")
df = pd.read_csv(fichier)

# Conserver les étiquettes originales et créer une étiquette par fenêtre
df["bearing_fault_severity_window"] = df["bearing_fault_severity"]

# Sans défaut actif, la sévérité de la fenêtre est "aucun"
df.loc[
    df["bearing_fault_active"] == 0,
    "bearing_fault_severity_window"
] = "aucun"

# Vérifier la cohérence après correction
print("=== ÉTAT DU ROULEMENT APRÈS CORRECTION ===")
print(
    pd.crosstab(
        df["bearing_fault_active"],
        df["bearing_fault_severity_window"],
        margins=True
    ).to_string()
)

# Vérifier les incohérences restantes
incoherences = df[
    (
        (df["bearing_fault_active"] == 0)
        & (df["bearing_fault_severity_window"] != "aucun")
    )
    |
    (
        (df["bearing_fault_active"] == 1)
        & (df["bearing_fault_severity_window"] == "aucun")
    )
]

print("\nNombre d'incohérences restantes :", len(incoherences))

# Enregistrer un nouveau fichier sans écraser l'original
sortie = Path("resultats_step13/indicateurs_vibratoires_5s_corriges.csv")
df.to_csv(sortie, index=False)

print(f"\nFichier corrigé créé : {sortie}")
print(f"Nombre de fenêtres : {len(df)}")
