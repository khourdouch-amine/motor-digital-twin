
import pandas as pd
from pathlib import Path

# Charger les indicateurs vibratoires
fichier = Path("resultats_step13/indicateurs_vibratoires_5s.csv")
df = pd.read_csv(fichier)

print("=== COHÉRENCE DES ÉTIQUETTES ===")

# Croiser l'état du défaut avec sa sévérité
print("\nÉtat du roulement selon la sévérité :")
print(
    pd.crosstab(
        df["bearing_fault_active"],
        df["bearing_fault_severity"],
        margins=True
    ).to_string()
)

# Vérifier les incohérences entre état et sévérité
incoherent_sans_defaut = df[
    (df["bearing_fault_active"] == 0)
    & (df["bearing_fault_severity"] != "aucun")
]

incoherent_avec_defaut = df[
    (df["bearing_fault_active"] == 1)
    & (df["bearing_fault_severity"] == "aucun")
]

print(
    "\nFenêtres sans défaut mais avec une sévérité renseignée :",
    len(incoherent_sans_defaut)
)

print(
    "Fenêtres avec défaut mais sévérité 'aucun' :",
    len(incoherent_avec_defaut)
)

# Vérifier la correspondance entre type de scénario et classe
print("\nClasse de défaut selon le type de scénario :")
print(
    pd.crosstab(
        df["scenario_type"],
        df["fault_class"]
    ).to_string()
)

# Vérifier les valeurs autorisées
print("\nClasses présentes :", sorted(df["fault_class"].unique()))
print(
    "États de roulement présents :",
    sorted(df["bearing_fault_active"].unique())
)

print("\nVérification terminée.")
