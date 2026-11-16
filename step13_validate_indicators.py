
import pandas as pd
from pathlib import Path

# Charger les indicateurs vibratoires calculés
fichier = Path("resultats_step13/indicateurs_vibratoires_5s.csv")
df = pd.read_csv(fichier)

print("=== VALIDATION DES INDICATEURS VIBRATOIRES ===")

# Dimensions du fichier
print(f"\nNombre de lignes : {len(df)}")
print(f"Nombre de colonnes : {len(df.columns)}")
print(f"Nombre de scénarios : {df['scenario_id'].nunique()}")

# Vérifier les valeurs manquantes
print("\nValeurs manquantes par colonne :")
print(df.isna().sum().to_string())

# Vérifier les doublons
print(f"\nNombre de lignes dupliquées : {df.duplicated().sum()}")

# Vérifier les valeurs physiques et numériques
colonnes_indicateurs = [
    "rms_m_s2",
    "ecart_type_m_s2",
    "facteur_crete",
]

print("\nNombre de valeurs non finies par indicateur :")
for colonne in colonnes_indicateurs:
    valeurs = pd.to_numeric(df[colonne], errors="coerce")
    print(f"{colonne} : {(~valeurs.map(lambda x: pd.notna(x) and abs(x) != float('inf'))).sum()}")

print("\nValeurs négatives :")
for colonne in colonnes_indicateurs:
    print(f"{colonne} : {(df[colonne] < 0).sum()}")

# Vérifier le nombre de fenêtres par scénario
print("\nNombre de fenêtres par scénario :")
print(df.groupby("scenario_id").size().to_string())

# Vérifier les classes et sévérités
print("\nRépartition des classes :")
print(df["fault_class"].value_counts().sort_index().to_string())

print("\nRépartition des sévérités :")
print(df["bearing_fault_severity"].value_counts().to_string())
