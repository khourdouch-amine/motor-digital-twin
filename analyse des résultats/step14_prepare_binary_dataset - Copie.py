
import pandas as pd
from pathlib import Path

# Charger les indicateurs corrigés
fichier = Path(
    "resultats_step13/indicateurs_vibratoires_5s_corriges.csv"
)
df = pd.read_csv(fichier)

# Créer une cible binaire pour le défaut de roulement
df["target_bearing_fault"] = df["bearing_fault_active"].astype(int)

# Sélectionner les variables d'entrée et les métadonnées
colonnes = [
    "scenario_id",
    "scenario_type",
    "temps_debut_s",
    "target_bearing_fault",
    "rms_m_s2",
    "ecart_type_m_s2",
    "facteur_crete",
]

dataset = df[colonnes].copy()

# Vérifier les données
print("=== DATASET DE DÉTECTION BINAIRE ===")
print(f"\nNombre de fenêtres : {len(dataset)}")
print(f"Nombre de scénarios : {dataset['scenario_id'].nunique()}")

print("\nRépartition de la cible :")
print(
    dataset["target_bearing_fault"]
    .value_counts()
    .sort_index()
    .rename(index={0: "Sans défaut actif", 1: "Défaut actif"})
    .to_string()
)

print("\nRépartition par scénario :")
print(
    dataset.groupby(
        ["scenario_id", "scenario_type", "target_bearing_fault"]
    ).size().to_string()
)

# Enregistrer dans un nouveau fichier
sortie = Path("resultats_step13/dataset_detection_binaire.csv")
dataset.to_csv(sortie, index=False)

print(f"\nFichier créé : {sortie}")
