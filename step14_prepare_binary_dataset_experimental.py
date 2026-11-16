
from pathlib import Path

import pandas as pd


# ============================================================
# 1. Chemins des fichiers
# ============================================================

fichier_entree = Path(
    "resultats_step13_experimental/"
    "indicateurs_vibratoires_5s.csv"
)

dossier_sortie = Path("resultats_step14_experimental")
dossier_sortie.mkdir(parents=True, exist_ok=True)

fichier_sortie = (
    dossier_sortie / "dataset_detection_binaire.csv"
)


# ============================================================
# 2. Chargement et vérification
# ============================================================

if not fichier_entree.exists():
    raise FileNotFoundError(
        f"Fichier introuvable : {fichier_entree}"
    )

df = pd.read_csv(fichier_entree)

colonnes_requises = [
    "scenario_id",
    "scenario_type",
    "temps_debut_s",
    "bearing_fault_active",
    "rms_m_s2",
    "ecart_type_m_s2",
    "facteur_crete",
]

colonnes_absentes = [
    colonne
    for colonne in colonnes_requises
    if colonne not in df.columns
]

if colonnes_absentes:
    raise ValueError(
        f"Colonnes absentes : {colonnes_absentes}"
    )

if df[colonnes_requises].isna().any().any():
    raise ValueError(
        "Des valeurs manquantes sont présentes."
    )


# ============================================================
# 3. Construction de la cible binaire
# ============================================================

df["target_bearing_fault"] = (
    df["bearing_fault_active"].astype(int)
)

if set(df["target_bearing_fault"].unique()) != {0, 1}:
    raise ValueError(
        "Le dataset doit contenir les deux classes."
    )


# ============================================================
# 4. Sélection des variables et métadonnées
# ============================================================

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


# ============================================================
# 5. Enregistrement
# ============================================================

dataset.to_csv(fichier_sortie, index=False)


# ============================================================
# 6. Rapport de contrôle
# ============================================================

print("=== DATASET BINAIRE EXPÉRIMENTAL ===")
print(f"Fichier créé : {fichier_sortie}")
print(f"Nombre de fenêtres : {len(dataset)}")
print(
    "Nombre de scénarios : "
    f"{dataset['scenario_id'].nunique()}"
)

print("\nRépartition de la cible :")
print(
    dataset["target_bearing_fault"]
    .value_counts()
    .sort_index()
    .rename(index={
        0: "Sans défaut actif",
        1: "Défaut actif",
    })
    .to_string()
)

print("\nRépartition par scénario :")
print(
    dataset.groupby(
        ["scenario_id", "scenario_type", "target_bearing_fault"]
    ).size().to_string()
)

print("\nDataset binaire expérimental créé avec succès.")
