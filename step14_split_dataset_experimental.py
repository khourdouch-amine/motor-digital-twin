
from pathlib import Path

import pandas as pd


# ============================================================
# 1. Chemins des fichiers
# ============================================================

fichier_entree = Path(
    "resultats_step14_experimental/"
    "dataset_detection_binaire.csv"
)

dossier_sortie = Path("resultats_step14_experimental")
dossier_sortie.mkdir(parents=True, exist_ok=True)

fichier_train = dossier_sortie / "dataset_train.csv"
fichier_test = dossier_sortie / "dataset_test.csv"


# ============================================================
# 2. Chargement et validation
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
    "target_bearing_fault",
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

if set(df["target_bearing_fault"].unique()) != {0, 1}:
    raise ValueError(
        "Le dataset doit contenir les deux classes."
    )


# ============================================================
# 3. Séparation par scénario
# ============================================================

# Réserver quatre scénarios contenant des défauts pour le test.
# Les scénarios 13 à 20 contiennent les deux états du roulement.
scenarios_test = [14, 16, 18, 20]

ids_disponibles = set(df["scenario_id"].unique())

if not set(scenarios_test).issubset(ids_disponibles):
    raise ValueError(
        "Certains scénarios de test sont absents du dataset."
    )

masque_test = df["scenario_id"].isin(scenarios_test)

df_test = df.loc[masque_test].copy()
df_train = df.loc[~masque_test].copy()


# ============================================================
# 4. Vérification de l'absence de fuite
# ============================================================

ids_train = set(df_train["scenario_id"].unique())
ids_test = set(df_test["scenario_id"].unique())

if ids_train.intersection(ids_test):
    raise ValueError(
        "Fuite détectée : un scénario apparaît dans les deux ensembles."
    )

for nom, ensemble in [
    ("ENTRAÎNEMENT", df_train),
    ("TEST", df_test),
]:
    distribution = (
        ensemble["target_bearing_fault"]
        .value_counts()
        .reindex([0, 1], fill_value=0)
    )

    if (distribution == 0).any():
        raise ValueError(
            f"L'ensemble {nom} ne contient pas les deux classes."
        )


# ============================================================
# 5. Sauvegarde des ensembles
# ============================================================

df_train.to_csv(fichier_train, index=False)
df_test.to_csv(fichier_test, index=False)


# ============================================================
# 6. Rapport
# ============================================================

print("=== SÉPARATION EXPÉRIMENTALE ===")
print(f"Fenêtres d'entraînement : {len(df_train)}")
print(f"Fenêtres de test        : {len(df_test)}")
print(f"Scénarios d'entraînement: {sorted(ids_train)}")
print(f"Scénarios de test       : {sorted(ids_test)}")
print(
    "Scénarios communs       : "
    f"{len(ids_train.intersection(ids_test))}"
)

for nom, ensemble in [
    ("ENTRAÎNEMENT", df_train),
    ("TEST", df_test),
]:
    print(f"\n=== DISTRIBUTION DES CLASSES : {nom} ===")
    print(
        ensemble["target_bearing_fault"]
        .value_counts()
        .reindex([0, 1], fill_value=0)
        .rename(index={
            0: "Sans défaut actif",
            1: "Défaut actif",
        })
        .to_string()
    )

print("\n=== SCÉNARIOS DE TEST PAR TYPE ===")
print(
    df_test.groupby("scenario_type")["scenario_id"]
    .nunique()
    .to_string()
)

print("\nFichiers créés :")
print(f"- {fichier_train}")
print(f"- {fichier_test}")
