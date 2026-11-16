
from pathlib import Path

import pandas as pd


# ============================================================
# 1. Chemins des fichiers
# ============================================================

fichier_entree = Path(
    "resultats_step13/dataset_detection_binaire.csv"
)

dossier_sortie = Path("resultats_step14")
dossier_sortie.mkdir(parents=True, exist_ok=True)

fichier_train = dossier_sortie / "dataset_train.csv"
fichier_test = dossier_sortie / "dataset_test.csv"


# ============================================================
# 2. Chargement et vérification des données
# ============================================================

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

if df["scenario_id"].isna().any():
    raise ValueError("Certains identifiants de scénario sont manquants.")

if not set(df["target_bearing_fault"].unique()).issubset({0, 1}):
    raise ValueError("La cible doit contenir uniquement les valeurs 0 et 1.")


# ============================================================
# 3. Définition des scénarios d'entraînement et de test
# ============================================================

scenarios_test = [2, 6, 10, 14, 16, 18, 20]

masque_test = df["scenario_id"].isin(scenarios_test)

df_test = df.loc[masque_test].copy()
df_train = df.loc[~masque_test].copy()


# ============================================================
# 4. Vérification de l'absence de fuite entre les scénarios
# ============================================================

ids_train = set(df_train["scenario_id"].unique())
ids_test = set(df_test["scenario_id"].unique())

chevauchement = ids_train.intersection(ids_test)

if chevauchement:
    raise ValueError(
        f"Fuite de scénarios détectée : {chevauchement}"
    )

if ids_test != set(scenarios_test):
    raise ValueError(
        "Les scénarios de test attendus ne correspondent pas aux données."
    )


# ============================================================
# 5. Enregistrement des deux ensembles
# ============================================================

df_train.to_csv(fichier_train, index=False)
df_test.to_csv(fichier_test, index=False)


# ============================================================
# 6. Rapport de séparation
# ============================================================

print("=== SÉPARATION DES DONNÉES ===")
print(f"Nombre de fenêtres d'entraînement : {len(df_train)}")
print(f"Nombre de fenêtres de test        : {len(df_test)}")
print(f"Scénarios d'entraînement          : {sorted(ids_train)}")
print(f"Scénarios de test                 : {sorted(ids_test)}")
print(f"Chevauchement entre les groupes   : {len(chevauchement)}")

for nom, ensemble in [
    ("ENTRAÎNEMENT", df_train),
    ("TEST", df_test),
]:
    print(f"\n=== DISTRIBUTION DES CLASSES : {nom} ===")

    distribution = (
        ensemble["target_bearing_fault"]
        .value_counts()
        .reindex([0, 1], fill_value=0)
    )

    print(f"Sans défaut actif (0) : {distribution.loc[0]}")
    print(f"Avec défaut actif (1) : {distribution.loc[1]}")

    if distribution.loc[0] == 0 or distribution.loc[1] == 0:
        raise ValueError(
            f"L'ensemble {nom} ne contient pas les deux classes."
        )

print("\n=== DISTRIBUTION PAR TYPE DE SCÉNARIO ===")
print(
    df_test.groupby("scenario_type")["scenario_id"]
    .nunique()
    .to_string()
)

print("\nFichiers enregistrés :")
print(f"- {fichier_train}")
print(f"- {fichier_test}")
