 
from pathlib import Path
import pandas as pd

# Chemin vers les jeux de données existants
dossier = Path("resultats_step14_experimental")

fichiers = [
    "dataset_train.csv",
    "dataset_test.csv",
]

for nom_fichier in fichiers:
    chemin = dossier / nom_fichier

    print("\n" + "=" * 70)
    print(f"FICHIER : {nom_fichier}")
    print("=" * 70)

    if not chemin.exists():
        print(f"ERREUR : fichier introuvable : {chemin}")
        continue

    df = pd.read_csv(chemin)

    print(f"Dimensions : {df.shape}")
    print(f"Colonnes : {df.columns.tolist()}")

    if "scenario_id" in df.columns:
        print(
            "Scénarios :",
            sorted(df["scenario_id"].dropna().unique().tolist()),
        )
    else:
        print("Colonne scenario_id absente.")

    if "scenario_type" in df.columns:
        print(
            "Types de scénarios :",
            df["scenario_type"].value_counts().to_dict(),
        )
    else:
        print("Colonne scenario_type absente.")

    if "target_bearing_fault" in df.columns:
        print(
            "Répartition des classes :",
            df["target_bearing_fault"].value_counts().to_dict(),
        )
    else:
        print("Colonne target_bearing_fault absente.")

# Vérification du chevauchement des scénarios
print("\n" + "=" * 70)
print("VÉRIFICATION DU CHEVAUCHEMENT")
print("=" * 70)

train_path = dossier / "dataset_train.csv"
test_path = dossier / "dataset_test.csv"

if train_path.exists() and test_path.exists():
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    if "scenario_id" in train.columns and "scenario_id" in test.columns:
        ids_train = set(train["scenario_id"].dropna().unique())
        ids_test = set(test["scenario_id"].dropna().unique())

        print("Scénarios communs :", sorted(ids_train & ids_test))
        print("Scénarios train uniquement :", sorted(ids_train - ids_test))
        print("Scénarios test uniquement :", sorted(ids_test - ids_train))

        if ids_train & ids_test:
            print(
                "ATTENTION : des scénarios sont partagés entre train et test."
            )
        else:
            print(
                "Aucun scénario partagé : séparation par scénario respectée."
            )
    else:
        print("Impossible de vérifier : colonne scenario_id absente.")
