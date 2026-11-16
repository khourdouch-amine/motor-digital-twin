from pathlib import Path
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Chemins des fichiers
FICHIER_ENTREE = Path("resultats_step20_experimental/predictions_completes_test.csv")
DOSSIER_SORTIE = Path("resultats_step29_persistance_experimental")
FICHIER_DETAIL = DOSSIER_SORTIE / "comparaison_predictions_persistance.csv"
FICHIER_METRIQUES = DOSSIER_SORTIE / "metriques_persistance.csv"

SEUIL = 0.50

# Lecture et validation des prédictions déjà calculées
df = pd.read_csv(FICHIER_ENTREE)

colonnes_requises = {
    "scenario_id",
    "temps_debut_s",
    "target_bearing_fault",
    "probabilite_defaut",
}

manquantes = colonnes_requises - set(df.columns)
if manquantes:
    raise ValueError(f"Colonnes manquantes : {sorted(manquantes)}")

df = df.sort_values(
    ["scenario_id", "temps_debut_s"]
).reset_index(drop=True)

# La règle de référence est la prédiction par fenêtre existante.
df["prediction_reference"] = (
    df["probabilite_defaut"] >= SEUIL
).astype(int)

# Calcul des règles temporelles séparément pour chaque scénario.
# Les fenêtres ne sont jamais mélangées entre deux scénarios.
df["prediction_2_sur_3"] = 0
df["prediction_3_sur_3"] = 0

for scenario_id, indices in df.groupby("scenario_id", sort=False).groups.items():
    indices = list(indices)
    probabilites = df.loc[indices, "probabilite_defaut"].to_numpy()
    decisions = (probabilites >= SEUIL).astype(int)

    for position, index in enumerate(indices):
        debut = max(0, position - 2)
        historique = decisions[debut:position + 1]

        # 2 sur 3 : au moins deux décisions positives parmi
        # les trois dernières fenêtres disponibles.
        df.loc[index, "prediction_2_sur_3"] = int(
            historique.sum() >= 2
        )

        # 3 sur 3 : les trois dernières fenêtres doivent être positives.
        df.loc[index, "prediction_3_sur_3"] = int(
            len(historique) == 3 and historique.sum() == 3
        )

# Calcul des métriques
regles = {
    "reference_seuil_050": "prediction_reference",
    "persistance_2_sur_3": "prediction_2_sur_3",
    "persistance_3_sur_3": "prediction_3_sur_3",
}

resultats = []

y_true = df["target_bearing_fault"].astype(int)

for nom, colonne in regles.items():
    y_pred = df[colonne].astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    resultats.append({
        "regle": nom,
        "seuil": SEUIL,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_defaut": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "rappel_defaut": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1_defaut": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "vrais_negatifs": tn,
        "fausses_alertes": fp,
        "defauts_manques": fn,
        "vrais_positifs": tp,
    })

# Sauvegarde dans un dossier distinct
DOSSIER_SORTIE.mkdir(parents=True, exist_ok=True)
df.to_csv(FICHIER_DETAIL, index=False)
pd.DataFrame(resultats).to_csv(FICHIER_METRIQUES, index=False)

print("Comparaison terminée.")
print(f"Nombre de fenêtres : {len(df)}")
print(f"Scénarios : {sorted(df['scenario_id'].unique().tolist())}")
print("\nMétriques comparatives :")
print(pd.DataFrame(resultats).round(4).to_string(index=False))
print(f"\nFichier détaillé : {FICHIER_DETAIL}")
print(f"Fichier des métriques : {FICHIER_METRIQUES}")
