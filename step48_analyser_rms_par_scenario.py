from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Définir les chemins des fichiers
DOSSIER = Path(__file__).resolve().parent
FICHIER_ENTREE = (
    DOSSIER
    / "resultats_step47_experimental"
    / "predictions_detaillees.csv"
)
DOSSIER_SORTIE = DOSSIER / "resultats_step48_experimental"
DOSSIER_SORTIE.mkdir(parents=True, exist_ok=True)

# Charger les prédictions du modèle RMS seul
df = pd.read_csv(FICHIER_ENTREE)
df = df[df["modele"] == "RMS_seul"].copy()

if df.empty:
    raise ValueError("Aucune prédiction du modèle RMS_seul n'a été trouvée.")

# Vérifier les colonnes nécessaires
colonnes = [
    "scenario_id",
    "target_bearing_fault",
    "prediction_seuil_050",
]
manquantes = [c for c in colonnes if c not in df.columns]

if manquantes:
    raise ValueError(f"Colonnes manquantes : {manquantes}")

# Calculer les métriques pour chaque scénario
resultats = []

for scenario_id, groupe in df.groupby("scenario_id"):
    y_reel = groupe["target_bearing_fault"].astype(int)
    y_pred = groupe["prediction_seuil_050"].astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_reel, y_pred, labels=[0, 1]
    ).ravel()

    resultats.append({
        "scenario_id": int(scenario_id),
        "nombre_fenetres": len(groupe),
        "nombre_fenetres_defaut": int(y_reel.sum()),
        "accuracy": accuracy_score(y_reel, y_pred),
        "precision": precision_score(
            y_reel, y_pred, zero_division=0
        ),
        "rappel": recall_score(
            y_reel, y_pred, zero_division=0
        ),
        "f1": f1_score(y_reel, y_pred, zero_division=0),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    })

resultats_df = pd.DataFrame(resultats).sort_values("scenario_id")

# Enregistrer les résultats
fichier_sortie = DOSSIER_SORTIE / "performances_par_scenario.csv"
resultats_df.to_csv(fichier_sortie, index=False)

# Afficher les résultats
print("=" * 72)
print("ETAPE 48 — PERFORMANCES DU MODELE RMS SEUL PAR SCENARIO")
print("=" * 72)
print(f"Nombre total de fenetres : {len(df)}")
print(f"Nombre de scenarios : {resultats_df['scenario_id'].nunique()}")
print()
print(resultats_df.round(4).to_string(index=False))
print()
print(f"Resultats enregistres dans : {DOSSIER_SORTIE.resolve()}")
print("Les donnees utilisees restent entierement synthetiques.")
