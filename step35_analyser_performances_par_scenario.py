import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score

# Définition des chemins
fichier_entree = Path("resultats_step34_experimental/predictions_par_seuil.csv")
dossier_sortie = Path("resultats_step35_experimental")
dossier_sortie.mkdir(parents=True, exist_ok=True)

# Chargement des prédictions
df = pd.read_csv(fichier_entree)

# Vérification des colonnes nécessaires
colonnes_requises = [
    "scenario_id",
    "scenario_type",
    "target_bearing_fault",
    "prediction_seuil_20",
    "prediction_seuil_50"
]

for colonne in colonnes_requises:
    if colonne not in df.columns:
        raise ValueError(f"Colonne manquante : {colonne}")

# Calcul des métriques par scénario et par seuil
resultats = []

for (scenario_id, scenario_type), groupe in df.groupby(
    ["scenario_id", "scenario_type"], sort=True
):
    y_true = groupe["target_bearing_fault"].astype(int)

    for seuil, colonne_prediction in [
        (0.20, "prediction_seuil_20"),
        (0.50, "prediction_seuil_50")
    ]:
        y_pred = groupe[colonne_prediction].astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).ravel()

        resultats.append({
            "scenario_id": scenario_id,
            "scenario_type": scenario_type,
            "seuil": seuil,
            "nombre_fenetres": len(groupe),
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
            "vrais_positifs": tp
        })

# Enregistrement des résultats
resultats_df = pd.DataFrame(resultats)
fichier_sortie = dossier_sortie / "metriques_par_scenario_et_seuil.csv"
resultats_df.to_csv(fichier_sortie, index=False, encoding="utf-8-sig")

# Affichage d'un tableau lisible
colonnes_affichage = [
    "scenario_id", "scenario_type", "seuil",
    "accuracy", "balanced_accuracy", "precision_defaut",
    "rappel_defaut", "f1_defaut",
    "fausses_alertes", "defauts_manques"
]

print("\nANALYSE PAR SCENARIO ET PAR SEUIL")
print(resultats_df[colonnes_affichage].to_string(
    index=False, float_format=lambda x: f"{x:.4f}"
))
print(f"\nFichier sauvegarde : {fichier_sortie}")
