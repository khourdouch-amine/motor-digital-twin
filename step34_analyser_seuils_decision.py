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

# Fichiers d'entrée et de sortie
FICHIER_ENTREE = Path(
    "resultats_step20_experimental/predictions_completes_test.csv"
)
DOSSIER_SORTIE = Path("resultats_step34_experimental")
FICHIER_METRIQUES = DOSSIER_SORTIE / "analyse_seuils_test.csv"
FICHIER_PREDICTIONS = DOSSIER_SORTIE / "predictions_par_seuil.csv"

# Seuils à examiner
SEUILS = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]

# Charger les probabilités déjà calculées
df = pd.read_csv(FICHIER_ENTREE)

colonnes_requises = {
    "scenario_id",
    "target_bearing_fault",
    "probabilite_defaut",
}
manquantes = colonnes_requises - set(df.columns)

if manquantes:
    raise ValueError(f"Colonnes manquantes : {sorted(manquantes)}")

y_true = df["target_bearing_fault"].astype(int)
lignes_metriques = []
predictions = df.copy()

# Évaluer chaque seuil sur les mêmes fenêtres
for seuil in SEUILS:
    colonne = f"prediction_seuil_{int(seuil * 100):02d}"
    y_pred = (df["probabilite_defaut"] >= seuil).astype(int)
    predictions[colonne] = y_pred

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    lignes_metriques.append({
        "seuil": seuil,
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

# Sauvegarder dans un nouveau dossier
DOSSIER_SORTIE.mkdir(parents=True, exist_ok=True)

metriques = pd.DataFrame(lignes_metriques)
metriques.to_csv(FICHIER_METRIQUES, index=False)
predictions.to_csv(FICHIER_PREDICTIONS, index=False)

print("Analyse des seuils terminée.")
print(f"Nombre de fenêtres : {len(df)}")
print(f"Scénarios de test : {sorted(df['scenario_id'].unique().tolist())}")
print("\nRésultats comparatifs :")
print(metriques.round(4).to_string(index=False))
print(f"\nMétriques : {FICHIER_METRIQUES}")
print(f"Prédictions : {FICHIER_PREDICTIONS}")
