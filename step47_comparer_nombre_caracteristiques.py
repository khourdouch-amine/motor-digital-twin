from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# ============================================================
# 1. Chargement des données
# ============================================================

BASE = Path(__file__).resolve().parent
SORTIE = BASE / "resultats_step47_experimental"
SORTIE.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(
    BASE / "resultats_step14_experimental" / "dataset_train.csv"
)
test = pd.read_csv(
    BASE / "resultats_step14_experimental" / "dataset_test.csv"
)

# ============================================================
# 2. Vérifications
# ============================================================

ids_train = set(train["scenario_id"].unique())
ids_test = set(test["scenario_id"].unique())

if ids_train & ids_test:
    raise ValueError("Chevauchement détecté entre train et test.")

cible = "target_bearing_fault"
y_train = train[cible].astype(int)
y_test = test[cible].astype(int)

groupes = {
    "RMS_seul": ["rms_m_s2"],
    "RMS_et_ecart_type": ["rms_m_s2", "ecart_type_m_s2"],
    "RMS_et_facteur_crete": ["rms_m_s2", "facteur_crete"],
    "Reference_trois_variables": [
        "rms_m_s2",
        "ecart_type_m_s2",
        "facteur_crete",
    ],
}

# ============================================================
# 3. Comparaison des modèles
# ============================================================

resultats = []
predictions_detaillees = []

for nom, variables in groupes.items():
    modele = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    modele.fit(train[variables], y_train)

    probabilites = modele.predict_proba(test[variables])[:, 1]
    predictions = (probabilites >= 0.50).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    resultats.append({
        "modele": nom,
        "variables": ", ".join(variables),
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "rappel": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilites),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    })

    detail = test[
        ["scenario_id", "temps_debut_s", cible]
    ].copy()
    detail.insert(0, "modele", nom)
    detail["probabilite_defaut"] = probabilites
    detail["prediction_seuil_050"] = predictions
    predictions_detaillees.append(detail)

# ============================================================
# 4. Sauvegarde et affichage
# ============================================================

df_resultats = pd.DataFrame(resultats).sort_values(
    "f1", ascending=False
)

df_resultats.to_csv(
    SORTIE / "comparaison_nombre_caracteristiques.csv",
    index=False,
    encoding="utf-8-sig",
)

pd.concat(predictions_detaillees, ignore_index=True).to_csv(
    SORTIE / "predictions_detaillees.csv",
    index=False,
    encoding="utf-8-sig",
)

print("=" * 68)
print("ÉTAPE 47 — LE RMS SUFFIT-IL À LA DÉTECTION ?")
print("=" * 68)
print("Fenêtres train :", len(train))
print("Fenêtres test :", len(test))
print("\nRésultats sur le test indépendant :")
print(
    df_resultats[
        ["modele", "accuracy", "balanced_accuracy", "precision",
         "rappel", "f1", "roc_auc", "FP", "FN"]
    ].round(4).to_string(index=False)
)
print("\nRésultats enregistrés dans :", SORTIE)
print("Les données utilisées restent entièrement synthétiques.")
