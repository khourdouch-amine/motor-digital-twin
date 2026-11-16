from pathlib import Path

import numpy as np
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
# 1. Chemins des fichiers
# ============================================================

BASE = Path(__file__).resolve().parent
SORTIE = BASE / "resultats_step45_experimental"
SORTIE.mkdir(parents=True, exist_ok=True)

F_SOURCE = BASE / "scenarios_step9_experimental" / "dataset_scenarios_complet.csv"
F_TRAIN = BASE / "resultats_step14_experimental" / "dataset_train.csv"
F_TEST = BASE / "resultats_step14_experimental" / "dataset_test.csv"

for fichier in [F_SOURCE, F_TRAIN, F_TEST]:
    if not fichier.exists():
        raise FileNotFoundError(f"Fichier introuvable : {fichier}")

source = pd.read_csv(F_SOURCE)
train = pd.read_csv(F_TRAIN)
test = pd.read_csv(F_TEST)

# ============================================================
# 2. Vérifications initiales
# ============================================================

colonnes_source = {
    "scenario_id", "time_s", "vibration_sensor_m_s2",
    "bearing_fault_active",
}
colonnes_dataset = {
    "scenario_id", "temps_debut_s", "target_bearing_fault",
    "rms_m_s2", "ecart_type_m_s2", "facteur_crete",
}

if not colonnes_source.issubset(source.columns):
    raise ValueError(
        f"Colonnes sources absentes : {colonnes_source - set(source.columns)}"
    )

for nom, df in [("train", train), ("test", test)]:
    if not colonnes_dataset.issubset(df.columns):
        raise ValueError(
            f"Colonnes absentes du dataset {nom} : "
            f"{colonnes_dataset - set(df.columns)}"
        )

ids_train = set(train["scenario_id"].unique())
ids_test = set(test["scenario_id"].unique())

if ids_train & ids_test:
    raise ValueError("Des scénarios apparaissent à la fois dans train et test.")

if train.isna().any().any() or test.isna().any().any():
    raise ValueError("Valeurs manquantes détectées dans train ou test.")

source = source.sort_values(["scenario_id", "time_s"])

# ============================================================
# 3. Calcul des caractéristiques sur des fenêtres de 5 secondes
# ============================================================

fenetres = pd.concat(
    [
        train[["scenario_id", "temps_debut_s"]],
        test[["scenario_id", "temps_debut_s"]],
    ],
    ignore_index=True,
).drop_duplicates()

caracteristiques = []

for ligne in fenetres.itertuples(index=False):
    scenario_id = int(ligne.scenario_id)
    debut = float(ligne.temps_debut_s)
    fin = debut + 5.0

    masque = (
        (source["scenario_id"] == scenario_id)
        & (source["time_s"] >= debut)
        & (source["time_s"] < fin)
    )
    signal = source.loc[
        masque, "vibration_sensor_m_s2"
    ].to_numpy(dtype=float)

    if len(signal) != 100:
        raise ValueError(
            f"Fenêtre invalide : scénario {scenario_id}, "
            f"t={debut}, nombre d'échantillons={len(signal)}."
        )

    centre = signal - np.mean(signal)
    rms_brut = float(np.sqrt(np.mean(signal ** 2)))
    rms_centre = float(np.sqrt(np.mean(centre ** 2)))

    facteur_brut = (
        float(np.max(np.abs(signal)) / rms_brut)
        if rms_brut > 0 else 0.0
    )
    facteur_centre = (
        float(np.max(np.abs(centre)) / rms_centre)
        if rms_centre > 0 else 0.0
    )

    caracteristiques.append({
        "scenario_id": scenario_id,
        "temps_debut_s": debut,
        "rms_brut": rms_brut,
        "rms_centre": rms_centre,
        "ecart_type": float(np.std(signal)),
        "facteur_crete_brut": facteur_brut,
        "facteur_crete_centre": facteur_centre,
        "amplitude_crete_a_crete": float(np.ptp(signal)),
    })

features = pd.DataFrame(caracteristiques)

# Fusion contrôlée : une seule ligne par fenêtre
train = train.merge(
    features,
    on=["scenario_id", "temps_debut_s"],
    how="left",
    validate="one_to_one",
)
test = test.merge(
    features,
    on=["scenario_id", "temps_debut_s"],
    how="left",
    validate="one_to_one",
)

nouvelles_colonnes = [
    "rms_brut", "rms_centre", "ecart_type",
    "facteur_crete_centre", "amplitude_crete_a_crete",
]

if train[nouvelles_colonnes].isna().any().any():
    raise ValueError("Caractéristiques manquantes dans train après fusion.")
if test[nouvelles_colonnes].isna().any().any():
    raise ValueError("Caractéristiques manquantes dans test après fusion.")

# ============================================================
# 4. Groupes de caractéristiques à comparer
# ============================================================

groupes = {
    "A_reference": [
        "rms_m_s2", "ecart_type_m_s2", "facteur_crete",
    ],
    "B_signal_centre": [
        "rms_centre", "ecart_type", "facteur_crete_centre",
    ],
    "C_complementaire": [
        "rms_brut", "ecart_type", "facteur_crete_centre",
        "amplitude_crete_a_crete",
    ],
}

y_train = train["target_bearing_fault"].astype(int)
y_test = test["target_bearing_fault"].astype(int)

resultats = []
predictions_detaillees = []

# ============================================================
# 5. Entraînement et évaluation
# ============================================================

for nom, colonnes in groupes.items():
    X_train = train[colonnes]
    X_test = test[colonnes]

    modele = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    modele.fit(X_train, y_train)

    probabilites = modele.predict_proba(X_test)[:, 1]
    predictions = (probabilites >= 0.50).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    resultats.append({
        "groupe": nom,
        "variables": ", ".join(colonnes),
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "rappel": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilites),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    })

    detail = test[
        ["scenario_id", "temps_debut_s", "target_bearing_fault"]
    ].copy()
    detail.insert(0, "groupe", nom)
    detail["probabilite_defaut"] = probabilites
    detail["prediction_seuil_050"] = predictions
    predictions_detaillees.append(detail)

# ============================================================
# 6. Sauvegarde des résultats
# ============================================================

df_resultats = pd.DataFrame(resultats).sort_values(
    "f1", ascending=False
)
df_resultats.to_csv(
    SORTIE / "comparaison_groupes_caracteristiques.csv",
    index=False,
    encoding="utf-8-sig",
)

pd.concat(predictions_detaillees, ignore_index=True).to_csv(
    SORTIE / "predictions_detaillees.csv",
    index=False,
    encoding="utf-8-sig",
)

features.to_csv(
    SORTIE / "caracteristiques_fenetres.csv",
    index=False,
    encoding="utf-8-sig",
)

# ============================================================
# 7. Rapport final
# ============================================================

print("=" * 70)
print("ÉTAPE 45 — COMPARAISON DES CARACTÉRISTIQUES")
print("=" * 70)
print(f"Fenêtres d'entraînement : {len(train)}")
print(f"Fenêtres de test : {len(test)}")
print(f"Scénarios train : {sorted(ids_train)}")
print(f"Scénarios test : {sorted(ids_test)}")
print("\nRésultats sur le test indépendant :")
print(
    df_resultats[
        ["groupe", "accuracy", "balanced_accuracy", "precision",
         "rappel", "f1", "roc_auc", "FP", "FN"]
    ].round(4).to_string(index=False)
)
print(f"\nRésultats enregistrés dans : {SORTIE}")
print(
    "Attention : les données sont synthétiques. "
    "Ces résultats ne démontrent pas une performance industrielle."
)
