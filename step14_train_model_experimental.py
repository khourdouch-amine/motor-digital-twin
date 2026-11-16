
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# 1. Chemins et configuration
# ============================================================

dossier = Path("resultats_step14_experimental")
dossier.mkdir(parents=True, exist_ok=True)

fichier_train = dossier / "dataset_train.csv"
fichier_test = dossier / "dataset_test.csv"

fichier_modele = (
    dossier / "modele_detection_roulement_experimental.joblib"
)
fichier_metriques = dossier / "metriques_evaluation_experimental.csv"
fichier_matrice = dossier / "matrice_confusion_experimental.png"
fichier_importance = dossier / "importance_indicateurs_experimental.png"


# ============================================================
# 2. Variables et cible
# ============================================================

variables = [
    "rms_m_s2",
    "ecart_type_m_s2",
    "facteur_crete",
]

cible = "target_bearing_fault"


# ============================================================
# 3. Chargement des ensembles séparés par scénario
# ============================================================

for fichier in [fichier_train, fichier_test]:
    if not fichier.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {fichier}"
        )

train = pd.read_csv(fichier_train)
test = pd.read_csv(fichier_test)

colonnes_requises = [
    "scenario_id",
    cible,
    *variables,
]

for nom, ensemble in [
    ("entraînement", train),
    ("test", test),
]:
    absentes = [
        colonne
        for colonne in colonnes_requises
        if colonne not in ensemble.columns
    ]

    if absentes:
        raise ValueError(
            f"Colonnes absentes dans {nom} : {absentes}"
        )

    if ensemble[colonnes_requises].isna().any().any():
        raise ValueError(
            f"Valeurs manquantes dans l'ensemble {nom}."
        )

    if not set(ensemble[cible].unique()).issubset({0, 1}):
        raise ValueError(
            f"La cible de {nom} doit contenir uniquement 0 et 1."
        )

    if set(ensemble[cible].unique()) != {0, 1}:
        raise ValueError(
            f"L'ensemble {nom} doit contenir les deux classes."
        )


# ============================================================
# 4. Vérification de l'absence de fuite par scénario
# ============================================================

ids_train = set(train["scenario_id"].unique())
ids_test = set(test["scenario_id"].unique())

if ids_train.intersection(ids_test):
    raise ValueError(
        "Fuite détectée : un scénario apparaît dans les deux ensembles."
    )

X_train = train[variables]
y_train = train[cible].astype(int)

X_test = test[variables]
y_test = test[cible].astype(int)

print("=== DIMENSIONS DES DONNÉES EXPÉRIMENTALES ===")
print(f"Entraînement : {len(X_train)} fenêtres")
print(f"Test         : {len(X_test)} fenêtres")
print(f"Variables    : {variables}")
print(f"Scénarios d'entraînement : {sorted(ids_train)}")
print(f"Scénarios de test        : {sorted(ids_test)}")
print("Chevauchement des scénarios : 0")


# ============================================================
# 5. Entraînement de la forêt aléatoire
# ============================================================

modele = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

modele.fit(X_train, y_train)

print("\nEntraînement terminé.")


# ============================================================
# 6. Prédictions sur le test
# ============================================================

y_pred = modele.predict(X_test)
probabilites = modele.predict_proba(X_test)[:, 1]


# ============================================================
# 7. Calcul des métriques
# ============================================================

metriques = {
    "accuracy": accuracy_score(y_test, y_pred),
    "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
    "precision_defaut": precision_score(
        y_test, y_pred, zero_division=0
    ),
    "recall_defaut": recall_score(
        y_test, y_pred, zero_division=0
    ),
    "f1_defaut": f1_score(
        y_test, y_pred, zero_division=0
    ),
    "roc_auc": roc_auc_score(y_test, probabilites),
}

print("\n=== MÉTRIQUES SUR LE TEST EXPÉRIMENTAL ===")
for nom, valeur in metriques.items():
    print(f"{nom:22s} : {valeur:.4f}")

print("\n=== RAPPORT DE CLASSIFICATION ===")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Sans défaut", "Défaut de roulement"],
        zero_division=0,
    )
)


# ============================================================
# 8. Matrice de confusion
# ============================================================

matrice = confusion_matrix(y_test, y_pred, labels=[0, 1])

print("=== MATRICE DE CONFUSION ===")
print("Lignes = classes réelles ; colonnes = classes prédites")
print(matrice)

affichage = ConfusionMatrixDisplay(
    confusion_matrix=matrice,
    display_labels=["Sans défaut", "Défaut"],
)

affichage.plot(values_format="d")
plt.title("Détection du défaut de roulement — test expérimental")
plt.tight_layout()
plt.savefig(fichier_matrice, dpi=200)
plt.close()


# ============================================================
# 9. Importance des indicateurs
# ============================================================

importances = pd.DataFrame({
    "indicateur": variables,
    "importance": modele.feature_importances_,
}).sort_values("importance", ascending=False)

print("\n=== IMPORTANCE DES INDICATEURS ===")
print(importances.to_string(index=False))

plt.figure(figsize=(8, 5))
plt.barh(
    importances["indicateur"][::-1],
    importances["importance"][::-1],
)
plt.xlabel("Importance selon la forêt aléatoire")
plt.title("Importance des indicateurs vibratoires expérimentaux")
plt.tight_layout()
plt.savefig(fichier_importance, dpi=200)
plt.close()


# ============================================================
# 10. Sauvegarde du modèle et des résultats
# ============================================================

joblib.dump(modele, fichier_modele)

pd.DataFrame([
    {"metrique": nom, "valeur": valeur}
    for nom, valeur in metriques.items()
]).to_csv(fichier_metriques, index=False)

print("\n=== FICHIERS ENREGISTRÉS ===")
print(f"Modèle               : {fichier_modele}")
print(f"Métriques            : {fichier_metriques}")
print(f"Matrice de confusion : {fichier_matrice}")
print(f"Importance           : {fichier_importance}")

print(
    "\nAttention : les données sont synthétiques. "
    "Les performances évaluent uniquement les scénarios simulés "
    "retenus pour le test et ne démontrent pas une performance "
    "industrielle réelle."
)
