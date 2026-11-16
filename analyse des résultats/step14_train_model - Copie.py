
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
from sklearn.utils.class_weight import compute_sample_weight


# ============================================================
# 1. Chemins et configuration
# ============================================================

dossier = Path("resultats_step14")
dossier.mkdir(parents=True, exist_ok=True)

fichier_train = dossier / "dataset_train.csv"
fichier_test = dossier / "dataset_test.csv"

fichier_modele = dossier / "modele_detection_roulement.joblib"
fichier_metriques = dossier / "metriques_evaluation.csv"
fichier_matrice = dossier / "matrice_confusion.png"
fichier_importance = dossier / "importance_indicateurs.png"

variables = [
    "rms_m_s2",
    "ecart_type_m_s2",
    "facteur_crete",
]

cible = "target_bearing_fault"


# ============================================================
# 2. Chargement des ensembles déjà séparés
# ============================================================

train = pd.read_csv(fichier_train)
test = pd.read_csv(fichier_test)

X_train = train[variables]
y_train = train[cible].astype(int)

X_test = test[variables]
y_test = test[cible].astype(int)

print("=== DIMENSIONS DES DONNÉES ===")
print(f"Entraînement : {X_train.shape[0]} fenêtres")
print(f"Test         : {X_test.shape[0]} fenêtres")
print(f"Variables    : {variables}")


# ============================================================
# 3. Vérifications avant entraînement
# ============================================================

if X_train.isna().any().any() or X_test.isna().any().any():
    raise ValueError("Les variables contiennent des valeurs manquantes.")

if set(y_train.unique()) != {0, 1}:
    raise ValueError("L'entraînement doit contenir les deux classes.")

if set(y_test.unique()) != {0, 1}:
    raise ValueError("Le test doit contenir les deux classes.")

ids_train = set(train["scenario_id"].unique())
ids_test = set(test["scenario_id"].unique())

if ids_train.intersection(ids_test):
    raise ValueError("Fuite détectée : un scénario apparaît dans les deux ensembles.")


# ============================================================
# 4. Entraînement de la forêt aléatoire
# ============================================================

modele = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

poids = compute_sample_weight(
    class_weight="balanced",
    y=y_train,
)

modele.fit(
    X_train,
    y_train,
    sample_weight=poids,
)

print("\nEntraînement terminé.")


# ============================================================
# 5. Prédictions sur les scénarios de test
# ============================================================

y_pred = modele.predict(X_test)
probabilites = modele.predict_proba(X_test)[:, 1]


# ============================================================
# 6. Calcul des métriques
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

print("\n=== MÉTRIQUES SUR LE TEST ===")
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
# 7. Matrice de confusion
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
plt.title("Détection du défaut de roulement — test par scénario")
plt.tight_layout()
plt.savefig(fichier_matrice, dpi=200)
plt.close()


# ============================================================
# 8. Importance des indicateurs
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
plt.title("Importance des indicateurs vibratoires")
plt.tight_layout()
plt.savefig(fichier_importance, dpi=200)
plt.close()


# ============================================================
# 9. Enregistrement du modèle et des résultats
# ============================================================

joblib.dump(modele, fichier_modele)

pd.DataFrame(
    [{"metrique": nom, "valeur": valeur}
     for nom, valeur in metriques.items()]
).to_csv(fichier_metriques, index=False)

print("\n=== FICHIERS ENREGISTRÉS ===")
print(f"Modèle              : {fichier_modele}")
print(f"Métriques           : {fichier_metriques}")
print(f"Matrice de confusion: {fichier_matrice}")
print(f"Importance          : {fichier_importance}")

print(
    "\nAttention : les données sont synthétiques. "
    "Les performances mesurent uniquement la capacité du modèle "
    "à distinguer les scénarios simulés retenus pour le test."
)
