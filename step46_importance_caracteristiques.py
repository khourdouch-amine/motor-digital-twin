from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, make_scorer

# ============================================================
# 1. Chemins
# ============================================================

BASE = Path(__file__).resolve().parent
SORTIE = BASE / "resultats_step46_experimental"
SORTIE.mkdir(parents=True, exist_ok=True)

F_TRAIN = BASE / "resultats_step14_experimental" / "dataset_train.csv"
F_TEST = BASE / "resultats_step14_experimental" / "dataset_test.csv"

train = pd.read_csv(F_TRAIN)
test = pd.read_csv(F_TEST)

# ============================================================
# 2. Variables de référence
# ============================================================

variables = [
    "rms_m_s2",
    "ecart_type_m_s2",
    "facteur_crete",
]
cible = "target_bearing_fault"

X_train = train[variables]
y_train = train[cible].astype(int)

X_test = test[variables]
y_test = test[cible].astype(int)

if set(train["scenario_id"]) & set(test["scenario_id"]):
    raise ValueError("Les scénarios d'entraînement et de test se chevauchent.")

# ============================================================
# 3. Entraînement du modèle de référence
# ============================================================

modele = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
modele.fit(X_train, y_train)

# ============================================================
# 4. Importance interne du modèle
# ============================================================

importance_interne = pd.DataFrame({
    "caracteristique": variables,
    "importance": modele.feature_importances_,
})

importance_interne = importance_interne.sort_values(
    "importance", ascending=False
)

importance_interne.to_csv(
    SORTIE / "importance_interne.csv",
    index=False,
    encoding="utf-8-sig",
)

# ============================================================
# 5. Importance par permutation sur le test indépendant
# ============================================================

permutation = permutation_importance(
    modele,
    X_test,
    y_test,
    scoring=make_scorer(f1_score, zero_division=0),
    n_repeats=30,
    random_state=42,
    n_jobs=-1,
)

importance_permutation = pd.DataFrame({
    "caracteristique": variables,
    "baisse_f1_moyenne": permutation.importances_mean,
    "ecart_type_baisse_f1": permutation.importances_std,
}).sort_values("baisse_f1_moyenne", ascending=False)

importance_permutation.to_csv(
    SORTIE / "importance_permutation_test.csv",
    index=False,
    encoding="utf-8-sig",
)

# ============================================================
# 6. Graphique
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].barh(
    importance_interne["caracteristique"],
    importance_interne["importance"],
)
axes[0].invert_yaxis()
axes[0].set_title("Importance interne")
axes[0].set_xlabel("Importance")

axes[1].barh(
    importance_permutation["caracteristique"],
    importance_permutation["baisse_f1_moyenne"],
    xerr=importance_permutation["ecart_type_baisse_f1"],
)
axes[1].invert_yaxis()
axes[1].set_title("Importance par permutation — test")
axes[1].set_xlabel("Baisse moyenne du F1")

for axe in axes:
    axe.grid(axis="x", alpha=0.3)

fig.tight_layout()
fig.savefig(
    SORTIE / "importance_caracteristiques.png",
    dpi=160,
    bbox_inches="tight",
)
plt.close(fig)

# ============================================================
# 7. Rapport
# ============================================================

print("=" * 65)
print("ÉTAPE 46 — IMPORTANCE DES CARACTÉRISTIQUES")
print("=" * 65)

print("\nImportance interne :")
print(importance_interne.round(5).to_string(index=False))

print("\nImportance par permutation sur le test indépendant :")
print(importance_permutation.round(5).to_string(index=False))

print("\nFichiers enregistrés dans :", SORTIE)
print(
    "Attention : l'importance dépend du modèle, des variables et "
    "des données synthétiques utilisées."
)
