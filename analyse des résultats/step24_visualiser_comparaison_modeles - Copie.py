
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# Chemins des fichiers
DOSSIER_PROJET = Path(__file__).resolve().parent
DOSSIER_RESULTATS = DOSSIER_PROJET / "resultats_step23_experimental"
FICHIER_ENTREE = DOSSIER_RESULTATS / "resume_validation_croisee.csv"
DOSSIER_SORTIE = DOSSIER_PROJET / "resultats_step24_experimental"

DOSSIER_SORTIE.mkdir(parents=True, exist_ok=True)

# Chargement des résultats de validation croisée
if not FICHIER_ENTREE.exists():
    raise FileNotFoundError(
        f"Fichier introuvable : {FICHIER_ENTREE}"
    )

df = pd.read_csv(FICHIER_ENTREE)

# Vérification des colonnes nécessaires
colonnes_requises = [
    "modele",
    "accuracy_mean",
    "balanced_accuracy_mean",
    "f1_defaut_mean",
    "recall_defaut_mean",
]

colonnes_manquantes = [
    colonne
    for colonne in colonnes_requises
    if colonne not in df.columns
]

if colonnes_manquantes:
    raise ValueError(
        f"Colonnes manquantes dans le CSV : {colonnes_manquantes}"
    )

# Conversion des métriques en valeurs numériques
for colonne in colonnes_requises[1:]:
    df[colonne] = pd.to_numeric(df[colonne], errors="raise")

# Vérification des valeurs
if df[colonnes_requises[1:]].isna().any().any():
    raise ValueError("Des valeurs manquantes existent dans les métriques.")

if not df[colonnes_requises[1:]].apply(
    lambda colonne: colonne.between(0, 1).all()
).all():
    raise ValueError("Les métriques doivent être comprises entre 0 et 1.")

# Graphique comparatif
metriques = {
    "accuracy_mean": "Accuracy",
    "balanced_accuracy_mean": "Balanced accuracy",
    "f1_defaut_mean": "F1 — défaut",
    "recall_defaut_mean": "Rappel — défaut",
}

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
axes = axes.flatten()

for axe, (colonne, titre) in zip(axes, metriques.items()):
    valeurs = df[colonne] * 100

    barres = axe.bar(
        df["modele"],
        valeurs,
        width=0.65,
    )

    axe.set_title(titre, fontsize=12, fontweight="bold")
    axe.set_ylabel("Score (%)")
    axe.set_ylim(0, 105)
    axe.tick_params(axis="x", rotation=12)
    axe.grid(axis="y", linestyle="--", alpha=0.35)
    axe.set_axisbelow(True)

    for barre, valeur in zip(barres, valeurs):
        axe.text(
            barre.get_x() + barre.get_width() / 2,
            barre.get_height() + 1,
            f"{valeur:.2f} %",
            ha="center",
            va="bottom",
            fontsize=9,
        )

fig.suptitle(
    "Comparaison des modèles — Validation croisée groupée",
    fontsize=15,
    fontweight="bold",
)

fig.text(
    0.5,
    0.015,
    "Données synthétiques — validation par scénario, 4 plis",
    ha="center",
    fontsize=10,
)

fig.tight_layout(rect=[0, 0.04, 1, 0.95])

# Enregistrement du graphique
fichier_graphique = DOSSIER_SORTIE / "comparaison_modeles_validation_croisee.png"
fig.savefig(fichier_graphique, dpi=300, bbox_inches="tight")
plt.close(fig)

print("=" * 70)
print("ÉTAPE 24 — VISUALISATION COMPARATIVE")
print("=" * 70)
print(f"Modèles comparés : {len(df)}")
print(f"Graphique enregistré : {fichier_graphique}")
print("Aucun fichier précédent n'a été modifié.")
