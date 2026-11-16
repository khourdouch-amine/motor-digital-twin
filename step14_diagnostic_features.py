
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. Chargement des données d'entraînement et de test
# ============================================================

dossier = Path("resultats_step14")

train = pd.read_csv(dossier / "dataset_train.csv")
test = pd.read_csv(dossier / "dataset_test.csv")

df = pd.concat(
    [
        train.assign(ensemble="Entraînement"),
        test.assign(ensemble="Test"),
    ],
    ignore_index=True,
)

indicateurs = [
    "rms_m_s2",
    "ecart_type_m_s2",
    "facteur_crete",
]

cible = "target_bearing_fault"


# ============================================================
# 2. Comparaison statistique des deux classes
# ============================================================

print("=== STATISTIQUES PAR CLASSE ===")

resume = (
    df.groupby(cible)[indicateurs]
    .agg(["min", "mean", "median", "max", "std"])
    .round(4)
)

print(resume.to_string())


# ============================================================
# 3. Recherche d'un chevauchement des intervalles
# ============================================================

print("\n=== CHEVAUCHEMENT DES INTERVALLES MIN-MAX ===")

for indicateur in indicateurs:
    sans_defaut = df.loc[df[cible] == 0, indicateur]
    avec_defaut = df.loc[df[cible] == 1, indicateur]

    minimum_commun = max(
        sans_defaut.min(),
        avec_defaut.min(),
    )

    maximum_commun = min(
        sans_defaut.max(),
        avec_defaut.max(),
    )

    chevauchement = minimum_commun <= maximum_commun

    print(f"\nIndicateur : {indicateur}")
    print(
        f"Sans défaut : [{sans_defaut.min():.4f}, "
        f"{sans_defaut.max():.4f}]"
    )
    print(
        f"Avec défaut : [{avec_defaut.min():.4f}, "
        f"{avec_defaut.max():.4f}]"
    )
    print(f"Chevauchement min-max : {chevauchement}")


# ============================================================
# 4. Visualisation des distributions
# ============================================================

fig, axes = plt.subplots(
    nrows=1,
    ncols=3,
    figsize=(15, 5),
)

for axe, indicateur in zip(axes, indicateurs):
    donnees = [
        df.loc[df[cible] == 0, indicateur],
        df.loc[df[cible] == 1, indicateur],
    ]

    axe.boxplot(
        donnees,
        tick_labels=["Sans défaut", "Défaut actif"],
    )

    axe.set_title(indicateur)
    axe.set_ylabel("Valeur")
    axe.grid(axis="y", alpha=0.3)

plt.suptitle(
    "Distribution des indicateurs selon l'état du roulement"
)
plt.tight_layout()

fichier_figure = dossier / "diagnostic_distributions.png"
plt.savefig(fichier_figure, dpi=200)
plt.close()

print(f"\nGraphique enregistré : {fichier_figure}")
print(
    "\nImportant : les données sont synthétiques. "
    "Un chevauchement absent ou faible peut refléter les règles "
    "de génération des défauts plutôt qu'une séparation réaliste."
)
