
"""
Étape 21 — Analyse des indicateurs vibratoires associés aux erreurs
Projet : Jumeau numérique d'un moteur électrique

Compare les indicateurs entre :
- vrais négatifs ;
- fausses alertes ;
- défauts manqués ;
- vrais positifs.

Les fichiers des étapes précédentes ne sont pas modifiés.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# 1. CHEMINS
# ============================================================

DOSSIER = Path(__file__).resolve().parent

FICHIER_ENTREE = (
    DOSSIER
    / "resultats_step20_experimental"
    / "predictions_completes_test.csv"
)

DOSSIER_SORTIE = (
    DOSSIER
    / "resultats_step21_experimental"
)

DOSSIER_SORTIE.mkdir(parents=True, exist_ok=True)

INDICATEURS = [
    "rms_m_s2",
    "ecart_type_m_s2",
    "facteur_crete",
]

ORDRE_RESULTATS = [
    "vrai_negatif",
    "fausse_alerte",
    "defaut_manque",
    "vrai_positif",
]

NOMS_RESULTATS = {
    "vrai_negatif": "Vrais négatifs",
    "fausse_alerte": "Fausses alertes",
    "defaut_manque": "Défauts manqués",
    "vrai_positif": "Vrais positifs",
}


# ============================================================
# 2. CHARGEMENT ET CONTRÔLES
# ============================================================

print("=" * 70)
print("ÉTAPE 21 — ANALYSE DES INDICATEURS VIBRATOIRES")
print("=" * 70)

if not FICHIER_ENTREE.exists():
    raise FileNotFoundError(
        f"Fichier introuvable : {FICHIER_ENTREE}"
    )

df = pd.read_csv(FICHIER_ENTREE)

colonnes_requises = INDICATEURS + [
    "type_resultat",
    "scenario_id",
    "target_bearing_fault",
    "prediction",
    "probabilite_defaut",
]

manquantes = [
    colonne
    for colonne in colonnes_requises
    if colonne not in df.columns
]

if manquantes:
    raise ValueError(
        f"Colonnes manquantes : {manquantes}"
    )

if df[INDICATEURS].isna().any().any():
    raise ValueError(
        "Les indicateurs contiennent des valeurs manquantes."
    )

print(f"\nNombre de fenêtres analysées : {len(df)}")


# ============================================================
# 3. EFFECTIFS PAR TYPE DE RÉSULTAT
# ============================================================

print("\n" + "=" * 70)
print("EFFECTIFS PAR TYPE DE RÉSULTAT")
print("=" * 70)

effectifs = (
    df["type_resultat"]
    .value_counts()
    .reindex(ORDRE_RESULTATS, fill_value=0)
)

for resultat, nombre in effectifs.items():
    print(
        f"{NOMS_RESULTATS.get(resultat, resultat)} : {nombre}"
    )


# ============================================================
# 4. STATISTIQUES DES INDICATEURS
# ============================================================

statistiques = df.groupby(
    "type_resultat"
)[INDICATEURS].agg(
    ["count", "mean", "std", "min", "median", "max"]
)

statistiques = statistiques.reindex(ORDRE_RESULTATS)

print("\n" + "=" * 70)
print("STATISTIQUES DES INDICATEURS")
print("=" * 70)
print(statistiques.round(4).to_string())

fichier_statistiques = (
    DOSSIER_SORTIE
    / "statistiques_indicateurs_par_resultat.csv"
)

statistiques.to_csv(
    fichier_statistiques,
    encoding="utf-8-sig",
)


# ============================================================
# 5. TABLEAU COMPACT DES MOYENNES
# ============================================================

moyennes = (
    df.groupby("type_resultat")[INDICATEURS]
    .mean()
    .reindex(ORDRE_RESULTATS)
)

moyennes.index = [
    NOMS_RESULTATS.get(valeur, valeur)
    for valeur in moyennes.index
]

print("\n" + "=" * 70)
print("MOYENNES DES INDICATEURS")
print("=" * 70)
print(moyennes.round(4).to_string())

fichier_moyennes = (
    DOSSIER_SORTIE
    / "moyennes_indicateurs_par_resultat.csv"
)

moyennes.to_csv(
    fichier_moyennes,
    encoding="utf-8-sig",
)


# ============================================================
# 6. GRAPHIQUES COMPARATIFS
# ============================================================

noms_indicateurs = {
    "rms_m_s2": "RMS (m/s²)",
    "ecart_type_m_s2": "Écart-type (m/s²)",
    "facteur_crete": "Facteur de crête",
}

figure, axes = plt.subplots(
    1,
    3,
    figsize=(16, 5),
)

for axe, indicateur in zip(axes, INDICATEURS):
    donnees = [
        df.loc[
            df["type_resultat"] == resultat,
            indicateur,
        ].dropna().to_numpy()
        for resultat in ORDRE_RESULTATS
    ]

    axe.boxplot(
        donnees,
        tick_labels=[
            "Vrais\nnégatifs",
            "Fausses\nalertes",
            "Défauts\nmanqués",
            "Vrais\npositifs",
        ],
        showmeans=True,
    )

    axe.set_title(
        noms_indicateurs[indicateur]
    )
    axe.grid(
        axis="y",
        alpha=0.3,
    )

figure.suptitle(
    "Distribution des indicateurs selon le résultat de classification"
)

figure.tight_layout()

fichier_graphique = (
    DOSSIER_SORTIE
    / "comparaison_indicateurs_erreurs.png"
)

figure.savefig(
    fichier_graphique,
    dpi=200,
    bbox_inches="tight",
)

plt.close(figure)


# ============================================================
# 7. DÉTAIL DES ERREURS
# ============================================================

erreurs = df[
    df["type_resultat"].isin(
        ["fausse_alerte", "defaut_manque"]
    )
].copy()

colonnes_erreurs = [
    "scenario_id",
    "scenario_type",
    "temps_debut_s",
    "target_bearing_fault",
    "prediction",
    "probabilite_defaut",
] + INDICATEURS

colonnes_erreurs = [
    colonne
    for colonne in colonnes_erreurs
    if colonne in erreurs.columns
]

print("\n" + "=" * 70)
print("DÉTAIL DES ERREURS")
print("=" * 70)

if erreurs.empty:
    print("Aucune erreur détectée.")
else:
    print(
        erreurs[colonnes_erreurs].to_string(
            index=False,
            float_format=lambda valeur: f"{valeur:.4f}",
        )
    )

fichier_erreurs = (
    DOSSIER_SORTIE
    / "details_erreurs.csv"
)

erreurs[colonnes_erreurs].to_csv(
    fichier_erreurs,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 8. RÉSUMÉ
# ============================================================

print("\n" + "=" * 70)
print("FICHIERS ENREGISTRÉS")
print("=" * 70)

print(f"Statistiques détaillées : {fichier_statistiques}")
print(f"Moyennes                : {fichier_moyennes}")
print(f"Graphique               : {fichier_graphique}")
print(f"Détails des erreurs     : {fichier_erreurs}")

print(
    "\nAttention : les données sont synthétiques. "
    "Les résultats ne constituent pas une validation industrielle."
)
