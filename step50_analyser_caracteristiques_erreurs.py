from pathlib import Path

import pandas as pd

# Définir les chemins des fichiers
DOSSIER = Path(__file__).resolve().parent
FICHIER_TEST = DOSSIER / "resultats_step14_experimental" / "dataset_test.csv"
FICHIER_PREDICTIONS = (
    DOSSIER
    / "resultats_step47_experimental"
    / "predictions_detaillees.csv"
)
DOSSIER_SORTIE = DOSSIER / "resultats_step50_experimental"
DOSSIER_SORTIE.mkdir(parents=True, exist_ok=True)

# Charger les caractéristiques et les prédictions
test = pd.read_csv(FICHIER_TEST)
predictions = pd.read_csv(FICHIER_PREDICTIONS)

predictions = predictions[predictions["modele"] == "RMS_seul"].copy()

# Vérifier l'unicité des clés avant la fusion
cles = ["scenario_id", "temps_debut_s"]

if test.duplicated(cles).any():
    raise ValueError("Le jeu de test contient des clés dupliquées.")

if predictions.duplicated(cles).any():
    raise ValueError("Les prédictions contiennent des clés dupliquées.")

# Fusionner les données par scénario et fenêtre temporelle
df = test.merge(
    predictions[
        cles + ["probabilite_defaut", "prediction_seuil_050"]
    ],
    on=cles,
    how="left",
    validate="one_to_one",
    indicator=True,
)

if not (df["_merge"] == "both").all():
    raise ValueError("Certaines fenêtres n'ont pas de prédiction associée.")

df = df.drop(columns="_merge")

# Identifier les types de résultats
df["resultat"] = "Vrai negatif"
df.loc[
    (df["target_bearing_fault"] == 1)
    & (df["prediction_seuil_050"] == 1),
    "resultat",
] = "Vrai positif"
df.loc[
    (df["target_bearing_fault"] == 1)
    & (df["prediction_seuil_050"] == 0),
    "resultat",
] = "Faux negatif"
df.loc[
    (df["target_bearing_fault"] == 0)
    & (df["prediction_seuil_050"] == 1),
    "resultat",
] = "Faux positif"

# Extraire les erreurs et calculer les statistiques comparatives
indicateurs = ["rms_m_s2", "ecart_type_m_s2", "facteur_crete"]
erreurs = df[df["resultat"].isin(["Faux negatif", "Faux positif"])].copy()

colonnes_erreurs = [
    "scenario_id",
    "scenario_type",
    "temps_debut_s",
    "target_bearing_fault",
    *indicateurs,
    "probabilite_defaut",
    "prediction_seuil_050",
    "resultat",
]
erreurs = erreurs[colonnes_erreurs].sort_values(
    ["scenario_id", "temps_debut_s"]
)

# Comparer les moyennes des indicateurs selon le résultat
resume = (
    df.groupby("resultat")[indicateurs]
    .agg(["count", "mean", "std", "min", "max"])
    .round(5)
)

# Enregistrer les résultats
fichier_erreurs = DOSSIER_SORTIE / "caracteristiques_erreurs.csv"
fichier_resume = DOSSIER_SORTIE / "statistiques_par_resultat.csv"

erreurs.to_csv(fichier_erreurs, index=False)
resume.to_csv(fichier_resume)

# Afficher les résultats
print("=" * 76)
print("ETAPE 50 — CARACTERISTIQUES DES ERREURS DU MODELE RMS SEUL")
print("=" * 76)
print("\nFenêtres mal classées :")
print(erreurs.to_string(index=False))
print("\nStatistiques des indicateurs par résultat de classification :")
print(resume.to_string())
print(f"\nNombre de fenêtres analysées : {len(df)}")
print(f"Nombre d'erreurs : {len(erreurs)}")
print(f"\nFichiers enregistrés dans : {DOSSIER_SORTIE.resolve()}")
print("Les donnees utilisees restent entierement synthetiques.")
