from pathlib import Path

import pandas as pd

# Définir les chemins
DOSSIER = Path(__file__).resolve().parent
FICHIER_ENTREE = (
    DOSSIER
    / "resultats_step47_experimental"
    / "predictions_detaillees.csv"
)
DOSSIER_SORTIE = DOSSIER / "resultats_step49_experimental"
DOSSIER_SORTIE.mkdir(parents=True, exist_ok=True)

# Charger les prédictions du modèle RMS seul
df = pd.read_csv(FICHIER_ENTREE)
df = df[df["modele"] == "RMS_seul"].copy()

# Séparer les erreurs de classification
df["type_erreur"] = ""

df.loc[
    (df["target_bearing_fault"] == 1)
    & (df["prediction_seuil_050"] == 0),
    "type_erreur",
] = "Faux negatif"

df.loc[
    (df["target_bearing_fault"] == 0)
    & (df["prediction_seuil_050"] == 1),
    "type_erreur",
] = "Faux positif"

erreurs = df[df["type_erreur"] != ""].copy()
erreurs = erreurs.sort_values(["scenario_id", "temps_debut_s"])

# Enregistrer les erreurs
fichier_sortie = DOSSIER_SORTIE / "erreurs_modele_rms.csv"
erreurs.to_csv(fichier_sortie, index=False)

# Afficher le bilan
print("=" * 72)
print("ETAPE 49 — ANALYSE DES ERREURS DU MODELE RMS SEUL")
print("=" * 72)
print(f"Nombre total de fenetres : {len(df)}")
print(f"Nombre d'erreurs : {len(erreurs)}")
print()

if erreurs.empty:
    print("Aucune erreur de classification.")
else:
    print(erreurs[
        [
            "scenario_id",
            "temps_debut_s",
            "target_bearing_fault",
            "probabilite_defaut",
            "prediction_seuil_050",
            "type_erreur",
        ]
    ].to_string(index=False))

print()
print("Bilan par type d'erreur :")
print(erreurs["type_erreur"].value_counts().to_string())
print()
print(f"Resultats enregistres dans : {DOSSIER_SORTIE.resolve()}")
print("Les donnees utilisees restent entierement synthetiques.")
