import pandas as pd
from pathlib import Path

# Charger les données existantes
fichier = Path("resultats_step34_experimental/predictions_par_seuil.csv")
df = pd.read_csv(fichier)
df = df.sort_values(["scenario_id", "temps_debut_s"]).copy()

# Calculer les variations de probabilité dans chaque scénario
df["variation_probabilite"] = (
    df.groupby("scenario_id")["probabilite_defaut"].diff().abs()
)

# Identifier les prédictions erronées au seuil de référence
df["erreur_seuil_50"] = (
    df["target_bearing_fault"] != df["prediction_seuil_50"]
).astype(int)

# Résumer les indicateurs par type de résultat
df["groupe_resultat"] = "Vrai négatif"
df.loc[
    (df["target_bearing_fault"] == 0) &
    (df["prediction_seuil_50"] == 1),
    "groupe_resultat"
] = "Fausse alerte"
df.loc[
    (df["target_bearing_fault"] == 1) &
    (df["prediction_seuil_50"] == 0),
    "groupe_resultat"
] = "Défaut manqué"
df.loc[
    (df["target_bearing_fault"] == 1) &
    (df["prediction_seuil_50"] == 1),
    "groupe_resultat"
] = "Vrai positif"

indicateurs = [
    "rms_m_s2",
    "ecart_type_m_s2",
    "facteur_crete",
    "probabilite_defaut",
    "variation_probabilite",
]

resume = df.groupby("groupe_resultat")[indicateurs].agg(
    ["count", "mean", "std", "min", "max"]
)

# Créer un dossier de résultats séparé
dossier = Path("resultats_step37_experimental")
dossier.mkdir(parents=True, exist_ok=True)

# Enregistrer les résultats
df.to_csv(
    dossier / "donnees_avec_variations.csv",
    index=False,
    encoding="utf-8-sig"
)
resume.to_csv(
    dossier / "resume_indicateurs_par_resultat.csv",
    encoding="utf-8-sig"
)

print("Résumé des indicateurs par résultat au seuil de 0,50")
print(resume.to_string())
print("\nNombre de variations de probabilité supérieures ou égales à 0,50 :",
      int((df["variation_probabilite"] >= 0.50).sum()))
print("\nFichiers enregistrés dans :", dossier)
