import pandas as pd
from pathlib import Path

# Chargement des prédictions existantes
fichier = Path("resultats_step34_experimental/predictions_par_seuil.csv")
df = pd.read_csv(fichier)

# Fenêtres à examiner et intervalle temporel autour de chaque erreur
erreurs = [
    (14, 185),
    (14, 190),
    (14, 235),
    (20, 255),
    (20, 560),
]

colonnes = [
    "scenario_id",
    "scenario_type",
    "temps_debut_s",
    "target_bearing_fault",
    "probabilite_defaut",
    "prediction_seuil_20",
    "prediction_seuil_50",
    "rms_m_s2",
    "ecart_type_m_s2",
    "facteur_crete",
]

resultats = []

for scenario_id, temps_erreur in erreurs:
    masque = (
        (df["scenario_id"] == scenario_id)
        & (df["temps_debut_s"] >= temps_erreur - 15)
        & (df["temps_debut_s"] <= temps_erreur + 15)
    )

    fenetres = df.loc[masque, colonnes].copy()
    fenetres["temps_erreur_reference_s"] = temps_erreur
    resultats.append(fenetres)

# Création du dossier de sortie séparé
dossier = Path("resultats_step36_experimental")
dossier.mkdir(parents=True, exist_ok=True)

# Regroupement, affichage et sauvegarde
analyse = pd.concat(resultats, ignore_index=True)
analyse = analyse.sort_values(
    ["scenario_id", "temps_erreur_reference_s", "temps_debut_s"]
)

sortie = dossier / "fenetres_autour_erreurs.csv"
analyse.to_csv(sortie, index=False, encoding="utf-8-sig")

print("Analyse temporelle locale")
print("Nombre de lignes :", len(analyse))
print(analyse.to_string(index=False))
print("\nFichier sauvegarde :", sortie)
