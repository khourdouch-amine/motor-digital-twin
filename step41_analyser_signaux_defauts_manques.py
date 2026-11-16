import pandas as pd
from pathlib import Path

# Fichiers source et prédictions
fichier_source = Path(
    "scenarios_step9_experimental/dataset_scenarios_complet.csv"
)
fichier_predictions = Path(
    "resultats_step34_experimental/predictions_par_seuil.csv"
)

source = pd.read_csv(fichier_source)
predictions = pd.read_csv(fichier_predictions)

# Défauts manqués au seuil de référence de 0,50
erreurs = predictions[
    (predictions["target_bearing_fault"] == 1)
    & (predictions["prediction_seuil_50"] == 0)
].copy()

colonnes_signaux = [
    "vibration_sensor_m_s2",
    "bearing_fault_active",
    "bearing_vibration_increase_m_s2",
    "load_factor_simulated",
]

resultats = []

for _, erreur in erreurs.iterrows():
    scenario_id = int(erreur["scenario_id"])
    debut = float(erreur["temps_debut_s"])
    fin = debut + 5.0

    # Extraire les échantillons correspondant à la fenêtre de 5 secondes
    fenetre = source[
        (source["scenario_id"] == scenario_id)
        & (source["time_s"] >= debut)
        & (source["time_s"] < fin)
    ].copy()

    if fenetre.empty:
        continue

    ligne = {
        "scenario_id": scenario_id,
        "temps_debut_s": debut,
        "temps_fin_s": fin,
        "nombre_echantillons": len(fenetre),
        "probabilite_modele": erreur["probabilite_defaut"],
        "rms_features_modele": erreur["rms_m_s2"],
        "ecart_type_features_modele": erreur["ecart_type_m_s2"],
        "facteur_crete_modele": erreur["facteur_crete"],
    }

    for colonne in colonnes_signaux:
        ligne[f"{colonne}_moyenne"] = fenetre[colonne].mean()
        ligne[f"{colonne}_min"] = fenetre[colonne].min()
        ligne[f"{colonne}_max"] = fenetre[colonne].max()

    ligne["proportion_defaut_actif"] = fenetre[
        "bearing_fault_active"
    ].mean()

    resultats.append(ligne)

# Enregistrer dans un nouveau dossier
dossier = Path("resultats_step41_experimental")
dossier.mkdir(parents=True, exist_ok=True)

rapport = pd.DataFrame(resultats)
sortie = dossier / "analyse_signaux_defauts_manques.csv"
rapport.to_csv(sortie, index=False, encoding="utf-8-sig")

print("Analyse des signaux sources sur les fenêtres manquées")
print("Nombre de fenêtres analysées :", len(rapport))
print(rapport.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nFichier sauvegardé :", sortie)
