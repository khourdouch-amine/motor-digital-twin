
import numpy as np
import pandas as pd
from pathlib import Path

# Charger le dataset original
fichier = Path("scenarios_step9/dataset_scenarios_complet.csv")
df = pd.read_csv(fichier)

# Paramètres des fenêtres temporelles
duree_fenetre_s = 5.0
pas_s = 0.05
taille_fenetre = int(duree_fenetre_s / pas_s)

resultats = []

# Analyser chaque scénario indépendamment
for scenario_id, groupe in df.groupby("scenario_id", sort=True):
    groupe = groupe.sort_values("time_s").reset_index(drop=True)

    for debut in range(0, len(groupe), taille_fenetre):
        fenetre = groupe.iloc[debut:debut + taille_fenetre]

        # Ignorer les fenêtres incomplètes
        if len(fenetre) != taille_fenetre:
            continue

        # Ignorer les fenêtres traversant une transition de défaut
        etats = fenetre["bearing_fault_active"].unique()
        if len(etats) != 1:
            continue

        signal = fenetre["vibration_sensor_m_s2"].to_numpy()

        # Calculer les indicateurs vibratoires
        rms = np.sqrt(np.mean(signal ** 2))
        ecart_type = np.std(signal)
        facteur_crete = (
            np.max(np.abs(signal)) / rms if rms > 0 else np.nan
        )

        resultats.append({
            "scenario_id": scenario_id,
            "scenario_type": fenetre["scenario_type"].iloc[0],
            "temps_debut_s": fenetre["time_s"].iloc[0],
            "bearing_fault_active": int(etats[0]),
            "bearing_fault_severity": fenetre[
                "bearing_fault_severity"
            ].iloc[0],
            "fault_class": int(fenetre["fault_class"].iloc[0]),
            "rms_m_s2": rms,
            "ecart_type_m_s2": ecart_type,
            "facteur_crete": facteur_crete,
        })

# Vérifier que des résultats ont été produits
if not resultats:
    raise ValueError("Aucune fenêtre valide n'a été trouvée.")

indicateurs = pd.DataFrame(resultats)

# Créer le dossier de sortie
dossier = Path("resultats_step13")
dossier.mkdir(exist_ok=True)

# Enregistrer les indicateurs dans un nouveau CSV
sortie = dossier / "indicateurs_vibratoires_5s.csv"
indicateurs.to_csv(sortie, index=False)

print(f"\nFichier créé : {sortie}")
print(f"Nombre de fenêtres : {len(indicateurs)}")
print(f"Nombre de scénarios : {indicateurs['scenario_id'].nunique()}")

print("\nRMS MOYEN PAR TYPE DE SCÉNARIO")
print(
    indicateurs.groupby("scenario_type")["rms_m_s2"]
    .agg(["count", "mean", "std"])
    .round(4)
    .to_string()
)

print("\nRMS MOYEN PAR SÉVÉRITÉ DU DÉFAUT")
defauts = indicateurs[indicateurs["bearing_fault_active"] == 1]

print(
    defauts.groupby("bearing_fault_severity")["rms_m_s2"]
    .agg(["count", "mean", "std"])
    .round(4)
    .to_string()
)
