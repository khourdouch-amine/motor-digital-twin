import numpy as np
import pandas as pd
from pathlib import Path

source = pd.read_csv(
    "scenarios_step9_experimental/dataset_scenarios_complet.csv"
)

erreurs = [
    (14, 185.0),
    (14, 190.0),
    (14, 235.0),
    (20, 255.0),
]

resultats = []

for scenario_id, debut in erreurs:
    fin = debut + 5.0

    fenetre = source[
        (source["scenario_id"] == scenario_id)
        & (source["time_s"] >= debut)
        & (source["time_s"] < fin)
    ]

    signal = fenetre["vibration_sensor_m_s2"].to_numpy(dtype=float)

    if len(signal) == 0:
        continue

    moyenne = np.mean(signal)
    rms_brut = np.sqrt(np.mean(signal ** 2))
    signal_centre = signal - moyenne
    rms_centre = np.sqrt(np.mean(signal_centre ** 2))
    ecart_type = np.std(signal)
    facteur_crete_brut = np.max(np.abs(signal)) / rms_brut

    resultats.append({
        "scenario_id": scenario_id,
        "temps_debut_s": debut,
        "nombre_echantillons": len(signal),
        "moyenne_signal": moyenne,
        "rms_brut": rms_brut,
        "rms_centre": rms_centre,
        "ecart_type": ecart_type,
        "facteur_crete_brut": facteur_crete_brut,
    })

rapport = pd.DataFrame(resultats)

dossier = Path("resultats_step42_experimental")
dossier.mkdir(parents=True, exist_ok=True)

sortie = dossier / "verification_indicateurs.csv"
rapport.to_csv(sortie, index=False, encoding="utf-8-sig")

print("Vérification des indicateurs sur les fenêtres manquées")
print(rapport.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
print("\nFichier sauvegardé :", sortie)
