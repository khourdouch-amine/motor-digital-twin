
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. Chemins des fichiers
# ============================================================

fichier_entree = Path(
    "scenarios_step9_experimental/dataset_scenarios_complet.csv"
)

dossier_sortie = Path("resultats_step13_experimental")
dossier_sortie.mkdir(parents=True, exist_ok=True)

fichier_sortie = (
    dossier_sortie / "indicateurs_vibratoires_5s.csv"
)


# ============================================================
# 2. Paramètres de calcul
# ============================================================

duree_fenetre_s = 5.0
pas_s = 0.05
taille_fenetre = int(round(duree_fenetre_s / pas_s))

if not fichier_entree.exists():
    raise FileNotFoundError(
        f"Dataset introuvable : {fichier_entree}"
    )


# ============================================================
# 3. Chargement et vérification des données
# ============================================================

df = pd.read_csv(fichier_entree)

colonnes_requises = [
    "scenario_id",
    "scenario_type",
    "time_s",
    "vibration_sensor_m_s2",
    "bearing_fault_active",
    "bearing_fault_severity",
    "fault_class",
]

colonnes_absentes = [
    colonne
    for colonne in colonnes_requises
    if colonne not in df.columns
]

if colonnes_absentes:
    raise ValueError(
        f"Colonnes absentes : {colonnes_absentes}"
    )

if df[colonnes_requises].isna().any().any():
    raise ValueError(
        "Des valeurs manquantes sont présentes dans les colonnes requises."
    )

if not set(df["bearing_fault_active"].unique()).issubset({0, 1}):
    raise ValueError(
        "bearing_fault_active doit contenir uniquement 0 et 1."
    )


# ============================================================
# 4. Calcul des indicateurs par fenêtre de 5 secondes
# ============================================================

resultats = []

for scenario_id, groupe in df.groupby("scenario_id", sort=True):
    groupe = (
        groupe.sort_values("time_s")
        .reset_index(drop=True)
    )

    # Vérifier la cohérence temporelle du scénario
    temps = groupe["time_s"].to_numpy()

    if len(temps) > 1 and not np.allclose(
        np.diff(temps), pas_s, atol=1e-6, rtol=0
    ):
        raise ValueError(
            f"Pas temporel inattendu dans le scénario {scenario_id}."
        )

    for debut in range(0, len(groupe), taille_fenetre):
        fenetre = groupe.iloc[
            debut:debut + taille_fenetre
        ]

        # Ignorer les fenêtres incomplètes
        if len(fenetre) != taille_fenetre:
            continue

        # Ignorer les fenêtres traversant une transition de défaut
        etats = fenetre["bearing_fault_active"].unique()

        if len(etats) != 1:
            continue

        signal = fenetre[
            "vibration_sensor_m_s2"
        ].to_numpy(dtype=float)

        if not np.isfinite(signal).all():
            continue

        # Calcul des indicateurs vibratoires
        rms = float(np.sqrt(np.mean(signal ** 2)))
        ecart_type = float(np.std(signal))

        facteur_crete = (
            float(np.max(np.abs(signal)) / rms)
            if rms > 0
            else np.nan
        )

        resultats.append({
            "scenario_id": int(scenario_id),
            "scenario_type": fenetre["scenario_type"].iloc[0],
            "temps_debut_s": float(fenetre["time_s"].iloc[0]),
            "bearing_fault_active": int(etats[0]),
            "bearing_fault_severity": (
                fenetre["bearing_fault_severity"].iloc[0]
            ),
            "fault_class": int(fenetre["fault_class"].iloc[0]),
            "rms_m_s2": rms,
            "ecart_type_m_s2": ecart_type,
            "facteur_crete": facteur_crete,
        })


# ============================================================
# 5. Vérification et sauvegarde
# ============================================================

if not resultats:
    raise ValueError(
        "Aucune fenêtre valide n'a été trouvée."
    )

indicateurs = pd.DataFrame(resultats)

indicateurs.to_csv(fichier_sortie, index=False)


# ============================================================
# 6. Rapport de synthèse
# ============================================================

print("=== INDICATEURS VIBRATOIRES EXPÉRIMENTAUX ===")
print(f"Dataset source : {fichier_entree}")
print(f"Fichier créé   : {fichier_sortie}")
print(f"Nombre de fenêtres : {len(indicateurs)}")
print(
    "Nombre de scénarios : "
    f"{indicateurs['scenario_id'].nunique()}"
)

print("\n=== RÉPARTITION PAR ÉTAT DU ROULEMENT ===")
print(
    indicateurs.groupby("bearing_fault_active")["rms_m_s2"]
    .agg(["count", "mean", "std", "min", "max"])
    .round(4)
    .to_string()
)

print("\n=== RMS MOYEN PAR TYPE DE SCÉNARIO ===")
print(
    indicateurs.groupby("scenario_type")["rms_m_s2"]
    .agg(["count", "mean", "std"])
    .round(4)
    .to_string()
)

print("\n=== RMS PAR SÉVÉRITÉ DU DÉFAUT ===")
defauts = indicateurs[
    indicateurs["bearing_fault_active"] == 1
]

if not defauts.empty:
    print(
        defauts.groupby("bearing_fault_severity")["rms_m_s2"]
        .agg(["count", "mean", "std"])
        .round(4)
        .to_string()
    )

print("\nCalcul terminé. Les fichiers de référence n'ont pas été modifiés.")
