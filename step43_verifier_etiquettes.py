import pandas as pd
from pathlib import Path

# Charger les indicateurs et le dataset source
indicateurs = pd.read_csv(
    "resultats_step13_experimental/indicateurs_vibratoires_5s.csv"
)

source = pd.read_csv(
    "scenarios_step9_experimental/dataset_scenarios_complet.csv"
)

# Vérifier chaque fenêtre dans les données sources
resultats = []

for _, ligne in indicateurs.iterrows():
    scenario_id = int(ligne["scenario_id"])
    debut = float(ligne["temps_debut_s"])
    fin = debut + 5.0

    fenetre = source[
        (source["scenario_id"] == scenario_id)
        & (source["time_s"] >= debut)
        & (source["time_s"] < fin)
    ]

    if len(fenetre) == 0:
        continue

    etats = fenetre["bearing_fault_active"].unique()
    etat_attendu = int(ligne["bearing_fault_active"])

    resultats.append({
        "scenario_id": scenario_id,
        "temps_debut_s": debut,
        "nombre_echantillons": len(fenetre),
        "etat_indicateur": etat_attendu,
        "nombre_etats_source": len(etats),
        "etat_source_unique": int(etats[0]) if len(etats) == 1 else -1,
        "etiquette_coherente": (
            len(etats) == 1 and int(etats[0]) == etat_attendu
        ),
    })

rapport = pd.DataFrame(resultats)

dossier = Path("resultats_step43_experimental")
dossier.mkdir(parents=True, exist_ok=True)

sortie = dossier / "verification_etiquettes.csv"
rapport.to_csv(sortie, index=False, encoding="utf-8-sig")

print("=== CONTRÔLE DES ÉTIQUETTES ===")
print("Fenêtres vérifiées :", len(rapport))
print("Étiquettes cohérentes :", int(rapport["etiquette_coherente"].sum()))
print("Étiquettes incohérentes :", int((~rapport["etiquette_coherente"]).sum()))
print("Fenêtres avec un nombre d'échantillons différent de 100 :",
      int((rapport["nombre_echantillons"] != 100).sum()))
print("Fenêtres contenant plusieurs états du défaut :",
      int((rapport["nombre_etats_source"] != 1).sum()))
print("\nFichier sauvegardé :", sortie)
