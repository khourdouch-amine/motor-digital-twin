from pathlib import Path
import pandas as pd

# Fichiers d'entrée et de sortie
entree = Path("resultats_step29_persistance_experimental/comparaison_predictions_persistance.csv")
sortie = Path("resultats_step29_persistance_experimental/bilan_par_scenario.csv")

df = pd.read_csv(entree)

# Méthodes à comparer
regles = {
    "reference": "prediction_reference",
    "persistance_2_sur_3": "prediction_2_sur_3",
    "persistance_3_sur_3": "prediction_3_sur_3",
}

lignes = []

# Calcul des résultats pour chaque méthode et chaque scénario
for nom_regle, colonne in regles.items():
    for scenario_id, groupe in df.groupby("scenario_id"):
        reel = groupe["target_bearing_fault"].astype(int)
        predit = groupe[colonne].astype(int)

        lignes.append({
            "regle": nom_regle,
            "scenario_id": scenario_id,
            "vrais_negatifs": int(((reel == 0) & (predit == 0)).sum()),
            "fausses_alertes": int(((reel == 0) & (predit == 1)).sum()),
            "defauts_manques": int(((reel == 1) & (predit == 0)).sum()),
            "vrais_positifs": int(((reel == 1) & (predit == 1)).sum()),
        })

# Enregistrement et affichage
resultats = pd.DataFrame(lignes)
resultats.to_csv(sortie, index=False)

print("Bilan par scénario :")
print(resultats.to_string(index=False))
print(f"\nFichier créé : {sortie}")
