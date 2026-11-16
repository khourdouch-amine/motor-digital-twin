import pandas as pd
from pathlib import Path

# Charger les prédictions existantes
fichier = Path("resultats_step34_experimental/predictions_par_seuil.csv")
df = pd.read_csv(fichier)

# Identifier les défauts manqués au seuil de référence
erreurs = df[
    (df["target_bearing_fault"] == 1) &
    (df["prediction_seuil_50"] == 0)
].copy()

# Caractéristiques à comparer
indicateurs = [
    "rms_m_s2",
    "ecart_type_m_s2",
    "facteur_crete",
    "probabilite_defaut",
]

resultats = []

for _, erreur in erreurs.iterrows():
    scenario = erreur["scenario_id"]
    temps = erreur["temps_debut_s"]

    # Fenêtres correctement détectées dans le même scénario,
    # dans un voisinage temporel de 30 secondes
    voisins = df[
        (df["scenario_id"] == scenario) &
        (df["target_bearing_fault"] == 1) &
        (df["prediction_seuil_50"] == 1) &
        ((df["temps_debut_s"] - temps).abs() <= 30)
    ].copy()

    ligne = {
        "scenario_id": scenario,
        "temps_defaut_manque_s": temps,
    }

    for indicateur in indicateurs:
        ligne[f"valeur_erreur_{indicateur}"] = erreur[indicateur]

        if len(voisins) > 0:
            ligne[f"moyenne_voisins_{indicateur}"] = voisins[indicateur].mean()
        else:
            ligne[f"moyenne_voisins_{indicateur}"] = float("nan")

    ligne["nombre_voisins_detectes"] = len(voisins)
    resultats.append(ligne)

# Sauvegarder dans un dossier séparé
dossier = Path("resultats_step39_experimental")
dossier.mkdir(parents=True, exist_ok=True)

sortie = dossier / "comparaison_defauts_manques_voisins.csv"
rapport = pd.DataFrame(resultats)
rapport.to_csv(sortie, index=False, encoding="utf-8-sig")

print("Comparaison des défauts manqués et des fenêtres voisines détectées")
print(rapport.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nFichier sauvegardé :", sortie)
