from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Définir les chemins du projet
DOSSIER = Path(__file__).resolve().parent
DOSSIER_ENTREE = DOSSIER / "resultats_step53_experimental"
DOSSIER_SORTIE = DOSSIER / "resultats_step54_experimental"
DOSSIER_SORTIE.mkdir(parents=True, exist_ok=True)

FICHIER_MODELES = DOSSIER_ENTREE / "synthese_performances_modeles.csv"
FICHIER_ROBUSTESSE = DOSSIER_ENTREE / "synthese_robustesse.csv"

# Charger les résultats
modeles = pd.read_csv(FICHIER_MODELES)
robustesse = pd.read_csv(FICHIER_ROBUSTESSE)

# Vérifier les colonnes requises
colonnes_modeles = ["modele", "f1", "rappel"]
colonnes_robustesse = [
    "niveau_bruit",
    "f1_defaut_mean",
    "recall_defaut_mean",
    "defauts_manques_mean",
]

for colonne in colonnes_modeles:
    if colonne not in modeles.columns:
        raise ValueError(f"Colonne absente du fichier modèles : {colonne}")

for colonne in colonnes_robustesse:
    if colonne not in robustesse.columns:
        raise ValueError(f"Colonne absente du fichier robustesse : {colonne}")

# Graphique 1 : F1-score et rappel selon le bruit
plt.figure(figsize=(9, 5.5))
plt.plot(
    robustesse["niveau_bruit"] * 100,
    robustesse["f1_defaut_mean"] * 100,
    marker="o",
    label="F1-score moyen",
)
plt.plot(
    robustesse["niveau_bruit"] * 100,
    robustesse["recall_defaut_mean"] * 100,
    marker="s",
    label="Rappel moyen",
)
plt.xlabel("Bruit ajouté (% de l'écart-type des caractéristiques)")
plt.ylabel("Performance (%)")
plt.title("Robustesse du modèle de référence au bruit simulé")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(DOSSIER_SORTIE / "graphique_robustesse_bruit.png", dpi=300)
plt.close()

# Graphique 2 : comparaison des modèles
positions = range(len(modeles))
largeur = 0.36

plt.figure(figsize=(10, 5.5))
plt.bar(
    [p - largeur / 2 for p in positions],
    modeles["f1"] * 100,
    width=largeur,
    label="F1-score",
)
plt.bar(
    [p + largeur / 2 for p in positions],
    modeles["rappel"] * 100,
    width=largeur,
    label="Rappel",
)
plt.xticks(
    list(positions),
    modeles["modele"],
    rotation=15,
    ha="right",
)
plt.ylabel("Performance (%)")
plt.title("Comparaison des modèles sur le test indépendant")
plt.ylim(0, 105)
plt.grid(axis="y", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(DOSSIER_SORTIE / "comparaison_modeles.png", dpi=300)
plt.close()

# Graphique 3 : défauts manqués moyens selon le bruit
plt.figure(figsize=(9, 5.5))
plt.errorbar(
    robustesse["niveau_bruit"] * 100,
    robustesse["defauts_manques_mean"],
    yerr=robustesse["defauts_manques_std"],
    marker="o",
    capsize=4,
)
plt.xlabel("Bruit ajouté (% de l'écart-type des caractéristiques)")
plt.ylabel("Défauts manqués moyens sur 472 fenêtres")
plt.title("Influence du bruit sur les défauts manqués")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(DOSSIER_SORTIE / "defauts_manques_selon_bruit.png", dpi=300)
plt.close()

print("=" * 72)
print("ETAPE 54 — GRAPHIQUES DE SYNTHESE")
print("=" * 72)
for fichier in sorted(DOSSIER_SORTIE.glob("*.png")):
    print(f"{fichier.name} : {fichier.stat().st_size} octets")
print(f"\nGraphiques enregistrés dans : {DOSSIER_SORTIE.resolve()}")
print("Les données utilisées restent entièrement synthétiques.")
