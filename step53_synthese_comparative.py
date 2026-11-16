from pathlib import Path

import pandas as pd

# Définir les chemins du projet
DOSSIER = Path(__file__).resolve().parent
DOSSIER_SORTIE = DOSSIER / "resultats_step53_experimental"
DOSSIER_SORTIE.mkdir(parents=True, exist_ok=True)

FICHIER_MODELES = (
    DOSSIER / "resultats_step47_experimental"
    / "comparaison_nombre_caracteristiques.csv"
)
FICHIER_ROBUSTESSE = (
    DOSSIER / "resultats_step52_experimental"
    / "robustesse_repetitions_resume.csv"
)

# Vérifier la présence des fichiers
for chemin in [FICHIER_MODELES, FICHIER_ROBUSTESSE]:
    if not chemin.exists():
        raise FileNotFoundError(f"Fichier introuvable : {chemin}")

# Charger les résultats
modeles = pd.read_csv(FICHIER_MODELES)
robustesse = pd.read_csv(FICHIER_ROBUSTESSE)

print("=" * 76)
print("ETAPE 53 — SYNTHESE COMPARATIVE")
print("=" * 76)

print("\nColonnes du tableau de comparaison des modèles :")
print(modeles.columns.tolist())
print("\nRésultats des modèles :")
print(modeles.to_string(index=False))

print("\nRésultats moyens de robustesse :")
print(robustesse.to_string(index=False))

# Enregistrer une copie de synthèse des sources
modeles.to_csv(
    DOSSIER_SORTIE / "synthese_performances_modeles.csv",
    index=False,
    encoding="utf-8-sig",
)
robustesse.to_csv(
    DOSSIER_SORTIE / "synthese_robustesse.csv",
    index=False,
    encoding="utf-8-sig",
)

print(f"\nFichiers de synthèse enregistrés dans : {DOSSIER_SORTIE.resolve()}")
print("Les données utilisées restent entièrement synthétiques.")
