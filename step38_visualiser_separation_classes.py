import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Charger les données annotées de l'étape précédente
fichier = Path("resultats_step37_experimental/donnees_avec_variations.csv")
df = pd.read_csv(fichier)

# Définir les caractéristiques à comparer
caracteristiques = [
    ("rms_m_s2", "RMS vibratoire (m/s²)"),
    ("ecart_type_m_s2", "Écart-type vibratoire (m/s²)"),
    ("facteur_crete", "Facteur de crête"),
]

# Créer un dossier de résultats indépendant
dossier = Path("resultats_step38_experimental")
dossier.mkdir(parents=True, exist_ok=True)

# Définir l'ordre des catégories
categories = [
    "Vrai négatif",
    "Fausse alerte",
    "Défaut manqué",
    "Vrai positif",
]

# Tracer un graphique par caractéristique
for colonne, titre in caracteristiques:
    donnees = [
        df.loc[df["groupe_resultat"] == categorie, colonne].dropna().values
        for categorie in categories
    ]

    plt.figure(figsize=(10, 6))
    plt.boxplot(donnees, tick_labels=categories, showmeans=True)
    plt.title(f"Distribution de {titre} selon le résultat du modèle")
    plt.ylabel(titre)
    plt.xticks(rotation=15)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    nom = colonne + "_par_resultat.png"
    plt.savefig(dossier / nom, dpi=160)
    plt.close()

print("Graphiques créés dans :", dossier)
for colonne, _ in caracteristiques:
    print("-", colonne + "_par_resultat.png")
