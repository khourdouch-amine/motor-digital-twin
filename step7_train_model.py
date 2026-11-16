
# ============================================================
# ETAPE 7 : ENTRAINEMENT DU PREMIER MODELE D'INTELLIGENCE ARTIFICIELLE
# Objectif : classifier les observations normales et anormales
# Algorithme : Random Forest
# ============================================================

# Importer Path pour gérer les chemins des fichiers.
from pathlib import Path

# Importer pandas pour manipuler les données.
import pandas as pd

# Importer matplotlib pour tracer les résultats.
import matplotlib.pyplot as plt

# Importer les outils de séparation des données et d'évaluation.
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# Importer l'algorithme Random Forest.
from sklearn.ensemble import RandomForestClassifier

# ------------------------------------------------------------
# 1. LOCALISATION ET CHARGEMENT DES DONNEES
# ------------------------------------------------------------

# Identifier le dossier dans lequel se trouve ce script.
DOSSIER = Path(__file__).resolve().parent

# Définir le chemin du jeu de données préparé à l'étape 6.
FICHIER = DOSSIER / "motor_ai_dataset.csv"

# Vérifier que le fichier existe avant de continuer.
if not FICHIER.exists():
    raise FileNotFoundError(
        f"Fichier introuvable : {FICHIER}. "
        "Exécute d'abord step6_prepare_data.py."
    )

# Charger le jeu de données dans un DataFrame.
data = pd.read_csv(FICHIER)

# Afficher les dimensions des données chargées.
print("\n========== CHARGEMENT DES DONNEES ==========")
print(f"Nombre de lignes : {len(data)}")
print(f"Nombre de colonnes : {len(data.columns)}")

# ------------------------------------------------------------
# 2. DEFINITION DES VARIABLES D'ENTREE ET DE LA CIBLE
# ------------------------------------------------------------

# Sélectionner uniquement les quatre mesures des capteurs.
# Les indicateurs de défaut ne sont volontairement pas utilisés.
colonnes_entree = [
    "speed_sensor_rpm",
    "temperature_sensor_c",
    "current_sensor_a",
    "vibration_sensor_m_s2",
]

# Construire la matrice des variables d'entrée.
X = data[colonnes_entree]

# Sélectionner la cible à prédire.
# 0 correspond à une observation normale.
# 1 correspond à une observation présentant un défaut simulé.
y = data["fault_label"]

# Afficher la répartition des classes.
print("\n========== REPARTITION DES CLASSES ==========")
print(y.value_counts().sort_index())

# ------------------------------------------------------------
# 3. SEPARATION ENTRE ENTRAINEMENT ET TEST
# ------------------------------------------------------------

# Réserver 20 % des observations pour évaluer le modèle.
# random_state fixe le tirage pour rendre l'expérience reproductible.
# stratify conserve approximativement la proportion des deux classes.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Afficher les dimensions des ensembles obtenus.
print("\n========== SEPARATION DES DONNEES ==========")
print(f"Observations d'entraînement : {len(X_train)}")
print(f"Observations de test : {len(X_test)}")

# ------------------------------------------------------------
# 4. CREATION ET ENTRAINEMENT DU MODELE
# ------------------------------------------------------------

# Créer une forêt de 100 arbres de décision.
# n_jobs=-1 permet d'utiliser les cœurs disponibles du processeur.
# random_state garantit la reproductibilité du modèle.
modele = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)

# Entraîner le modèle sur les données d'entraînement.
print("\n========== ENTRAINEMENT DU MODELE ==========")
modele.fit(X_train, y_train)

# Indiquer que l'entraînement est terminé.
print("Entraînement terminé.")

# ------------------------------------------------------------
# 5. PREDICTIONS SUR LES DONNEES DE TEST
# ------------------------------------------------------------

# Prédire la classe de chaque observation de test.
y_pred = modele.predict(X_test)

# Calculer la proportion de prédictions correctes.
accuracy = accuracy_score(y_test, y_pred)

# Afficher la précision globale.
print("\n========== EVALUATION DU MODELE ==========")
print(f"Accuracy : {accuracy:.4f}")
print(f"Accuracy en pourcentage : {accuracy * 100:.2f} %")

# Afficher précision, rappel et F1-score pour chaque classe.
print("\nRapport de classification :")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Normal", "Anomalie"],
        zero_division=0,
    )
)

# Calculer la matrice de confusion.
matrice = confusion_matrix(y_test, y_pred, labels=[0, 1])

# Afficher les quatre cases de la matrice de confusion.
print("Matrice de confusion :")
print(matrice)

# ------------------------------------------------------------
# 6. IMPORTANCE DES VARIABLES
# ------------------------------------------------------------

# Extraire l'importance attribuée à chaque variable par la forêt.
importances = modele.feature_importances_

# Construire un tableau des importances.
importance_df = pd.DataFrame({
    "Variable": colonnes_entree,
    "Importance": importances,
})

# Trier les variables de la plus importante à la moins importante.
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False,
)

# Afficher les importances calculées.
print("\n========== IMPORTANCE DES VARIABLES ==========")
print(importance_df.to_string(index=False))

# ------------------------------------------------------------
# 7. VISUALISATION DES RESULTATS
# ------------------------------------------------------------

# Créer une fenêtre contenant deux graphiques côte à côte.
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Tracer la matrice de confusion sur le premier graphique.
affichage = ConfusionMatrixDisplay(
    confusion_matrix=matrice,
    display_labels=["Normal", "Anomalie"],
)

# Afficher la matrice sans créer une nouvelle fenêtre.
affichage.plot(
    ax=axes[0],
    colorbar=False,
    values_format="d",
)

# Ajouter un titre au premier graphique.
axes[0].set_title("Matrice de confusion")

# Tracer l'importance des variables sur le second graphique.
axes[1].barh(
    importance_df["Variable"],
    importance_df["Importance"],
)

# Placer la variable la plus importante en haut.
axes[1].invert_yaxis()

# Ajouter un titre au second graphique.
axes[1].set_title("Importance des variables")

# Nommer l'axe horizontal du second graphique.
axes[1].set_xlabel("Importance relative")

# Ajuster les espacements entre les graphiques.
plt.tight_layout()

# Enregistrer la figure dans le dossier du projet.
figure_path = DOSSIER / "step7_model_results.png"
plt.savefig(figure_path, dpi=300, bbox_inches="tight")

# Afficher les deux graphiques dans une seule fenêtre.
plt.show()

# Confirmer l'emplacement de la figure enregistrée.
print(f"\nFigure enregistrée : {figure_path}")

# ------------------------------------------------------------
# FIN DE L'ETAPE 7
# ------------------------------------------------------------
