
# ============================================================
# ETAPE 8 : EVALUATION TEMPORELLE DU MODELE
# Objectif : tester le modèle sur une période ultérieure
# Algorithme : Random Forest
# ============================================================

# Importer Path pour gérer les chemins des fichiers.
from pathlib import Path

# Importer pandas pour manipuler les données.
import pandas as pd

# Importer matplotlib pour créer les graphiques.
import matplotlib.pyplot as plt

# Importer le modèle de classification.
from sklearn.ensemble import RandomForestClassifier

# Importer les métriques d'évaluation.
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ------------------------------------------------------------
# 1. CHARGEMENT DES DONNEES
# ------------------------------------------------------------

# Identifier le dossier contenant ce script.
DOSSIER = Path(__file__).resolve().parent

# Définir le chemin du fichier de données.
FICHIER = DOSSIER / "motor_ai_dataset.csv"

# Vérifier que le fichier existe.
if not FICHIER.exists():
    raise FileNotFoundError(
        f"Fichier introuvable : {FICHIER}"
    )

# Charger les données.
data = pd.read_csv(FICHIER)

# Trier les observations selon le temps.
data = data.sort_values("time_s").reset_index(drop=True)

# Afficher les dimensions du jeu de données.
print("\n========== DONNEES CHARGEES ==========")
print(f"Nombre total d'observations : {len(data)}")

# ------------------------------------------------------------
# 2. DEFINITION DES VARIABLES
# ------------------------------------------------------------

# Définir les quatre mesures disponibles.
colonnes_entree = [
    "speed_sensor_rpm",
    "temperature_sensor_c",
    "current_sensor_a",
    "vibration_sensor_m_s2",
]

# Extraire les variables d'entrée.
X = data[colonnes_entree]

# Extraire la classe à prédire.
y = data["fault_label"]

# Afficher les classes présentes.
print("\nRépartition des classes :")
print(y.value_counts().sort_index())

# ------------------------------------------------------------
# 3. SEPARATION CHRONOLOGIQUE
# ------------------------------------------------------------

# Définir le point de séparation à 70 % des observations.
indice_coupure = int(len(data) * 0.70)

# Sélectionner les observations antérieures pour l'entraînement.
X_train = X.iloc[:indice_coupure]
y_train = y.iloc[:indice_coupure]

# Sélectionner les observations ultérieures pour le test.
X_test = X.iloc[indice_coupure:]
y_test = y.iloc[indice_coupure:]

# Conserver les informations temporelles du test.
temps_test = data["time_s"].iloc[indice_coupure:]

# Afficher les périodes utilisées.
print("\n========== SEPARATION TEMPORELLE ==========")
print(f"Observations d'entraînement : {len(X_train)}")
print(f"Observations de test : {len(X_test)}")
print(
    f"Période d'entraînement : "
    f"{data['time_s'].iloc[0]:.2f} à "
    f"{data['time_s'].iloc[indice_coupure - 1]:.2f} s"
)
print(
    f"Période de test : "
    f"{temps_test.iloc[0]:.2f} à "
    f"{temps_test.iloc[-1]:.2f} s"
)

# Afficher la répartition des classes dans chaque ensemble.
print("\nClasses dans l'entraînement :")
print(y_train.value_counts().sort_index())

print("\nClasses dans le test :")
print(y_test.value_counts().sort_index())

# ------------------------------------------------------------
# 4. VERIFICATION DE LA REPRESENTATION DES CLASSES
# ------------------------------------------------------------

# Vérifier que l'entraînement contient les deux classes.
if y_train.nunique() < 2:
    raise ValueError(
        "L'entraînement ne contient pas les deux classes."
    )

# Vérifier que le test contient les deux classes.
if y_test.nunique() < 2:
    print(
        "\nATTENTION : le test ne contient pas les deux classes. "
        "Certaines métriques ne seront pas interprétables."
    )

# ------------------------------------------------------------
# 5. ENTRAINEMENT DU MODELE
# ------------------------------------------------------------

# Créer une Random Forest de 100 arbres.
modele = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)

# Entraîner le modèle uniquement sur la période d'entraînement.
print("\n========== ENTRAINEMENT ==========")
modele.fit(X_train, y_train)

# Prédire les classes de la période de test.
y_pred = modele.predict(X_test)

# ------------------------------------------------------------
# 6. EVALUATION GLOBALE
# ------------------------------------------------------------

# Calculer l'accuracy sur la période de test.
accuracy = accuracy_score(y_test, y_pred)

# Afficher le résultat.
print("\n========== RESULTATS TEMPORELS ==========")
print(f"Accuracy : {accuracy:.4f}")
print(f"Accuracy en pourcentage : {accuracy * 100:.2f} %")

# Afficher les métriques par classe.
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
matrice = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1],
)

# Afficher la matrice.
print("Matrice de confusion :")
print(matrice)

# ------------------------------------------------------------
# 7. ANALYSE PAR SCENARIO
# ------------------------------------------------------------

# Copier les informations nécessaires dans un tableau de résultats.
resultats = data.iloc[indice_coupure:].copy()

# Ajouter la classe prédite par le modèle.
resultats["prediction"] = y_pred

# Afficher les résultats selon les indicateurs de défaut.
print("\n========== ANALYSE PAR SCENARIO ==========")

# Analyser séparément les trois scénarios observés.
for surcharge, roulement, nom in [
    (0, 0, "Fonctionnement normal"),
    (1, 0, "Surcharge"),
    (0, 1, "Défaut de roulement"),
]:

    # Sélectionner les observations correspondant au scénario.
    masque = (
        (resultats["overload_active"] == surcharge)
        & (resultats["bearing_fault_active"] == roulement)
    )

    # Extraire les observations sélectionnées.
    scenario = resultats.loc[masque]

    # Vérifier que le scénario existe dans la période de test.
    if len(scenario) == 0:
        print(f"\n{nom} : aucune observation dans le test.")
        continue

    # Calculer le taux de classification correcte.
    taux_correct = (
        scenario["prediction"] == scenario["fault_label"]
    ).mean()

    # Afficher les résultats du scénario.
    print(f"\n{nom}")
    print(f"Nombre d'observations : {len(scenario)}")
    print(f"Taux correct : {taux_correct * 100:.2f} %")

    # Afficher la proportion de prédictions normales.
    print(
        "Prédictions normales : "
        f"{(scenario['prediction'] == 0).sum()}"
    )

    # Afficher la proportion de prédictions anormales.
    print(
        "Prédictions anomalies : "
        f"{(scenario['prediction'] == 1).sum()}"
    )

# ------------------------------------------------------------
# 8. VISUALISATION DES RESULTATS
# ------------------------------------------------------------

# Créer une fenêtre avec deux graphiques.
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Afficher la matrice de confusion dans le premier graphique.
affichage = ConfusionMatrixDisplay(
    confusion_matrix=matrice,
    display_labels=["Normal", "Anomalie"],
)

affichage.plot(
    ax=axes[0],
    colorbar=False,
    values_format="d",
)

# Ajouter le titre du premier graphique.
axes[0].set_title("Evaluation temporelle")

# Tracer les classes réelles dans le second graphique.
axes[1].plot(
    temps_test,
    y_test.to_numpy(),
    label="Classe réelle",
    linewidth=2,
)

# Tracer les classes prédites.
axes[1].plot(
    temps_test,
    y_pred,
    label="Classe prédite",
    alpha=0.7,
)

# Ajouter les titres et les légendes.
axes[1].set_title("Classes réelles et prédites")
axes[1].set_xlabel("Temps (s)")
axes[1].set_ylabel("Classe")
axes[1].set_yticks([0, 1])
axes[1].legend()

# Ajouter une grille au graphique temporel.
axes[1].grid(True, alpha=0.3)

# Ajuster la disposition des graphiques.
plt.tight_layout()

# Définir le chemin de sauvegarde de la figure.
figure_path = DOSSIER / "step8_temporal_evaluation.png"

# Enregistrer la figure avec une bonne résolution.
plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight",
)

# Afficher la fenêtre des graphiques.
plt.show()

# Confirmer l'enregistrement de la figure.
print(f"\nFigure enregistrée : {figure_path}")

# ============================================================
# FIN DE L'ETAPE 8
# ============================================================