
"""
ÉTAPE 9 — GÉNÉRATION DE SCÉNARIOS INDÉPENDANTS
================================================

Projet :
Conception et simulation d'un jumeau numérique d'un moteur électrique
intégrant l'IoT industriel, la maintenance prédictive par IA
et le suivi énergétique.

Objectifs :
- Générer plusieurs simulations indépendantes.
- Simuler cinq catégories de fonctionnement.
- Produire des mesures synthétiques de capteurs.
- Créer des étiquettes de défauts pour l'apprentissage automatique.
- Sauvegarder les données dans des fichiers CSV.

Important :
Les équations utilisées constituent un modèle simplifié.
Les résultats ne représentent pas des mesures industrielles réelles.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION GÉNÉRALE
# ============================================================

# Dossier contenant le script
DOSSIER_PROJET = Path(__file__).resolve().parent

# Dossier de sauvegarde des résultats
DOSSIER_RESULTATS = DOSSIER_PROJET / "scenarios_step9"
DOSSIER_RESULTATS.mkdir(parents=True, exist_ok=True)

# Paramètres temporels
DUREE_SIMULATION_S = 600.0
PAS_TEMPS_S = 0.05

# Paramètres nominaux du moteur électrique
PUISSANCE_MECANIQUE_W = 5500.0
RENDEMENT_NOMINAL = 0.88
FREQUENCE_HZ = 50.0
NOMBRE_POLES = 4
VITESSE_NOMINALE_RPM = 1450.0
COURANT_NOMINAL_A = 11.0

# Calcul de la vitesse synchrone
VITESSE_SYNCHRONE_RPM = (
    120.0 * FREQUENCE_HZ / NOMBRE_POLES
)

# Calcul du couple nominal
VITESSE_ANGULAIRE_NOMINALE = (
    2.0 * np.pi * VITESSE_NOMINALE_RPM / 60.0
)

COUPLE_NOMINAL_NM = (
    PUISSANCE_MECANIQUE_W / VITESSE_ANGULAIRE_NOMINALE
)

# Paramètres thermiques simplifiés
TEMPERATURE_AMBIANTE_C = 25.0
CAPACITE_THERMIQUE_J_K = 12000.0
COEFFICIENT_THERMIQUE_W_K = 35.0

# Estimation des pertes nominales
PERTES_NOMINALES_W = (
    PUISSANCE_MECANIQUE_W / RENDEMENT_NOMINAL
    - PUISSANCE_MECANIQUE_W
)

# Contributions supplémentaires aux pertes en présence de défauts.
# Ces coefficients sont des hypothèses de simulation,
# et non des paramètres identifiés expérimentalement.
PERTES_SUPPLEMENTAIRES_SURCHARGE_W = 150.0
PERTES_SUPPLEMENTAIRES_ROULEMENT_W = 100.0

# Constante de temps du modèle mécanique
# ============================================================
# PARAMÈTRES DU MODÈLE MÉCANIQUE AMÉLIORÉ
# ============================================================

# Inertie équivalente moteur-charge.
# Valeur initiale hypothétique à ajuster avec des données réelles.
INERTIE_EQUIVALENTE_KG_M2 = 0.05

# Coefficient de frottement visqueux.
# Valeur initiale hypothétique, exprimée en N.m.s/rad.
COEFFICIENT_FROTTEMENT_NMS_RAD = 0.002

# Couple supplémentaire associé à un défaut de roulement.
# Hypothèse de simulation, non issue d'une mesure expérimentale.
COUPLE_DEFAUT_ROULEMENT_NM = 1.0

# Coefficient simplifié reliant le glissement au couple moteur.
# Il devra être calibré ou remplacé par un modèle moteur plus détaillé.
COEFFICIENT_COUPLE_GLISSEMENT_NM_PAR_RAD_S = 2.0

# Graine aléatoire pour reproduire les simulations
GRAINE_ALEATOIRE = 42

# Nombre de simulations par catégorie
CATEGORIES = {
    "normal": 4,
    "charge_variable": 4,
    "surcharge": 4,
    "defaut_roulement": 4,
    "defauts_combines": 4,
}


# ============================================================
# 2. FONCTION DE SIMULATION
# ============================================================

def simuler_scenario(identifiant, categorie, graine):
    """
    Simule un scénario indépendant de fonctionnement du moteur.

    Paramètres :
        identifiant : numéro unique du scénario
        categorie : catégorie de fonctionnement
        graine : graine du générateur aléatoire

    Retour :
        DataFrame contenant les mesures et les étiquettes.
    """

    # Générateur aléatoire indépendant pour chaque scénario
    rng = np.random.default_rng(graine)

    # Axe temporel
    temps = np.arange(
        0.0,
        DUREE_SIMULATION_S + PAS_TEMPS_S / 2.0,
        PAS_TEMPS_S,
    )

    nombre_points = len(temps)

    # --------------------------------------------------------
    # 2.1. Paramètres propres à chaque simulation
    # --------------------------------------------------------

    # Charge de base comprise entre 65 % et 100 %
    charge_base = rng.uniform(0.65, 1.00)

    # Niveau de surcharge
    niveau_surcharge = rng.uniform(1.20, 1.45)

    # Instants de début des événements
    debut_surcharge = rng.uniform(100.0, 350.0)
    debut_defaut = rng.uniform(100.0, 350.0)

    # Durées des événements
    duree_surcharge = rng.uniform(60.0, 150.0)
    duree_defaut = rng.uniform(120.0, 250.0)

    # Intensité de l'augmentation des vibrations
    intensite_vibration = rng.uniform(0.8, 1.8)

    # Bruits des capteurs
    bruit_vitesse = rng.uniform(1.5, 2.5)
    bruit_courant = rng.uniform(0.15, 0.30)
    bruit_temperature = rng.uniform(0.3, 0.7)
    bruit_vibration = rng.uniform(0.04, 0.08)

    # --------------------------------------------------------
    # 2.2. Activation des défauts
    # --------------------------------------------------------

    surcharge_active = np.zeros(
        nombre_points,
        dtype=int,
    )

    defaut_roulement_actif = np.zeros(
        nombre_points,
        dtype=int,
    )

    # Activation de la surcharge pour les catégories concernées
    if categorie in ("surcharge", "defauts_combines"):

        surcharge_active = (
            (temps >= debut_surcharge)
            & (temps < debut_surcharge + duree_surcharge)
        ).astype(int)

    # Activation du défaut de roulement
    if categorie in ("defaut_roulement", "defauts_combines"):

        defaut_roulement_actif = (
            (temps >= debut_defaut)
            & (temps < debut_defaut + duree_defaut)
        ).astype(int)

    # --------------------------------------------------------
    # 2.3. Profil de charge
    # --------------------------------------------------------

    # Charge de base
    charge = np.full(
        nombre_points,
        charge_base,
        dtype=float,
    )

    # Variation lente de charge
    phase = rng.uniform(0.0, 2.0 * np.pi)

    variation_lente = (
        0.04
        * np.sin(2.0 * np.pi * temps / 90.0 + phase)
    )

    charge += variation_lente

    # Variations supplémentaires pour la catégorie
    # « charge variable »
    if categorie == "charge_variable":

        variation_rapide = (
            0.12
            * np.sin(2.0 * np.pi * temps / 25.0 + phase)
        )

        # Changements aléatoires toutes les 10 secondes
        temps_points = np.arange(
            0.0,
            DUREE_SIMULATION_S + 10.0,
            10.0,
        )

        changements = rng.uniform(
            -0.08,
            0.08,
            size=len(temps_points),
        )

        variation_aleatoire = np.interp(
            temps,
            temps_points,
            changements,
        )

        charge += variation_rapide + variation_aleatoire

    # Application de la surcharge
    charge = np.where(
        surcharge_active == 1,
        niveau_surcharge,
        charge,
    )

    # Limitation aux valeurs retenues pour ce modèle simplifié
    charge = np.clip(
        charge,
        0.40,
        1.50,
    )

    # --------------------------------------------------------
    # 2.4. Modèle de vitesse
    # --------------------------------------------------------

    # Approximation de la vitesse en fonction de la charge
    vitesse_cible = (
        VITESSE_SYNCHRONE_RPM
        - (
            VITESSE_SYNCHRONE_RPM
            - VITESSE_NOMINALE_RPM
        ) * charge
    )

    # Réponse mécanique de premier ordre
    vitesse_reelle = np.zeros(nombre_points)

    vitesse_reelle[0] = vitesse_cible[0]

    alpha_vitesse = (
        PAS_TEMPS_S
        / (
            CONSTANTE_TEMPS_VITESSE_S
            + PAS_TEMPS_S
        )
    )

    for i in range(1, nombre_points):

        vitesse_reelle[i] = (
            vitesse_reelle[i - 1]
            + alpha_vitesse
            * (
                vitesse_cible[i]
                - vitesse_reelle[i - 1]
            )
        )

    # Mesure bruitée de la vitesse
    vitesse_mesuree = (
        vitesse_reelle
        + rng.normal(
            0.0,
            bruit_vitesse,
            nombre_points,
        )
    )

    # --------------------------------------------------------
    # 2.5. Modèle de courant
    # --------------------------------------------------------

    # Modèle approximatif : le courant augmente avec la charge.
    # Il ne s'agit pas d'un calcul électrique complet.
    courant_reel = (
        COURANT_NOMINAL_A
        * (0.15 + 0.85 * charge)
    )

    courant_mesure = (
        courant_reel
        + rng.normal(
            0.0,
            bruit_courant,
            nombre_points,
        )
    )

    # --------------------------------------------------------
    # 2.6. Modèle thermique
    # --------------------------------------------------------

    temperature_reelle = np.zeros(nombre_points)

    temperature_reelle[0] = TEMPERATURE_AMBIANTE_C

    # Pertes de base : dépendance simplifiée à la charge.
    pertes = PERTES_NOMINALES_W * charge**2

    # Pertes supplémentaires pendant les défauts.
    # Ces valeurs sont des hypothèses de simulation.
    pertes += (
        PERTES_SUPPLEMENTAIRES_SURCHARGE_W
        * surcharge_active
    )

    pertes += (
        PERTES_SUPPLEMENTAIRES_ROULEMENT_W
        * defaut_roulement_actif
    )

    # Calcul de l'évolution de la température
    for i in range(1, nombre_points):

        flux_thermique_sortant = (
            COEFFICIENT_THERMIQUE_W_K
            * (
                temperature_reelle[i - 1]
                - TEMPERATURE_AMBIANTE_C
            )
        )

        variation_temperature = (
            (
                pertes[i]
                - flux_thermique_sortant
            )
            / CAPACITE_THERMIQUE_J_K
        )

        temperature_reelle[i] = (
            temperature_reelle[i - 1]
            + variation_temperature * PAS_TEMPS_S
        )

    # Mesure bruitée de la température
    temperature_mesuree = (
        temperature_reelle
        + rng.normal(
            0.0,
            bruit_temperature,
            nombre_points,
        )
    )

    # --------------------------------------------------------
    # 2.7. Modèle de vibration
    # --------------------------------------------------------

    # Niveau RMS synthétique de vibration
    vibration_base = (
        0.45
        + 0.10 * charge
        + 0.03
        * np.abs(
            np.sin(2.0 * np.pi * temps / 12.0)
        )
    )

    # Le défaut de roulement augmente les vibrations
    augmentation_vibration = (
        intensite_vibration
        * defaut_roulement_actif
    )

    vibration_reelle = (
        vibration_base + augmentation_vibration
    )

    vibration_mesuree = (
        vibration_reelle
        + rng.normal(
            0.0,
            bruit_vibration,
            nombre_points,
        )
    )

    # Une mesure RMS ne peut pas être négative
    vibration_mesuree = np.maximum(
        vibration_mesuree,
        0.0,
    )

    # --------------------------------------------------------
    # 2.8. Étiquettes des classes
    # --------------------------------------------------------

    # 0 : fonctionnement normal
    # 1 : surcharge
    # 2 : défaut de roulement
    # 3 : surcharge et défaut de roulement simultanés

    surcharge = surcharge_active.astype(bool)
    roulement = defaut_roulement_actif.astype(bool)

    etiquette_classe = np.zeros(
        nombre_points,
        dtype=int,
    )

    etiquette_classe[
        surcharge & ~roulement
    ] = 1

    etiquette_classe[
        ~surcharge & roulement
    ] = 2

    etiquette_classe[
        surcharge & roulement
    ] = 3

    noms_classes = {
        0: "normal",
        1: "surcharge",
        2: "defaut_roulement",
        3: "defauts_combines",
    }

    etiquette_texte = [
        noms_classes[int(valeur)]
        for valeur in etiquette_classe
    ]

    # Étiquette binaire :
    # 0 = aucun défaut actif
    # 1 = au moins un défaut actif
    etiquette_binaire = (
        etiquette_classe != 0
    ).astype(int)

    # --------------------------------------------------------
    # 2.9. Assemblage des données
    # --------------------------------------------------------

    donnees = pd.DataFrame({
        "scenario_id": identifiant,
        "scenario_type": categorie,
        "time_s": temps,

        # Mesures de capteurs
        "speed_sensor_rpm": vitesse_mesuree,
        "temperature_sensor_c": temperature_mesuree,
        "current_sensor_a": courant_mesure,
        "vibration_sensor_m_s2": vibration_mesuree,

        # Étiquettes utilisées pour l'apprentissage supervisé
        "fault_label": etiquette_binaire,
        "fault_class": etiquette_classe,
        "fault_class_name": etiquette_texte,

        # Métadonnées de simulation.
        # Ne pas utiliser ces colonnes comme entrées du modèle IA.
        "overload_active": surcharge_active,
        "bearing_fault_active": defaut_roulement_actif,
        "load_factor_simulated": charge,
    })

    return donnees


# ============================================================
# 3. GÉNÉRATION DES SCÉNARIOS
# ============================================================

def main():

    print("=" * 70)
    print("ÉTAPE 9 — GÉNÉRATION DE SCÉNARIOS INDÉPENDANTS")
    print("=" * 70)

    print(f"Durée d'un scénario : {DUREE_SIMULATION_S:.0f} s")
    print(f"Pas de temps : {PAS_TEMPS_S:.3f} s")
    print(f"Nombre de scénarios : {sum(CATEGORIES.values())}")
    print(f"Dossier de sortie : {DOSSIER_RESULTATS}")
    print()

    tous_les_scenarios = []
    identifiant = 1

    # Générer quatre simulations indépendantes par catégorie
    for categorie, nombre in CATEGORIES.items():

        for repetition in range(1, nombre + 1):

            print(
                f"Génération du scénario {identifiant:02d} : "
                f"{categorie} ({repetition}/{nombre})"
            )

            donnees = simuler_scenario(
                identifiant=identifiant,
                categorie=categorie,
                graine=GRAINE_ALEATOIRE + identifiant * 100,
            )

            # Sauvegarder chaque simulation individuellement
            nom_fichier = (
                f"scenario_{identifiant:02d}_{categorie}.csv"
            )

            chemin_fichier = (
                DOSSIER_RESULTATS / nom_fichier
            )

            donnees.to_csv(
                chemin_fichier,
                index=False,
                float_format="%.5f",
            )

            tous_les_scenarios.append(donnees)

            identifiant += 1

    # --------------------------------------------------------
    # 3.1. Regrouper toutes les simulations
    # --------------------------------------------------------

    dataset_complet = pd.concat(
        tous_les_scenarios,
        ignore_index=True,
    )

    chemin_dataset = (
        DOSSIER_RESULTATS
        / "dataset_scenarios_complet.csv"
    )

    dataset_complet.to_csv(
        chemin_dataset,
        index=False,
        float_format="%.5f",
    )

    # --------------------------------------------------------
    # 3.2. Afficher le résumé
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("RÉSUMÉ DU DATASET")
    print("-" * 70)

    print(
        f"Nombre total de lignes : "
        f"{len(dataset_complet):,}"
    )

    print(
        f"Nombre de scénarios : "
        f"{dataset_complet['scenario_id'].nunique()}"
    )

    print("\nNombre de simulations par catégorie :")

    print(
        dataset_complet.groupby(
            "scenario_type"
        )["scenario_id"].nunique().to_string()
    )

    print("\nRépartition des classes :")

    print(
        dataset_complet["fault_class_name"]
        .value_counts()
        .to_string()
    )

    print("\nFichier CSV complet :")
    print(chemin_dataset)

    # --------------------------------------------------------
    # 3.3. Comparer les scénarios graphiquement
    # --------------------------------------------------------

    # Sélectionner la première simulation de chaque catégorie
    premiers_scenarios = (
        dataset_complet
        .groupby("scenario_type", sort=False)["scenario_id"]
        .first()
        .to_dict()
    )

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(13, 12),
        sharex=True,
    )

    variables = [
        (
            "speed_sensor_rpm",
            "Vitesse mesurée (tr/min)",
        ),
        (
            "current_sensor_a",
            "Courant mesuré (A)",
        ),
        (
            "temperature_sensor_c",
            "Température mesurée (°C)",
        ),
        (
            "vibration_sensor_m_s2",
            "Vibration RMS mesurée (m/s²)",
        ),
    ]

    for axe, (variable, titre) in zip(
        axes,
        variables,
    ):

        for categorie, id_scenario in premiers_scenarios.items():

            sous_ensemble = dataset_complet[
                dataset_complet["scenario_id"] == id_scenario
            ]

            axe.plot(
                sous_ensemble["time_s"],
                sous_ensemble[variable],
                label=categorie,
                linewidth=1.0,
            )

        axe.set_ylabel(titre)
        axe.grid(True, alpha=0.3)
        axe.legend(loc="best", ncol=3)

    axes[-1].set_xlabel("Temps (s)")

    fig.suptitle(
        "Étape 9 — Comparaison des scénarios du moteur",
        fontsize=14,
    )

    fig.tight_layout()

    chemin_figure = (
        DOSSIER_RESULTATS
        / "step9_comparaison_scenarios.png"
    )

    fig.savefig(
        chemin_figure,
        dpi=150,
        bbox_inches="tight",
    )

    plt.show()

    print("\nFigure enregistrée :")
    print(chemin_figure)

    print("\nÉtape 9 terminée avec succès.")


# ============================================================
# 4. POINT D'ENTRÉE DU PROGRAMME
# ============================================================

if __name__ == "__main__":
    main()
