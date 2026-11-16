
# ============================================================
# JUMEAU NUMERIQUE D'UN MOTEUR ELECTRIQUE
# ETAPE 4 : SIMULATION DES CAPTEURS ET GENERATION DES DONNEES
# ============================================================

import numpy as np  # Calculs numeriques
import pandas as pd  # Creation et exportation des tableaux de donnees
import matplotlib.pyplot as plt  # Affichage des graphiques

# ============================================================
# 1. PARAMETRES DU MOTEUR
# ============================================================

motor_name = "Moteur asynchrone triphase"  # Type de moteur
motor_id = "MOTOR_001"  # Identifiant du moteur simule
motor_power_kw = 5.5  # Puissance mecanique nominale en kW
motor_voltage_v = 400.0  # Tension nominale en V
motor_frequency_hz = 50.0  # Frequence electrique en Hz
motor_poles = 4  # Nombre de poles magnetiques
motor_efficiency = 0.88  # Rendement nominal suppose
motor_power_factor = 0.82  # Facteur de puissance suppose
motor_nominal_speed_rpm = 1450.0  # Vitesse nominale en tr/min

# Convertit la puissance mecanique en watts.
motor_power_w = motor_power_kw * 1000.0

# Calcule la vitesse synchrone theorique en tr/min.
motor_synchronous_speed_rpm = (
    120.0 * motor_frequency_hz / motor_poles
)

# Convertit la vitesse nominale en rad/s.
motor_nominal_speed_rad_s = (
    motor_nominal_speed_rpm * 2.0 * np.pi / 60.0
)

# Calcule le couple mecanique nominal en N.m.
motor_nominal_torque_nm = (
    motor_power_w / motor_nominal_speed_rad_s
)

# Estime la puissance electrique absorbee en fonctionnement nominal.
motor_input_power_w = motor_power_w / motor_efficiency

# Estime les pertes nominales totales en watts.
motor_nominal_losses_w = motor_input_power_w - motor_power_w

# Estime le courant nominal d'un systeme triphase equilibre.
motor_nominal_current_a = (
    motor_input_power_w
    / (
        np.sqrt(3.0)
        * motor_voltage_v
        * motor_power_factor
    )
)

# ============================================================
# 2. PARAMETRES MECANIQUES
# ============================================================

inertia_kg_m2 = 0.02  # Inertie equivalente supposee en kg.m^2
friction_coefficient = 0.002  # Frottement visqueux en N.m.s/rad

# Definit le couple de demarrage suppose.
starting_torque_factor = 1.8  # Couple de demarrage / couple nominal
starting_torque_nm = (
    starting_torque_factor * motor_nominal_torque_nm
)

# ============================================================
# 3. PARAMETRES THERMIQUES
# ============================================================

ambient_temperature_c = 25.0  # Temperature ambiante en degres C
initial_temperature_c = 25.0  # Temperature initiale du moteur
thermal_capacity_j_k = 12000.0  # Capacite thermique equivalente en J/K
heat_transfer_w_k = 35.0  # Coefficient global de dissipation en W/K

# Part supposee des pertes qui ne varie pas avec le carre de la charge.
constant_loss_fraction = 0.30

# Part supposee des pertes qui varie avec le carre de la charge.
variable_loss_fraction = 0.70

# ============================================================
# 4. PARAMETRES DE SIMULATION
# ============================================================

simulation_duration_s = 600.0  # Duree de simulation : 10 minutes
time_step_s = 0.01  # Pas de calcul de 10 millisecondes
start_time_s = 1.0  # Instant de demarrage du moteur

# Construit le vecteur temps avec un pas regulier.
time_s = np.arange(
    0.0,
    simulation_duration_s + time_step_s,
    time_step_s
)

# Determine le nombre de points de simulation.
number_of_points = len(time_s)

# ============================================================
# 5. PARAMETRES DES CAPTEURS VIRTUELS
# ============================================================

# Ecart-type du bruit du capteur de vitesse, en tr/min.
speed_noise_std_rpm = 2.0

# Ecart-type du bruit du capteur de courant, en amperes.
current_noise_std_a = 0.2

# Ecart-type du bruit du capteur de temperature, en degres C.
temperature_noise_std_c = 0.5

# Ecart-type du bruit du capteur de vibrations, en m/s^2.
vibration_noise_std_m_s2 = 0.05

# Niveau RMS de vibration de reference suppose, en m/s^2.
normal_vibration_rms_m_s2 = 0.5

# Part du courant nominal consideree comme courant a faible charge.
# Cette approximation ne remplace pas un modele electrique complet.
no_load_current_fraction = 0.30

# ============================================================
# 6. PARAMETRES DES DEFAUTS SIMULES
# ============================================================

# Instant de debut de la surcharge, en secondes.
overload_start_s = 200.0

# Instant de fin de la surcharge, en secondes.
overload_end_s = 350.0

# Charge appliquee pendant la surcharge : 130 % du nominal.
overload_factor = 1.30

# Instant de debut du defaut de roulement, en secondes.
bearing_fault_start_s = 400.0

# Augmentation supposee des vibrations en cas de defaut de roulement.
bearing_fault_extra_vibration_m_s2 = 1.5

# ============================================================
# 7. INITIALISATION DES VARIABLES DU MODELE
# ============================================================

# Initialise la vitesse angulaire en rad/s.
angular_speed_rad_s = np.zeros(number_of_points)

# Initialise la vitesse mecanique en tr/min.
speed_rpm = np.zeros(number_of_points)

# Initialise le couple moteur en N.m.
motor_torque_nm = np.zeros(number_of_points)

# Initialise le couple de frottement en N.m.
friction_torque_nm = np.zeros(number_of_points)

# Initialise l'acceleration angulaire en rad/s^2.
angular_acceleration_rad_s2 = np.zeros(number_of_points)

# Initialise le couple de charge en N.m.
load_torque_nm = np.zeros(number_of_points)

# Initialise la temperature du moteur en degres C.
temperature_c = np.zeros(number_of_points)

# Initialise les pertes thermiques en watts.
thermal_losses_w = np.zeros(number_of_points)

# Initialise le facteur de charge a chaque instant.
load_factor_history = np.zeros(number_of_points)

# Definit la temperature initiale.
temperature_c[0] = initial_temperature_c

# Initialise le generateur aleatoire pour rendre les resultats reproductibles.
rng = np.random.default_rng(seed=42)

# ============================================================
# 8. SIMULATION MECANIQUE ET THERMIQUE
# ============================================================

# Parcourt tous les instants de la simulation.
for i in range(1, number_of_points):

    # Recupere le temps courant.
    current_time_s = time_s[i]

    # Recupere la vitesse et la temperature precedentes.
    previous_speed = angular_speed_rad_s[i - 1]
    previous_temperature = temperature_c[i - 1]

    # Determine si le moteur est en fonctionnement.
    motor_is_running = current_time_s >= start_time_s

    # Active la surcharge uniquement dans la periode choisie.
    if overload_start_s <= current_time_s < overload_end_s:
        current_load_factor = overload_factor
    else:
        current_load_factor = 1.0

    # Enregistre le facteur de charge courant.
    load_factor_history[i] = (
        current_load_factor if motor_is_running else 0.0
    )

    # Calcule le couple de charge applique a l'arbre.
    if motor_is_running:
        load_torque_nm[i] = (
            current_load_factor * motor_nominal_torque_nm
        )
    else:
        load_torque_nm[i] = 0.0

    # --------------------------------------------------------
    # 8A. MODELE MECANIQUE
    # --------------------------------------------------------

    # Si le moteur est arrete, sa vitesse et ses couples sont nuls.
    if not motor_is_running:

        # Maintient le rotor a l'arret avant le demarrage.
        angular_speed_rad_s[i] = 0.0

        # Met les couples mecaniques a zero.
        motor_torque_nm[i] = 0.0
        friction_torque_nm[i] = 0.0

        # Met l'acceleration a zero avant le demarrage.
        angular_acceleration_rad_s2[i] = 0.0

    else:

        # Calcule le couple moteur requis a la vitesse nominale
        # pour la charge actuelle, en incluant les frottements.
        torque_at_nominal_speed_nm = (
            load_torque_nm[i]
            + friction_coefficient * motor_nominal_speed_rad_s
        )

        # Calcule le rapport de vitesse par rapport a la vitesse nominale.
        speed_ratio = (
            previous_speed / motor_nominal_speed_rad_s
        )

        # Approxime la relation couple-vitesse par une droite.
        torque_candidate_nm = (
            starting_torque_nm
            + (
                torque_at_nominal_speed_nm - starting_torque_nm
            ) * speed_ratio
        )

        # Evite un couple moteur negatif dans ce modele simplifie.
        motor_torque_nm[i] = max(0.0, torque_candidate_nm)

        # Calcule le couple de frottement visqueux.
        friction_torque_nm[i] = (
            friction_coefficient * previous_speed
        )

        # Calcule le couple resultant sur le rotor.
        net_torque_nm = (
            motor_torque_nm[i]
            - load_torque_nm[i]
            - friction_torque_nm[i]
        )

        # Calcule l'acceleration angulaire avec la loi de Newton en rotation.
        angular_acceleration_rad_s2[i] = (
            net_torque_nm / inertia_kg_m2
        )

        # Integre l'acceleration pour obtenir la nouvelle vitesse.
        angular_speed_rad_s[i] = (
            previous_speed
            + angular_acceleration_rad_s2[i] * time_step_s
        )

        # Empeche une vitesse negative dans le modele de demarrage.
        angular_speed_rad_s[i] = max(
            0.0, angular_speed_rad_s[i]
        )

    # Convertit la vitesse angulaire en tours/minute.
    speed_rpm[i] = (
        angular_speed_rad_s[i] * 60.0 / (2.0 * np.pi)
    )

    # --------------------------------------------------------
    # 8B. MODELE THERMIQUE
    # --------------------------------------------------------

    # Calcule les pertes thermiques si le moteur est en marche.
    if motor_is_running:

        # Calcule la partie constante des pertes.
        constant_losses_w = (
            constant_loss_fraction * motor_nominal_losses_w
        )

        # Calcule la partie variable des pertes selon le carre de la charge.
        variable_losses_w = (
            variable_loss_fraction
            * motor_nominal_losses_w
            * current_load_factor**2
        )

        # Additionne les pertes constantes et variables.
        thermal_losses_w[i] = (
            constant_losses_w + variable_losses_w
        )

    else:

        # Suppose que les pertes de fonctionnement sont nulles a l'arret.
        thermal_losses_w[i] = 0.0

    # Calcule la puissance thermique evacuee vers l'environnement.
    heat_dissipation_w = (
        heat_transfer_w_k
        * (previous_temperature - ambient_temperature_c)
    )

    # Calcule la variation de temperature par seconde.
    temperature_rate_c_s = (
        thermal_losses_w[i] - heat_dissipation_w
    ) / thermal_capacity_j_k

    # Met a jour la temperature par integration numerique.
    temperature_c[i] = (
        previous_temperature
        + temperature_rate_c_s * time_step_s
    )

# ============================================================
# 9. GENERATION DES MESURES DES CAPTEURS
# ============================================================

# Simule le bruit du capteur de vitesse.
speed_sensor_rpm = (
    speed_rpm
    + rng.normal(0.0, speed_noise_std_rpm, number_of_points)
)

# Impose une vitesse mesuree non negative.
speed_sensor_rpm = np.maximum(0.0, speed_sensor_rpm)

# Estime le courant a partir du facteur de charge.
# Le courant a vide est approxime par une fraction du courant nominal.
current_estimated_a = (
    motor_nominal_current_a
    * (
        no_load_current_fraction
        + (1.0 - no_load_current_fraction) * load_factor_history
    )
)

# Ajoute un bruit aleatoire au courant estime.
current_sensor_a = (
    current_estimated_a
    + rng.normal(0.0, current_noise_std_a, number_of_points)
)

# Impose un courant mesure non negatif.
current_sensor_a = np.maximum(0.0, current_sensor_a)

# Simule le bruit du capteur de temperature.
temperature_sensor_c = (
    temperature_c
    + rng.normal(0.0, temperature_noise_std_c, number_of_points)
)

# Simule une vibration de base avec un faible bruit aleatoire.
vibration_sensor_m_s2 = (
    normal_vibration_rms_m_s2
    + rng.normal(0.0, vibration_noise_std_m_s2, number_of_points)
)

# Identifie les instants apres l'apparition du defaut de roulement.
bearing_fault_active = time_s >= bearing_fault_start_s

# Ajoute une augmentation de vibration pendant le defaut simule.
vibration_sensor_m_s2[bearing_fault_active] += (
    bearing_fault_extra_vibration_m_s2
)

# Evite des valeurs de vibration negatives.
vibration_sensor_m_s2 = np.maximum(
    0.0, vibration_sensor_m_s2
)

# Cree une variable indiquant si la surcharge est active.
overload_active = (
    (time_s >= overload_start_s)
    & (time_s < overload_end_s)
)

# Cree une variable indiquant si le defaut de roulement est actif.
bearing_fault_active_int = bearing_fault_active.astype(int)

# Cree une variable indiquant si le moteur est en marche.
motor_running = (time_s >= start_time_s).astype(int)

# ============================================================
# 10. CREATION DU TABLEAU DE DONNEES
# ============================================================

# Regroupe les temps, les variables simulees et les mesures des capteurs.
data = pd.DataFrame({
    "time_s": time_s,  # Temps de simulation en secondes
    "motor_running": motor_running,  # Etat du moteur : 0 arret, 1 marche
    "speed_model_rpm": speed_rpm,  # Vitesse calculee par le modele
    "speed_sensor_rpm": speed_sensor_rpm,  # Vitesse avec bruit de mesure
    "motor_torque_nm": motor_torque_nm,  # Couple moteur calcule
    "load_torque_nm": load_torque_nm,  # Couple resistant applique
    "angular_acceleration_rad_s2": angular_acceleration_rad_s2,
    "temperature_model_c": temperature_c,  # Temperature calculee
    "temperature_sensor_c": temperature_sensor_c,  # Temperature mesuree
    "thermal_losses_w": thermal_losses_w,  # Pertes thermiques estimees
    "load_factor": load_factor_history,  # Facteur de charge
    "current_sensor_a": current_sensor_a,  # Courant estime avec bruit
    "vibration_sensor_m_s2": vibration_sensor_m_s2,  # Vibration simulee
    "overload_active": overload_active.astype(int),  # Etiquette de surcharge
    "bearing_fault_active": bearing_fault_active_int  # Etiquette roulement
})

# Definit le nom du fichier CSV a generer.
csv_filename = "motor_sensor_data.csv"

# Enregistre toutes les donnees dans un fichier CSV sans index supplementaire.
data.to_csv(csv_filename, index=False)

# ============================================================
# 11. AFFICHAGE D'UN RESUME DANS LE TERMINAL
# ============================================================

# Affiche un separateur pour rendre la sortie lisible.
print("=" * 65)

# Affiche le titre du resume.
print("RESUME DE LA SIMULATION DES CAPTEURS")

# Affiche un separateur.
print("=" * 65)

# Affiche l'identifiant du moteur.
print(f"Identifiant : {motor_id}")

# Affiche le nombre de lignes de donnees generees.
print(f"Nombre de mesures generees : {len(data)}")

# Affiche la duree totale de simulation.
print(f"Duree simulee : {simulation_duration_s:.1f} secondes")

# Affiche le courant nominal estime.
print(f"Courant nominal estime : {motor_nominal_current_a:.2f} A")

# Affiche la vitesse finale simulee.
print(f"Vitesse finale simulee : {speed_rpm[-1]:.2f} tr/min")

# Affiche la temperature finale simulee.
print(f"Temperature finale simulee : {temperature_c[-1]:.2f} deg C")

# Affiche le chemin relatif du fichier exporte.
print(f"Fichier CSV genere : {csv_filename}")

# Affiche les premieres lignes pour verifier les donnees.
print("\nApercu des cinq premieres lignes :")
print(data.head())

# Affiche les noms des colonnes disponibles.
print("\nColonnes disponibles :")
print(data.columns.tolist())

# Termine le resume.
print("=" * 65)

# ============================================================
# 12. AFFICHAGE DES CINQ GRAPHIQUES DANS UNE SEULE FENETRE
# ============================================================

# Cree une seule fenetre avec cinq graphiques verticaux.
fig, axes = plt.subplots(
    nrows=5,  # Cinq graphiques
    ncols=1,  # Une seule colonne
    figsize=(13, 16),  # Taille globale de la fenetre
    sharex=True  # Partage le meme axe temporel
)

# ---------------- GRAPHIQUE 1 : VITESSE ----------------

# Trace la vitesse calculee par le modele.
axes[0].plot(
    time_s,
    speed_rpm,
    label="Vitesse du modele"
)

# Trace la vitesse mesuree avec bruit.
axes[0].plot(
    time_s,
    speed_sensor_rpm,
    alpha=0.5,
    label="Capteur de vitesse"
)

# Affiche la vitesse nominale comme reference.
axes[0].axhline(
    motor_nominal_speed_rpm,
    linestyle="--",
    label="Vitesse nominale"
)

# Definit le titre, l'axe vertical, la grille et la legende.
axes[0].set_title("1. Vitesse du moteur")
axes[0].set_ylabel("Vitesse (tr/min)")
axes[0].grid(True)
axes[0].legend()

# ---------------- GRAPHIQUE 2 : TEMPERATURE ----------------

# Trace la temperature calculee.
axes[1].plot(
    time_s,
    temperature_c,
    label="Temperature du modele"
)

# Trace la temperature mesuree par le capteur virtuel.
axes[1].plot(
    time_s,
    temperature_sensor_c,
    alpha=0.5,
    label="Capteur de temperature"
)

# Affiche la temperature ambiante.
axes[1].axhline(
    ambient_temperature_c,
    linestyle="--",
    label="Temperature ambiante"
)

# Definit le titre, l'axe vertical, la grille et la legende.
axes[1].set_title("2. Temperature du moteur")
axes[1].set_ylabel("Temperature (deg C)")
axes[1].grid(True)
axes[1].legend()

# ---------------- GRAPHIQUE 3 : COURANT ----------------

# Trace le courant mesure estime.
axes[2].plot(
    time_s,
    current_sensor_a,
    label="Courant du capteur virtuel"
)

# Affiche le courant nominal estime comme reference.
axes[2].axhline(
    motor_nominal_current_a,
    linestyle="--",
    label="Courant nominal estime"
)

# Definit le titre, l'axe vertical, la grille et la legende.
axes[2].set_title("3. Courant electrique estime")
axes[2].set_ylabel("Courant (A)")
axes[2].grid(True)
axes[2].legend()

# ---------------- GRAPHIQUE 4 : VIBRATIONS ----------------

# Trace les vibrations simulees.
axes[3].plot(
    time_s,
    vibration_sensor_m_s2,
    label="Capteur de vibrations virtuel"
)

# Marque l'instant d'apparition du defaut de roulement.
axes[3].axvline(
    bearing_fault_start_s,
    linestyle="--",
    label="Debut du defaut simule"
)

# Definit le titre, l'axe vertical, la grille et la legende.
axes[3].set_title("4. Vibrations simulees")
axes[3].set_ylabel("Acceleration (m/s²)")
axes[3].grid(True)
axes[3].legend()

# ---------------- GRAPHIQUE 5 : ETAT DES DEFAUTS ----------------

# Affiche la surcharge sous forme d'une variable binaire.
axes[4].step(
    time_s,
    overload_active.astype(int),
    where="post",
    label="Surcharge active"
)

# Affiche le defaut de roulement sous forme d'une variable binaire.
axes[4].step(
    time_s,
    bearing_fault_active_int,
    where="post",
    label="Defaut de roulement actif"
)

# Definit les graduations binaires 0 et 1.
axes[4].set_yticks([0, 1])

# Definit le titre et les etiquettes.
axes[4].set_title("5. Evenements de defaut imposes par le scenario")
axes[4].set_ylabel("Etat (0/1)")
axes[4].set_xlabel("Temps (s)")
axes[4].grid(True)
axes[4].legend()

# Ajuste les marges et les espacements.
fig.tight_layout()

# Affiche les cinq graphiques dans une seule fenetre.
plt.show()
