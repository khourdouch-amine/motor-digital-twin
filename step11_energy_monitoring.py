
# -*- coding: utf-8 -*-

"""
Étape 11 — Suivi énergétique du moteur électrique.

Ce programme exploite les scénarios synthétiques générés à l'étape 9
pour estimer la puissance, les pertes, le rendement et l'énergie consommée.

Attention :
Les résultats sont des estimations issues d'un modèle simplifié.
Ils ne représentent pas des mesures expérimentales réelles.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION DU PROJET
# ============================================================

DOSSIER_PROJET = Path(__file__).resolve().parent

DOSSIER_SCENARIOS = DOSSIER_PROJET / "scenarios_step9"

FICHIER_ENTREE = (
    DOSSIER_SCENARIOS / "dataset_scenarios_complet.csv"
)

DOSSIER_RESULTATS = DOSSIER_PROJET / "resultats_step11"

DOSSIER_RESULTATS.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. PARAMÈTRES NOMINAUX DU MOTEUR
# ============================================================

PUISSANCE_MECANIQUE_NOMINALE_W = 5500.0

TENSION_TRIPHASEE_V = 400.0

RENDEMENT_NOMINAL = 0.88

FACTEUR_PUISSANCE = 0.82

VITESSE_NOMINALE_RPM = 1450.0

# Puissance électrique nominale estimée à partir du rendement.
PUISSANCE_ELECTRIQUE_NOMINALE_W = (
    PUISSANCE_MECANIQUE_NOMINALE_W / RENDEMENT_NOMINAL
)

# Pertes nominales totales estimées.
PERTES_NOMINALES_W = (
    PUISSANCE_ELECTRIQUE_NOMINALE_W
    - PUISSANCE_MECANIQUE_NOMINALE_W
)

# Courant nominal estimé pour un réseau triphasé équilibré.
COURANT_NOMINAL_A = (
    PUISSANCE_ELECTRIQUE_NOMINALE_W
    / (
        np.sqrt(3.0)
        * TENSION_TRIPHASEE_V
        * FACTEUR_PUISSANCE
    )
)


# ============================================================
# 3. CHARGEMENT ET VALIDATION DES DONNÉES
# ============================================================

def charger_donnees():
    """Charge le fichier CSV et vérifie les colonnes nécessaires."""

    if not FICHIER_ENTREE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {FICHIER_ENTREE}"
        )

    donnees = pd.read_csv(FICHIER_ENTREE)

    colonnes_requises = [
        "scenario_id",
        "scenario_type",
        "time_s",
        "speed_sensor_rpm",
        "current_sensor_a",
        "load_factor_simulated",
    ]

    colonnes_manquantes = [
        colonne
        for colonne in colonnes_requises
        if colonne not in donnees.columns
    ]

    if colonnes_manquantes:
        raise ValueError(
            "Colonnes manquantes dans le CSV : "
            + ", ".join(colonnes_manquantes)
        )

    if donnees.empty:
        raise ValueError("Le fichier CSV ne contient aucune donnée.")

    return donnees


# ============================================================
# 4. CALCUL DES GRANDEURS ÉNERGÉTIQUES
# ============================================================

def calculer_grandeurs_energetiques(donnees):
    """
    Estime les grandeurs énergétiques à partir des signaux simulés.

    Le courant et le facteur de charge sont utilisés comme indicateurs
    de fonctionnement. Le modèle électrique reste approximatif.
    """

    resultat = donnees.copy()

    # --------------------------------------------------------
    # 4.1 Puissance électrique absorbée estimée
    # --------------------------------------------------------

    # Le courant est utilisé pour estimer la puissance électrique.
    # On suppose que le facteur de puissance reste constant.
    resultat["puissance_electrique_estimee_w"] = (
        np.sqrt(3.0)
        * TENSION_TRIPHASEE_V
        * resultat["current_sensor_a"]
        * FACTEUR_PUISSANCE
    )

    # --------------------------------------------------------
    # 4.2 Puissance mécanique utile estimée
    # --------------------------------------------------------

    # Le facteur de charge est limité à une plage physique plausible
    # pour éviter des valeurs négatives ou excessives dans ce modèle.
    facteur_charge = resultat["load_factor_simulated"].clip(
        lower=0.0,
        upper=1.5,
    )

    # La puissance utile est approximée par la puissance nominale
    # multipliée par le facteur de charge et le rapport de vitesse.
    rapport_vitesse = (
        resultat["speed_sensor_rpm"].clip(lower=0.0)
        / VITESSE_NOMINALE_RPM
    ).clip(lower=0.0, upper=1.1)

    resultat["puissance_mecanique_estimee_w"] = (
        PUISSANCE_MECANIQUE_NOMINALE_W
        * facteur_charge
        * rapport_vitesse
    )

    # --------------------------------------------------------
    # 4.3 Pertes estimées
    # --------------------------------------------------------

    # Les pertes sont la différence entre la puissance absorbée
    # et la puissance utile estimée.
    resultat["pertes_estimees_w"] = (
        resultat["puissance_electrique_estimee_w"]
        - resultat["puissance_mecanique_estimee_w"]
    )

    # --------------------------------------------------------
    # 4.4 Rendement estimé
    # --------------------------------------------------------

    # Le rendement n'est calculé que lorsque la puissance absorbée
    # est strictement positive.
    resultat["rendement_estime_pct"] = np.where(
        resultat["puissance_electrique_estimee_w"] > 0.0,
        100.0
        * resultat["puissance_mecanique_estimee_w"]
        / resultat["puissance_electrique_estimee_w"],
        np.nan,
    )

    # --------------------------------------------------------
    # 4.5 Énergie électrique cumulée
    # --------------------------------------------------------

    # Les données sont regroupées par scénario.
    # L'intervalle de temps est calculé à partir des horodatages.
    resultat["energie_electrique_cumulee_kwh"] = 0.0

    for identifiant, indices in resultat.groupby(
        "scenario_id",
        sort=False,
    ).groups.items():

        indices = list(indices)

        temps = resultat.loc[indices, "time_s"].to_numpy()

        puissance = resultat.loc[
            indices,
            "puissance_electrique_estimee_w",
        ].to_numpy()

        energie_cumulee = np.zeros(len(indices), dtype=float)

        if len(indices) > 1:
            intervalles = np.diff(temps)

            if np.any(intervalles < 0.0):
                raise ValueError(
                    f"Temps non croissant dans le scénario {identifiant}."
                )

            # Intégration numérique par la méthode des trapèzes.
            energie_incrementale_kwh = (
                0.5
                * (puissance[:-1] + puissance[1:])
                * intervalles
                / 3_600_000.0
            )

            energie_cumulee[1:] = np.cumsum(
                energie_incrementale_kwh
            )

        resultat.loc[
            indices,
            "energie_electrique_cumulee_kwh",
        ] = energie_cumulee

    return resultat


# ============================================================
# 5. SYNTHÈSE PAR SCÉNARIO
# ============================================================

def creer_synthese(donnees):
    """Calcule un bilan énergétique pour chaque simulation."""

    lignes = []

    for (identifiant, type_scenario), groupe in donnees.groupby(
        ["scenario_id", "scenario_type"],
        sort=True,
    ):

        groupe = groupe.sort_values("time_s")

        temps = groupe["time_s"].to_numpy()

        puissance_elec = groupe[
            "puissance_electrique_estimee_w"
        ].to_numpy()

        puissance_meca = groupe[
            "puissance_mecanique_estimee_w"
        ].to_numpy()

        pertes = groupe["pertes_estimees_w"].to_numpy()

        rendement = groupe["rendement_estime_pct"].to_numpy()

        # Intégration sur toute la durée du scénario.
        duree_s = float(temps[-1] - temps[0]) if len(temps) > 1 else 0.0

        energie_elec_kwh = (
            np.trapezoid(puissance_elec, temps) / 3_600_000.0
            if len(temps) > 1
            else 0.0
        )

        energie_meca_kwh = (
            np.trapezoid(puissance_meca, temps) / 3_600_000.0
            if len(temps) > 1
            else 0.0
        )

        energie_pertes_kwh = (
            np.trapezoid(pertes, temps) / 3_600_000.0
            if len(temps) > 1
            else 0.0
        )

        lignes.append({
            "scenario_id": int(identifiant),
            "scenario_type": type_scenario,
            "duree_s": duree_s,
            "puissance_electrique_moyenne_w": float(
                np.mean(puissance_elec)
            ),
            "puissance_mecanique_moyenne_w": float(
                np.mean(puissance_meca)
            ),
            "pertes_moyennes_w": float(np.mean(pertes)),
            "rendement_moyen_pct": float(
                np.nanmean(rendement)
            ),
            "energie_electrique_kwh": float(energie_elec_kwh),
            "energie_mecanique_kwh": float(energie_meca_kwh),
            "energie_pertes_kwh": float(energie_pertes_kwh),
            "courant_moyen_a": float(
                groupe["current_sensor_a"].mean()
            ),
            "vitesse_moyenne_rpm": float(
                groupe["speed_sensor_rpm"].mean()
            ),
        })

    return pd.DataFrame(lignes)


# ============================================================
# 6. EXPORT DES RÉSULTATS
# ============================================================

def sauvegarder_resultats(donnees, synthese):
    """Enregistre les résultats détaillés et la synthèse."""

    fichier_detail = (
        DOSSIER_RESULTATS / "donnees_energetiques_detaillees.csv"
    )

    fichier_synthese = (
        DOSSIER_RESULTATS / "bilan_energetique_par_scenario.csv"
    )

    donnees.to_csv(
        fichier_detail,
        index=False,
        encoding="utf-8-sig",
    )

    synthese.to_csv(
        fichier_synthese,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nFichiers enregistrés :")
    print(f"  Données détaillées : {fichier_detail}")
    print(f"  Bilan par scénario : {fichier_synthese}")


# ============================================================
# 7. GRAPHIQUES COMPARATIFS
# ============================================================

def creer_graphiques(synthese):
    """Crée une figure avec plusieurs indicateurs énergétiques."""

    fichier_figure = (
        DOSSIER_RESULTATS / "step11_bilan_energetique.png"
    )

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(15, 10),
    )

    fig.suptitle(
        "Étape 11 — Suivi énergétique du moteur simulé",
        fontsize=16,
        fontweight="bold",
    )

    # Puissance électrique moyenne.
    axes[0, 0].bar(
        synthese["scenario_id"].astype(str),
        synthese["puissance_electrique_moyenne_w"],
    )

    axes[0, 0].set_title("Puissance électrique moyenne")
    axes[0, 0].set_ylabel("Puissance (W)")
    axes[0, 0].set_xlabel("Identifiant du scénario")
    axes[0, 0].tick_params(axis="x", rotation=90)
    axes[0, 0].grid(axis="y", alpha=0.3)

    # Pertes moyennes estimées.
    axes[0, 1].bar(
        synthese["scenario_id"].astype(str),
        synthese["pertes_moyennes_w"],
    )

    axes[0, 1].set_title("Pertes moyennes estimées")
    axes[0, 1].set_ylabel("Pertes (W)")
    axes[0, 1].set_xlabel("Identifiant du scénario")
    axes[0, 1].tick_params(axis="x", rotation=90)
    axes[0, 1].grid(axis="y", alpha=0.3)

    # Énergie électrique consommée.
    axes[1, 0].bar(
        synthese["scenario_id"].astype(str),
        synthese["energie_electrique_kwh"],
    )

    axes[1, 0].set_title("Énergie électrique consommée")
    axes[1, 0].set_ylabel("Énergie (kWh)")
    axes[1, 0].set_xlabel("Identifiant du scénario")
    axes[1, 0].tick_params(axis="x", rotation=90)
    axes[1, 0].grid(axis="y", alpha=0.3)

    # Rendement moyen estimé.
    axes[1, 1].bar(
        synthese["scenario_id"].astype(str),
        synthese["rendement_moyen_pct"],
    )

    axes[1, 1].set_title("Rendement moyen estimé")
    axes[1, 1].set_ylabel("Rendement (%)")
    axes[1, 1].set_xlabel("Identifiant du scénario")
    axes[1, 1].tick_params(axis="x", rotation=90)
    axes[1, 1].grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    fig.savefig(
        fichier_figure,
        dpi=200,
        bbox_inches="tight",
    )

    print(f"\nGraphique enregistré : {fichier_figure}")

    plt.show()


# ============================================================
# 8. PROGRAMME PRINCIPAL
# ============================================================

def main():
    """Exécute le suivi énergétique de bout en bout."""

    print("=" * 75)
    print("ÉTAPE 11 — SUIVI ÉNERGÉTIQUE DU MOTEUR ÉLECTRIQUE")
    print("=" * 75)

    print("\nParamètres nominaux du moteur :")
    print(
        f"Puissance mécanique nominale : "
        f"{PUISSANCE_MECANIQUE_NOMINALE_W:.0f} W"
    )
    print(f"Tension triphasée : {TENSION_TRIPHASEE_V:.0f} V")
    print(f"Rendement nominal supposé : {RENDEMENT_NOMINAL * 100:.1f} %")
    print(f"Facteur de puissance supposé : {FACTEUR_PUISSANCE:.2f}")
    print(f"Courant nominal estimé : {COURANT_NOMINAL_A:.2f} A")
    print(f"Puissance électrique nominale estimée : "
          f"{PUISSANCE_ELECTRIQUE_NOMINALE_W:.2f} W")
    print(f"Pertes nominales estimées : {PERTES_NOMINALES_W:.2f} W")

    print("\nChargement des scénarios...")
    donnees = charger_donnees()

    print(f"Observations chargées : {len(donnees):,}")
    print(f"Nombre de scénarios : {donnees['scenario_id'].nunique()}")

    print("\nCalcul des grandeurs énergétiques...")
    donnees = calculer_grandeurs_energetiques(donnees)

    print("Calcul du bilan par scénario...")
    synthese = creer_synthese(donnees)

    sauvegarder_resultats(donnees, synthese)

    print("\nRésumé du bilan énergétique :")
    colonnes_affichage = [
        "scenario_id",
        "scenario_type",
        "puissance_electrique_moyenne_w",
        "pertes_moyennes_w",
        "rendement_moyen_pct",
        "energie_electrique_kwh",
    ]

    print(
        synthese[colonnes_affichage]
        .round(3)
        .to_string(index=False)
    )

    creer_graphiques(synthese)

    print("\n" + "=" * 75)
    print("ÉTAPE 11 TERMINÉE")
    print("=" * 75)

    print(
        "\nRappel : les résultats sont issus d'un modèle synthétique "
        "et nécessitent une validation physique."
    )


if __name__ == "__main__":
    main()
