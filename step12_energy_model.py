# -*- coding: utf-8 -*-

"""
Étape 12 — Modèle énergétique amélioré du moteur électrique.

Objectifs :
- Séparer les pertes constantes et les pertes variables.
- Calculer la puissance mécanique utile.
- Résoudre de manière cohérente le bilan énergétique.
- Estimer le courant théorique du moteur.
- Comparer le courant théorique au courant simulé des capteurs.
- Calculer le rendement et les énergies.
- Vérifier automatiquement la cohérence du bilan.
- Exporter les résultats et générer des graphiques.

Attention :
Le modèle est simplifié et repose sur des données synthétiques.
Il ne constitue pas une validation expérimentale d'un moteur réel.
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

DOSSIER_RESULTATS = DOSSIER_PROJET / "resultats_step12"

DOSSIER_RESULTATS.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. PARAMETRES NOMINAUX DU MOTEUR
# ============================================================

PUISSANCE_MECANIQUE_NOMINALE_W = 5500.0

TENSION_TRIPHASEE_V = 400.0

RENDEMENT_NOMINAL = 0.88

FACTEUR_PUISSANCE = 0.82

VITESSE_NOMINALE_RPM = 1450.0

# Hypothese de repartition des pertes nominales.
FRACTION_PERTES_CONSTANTES = 0.30

FRACTION_PERTES_VARIABLES = 0.70

# Puissance electrique nominale estimee.
PUISSANCE_ELECTRIQUE_NOMINALE_W = (
    PUISSANCE_MECANIQUE_NOMINALE_W
    / RENDEMENT_NOMINAL
)

# Pertes nominales totales.
PERTES_NOMINALES_W = (
    PUISSANCE_ELECTRIQUE_NOMINALE_W
    - PUISSANCE_MECANIQUE_NOMINALE_W
)

# Separation des pertes nominales.
PERTES_CONSTANTES_W = (
    FRACTION_PERTES_CONSTANTES
    * PERTES_NOMINALES_W
)

PERTES_VARIABLES_NOMINALES_W = (
    FRACTION_PERTES_VARIABLES
    * PERTES_NOMINALES_W
)

# Courant nominal calcule a partir de la puissance nominale.
COURANT_NOMINAL_A = (
    PUISSANCE_ELECTRIQUE_NOMINALE_W
    / (
        np.sqrt(3.0)
        * TENSION_TRIPHASEE_V
        * FACTEUR_PUISSANCE
    )
)

# Coefficient de conversion courant-puissance pour le modele
# triphase equilibre avec facteur de puissance constant.
COEFFICIENT_PUISSANCE_W_A = (
    np.sqrt(3.0)
    * TENSION_TRIPHASEE_V
    * FACTEUR_PUISSANCE
)

# Coefficient de la loi simplifiee des pertes variables :
# P_variables = COEFFICIENT_PERTES * I**2
COEFFICIENT_PERTES_W_A2 = (
    PERTES_VARIABLES_NOMINALES_W
    / COURANT_NOMINAL_A**2
)


# ============================================================
# 3. CHARGEMENT ET VALIDATION DES DONNEES
# ============================================================

def charger_donnees():
    """Charge le fichier des scenarios et valide les colonnes."""

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
            "Colonnes manquantes dans le fichier CSV : "
            + ", ".join(colonnes_manquantes)
        )

    if donnees.empty:
        raise ValueError(
            "Le fichier CSV ne contient aucune donnee."
        )

    colonnes_numeriques = [
        "scenario_id",
        "time_s",
        "speed_sensor_rpm",
        "current_sensor_a",
        "load_factor_simulated",
    ]

    for colonne in colonnes_numeriques:
        donnees[colonne] = pd.to_numeric(
            donnees[colonne],
            errors="raise",
        )

        if not np.isfinite(
            donnees[colonne].to_numpy(dtype=float)
        ).all():
            raise ValueError(
                f"Valeurs manquantes ou non finies dans {colonne}."
            )

    # Tri chronologique au sein de chaque scenario.
    donnees = donnees.sort_values(
        ["scenario_id", "time_s"]
    ).reset_index(drop=True)

    # Verification des horodatages.
    differences_temps = donnees.groupby(
        "scenario_id"
    )["time_s"].diff()

    if (differences_temps.dropna() <= 0.0).any():
        raise ValueError(
            "Les horodatages doivent etre strictement croissants "
            "dans chaque scenario."
        )

    return donnees


# ============================================================
# 4. RESOLUTION DU MODELE ENERGETIQUE
# ============================================================

def resoudre_courant_theorique(puissance_mecanique_w):
    """
    Resout le bilan energetique pour obtenir le courant theorique.

    Modele :
        P_elec = P_meca + P_constantes + k * I**2

    Relation electrique :
        P_elec = a * I

    D'ou :
        k * I**2 - a * I + (P_meca + P_constantes) = 0

    On retient la petite racine positive, qui correspond a la
    branche passant par le point nominal du moteur.
    """

    puissance_mecanique_w = np.asarray(
        puissance_mecanique_w,
        dtype=float,
    )

    coefficient_a = COEFFICIENT_PUISSANCE_W_A

    coefficient_k = COEFFICIENT_PERTES_W_A2

    terme_constant = (
        puissance_mecanique_w
        + PERTES_CONSTANTES_W
    )

    discriminant = (
        coefficient_a**2
        - 4.0 * coefficient_k * terme_constant
    )

    # Un discriminant negatif signifie que le modele simplifie
    # ne possede pas de solution reelle pour cette puissance.
    if np.any(discriminant < 0.0):
        raise ValueError(
            "Le modele energetique n'a pas de solution reelle "
            "pour certaines puissances mecaniques. "
            "Il faut revoir la plage de fonctionnement ou "
            "les hypotheses sur les pertes."
        )

    # Formule numeriquement plus stable pour la petite racine.
    courant_a = (
        2.0 * terme_constant
        / (
            coefficient_a
            + np.sqrt(discriminant)
        )
    )

    return courant_a


def calculer_grandeurs_energetiques(donnees):
    """
    Calcule les puissances, pertes, rendements et energies.

    Le courant capteur reste un signal de diagnostic.
    Le courant theorique est calcule par resolution du bilan.
    """

    resultat = donnees.copy()

    # --------------------------------------------------------
    # 4.1 Puissance mecanique utile estimee
    # --------------------------------------------------------

    facteur_charge = resultat[
        "load_factor_simulated"
    ].clip(
        lower=0.0,
        upper=1.5,
    )

    rapport_vitesse = (
        resultat["speed_sensor_rpm"].clip(lower=0.0)
        / VITESSE_NOMINALE_RPM
    ).clip(
        lower=0.0,
        upper=1.1,
    )

    resultat["puissance_mecanique_estimee_w"] = (
        PUISSANCE_MECANIQUE_NOMINALE_W
        * facteur_charge
        * rapport_vitesse
    )

    # --------------------------------------------------------
    # 4.2 Courant theorique et puissance electrique
    # --------------------------------------------------------

    puissance_mecanique = resultat[
        "puissance_mecanique_estimee_w"
    ].to_numpy()

    courant_theorique = resoudre_courant_theorique(
        puissance_mecanique
    )

    resultat["courant_theorique_a"] = courant_theorique

    resultat["puissance_electrique_estimee_w"] = (
        COEFFICIENT_PUISSANCE_W_A
        * courant_theorique
    )

    # --------------------------------------------------------
    # 4.3 Pertes constantes et variables
    # --------------------------------------------------------

    resultat["pertes_constantes_estimees_w"] = (
        PERTES_CONSTANTES_W
    )

    resultat["pertes_variables_estimees_w"] = (
        COEFFICIENT_PERTES_W_A2
        * courant_theorique**2
    )

    resultat["pertes_estimees_w"] = (
        resultat["pertes_constantes_estimees_w"]
        + resultat["pertes_variables_estimees_w"]
    )

    # --------------------------------------------------------
    # 4.4 Verification du bilan energetique
    # --------------------------------------------------------

    resultat["residu_bilan_energetique_w"] = (
        resultat["puissance_electrique_estimee_w"]
        - resultat["puissance_mecanique_estimee_w"]
        - resultat["pertes_estimees_w"]
    )

    # --------------------------------------------------------
    # 4.5 Rendement estime
    # --------------------------------------------------------

    puissance_electrique = resultat[
        "puissance_electrique_estimee_w"
    ].to_numpy()

    puissance_mecanique = resultat[
        "puissance_mecanique_estimee_w"
    ].to_numpy()

    resultat["rendement_estime_pct"] = np.divide(
        100.0 * puissance_mecanique,
        puissance_electrique,
        out=np.zeros_like(puissance_electrique),
        where=puissance_electrique > 0.0,
    )

    # --------------------------------------------------------
    # 4.6 Comparaison des courants
    # --------------------------------------------------------

    resultat["ecart_courant_a"] = (
        resultat["current_sensor_a"]
        - resultat["courant_theorique_a"]
    )

    resultat["ecart_courant_relatif_pct"] = np.divide(
        100.0 * resultat["ecart_courant_a"],
        resultat["courant_theorique_a"],
        out=np.full(len(resultat), np.nan),
        where=resultat["courant_theorique_a"].to_numpy() > 0.0,
    )

    # --------------------------------------------------------
    # 4.7 Energie cumulee par scenario
    # --------------------------------------------------------

    resultat["energie_electrique_cumulee_kwh"] = 0.0
    resultat["energie_mecanique_cumulee_kwh"] = 0.0
    resultat["energie_pertes_cumulee_kwh"] = 0.0

    for identifiant, indices_groupe in resultat.groupby(
        "scenario_id",
        sort=False,
    ).groups.items():

        indices = list(indices_groupe)

        groupe = resultat.loc[indices].sort_values("time_s")

        temps = groupe["time_s"].to_numpy()

        puissances = {
            "energie_electrique_cumulee_kwh":
                groupe["puissance_electrique_estimee_w"].to_numpy(),

            "energie_mecanique_cumulee_kwh":
                groupe["puissance_mecanique_estimee_w"].to_numpy(),

            "energie_pertes_cumulee_kwh":
                groupe["pertes_estimees_w"].to_numpy(),
        }

        for nom_colonne, puissance in puissances.items():

            energie = np.zeros(len(temps), dtype=float)

            if len(temps) > 1:
                increments = (
                    0.5
                    * (puissance[:-1] + puissance[1:])
                    * np.diff(temps)
                    / 3_600_000.0
                )

                energie[1:] = np.cumsum(increments)

            resultat.loc[
                groupe.index,
                nom_colonne,
            ] = energie

    return resultat


# ============================================================
# 5. SYNTHESE PAR SCENARIO
# ============================================================

def creer_synthese(donnees):
    """Construit le bilan energetique de chaque scenario."""

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

        pertes = groupe[
            "pertes_estimees_w"
        ].to_numpy()

        rendement = groupe[
            "rendement_estime_pct"
        ].to_numpy()

        courant_capteur = groupe[
            "current_sensor_a"
        ].to_numpy()

        courant_theorique = groupe[
            "courant_theorique_a"
        ].to_numpy()

        duree_s = (
            float(temps[-1] - temps[0])
            if len(temps) > 1
            else 0.0
        )

        def integrer_kwh(signal):
            if len(temps) < 2:
                return 0.0

            return float(
                np.trapezoid(signal, temps)
                / 3_600_000.0
            )

        lignes.append({
            "scenario_id": int(identifiant),
            "scenario_type": type_scenario,
            "duree_s": duree_s,

            "puissance_electrique_moyenne_w":
                float(np.mean(puissance_elec)),

            "puissance_mecanique_moyenne_w":
                float(np.mean(puissance_meca)),

            "pertes_moyennes_w":
                float(np.mean(pertes)),

            "pertes_constantes_moyennes_w":
                float(groupe["pertes_constantes_estimees_w"].mean()),

            "pertes_variables_moyennes_w":
                float(groupe["pertes_variables_estimees_w"].mean()),

            "rendement_moyen_pct":
                float(np.mean(rendement)),

            "energie_electrique_kwh":
                integrer_kwh(puissance_elec),

            "energie_mecanique_kwh":
                integrer_kwh(puissance_meca),

            "energie_pertes_kwh":
                integrer_kwh(pertes),

            "courant_capteur_moyen_a":
                float(np.mean(courant_capteur)),

            "courant_theorique_moyen_a":
                float(np.mean(courant_theorique)),

            "ecart_courant_absolu_moyen_a":
                float(np.mean(np.abs(
                    courant_capteur - courant_theorique
                ))),

            "residu_bilan_max_absolu_w":
                float(np.max(np.abs(
                    groupe["residu_bilan_energetique_w"]
                ))),
        })

    return pd.DataFrame(lignes)


# ============================================================
# 6. EXPORT DES RESULTATS
# ============================================================

def sauvegarder_resultats(donnees, synthese):
    """Enregistre les resultats detailles et la synthese."""

    fichier_detail = (
        DOSSIER_RESULTATS
        / "donnees_energetiques_detaillees.csv"
    )

    fichier_synthese = (
        DOSSIER_RESULTATS
        / "bilan_energetique_par_scenario.csv"
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

    print("\nFichiers enregistres :")
    print(f"  Donnees detaillees : {fichier_detail}")
    print(f"  Bilan par scenario : {fichier_synthese}")


# ============================================================
# 7. GRAPHIQUES COMPARATIFS
# ============================================================

def creer_graphiques(synthese):
    """Cree une figure avec quatre graphiques comparatifs."""

    fichier_figure = (
        DOSSIER_RESULTATS
        / "step12_bilan_energetique.png"
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 10),
    )

    fig.suptitle(
        "Etape 12 — Bilan energetique du moteur simule",
        fontsize=15,
        fontweight="bold",
    )

    etiquettes = synthese["scenario_id"].astype(str)

    # Puissances moyenne electrique et mecanique.
    axes[0, 0].bar(
        etiquettes,
        synthese["puissance_electrique_moyenne_w"],
        label="Puissance electrique",
    )

    axes[0, 0].bar(
        etiquettes,
        synthese["puissance_mecanique_moyenne_w"],
        label="Puissance mecanique",
    )

    axes[0, 0].set_title("Puissances moyennes")
    axes[0, 0].set_ylabel("Puissance (W)")
    axes[0, 0].set_xlabel("Identifiant du scenario")
    axes[0, 0].legend()
    axes[0, 0].grid(axis="y", alpha=0.3)

    # Pertes moyennes.
    axes[0, 1].bar(
        etiquettes,
        synthese["pertes_constantes_moyennes_w"],
        label="Pertes constantes",
    )

    axes[0, 1].bar(
        etiquettes,
        synthese["pertes_variables_moyennes_w"],
        bottom=synthese["pertes_constantes_moyennes_w"],
        label="Pertes variables",
    )

    axes[0, 1].set_title("Repartition des pertes moyennes")
    axes[0, 1].set_ylabel("Pertes (W)")
    axes[0, 1].set_xlabel("Identifiant du scenario")
    axes[0, 1].legend()
    axes[0, 1].grid(axis="y", alpha=0.3)

    # Energie electrique consommee.
    axes[1, 0].bar(
        etiquettes,
        synthese["energie_electrique_kwh"],
    )

    axes[1, 0].set_title("Energie electrique consommee")
    axes[1, 0].set_ylabel("Energie (kWh)")
    axes[1, 0].set_xlabel("Identifiant du scenario")
    axes[1, 0].grid(axis="y", alpha=0.3)

    # Rendement moyen.
    axes[1, 1].bar(
        etiquettes,
        synthese["rendement_moyen_pct"],
    )

    axes[1, 1].set_title("Rendement moyen estime")
    axes[1, 1].set_ylabel("Rendement (%)")
    axes[1, 1].set_xlabel("Identifiant du scenario")
    axes[1, 1].grid(axis="y", alpha=0.3)

    for axe in axes.flat:
        axe.tick_params(axis="x", rotation=90)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    fig.savefig(
        fichier_figure,
        dpi=200,
        bbox_inches="tight",
    )

    print(f"\nGraphique enregistre : {fichier_figure}")

    plt.show()


# ============================================================
# 8. PROGRAMME PRINCIPAL
# ============================================================

def main():
    """Execute le suivi energetique ameliore."""

    print("=" * 75)
    print("ETAPE 12 — MODELE ENERGETIQUE AMELIORE")
    print("=" * 75)

    print("\nParametres nominaux :")
    print(
        f"Puissance mecanique nominale : "
        f"{PUISSANCE_MECANIQUE_NOMINALE_W:.1f} W"
    )
    print(
        f"Puissance electrique nominale : "
        f"{PUISSANCE_ELECTRIQUE_NOMINALE_W:.1f} W"
    )
    print(
        f"Pertes nominales totales : "
        f"{PERTES_NOMINALES_W:.1f} W"
    )
    print(
        f"Pertes constantes : "
        f"{PERTES_CONSTANTES_W:.1f} W"
    )
    print(
        f"Pertes variables nominales : "
        f"{PERTES_VARIABLES_NOMINALES_W:.1f} W"
    )
    print(
        f"Courant nominal theorique : "
        f"{COURANT_NOMINAL_A:.2f} A"
    )

    print("\nChargement des donnees...")
    donnees = charger_donnees()

    print(f"Nombre d'observations : {len(donnees):,}")
    print(
        f"Nombre de scenarios : "
        f"{donnees['scenario_id'].nunique()}"
    )

    print("\nCalcul des grandeurs energetiques...")
    donnees = calculer_grandeurs_energetiques(donnees)

    print("Construction du bilan par scenario...")
    synthese = creer_synthese(donnees)

    sauvegarder_resultats(donnees, synthese)

    # Verification numerique du bilan energetique.
    residu_max = float(
        donnees["residu_bilan_energetique_w"]
        .abs()
        .max()
    )

    print("\nVerification du bilan energetique :")
    print(f"Residuel maximal : {residu_max:.10f} W")

    if residu_max < 1e-6:
        print("Bilan energetique coherent numeriquement.")
    else:
        print("Attention : verifier le bilan energetique.")

    print("\nSynthese par scenario :")

    colonnes = [
        "scenario_id",
        "scenario_type",
        "puissance_electrique_moyenne_w",
        "puissance_mecanique_moyenne_w",
        "pertes_moyennes_w",
        "rendement_moyen_pct",
        "energie_electrique_kwh",
        "courant_theorique_moyen_a",
        "ecart_courant_absolu_moyen_a",
    ]

    print(
        synthese[colonnes]
        .round(3)
        .to_string(index=False)
    )

    creer_graphiques(synthese)

    print("\n" + "=" * 75)
    print("ETAPE 12 TERMINEE")
    print("=" * 75)

    print(
        "\nImportant : les resultats reposent sur des scenarios "
        "synthetiques et des lois simplifiees. Une validation "
        "experimentale est necessaire avant toute utilisation industrielle."
    )


if __name__ == "__main__":
    main()