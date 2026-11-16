from pathlib import Path
import pandas as pd

SOURCE = Path("resultats_step53_experimental/synthese_robustesse.csv")
SORTIE = Path("resultats_step56_experimental")

def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Fichier source introuvable : {SOURCE}")

    df = pd.read_csv(SOURCE)
    colonnes = [
        "niveau_bruit",
        "accuracy_mean",
        "recall_defaut_mean",
        "defauts_manques_mean",
        "fausses_alertes_mean",
    ]

    manquantes = [c for c in colonnes if c not in df.columns]
    if manquantes:
        raise ValueError(
            f"Colonnes manquantes : {manquantes}\n"
            f"Colonnes disponibles : {list(df.columns)}"
        )

    df = df.sort_values("niveau_bruit").reset_index(drop=True)
    reference = df.iloc[0]
    final = df.iloc[-1]

    lignes = [
        {
            "axe": "Nature de l'étude",
            "resultat": "Données entièrement synthétiques",
            "discussion": (
                "Les performances mesurent le comportement du modèle dans "
                "l'environnement simulé. Elles ne démontrent pas une "
                "performance industrielle réelle."
            ),
            "limite": (
                "Acquérir des mesures réelles, documenter les capteurs et "
                "valider le modèle sur des moteurs non utilisés à l'entraînement."
            )
        },
        {
            "axe": "Exactitude",
            "resultat": (
                f"{reference['accuracy_mean']:.4f} à 0 % de bruit ; "
                f"{final['accuracy_mean']:.4f} à "
                f"{final['niveau_bruit']:.0%} de bruit"
            ),
            "discussion": (
                f"La variation observée est de "
                f"{final['accuracy_mean'] - reference['accuracy_mean']:.4f}. "
                "L'exactitude seule ne suffit pas pour évaluer un système "
                "de maintenance prédictive."
            ),
            "limite": (
                "Compléter l'analyse par le rappel, la précision, les faux "
                "positifs, les faux négatifs et une validation indépendante."
            )
        },
        {
            "axe": "Rappel des défauts",
            "resultat": (
                f"{reference['recall_defaut_mean']:.4f} à 0 % de bruit ; "
                f"{final['recall_defaut_mean']:.4f} à "
                f"{final['niveau_bruit']:.0%} de bruit"
            ),
            "discussion": (
                "Le rappel mesure la proportion des défauts présents qui "
                "sont détectés. Sa baisse indique une sensibilité accrue "
                "aux perturbations simulées."
            ),
            "limite": (
                "Définir les exigences de détection avec des spécialistes "
                "du moteur et de la sécurité."
            )
        },
        {
            "axe": "Défauts manqués",
            "resultat": (
                f"{reference['defauts_manques_mean']:.2f} en moyenne à 0 % "
                f"de bruit ; {final['defauts_manques_mean']:.2f} à "
                f"{final['niveau_bruit']:.0%}"
            ),
            "discussion": (
                "Le nombre moyen de défauts manqués augmente dans les "
                "perturbations simulées les plus fortes."
            ),
            "limite": (
                "Les nombres moyens proviennent des répétitions expérimentales "
                "simulées ; ils ne sont pas des taux de panne réels."
            )
        },
        {
            "axe": "Fausses alertes",
            "resultat": (
                f"{reference['fausses_alertes_mean']:.2f} en moyenne à 0 % "
                f"de bruit ; {final['fausses_alertes_mean']:.2f} à "
                f"{final['niveau_bruit']:.0%}"
            ),
            "discussion": (
                "Les fausses alertes peuvent augmenter avec le bruit et "
                "entraîner des inspections inutiles."
            ),
            "limite": (
                "Évaluer le compromis entre défauts manqués et fausses alertes "
                "en fonction du coût opérationnel."
            )
        },
        {
            "axe": "Généralisation",
            "resultat": "Généralisation industrielle non démontrée",
            "discussion": (
                "La séparation par scénarios réduit le risque de fuite entre "
                "fenêtres issues d'un même scénario, mais ne remplace pas une "
                "validation sur un équipement réel indépendant."
            ),
            "limite": (
                "Tester plusieurs moteurs, régimes de fonctionnement, charges, "
                "conditions ambiantes et défauts mesurés."
            )
        },
        {
            "axe": "Conclusion",
            "resultat": "Résultats encourageants dans le cadre simulé",
            "discussion": (
                "Le modèle semble distinguer les classes synthétiques et "
                "conserve des performances élevées sous certaines perturbations. "
                "La robustesse se dégrade lorsque le bruit simulé augmente."
            ),
            "limite": (
                "Ne pas revendiquer une validation terrain ni une aptitude "
                "au déploiement industriel avant des essais réels."
            )
        }
    ]

    SORTIE.mkdir(parents=True, exist_ok=True)
    rapport = pd.DataFrame(lignes)
    chemin = SORTIE / "discussion_scientifique.csv"
    rapport.to_csv(chemin, index=False, encoding="utf-8-sig")

    print("=" * 72)
    print("ETAPE 56 — DISCUSSION SCIENTIFIQUE")
    print("=" * 72)
    print(rapport[["axe", "resultat"]].to_string(index=False))
    print(f"\nRapport enregistré : {chemin.resolve()}")
    print("Les résultats restent limités aux données synthétiques.")

if __name__ == "__main__":
    main()
