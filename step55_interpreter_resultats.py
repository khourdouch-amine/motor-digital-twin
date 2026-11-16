from pathlib import Path
import pandas as pd

DOSSIER_SOURCE = Path("resultats_step53_experimental")
DOSSIER_SORTIE = Path("resultats_step55_experimental")

FICHIER_MODELES = DOSSIER_SOURCE / "synthese_performances_modeles.csv"
FICHIER_ROBUSTESSE = DOSSIER_SOURCE / "synthese_robustesse.csv"

def verifier_colonnes(df, colonnes, nom_fichier):
    manquantes = [c for c in colonnes if c not in df.columns]
    if manquantes:
        raise ValueError(
            f"Colonnes manquantes dans {nom_fichier} : {manquantes}\n"
            f"Colonnes disponibles : {list(df.columns)}"
        )

def trouver_ligne_reference(df):
    candidats = df.copy()
    if "modele" in candidats.columns:
        correspondances = candidats[
            candidats["modele"].astype(str).str.contains(
                "reference|random forest|forêt aléatoire|rf",
                case=False,
                regex=True
            )
        ]
        if not correspondances.empty:
            return correspondances.iloc[0]
    return candidats.iloc[0]

def main():
    if not FICHIER_MODELES.exists():
        raise FileNotFoundError(f"Fichier introuvable : {FICHIER_MODELES}")
    if not FICHIER_ROBUSTESSE.exists():
        raise FileNotFoundError(f"Fichier introuvable : {FICHIER_ROBUSTESSE}")

    DOSSIER_SORTIE.mkdir(parents=True, exist_ok=True)

    modeles = pd.read_csv(FICHIER_MODELES)
    robustesse = pd.read_csv(FICHIER_ROBUSTESSE)

    print("\n" + "=" * 72)
    print("ETAPE 55 — INTERPRETATION SCIENTIFIQUE DES RESULTATS")
    print("=" * 72)

    print("\nColonnes du tableau des modèles :")
    print(list(modeles.columns))
    print("\nColonnes du tableau de robustesse :")
    print(list(robustesse.columns))

    modeles.to_csv(
        DOSSIER_SORTIE / "copie_synthese_modeles.csv",
        index=False,
        encoding="utf-8-sig"
    )
    robustesse.to_csv(
        DOSSIER_SORTIE / "copie_synthese_robustesse.csv",
        index=False,
        encoding="utf-8-sig"
    )

    lignes = [
        {
            "theme": "Nature des données",
            "constat": "Les données sont entièrement synthétiques.",
            "interpretation": (
                "Les résultats décrivent le comportement du système simulé "
                "et ne prouvent pas une performance sur un moteur industriel réel."
            ),
            "limite": (
                "Une validation sur des mesures expérimentales réelles "
                "reste nécessaire."
            )
        }
    ]

    if not modeles.empty:
        ligne = trouver_ligne_reference(modeles)
        valeurs = ligne.to_dict()
        nom_modele = str(
            valeurs.get("modele", valeurs.get("model", "modèle de référence"))
        )

        for colonne, description in [
            ("accuracy", "Exactitude"),
            ("balanced_accuracy", "Exactitude équilibrée"),
            ("precision", "Précision"),
            ("recall", "Rappel"),
            ("f1", "Score F1"),
            ("roc_auc", "AUC ROC"),
            ("auc", "AUC ROC"),
            ("faux_positifs", "Faux positifs"),
            ("faux_negatifs", "Faux négatifs"),
            ("fp", "Faux positifs"),
            ("fn", "Faux négatifs"),
        ]:
            if colonne in valeurs and pd.notna(valeurs[colonne]):
                lignes.append({
                    "theme": description,
                    "constat": f"{nom_modele} : {valeurs[colonne]}",
                    "interpretation": (
                        "Indicateur calculé sur le jeu de test synthétique "
                        "selon le protocole expérimental du projet."
                    ),
                    "limite": (
                        "La valeur dépend des scénarios simulés, des variables "
                        "choisies et de la séparation entraînement-test."
                    )
                })

    if not robustesse.empty:
        print("\nAperçu des résultats de robustesse :")
        print(robustesse.head(10).to_string(index=False))

        colonnes_bruit = [
            c for c in robustesse.columns
            if "bruit" in c.lower() or "noise" in c.lower()
        ]
        colonnes_performance = [
            c for c in robustesse.columns
            if any(mot in c.lower() for mot in [
                "accuracy", "balanced", "precision", "recall", "f1", "auc"
            ])
        ]

        if colonnes_bruit and colonnes_performance:
            lignes.append({
                "theme": "Robustesse au bruit",
                "constat": (
                    "Les résultats de robustesse sont disponibles dans le "
                    "tableau dédié ; consulter les niveaux de bruit et les "
                    "indicateurs associés avant de conclure."
                ),
                "interpretation": (
                    "Une baisse des performances lorsque le bruit augmente "
                    "indiquerait une sensibilité aux perturbations simulées."
                ),
                "limite": (
                    "Le bruit ajouté aux caractéristiques est une perturbation "
                    "numérique synthétique, pas un modèle complet de capteur réel."
                )
            })

    rapport = pd.DataFrame(lignes)
    chemin_rapport = DOSSIER_SORTIE / "rapport_interpretation.csv"
    rapport.to_csv(chemin_rapport, index=False, encoding="utf-8-sig")

    print("\nRapport préliminaire créé :", chemin_rapport.resolve())
    print("Fichiers de synthèse copiés dans :", DOSSIER_SORTIE.resolve())
    print("\nImportant : les conclusions restent limitées aux données synthétiques.")
    print("Étape 55 — génération terminée.")

if __name__ == "__main__":
    main()
