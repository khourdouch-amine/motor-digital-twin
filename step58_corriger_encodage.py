from pathlib import Path

source = Path("resultats_step57_experimental/conclusion_generale.txt")
dossier = Path("resultats_step58_experimental")
destination = dossier / "conclusion_generale_corrigee.txt"

texte = source.read_text(encoding="utf-8")

# Corriger le texte si des caractères ont été mal décodés.
try:
    texte_corrige = texte.encode("latin1").decode("utf-8")
    if "numérique" not in texte_corrige and "numÃ©rique" in texte:
        texte_corrige = texte
except (UnicodeEncodeError, UnicodeDecodeError):
    texte_corrige = texte

dossier.mkdir(parents=True, exist_ok=True)
destination.write_text(texte_corrige, encoding="utf-8")

print("Étape 58 — Correction de l'encodage")
print(f"Document créé : {destination.resolve()}")
print(f"Texte accentué correctement détecté : {'numérique' in texte_corrige}")
