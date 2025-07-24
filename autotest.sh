#!/bin/bash

sudo apt install jq

# Chemin vers le fichier JSON
JSON_FILE="TestCase/config.json"

# Lancer le script Python
echo "[🚀] Lancement de src/main.py"
python3 TestCase/src/main.py

# Pause de 500 secondes
echo "[⏳] Attente de 500 secondes..."
sleep 500

# Mise à jour de la clé "model"
jq '.model = "gemma3:4b"' "$JSON_FILE" > tmp.json && mv tmp.json "$JSON_FILE"
echo "[✔] Modèle changé en gemma3:4b dans $JSON_FILE"

# Pause de 500 secondes
echo "[⏳] Attente de 500 secondes..."
sleep 500

# Lancer le script Python
echo "[🚀] Lancement de src/main.py"
python3 TestCase/src/main.py