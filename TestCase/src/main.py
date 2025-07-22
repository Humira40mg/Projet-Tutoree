import json
import time
import requests
from os import listdir, path
from time import sleep
from datetime import datetime

resultats: dict = {}

# Chargement config
with open("TestCase/config.json", "r", encoding="utf-8") as f:
    config: dict = json.load(f)

debut_global = time.time()

# Test pour chaque image
for image in listdir("TestCase/img"):

    absolutePathImg = path.abspath(path.join("TestCase/img", image))

    payload = {
        "model": config["model"],
        "system": config["systemprompt"],
        "prompt": f"{absolutePathImg} {config['testprompt']}",
        "stream": False
    }

    debut = time.time()
    response = requests.post("http://localhost:11434/api/generate", json=payload)
    fin = time.time()

    if response.ok:
        resultats[image] = {
            "response": response.json()["response"].strip(),
            "temps_analyse": round(fin - debut, 3)
        }
        print(f"Test pour {image} terminé en {round(fin - debut, 3)}s")
    else:
        raise Exception(f"Erreur Ollama API: {response.text}")

    sleep(1)

# Enregistrement des résultats
resultPath = f"resultat-{config['model']}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
with open(f"TestCase/output/{resultPath}", "w", encoding="utf-8") as f:
    json.dump(resultats, f, ensure_ascii=False, indent=4)

# Affichage temps global
fin_global = time.time()
print(f"Temps total de traitement: {round(fin_global - debut_global, 2)}s")
