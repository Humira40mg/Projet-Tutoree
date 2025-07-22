import json
import requests
from os import listdir, path
from time import sleep
from datetime import datetime

resultats: dict = {}

#chargement config
with open("TestCase/config.json", "r", encoding="utf-8") as f:
    config: dict = json.load(f)

#Test Pour Chaque image
for image in listdir("TestCase/img"):

    absolutePathImg = path.abspath(image)

    payload = {
                "model": config["model"],
                "system": config["systemprompt"],
                "prompt": f"{absolutePathImg} {config["testprompt"]}",
                "stream": False
            }
    response = requests.post(f"http://localhost:11434/api/generate", json=payload)
    if response.ok:
        resultats[image] = response.json()["response"].strip()
        print(f"Test pour {image} terminé")
    else:
        raise Exception(f"Error Ollama API: {response.text}")
    
    sleep(1)

resultPath = f"resultat-{config["model"]}-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.json"
with open(f"TestCase/output/{resultPath}", "w", encoding="utf-8") as f:
        json.dump(resultats, f, ensure_ascii=False, indent=4)
