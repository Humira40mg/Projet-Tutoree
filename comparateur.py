"""
Ce script à été fait pour récuperer tous les fichiers json résultat engendrés par TestCasr/src/main.py
et comparer les résultats afin de les trier de deux façons en deux listes : 
    - temps moyen le plus cours, 
    - temps total le plus cours
"""
import json
from os import listdir
from datetime import datetime

tempsMoyen = []
tempsTotal = []

# ajoute les résultats d'un fichier json aux deux listes
def pushToList(sub_dict: dict, resultfile):
    totalTime = 0
    for val in sub_dict.values() :
        totalTime += val["temps_analyse"]

    # reformat json
    res = {
        'model': resultfile,
        'avg': totalTime/len(list(sub_dict)),
        'total': totalTime,
        'reponses': sub_dict
    }

    tempsMoyen.append(res)
    tempsTotal.append(res)

# lecture de tous les résultats
for resultfile in listdir("TestCase/output"):
    with open(f"TestCase/output/{resultfile}", "r", encoding="utf-8") as f:
        pushToList(json.load(f), resultfile)

# trie des deux listes
tempsMoyen.sort(key=lambda x: x["avg"])
tempsTotal.sort(key=lambda x: x["total"])

# json final
rapport = {
    "tempsMoyen": tempsMoyen,
    "tempsTotal": tempsTotal
}

# écriture du rapport de comparaison
resultPath = f"rapport[NVMe]-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
with open(f"{resultPath}", "w", encoding="utf-8") as f:
    json.dump(rapport, f, ensure_ascii=False, indent=4)