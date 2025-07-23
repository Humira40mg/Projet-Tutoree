import json
from os import listdir
from datetime import datetime

tempsMoyen = []
tempsTotal = []

def pushToList(sub_dict: dict, resultfile):
    totalTime = 0
    for val in sub_dict.values() :
        totalTime += val["temps_analyse"]

    res = {
        'model': resultfile,
        'avg': totalTime/len(list(sub_dict)),
        'total': totalTime,
        'reponses': sub_dict
    }

    tempsMoyen.append(res)
    tempsTotal.append(res)

for resultfile in listdir("TestCase/output"):
    with open(f"TestCase/output/{resultfile}", "r", encoding="utf-8") as f:
        pushToList(json.load(f), resultfile)

tempsMoyen.sort(key=lambda x: x["avg"])
tempsTotal.sort(key=lambda x: x["total"])

rapport = {
    "tempsMoyen": tempsMoyen,
    "tempsTotal": tempsTotal
}

resultPath = f"rapport[NVMe]-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
with open(f"rapports/{resultPath}", "w", encoding="utf-8") as f:
    json.dump(rapport, f, ensure_ascii=False, indent=4)