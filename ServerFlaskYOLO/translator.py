"""
Module qui fait la traduction des données récupérée pour les restructurer
de façon à ce que celà soit comprhensible plus facilement par un humain.
Vu que les LLM sont entrainé sur le language humain ce sera aussi plus
facile pour le LLM de comprendre les information.

Le main est : stream3.py
"""
def detections_to_text(detections):
    if not detections:
        return "Aucun objet détecté autour du bateau."

    lines = ["Le bateau détecte les objets suivants :"]
    for det in detections:
        label = det["label"]
        conf = round(det["confidence"] * 100)
        x1, y1, x2, y2 = det["box"]
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        lines.append(
            f"- {label} (confiance : {conf}%) à la position [{center_x}, {center_y}]"
        )

    return "\n".join(lines)
