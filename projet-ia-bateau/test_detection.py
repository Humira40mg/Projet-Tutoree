import cv2
from object_detection import detect_objects

# Charge une image test (remplace par ton propre chemin si besoin)
image = cv2.imread("test.jpg")

# Vérifie que l'image est bien chargée
if image is None:
    print("Erreur : image non trouvée.")
    exit()

# Lancer la détection
detections = detect_objects(image)

# Affiche les résultats
for det in detections:
    print(f"Objet : {det['label']}, Confiance : {det['confidence']:.2f}, Box : {det['box']}")

# Affiche l'image avec les détections (optionnel)
for det in detections:
    x1, y1, x2, y2 = det["box"]
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, det["label"], (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

cv2.imshow("Image détectée", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
