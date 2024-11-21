import cv2


image_path = 'samples/train/0_x.png'
image = cv2.imread(image_path)

# Callback per ottenere il colore cliccato
def get_color(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Ottieni il colore BGR al punto cliccato
        b, g, r = image[y, x]
        print(f'BGR Color: ({b}, {g}, {r})')
        # Salva il colore per l'uso futuro
        with open("target_color.txt", "w") as f:
            f.write(f"{b},{g},{r}")
        cv2.destroyAllWindows()

# Mostra l'immagine e imposta il callback del mouse
cv2.imshow('Image', image)
cv2.setMouseCallback('Image', get_color)

cv2.waitKey(0)
cv2.destroyAllWindows()
