Le module utils_pl.py contient different fonctions pour le pretraitement des images, features extraction, resize des images, ...
Le module PL_FINAL contient le programme qui en input recois une image et en output nous affiche la plaque d'immatriculation ainsi que le "string" associé
	Pour cela, le programme se décompose en plusieurs partie:
		--- importation model
		--- Deplacement vers l'image input
		--- Resize de l'input en 224X224
		--- Features extraction de l'input
		--- Prediction avec le model charge au tout début
		--- OCR
		--- Affichage image