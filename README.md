# Kaggle PlanesNet

Projet de classification sur le dataset PlanesNet (détection d'avions dans des images satellite 20×20 RGB).

## Données

- **planesnet.json** : données (pixels R,G,B aplatis), labels (0 = pas d'avion, 1 = avion), scene_ids, locations.
- Les images font 20×20 pixels, 3 canaux (1200 valeurs par image).


## Lancer les expériences

Il faut avoir le fichier `Data/planesnet/planesnet.json` (ou un dossier d'images pour les scripts qui le supportent).

Exemples avec le runner :

bash
# Entraîner un CNN
python planesnet_runner.py --algo cnn --mode train --json Data/planesnet/planesnet.json

# Tester le meilleur checkpoint
python planesnet_runner.py --algo cnn --mode test --json Data/planesnet/planesnet.json

# Lancer tous les algorithmes supervisés en train
python planesnet_runner.py --algo all-supervised --mode train --json Data/planesnet/planesnet.json


Chaque script écrit dans son propre dossier (`runs/`, `runs_nn/`, `runs_knn/`, etc.) : modèles, métriques, courbes, matrices de confusion.

## Dépendances

- Python 3
- numpy, PIL, matplotlib
- PyTorch (pour CNN / MLP / ResNet)
- scikit-learn
- joblib (sauvegarde modèles KNN, Bayes)
