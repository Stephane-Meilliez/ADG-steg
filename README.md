# ADG-steg

Implémentation en Python d'**ADG (Adaptive Dynamic Grouping)**, l'algorithme de
stéganographie linguistique générative de Zhang et al. (2021). Ce dépôt fournit
l'encodeur/décodeur principal et un notebook de démonstration minimal. Il
s'inscrit dans une étude plus large sur la stéganographie linguistique
générative.



## Présentation

ADG dissimule une charge utile de bits dans du texte produit par un modèle de
langage. À chaque étape de génération, la distribution top-k du token suivant
est partitionnée en groupes de probabilités quasi égales ; les bits à cacher
sélectionnent un groupe, puis un token est échantillonné à l'intérieur. Le
message est donc porté par le choix des tokens, tandis que le texte produit
reste un échantillon plausible du modèle de langage.

ADG est prouvé sûr sous des hypothèses théoriques. Dans l'article initial,
l'algorithme est utilisé avec un modèle de langage conçu et entraîné par les
auteurs.

Ici nous l'utilisons avec des modèles pré-entraînés.



Le module implémente :

* `ADG_encode` — dissimule une charge utile de bits dans une séquence stego
* `ADG_decode` — récupère la charge utile à partir d'une séquence stego
* `ADG_generate_cover` — échantillonne une séquence sans message (référence)


## Utilisation 

Voir le notebook `demo_adg.ipynb` pour un exemple complet (encodage, décodage,
génération de cover).

## Prérequis

* Python 3.10+
* `torch`
* `transformers`

## Organisation du dépôt

```
src/ADG.py        implémentation principale d'ADG
src/__init__.py   marqueur de package
demo_adg.ipynb    démonstration d'usage minimale
```

## Référence

Zhang et al., *Provably Secure Generative Linguistic Steganography*, 2021.
 

## Licence

Distribué sous licence MIT. Voir le fichier `LICENSE`.

