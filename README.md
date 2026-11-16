M2 MALIA Deep Learning
Projet : Étude progressive des réseaux de neurones sur le jeu
de données HAR
Enseignant : Walid Bechkit
Année universitaire 2026–2027
Ce projet vise à construire et à analyser différentes architectures de réseaux de neurones multicouches
(MLP) pour un problème réel de classification : la reconnaissance d’activités humaines à partir de
données de capteurs (HAR dataset). L’objectif est de comprendre pas à pas comment la complexité
du réseau, le choix des fonctions d’activation, des optimiseurs et des techniques de régularisation
influencent les performances du modèle, et de mener cette étude avec une méthodologie rigoureuse :
une référence simple, un ensemble de validation, des résultats répétés sur plusieurs graines, et un
ensemble de test utilisé une seule fois.
Partie 1 : Jeu de données
1.1 Le jeu de données HAR (Human Activity Recognition) provient du projet Human Activity
Recognition Using Smartphones. Les fichiers d’entraînement et de test sont fournis ici
kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones sous la forme :
— ./HAR/train.csv
— ./HAR/test.csv
Chaque exemple correspond à une fenêtre de mesures enregistrée par le smartphone d’un volontaire ;
il est décrit par plusieurs centaines de caractéristiques calculées à partir des accéléromètres et des gyroscopes. La colonne subject indique le volontaire concerné. La colonne cible est Activity (6 classes correspondant aux activités : WALKING, STANDING, SITTING, LAYING, WALKING_UPSTAIRS,
WALKING_DOWNSTAIRS). Les volontaires du fichier de test ne figurent pas dans le fichier d’entraînement : on évalue donc le modèle sur des personnes nouvelles.
1.2 Commencez par une prise en main exploratoire des données. Vous devez :
— afficher le nombre d’observations, de variables et de volontaires (dans chaque fichier) ;
— examiner la distribution des classes ;
— calculer des statistiques descriptives (moyenne, écart-type, minimum, maximum) ;
— produire quelques visualisations simples (histogrammes, boxplots, corrélations).
Discutez des besoins éventuels en prétraitement (normalisation, encodage, suppression
de colonnes, etc.). Les variables sont-elles déjà mises à l’échelle ? La colonne subject doit-elle servir
d’entrée au modèle ?
1.3 Découpage et ensemble de validation Le fichier de test ne doit servir qu’une
seule fois, pour l’évaluation finale des configurations retenues. Tous les choix (architecture, activation,
1
optimiseur, pas d’apprentissage, régularisation, nombre d’époques) doivent être faits sur un ensemble
de validation extrait de train.csv.
— Construisez cet ensemble en réservant quelques volontaires entiers (par exemple 4 ou 5), et
non des lignes tirées au hasard. Expliquez pourquoi : deux fenêtres consécutives d’un même
volontaire se chevauchent et se ressemblent beaucoup.
— Comparez, pour un même modèle, la performance de validation obtenue avec un découpage aléatoire des lignes et avec un découpage par volontaire. Lequel est le plus proche de la performance
de test ? Concluez.
1.4 Une référence simple Avant tout réseau de neurones, entraînez une régression logistique multiclasse (par exemple avec scikit-learn, ou un réseau sans couche cachée en PyTorch). Ses
performances serviront de référence : chaque réseau devra être comparé à elle.
Partie 2 : Première architecture de réseau
2.1 Construisez un réseau de neurones artificiels comportant une seule couche cachée (par exemple
64 neurones). Utilisez une fonction d’activation au choix parmi : ReLU, Sigmoid, Tanh. Choisissez une
fonction de perte adaptée à un problème de classification multiclasse (ex. CrossEntropyLoss). Avant
d’entraîner, vérifiez la perte initiale (quelle valeur attendez-vous ?) et contrôlez que le réseau est capable
de surapprendre un petit lot de données.
2.2 Réalisez plusieurs essais :
— en faisant varier la fonction d’activation ;
— en testant plusieurs algorithmes d’optimisation : SGD, SGD avec momentum, Adam ; pour
chacun, le pas d’apprentissage doit être réglé (par exemple avec un LR range test ou une petite
grille logarithmique) ;
— en suivant les métriques accuracy et loss sur les ensembles d’entraînement et de validation
à chaque époque ;
— réfléchissez à la pertinence de l’accuracy par rapport au F1-score (macro et par classe). Quelle
métrique est la plus informative ici et pourquoi ? Votre réponse dépend-elle de la distribution
des classes observée en partie 1 ? D’autres métriques, comme la matrice de confusion, seront
demandées par la suite.
Commentez vos observations : convergence, stabilité, sous/sur-apprentissage éventuel.
2.3 Variabilité des résultats Les performances sur ce jeu de données sont élevées, et les
écarts entre configurations peuvent être faibles. Pour au moins deux configurations, répétez l’entraînement avec plusieurs graines aléatoires (au moins 3) et donnez la moyenne et l’écart-type des
scores. Un écart entre deux configurations plus petit que cette variabilité ne permet pas de conclure.
Appliquez ce principe à toutes vos comparaisons importantes.
Partie 3 : Extension de l’architecture
3.1 Augmentez progressivement la taille de la couche cachée (ex. 128, 256 neurones) et analysez
l’impact sur les performances et sur l’overfitting. Indiquez le nombre de paramètres de chaque modèle.
3.2 Implémentez ensuite des réseaux plus profonds (2 à 4 couches cachées). Testez différentes combinaisons de fonctions d’activation dans les couches intermédiaires et comparez les résultats obtenus
(accuracy, F1-score), en tenant compte du nombre de paramètres et du temps d’entraînement.
2
Partie 4 : Régularisation et robustesse
4.1 Si vous observez un sur-apprentissage, testez différentes techniques de régularisation comme par
exemple :
— Dropout (p = 0.1, 0.3, 0.5) ;
— Weight Decay (L2 regularization) ;
— Batch Normalization ;
— Early stopping (arrêt anticipé si le score de validation ne s’améliore plus).
4.2 Analysez l’effet de chaque méthode sur la stabilité et la généralisation du modèle. Gardez en
mémoire les 2 ou 3 configurations les plus performantes sur la validation pour la suite.
Partie 5 : Visualisation et analyse des performances
5.1 Courbes d’apprentissage Vous devez tracer l’évolution de la loss et de l’accuracy (ou
F1-score) sur le jeu d’entraînement et le jeu de validation au cours des époques. Cette visualisation permet de suivre la convergence de votre réseau et de détecter d’éventuels problèmes de surapprentissage
ou sous-apprentissage.
5.2 Évaluation finale et matrice de confusion Évaluez une seule fois, sur le jeu de
test, les 2 ou 3 configurations retenues ainsi que la référence de la partie 1, et comparez ces scores à
ceux obtenus en validation. Calculez et visualisez ensuite la matrice de confusion sur le jeu de test.
Cette analyse permet d’identifier quelles classes sont souvent confondues par votre modèle et d’aller
plus en profondeur dans la comparaison des performances entre les différentes architectures testées.
Vous commenterez vos observations et discuterez les classes les plus difficiles à prédire : la confusion
est-elle plausible du point de vue des capteurs ?
Partie 6 : Interprétabilité et réduction de dimension
6.1 Lecture et compréhension Réalisez quelques lectures sur les méthodes de réduction
de dimension telles que t-SNE et UMAP afin de comprendre leur principe et leurs limites.
6.2 Application sur le modèle Vous appliquerez ces méthodes sur les activations de la dernière couche cachée de votre réseau, et, pour comparaison, sur les variables d’entrée brutes. L’objectif
est de visualiser comment le modèle perçoit les classes et comment il projette les exemples dans un
espace latent discriminant. Même si la réduction de dimension rend l’interprétation approximative, ces
visualisations permettent de mieux comprendre le fonctionnement interne du modèle.
6.3 Analyse et comparaison Vous comparerez les visualisations obtenues avec la matrice
de confusion :
— Identifier si les classes qui se chevauchent dans la t-SNE/UMAP correspondent aux classes
souvent confondues dans la matrice de confusion.
— Discuter de ce que le réseau a changé par rapport à l’espace des variables d’entrée.
— Discuter des insights que ces visualisations apportent sur la capacité du modèle à séparer les
différentes activités.
— Commenter les limitations de ces méthodes et les précautions à prendre dans leur interprétation
(paramètres comme la perplexité, distances entre groupes, variabilité d’une exécution à l’autre).
3
Partie 7 : Ouvertures possibles (BONUS)
Pour aller plus loin (optionnel) :
— automatiser la recherche d’hyperparamètres (grid search, random search, autres) ;
— analyser l’importance des variables (feature importance ou ablation) ;
— tester d’autres techniques de normalisation (LayerNorm, etc.) ;
— signaux bruts : la version originale du jeu de données (dépôt UCI) contient aussi les signaux
bruts des capteurs (fenêtres de 128 mesures sur 9 canaux). Entraînez directement sur ces signaux
un réseau convolutif 1D ou un réseau récurrent (partie III du cours), et comparez-le au MLP sur
les caractéristiques calculées à la main. Ce point est réservé aux étudiants qui veulent anticiper
la partie III ; aucune pénalité s’il n’est pas traité.
Partie 8 : Votre rendu
Vous devez soumettre sur Moodle, dans un seul fichier ZIP nommé M2_DL_2026_NOM1_NOM2.zip.
1. Le rapport (format PDF, max 6 pages, détails ci-dessous) ;
2. Le code utilisé, commenté et structuré (2 versions, détails ci-dessous) ;
8.1 Organisation du code Le code doit être écrit sous forme de notebook Jupyter clair et
structuré, et les résultats doivent être reproductibles (graines aléatoires fixées). Deux versions sont
attendues :
1. Version 1 : Réseau à une seule couche cachée (baseline simple) ;
2. Version 2 : Réseau plus profond (3 couches cachées par exemple) avec les meilleures régularisations, visualisation et interprétabilité, et ouvertures (optimisation automatique, métriques
avancées, etc.).
Chaque version doit suivre une structure commune qui suit globalement les sections suivantes :
1. Imports des bibliothèques ;
2. Chargement et exploration des données ;
3. Découpage apprentissage / validation et préparation des DataLoaders ;
4. Définition du modèle et des fonctions d’entraînement/évaluation (inspirez-vous des fonctions
vues en TP, ex. train_epoch et evaluate) ;
5. Boucle d’entraînement principale ;
6. Visualisation et comparaison des résultats ;
7. Évaluation finale sur le jeu de test.
8.2 Rapport Le rapport doit être rédigé comme un court article scientifique, limité à 6 pages,
contenant :
— une introduction motivant la problématique et les choix faits ;
— une présentation du jeu de données, du découpage utilisé et des traitements appliqués ;
— une description de votre (ou vos) architectures, avec leur nombre de paramètres ;
— une analyse des résultats, illustrée par des tableaux et figures (moyennes et écarts-types quand
les entraînements sont répétés) ;
— une discussion critique sur les performances et la généralisation ;
— une conclusion résumant vos observations principales.
Le rapport peut être rédigé en français ou en anglais, au choix. L’utilisation de ChatGPT (ou autre
technologie similaire) est tolérée, à condition que vous indiquiez clairement :
4
— les endroits où vous l’avez utilisé ;
— comment vous avez vérifié et ajusté les résultats générés.
Bon courage ! L’objectif principal n’est pas d’obtenir la meilleure accuracy, mais de comprendre les
mécanismes de l’apprentissage profond et de justifier vos choix.
5