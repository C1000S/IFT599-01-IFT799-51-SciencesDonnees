# TP1 — Notes détaillées (concepts + implémentation)

Ce document explique, étape par étape, tout ce qui a été fait jusqu'ici : le raisonnement
derrière chaque décision, les formules utilisées, et comment le code les implémente.
Objectif : servir de base pour rédiger le rapport final et de référence si on doit
revenir sur une décision plus tard.

## 1. Le problème posé

Le jeu de données **TCGA-PANCAN-HiSeq-801x20531** contient les profils d'expression
génétique de 801 patients atteints de 5 types de cancer (BRCA, KIRC, COAD, LUAD, PRAD),
chaque patient étant décrit par 20531 gènes (variables numériques).

Question du TP : **est-ce que ces 5 types de cancer sont bien séparés** dans l'espace
des données ? On répond à cette question par deux méthodes complémentaires :

- **Méthode 1** : mesure quantitative de séparabilité (sans visualisation), via une
  fonction `Overlap()`.
- **Méthode 2** : visualisation (scatter plots) après sélection de bonnes paires de
  variables (variables originales, ACP, t-SNE).

## 2. Jeu de données réel — vérification

Fichiers extraits dans `TCGA-PANCAN-HiSeq-801x20531/` :
- `data.csv` : 801 lignes (patients) x 20531 colonnes (gènes), ~206 Mo
- `labels.csv` : la classe (type de cancer) de chaque patient

Répartition des classes vérifiée :

| Classe | Nombre de patients |
|---|---|
| BRCA | 300 |
| KIRC | 146 |
| LUAD | 141 |
| PRAD | 136 |
| COAD | 78 |
| **Total** | **801** |

**Remarque** : l'énoncé du TP mentionne "16382 variables", mais le fichier réel contient
20531 gènes. Ce n'est pas une erreur de notre pipeline — probablement une version du
dataset différente de celle utilisée par le professeur. À mentionner brièvement dans le
rapport pour justifier nos chiffres.

## 3. Étape 0 — Réduction de dimension (`src/data_loading.py`, `src/build_dataset.py`)

### Pourquoi réduire

- La distance de **Mahalanobis** nécessite d'inverser une matrice de covariance
  (p x p). Avec p=20531, c'est numériquement et mémoire-lourd, voire instable.
- L'**ACP** sur 20531 variables est faisable mais lourde selon la machine.

### Décisions de conception prises

1. **Covariance globale (poolée)** plutôt que par classe pour Mahalanobis : on calcule
   une seule matrice de covariance sur les 801 patients (toutes classes confondues),
   au lieu d'une matrice par classe. Raison : la plus petite classe (COAD, 78 patients)
   est trop petite pour estimer une covariance stable sur plusieurs centaines de
   variables (il faudrait p << 78). Avec une covariance globale sur 801 patients, on
   peut se permettre p de l'ordre de quelques centaines sans problème d'inversibilité.
2. **Filtrage des gènes à variance nulle avant tout** : des gènes constants
   n'apportent aucune information et peuvent dégrader l'estimation de covariance.
3. **Sélection aléatoire** (avec seed fixe = reproductible) parmi les gènes restants,
   plutôt qu'une sélection "intelligente" — pour éviter de biaiser artificiellement les
   résultats de la Méthode 1 en choisissant à l'avance des gènes qu'on sait
   discriminants.
4. **Standardisation** (centrer-réduire, moyenne 0 / écart-type 1) après la sélection,
   car les échelles d'expression génique varient énormément d'un gène à l'autre, ce qui
   fausserait les distances et l'ACP si on ne standardisait pas.

### Ce que fait le code

`data_loading.py` :
- `load_data()` : charge `data.csv` (matrice X) et `labels.csv` (vecteur y) avec pandas.
- `filter_near_zero_variance(X, threshold=1e-8)` : calcule la variance de chaque colonne
  (gène) sur les 801 patients et ne garde que celles au-dessus du seuil.
- `select_random_subset(X, n_vars, seed)` : tire aléatoirement `n_vars` colonnes parmi
  celles restantes, avec un générateur `numpy` seedé pour la reproductibilité.
- `standardize(X)` : centre-réduit chaque colonne indépendamment.

`build_dataset.py` : exécute le pipeline complet dans l'ordre (charger → filtrer variance
nulle → sous-échantillonner → standardiser → sauvegarder), et affiche les tailles à
chaque étape pour tracer ce qui se passe.

### Résultats de vérification obtenus

- Genes bruts : 20531 → après filtrage variance nulle : 20264 (267 gènes retirés, tous
  avec **variance exactement 0**, aucun cas limite ambigu au seuil choisi).
- Sous-ensemble final retenu : **200 gènes** (seed=42), sauvegardés dans
  `data_cache/selected_genes.txt` (liste des noms de gènes, pas les données — conforme
  à la consigne "ne pas fournir les données, seulement les dimensions choisies").
- Vérifié : après standardisation, moyenne des colonnes ≈ 0 (1e-17) et écart-type ≈ 1.
- Vérifié : les index de `X` (patients) et `y` (labels) sont parfaitement alignés.
- Nombre de conditionnement de la matrice de covariance (200x200) ≈ 625 → acceptable,
  pas d'instabilité numérique pour l'inversion Mahalanobis. Ratio échantillons/variables
  = 801/200 = 4x (correct, mais pourrait être plus confortable avec moins de variables,
  ex. 100 → ratio 8x, si on veut une covariance plus robuste).

## 4. Méthode 1 — Mesures de distance et Overlap (`src/distances.py`)

### Concepts et formules

Pour une classe C = {x1, ..., xn} de centre x̄_C (moyenne) :

**Distance intra-classe** (cohésion, "à quel point le nuage est étalé") :
```
dist_intra(C) = max{ dist(xi, x̄_C) | xi appartient à C }
```

**Distance inter-classe** entre C1 et C2 (séparation, "à quel point les nuages sont
proches") :
```
dist_inter(C1, C2) = min( dist(C1,C2), dist(C2,C1) )
  où dist(C1,C2) = min{ dist(xi, x̄_C2) | xi appartient à C1 }
     dist(C2,C1) = min{ dist(yj, x̄_C1) | yj appartient à C2 }
```

**Indicateur de chevauchement** :
```
Overlap(C1, C2) = ( dist_intra(C1) + dist_intra(C2) ) / ( 2 * dist_inter(C1, C2) )
```
Si Overlap < 1 → classes bien séparées (condition suffisante, pas nécessaire).

Deux métriques de distance testées :
- **Euclidienne** : traite toutes les variables également, ignore les corrélations.
- **Mahalanobis** : `dist(x,y) = sqrt( (x-y)^T * Σ^-1 * (x-y) )`, corrige pour la
  covariance entre variables (deux gènes très corrélés ne comptent pas "deux fois").

### Ce que fait le code

`distances.py` :
- `class_center(X)` : moyenne des lignes (= centre de la classe).
- `dist_intra(X, metric, VI)` : utilise `scipy.spatial.distance.cdist` pour calculer
  toutes les distances au centre d'un coup, puis prend le max. `VI` est la matrice de
  covariance inverse, seulement utilisée si `metric="mahalanobis"`.
- `dist_inter(X1, X2, metric, VI)` : calcule les deux distances min (X1→centre de X2,
  X2→centre de X1) et retourne le plus petit des deux.
- `overlap(intra1, intra2, inter)` : applique directement la formule.
- `pairwise_overlap(X, y, metric)` : calcule automatiquement, pour **toutes les paires**
  parmi les 5 classes, `dist_intra` de chaque classe, puis `dist_inter` et `Overlap`
  pour chaque combinaison. Retourne une matrice 5x5 symétrique d'Overlap, plus les
  détails intermédiaires (utile pour les tableaux du rapport).
  Pour Mahalanobis, la matrice de covariance (et son inverse `VI`) est calculée **une
  seule fois** sur l'ensemble des 801 patients (toutes classes confondues) — cohérent
  avec la décision de "covariance globale poolée" prise plus haut.

### Vérifications effectuées (avant de faire confiance aux résultats)

1. **Reproduction des calculs à la main** : un exemple numérique simple en 2D
   (3 points par classe) a été calculé manuellement (Overlap ≈ 0.11 pour des classes
   éloignées, Overlap ≈ 3.16 pour des classes qui se chevauchent), puis rejoué avec le
   code — résultats identiques au centième près. Confirme que les formules sont bien
   implémentées.
2. **Test sur données synthétiques bien séparées en haute dimension** : deux classes
   générées aléatoirement en 200D, centrées très loin l'une de l'autre (delta=15 sur
   chaque axe) → Overlap = 0.072 (< 1, correctement détecté comme séparé). Confirme que
   le code est capable de détecter une vraie séparation quand elle existe, et que ce
   n'est pas un problème structurel du code qui empêcherait toujours Overlap < 1.
3. **Vérification de la stabilité numérique de Mahalanobis** : nombre de conditionnement
   de la matrice de covariance ≈ 625 (raisonnable, pas de division par une valeur
   propre proche de zéro qui fausserait l'inversion).

### Résultats obtenus sur les vraies données (`src/run_method1.py`)

Avec les 200 gènes choisis aléatoirement, **toutes les paires de classes ont un
Overlap > 1** (entre ~2.0 et ~2.9), pour Euclidienne **et** Mahalanobis. Aucune paire
n'est "bien séparée" au sens strict du critère.

**Interprétation** (à développer dans le rapport) : c'est un exemple de **malédiction de
la dimensionnalité**. Sur 200 gènes pris au hasard, la grande majorité ne sont
probablement pas discriminants pour le type de cancer — ils ajoutent du bruit à
`dist_intra` sans faire progresser `dist_inter`. C'est précisément la motivation de la
Méthode 2 : au lieu d'utiliser toutes les variables en vrac, chercher activement les
quelques variables qui séparent bien (sélection gloutonne, ACP, t-SNE). Le contraste
"Méthode 1 sur variables brutes" (mauvais) vs "Méthode 2 avec sélection" (attendu
meilleur) est une des analyses comparatives demandées par l'énoncé.

**Correction (verifiee par calcul le 2026-09-16)** : une premiere observation notee ici
affirmait que "COAD-LUAD ressort comme la paire la moins pire avec les deux metriques"
— c'est **faux**, verifie apres coup. En triant les 10 paires par Overlap croissant
pour chaque metrique :
- Euclidienne : meilleure paire = **COAD vs KIRC** (2.062), pire = LUAD vs PRAD (2.857).
  COAD vs LUAD y est en fait 9e sur 10 (2.744, presque la pire).
- Mahalanobis : meilleure paire = **COAD vs LUAD** (2.162), pire = LUAD vs PRAD (2.797).

Les deux métriques **s'accordent sur la pire paire** (LUAD vs PRAD dans les deux cas)
mais **pas sur la meilleure** (COAD-KIRC en Euclidienne, COAD-LUAD en Mahalanobis) —
cohérent avec le fait que Mahalanobis tient compte des corrélations entre gènes,
contrairement à Euclidienne qui traite chaque variable indépendamment. C'est ce
désaccord de classement qu'il faut citer dans le rapport, pas une fausse concordance.

## 5. Méthode 2 — Sélection de paires de variables originales (`src/variable_selection.py`)

### Concept : Sequential Forward Selection (SFS)

Comme prévu, deux critères d'évaluation distincts sont utilisés (l'énoncé exige qu'ils
ne soient pas les mêmes) :

- **Critère global** (`global_score`) : Overlap moyen sur les 10 paires de classes.
  Sert à trouver LA paire de variables qui distingue le mieux les 5 classes
  simultanément.
- **Critère local** (`local_score`) : Overlap d'une seule paire de classes (ex. BRCA vs
  KIRC), calculé uniquement sur les patients de ces 2 classes. Sert à trouver la
  meilleure paire de variables spécifique à chaque combinaison de 2 classes.

L'algorithme `sfs_select_pair` :
1. Teste chaque variable seule (en 1D), garde celle qui minimise le score → `v1`.
2. Teste chaque variable restante combinée à `v1` (en 2D), garde celle qui minimise le
   score → `v2`.
3. Retourne la paire `(v1, v2)`. Le critère d'arrêt est simplement "on veut une paire",
   donc 2 étapes suffisent (pas besoin de critère d'arrêt plus complexe ici).

`select_global_pair` applique ça une fois avec le critère global.
`select_all_local_pairs` répète l'opération pour chacune des 10 paires de classes avec
le critère local.

### Bug numérique rencontré et corrigé

Lors du premier essai, un `RuntimeWarning: divide by zero` est apparu dans `overlap()`.
Cause : certaines paires de classes partagent exactement la même valeur minimale sur un
gène très épars (`dist_inter = 0`), rendant Overlap mathématiquement infini. Corrigé en
retournant explicitement `np.inf` dans ce cas (sémantiquement correct : chevauchement
total), au lieu de laisser la division par zéro déclencher un warning silencieux.

### Découverte importante : sensibilité aux outliers en basse dimension

Premier test surprenant : une paire de variables sélectionnée donnait un Overlap de
**29.5**, bien pire que le résultat en 200 dimensions de la Méthode 1 (~2-3) ! Vérifié
que ce n'est pas un bug : `dist_intra` utilise un **maximum**, donc un seul patient
avec une valeur extrême (ex. `gene_14110`, z-score = 7.55 alors que 88.1% des valeurs de
ce gène sont identiques, verifie par calcul exact — typique d'un gène très peu exprimé sauf chez de rares
patients) suffit à faire exploser la distance intra-classe **quand on ne regarde que
1-2 variables**. En 200 dimensions, ce même outlier est "dilué" dans la somme des 200
écarts au carré (distance euclidienne), donc moins dominant.

**Nuance à ajouter à l'explication de la Méthode 1** : ce n'est pas la malédiction de la
dimensionnalité qui explique la différence basse-dim vs haute-dim ici, mais bien la
**sensibilité aux valeurs extrêmes de la formule max/min** — un effet distinct, propre
à des données RNA-seq très asymétriques (beaucoup de gènes avec une majorité de valeurs
basses/nulles et de rares pics d'expression).

### Résultats obtenus (`src/run_method2_original_vars.py`, figures dans `figures/method2_original_vars/`)

- **Paire globale (5 classes)** : `(gene_16680, gene_11393)`, Overlap moyen = **16.72**
  → mauvais, attendu : difficile de séparer 5 classes à la fois avec seulement 2 gènes
  bruts choisis parmi 200 candidats aléatoires.
- **Paires locales (par couple de classes)** : résultats très hétérogènes —
  - 2 paires atteignent Overlap < 1 (bien séparées) : **COAD vs KIRC (0.83)** et
    **COAD vs PRAD (0.93)**.
  - Plusieurs restent très mauvaises : **KIRC vs LUAD (11.19)**, **BRCA vs LUAD
    (10.64)**, **BRCA vs COAD (7.76)**.
  - Les autres sont intermédiaires (1.15 à 2.07).

**Vérification visuelle** : le scatter plot COAD vs KIRC (Overlap=0.83) montre deux
nuages de points clairement séparés spatialement — cohérent avec le score. Le scatter
KIRC vs LUAD (Overlap=11.19) montre les deux classes empilées ensemble sur l'axe
`gene_14110` (le gène à outliers identifié plus haut) — le score élevé reflète un vrai
chevauchement visuel, pas une anomalie de calcul.

**Point pour le rapport** : ce contraste (certaines paires de classes se séparent très
bien avec seulement 2 gènes bien choisis, d'autres non) illustre que la difficulté de
séparation dépend de la paire de classes, pas seulement de la méthode — et motive de
comparer avec l'ACP (qui combine l'info de plusieurs gènes, diluant l'effet outlier
comme observé en 200D) pour voir si les paires difficiles s'améliorent.

## 5bis. Méthode 2 — ACP (`src/run_method2_pca.py`)

### Ce qui a été fait

`sklearn.decomposition.PCA` avec `n_components=15` appliqué sur les 200 gènes déjà
standardisés. Les 15 composantes expliquent **57.2%** de la variance totale (première
composante seule : 12.9%). Puis exactement le même algorithme de sélection SFS
(`select_global_pair`, `select_all_local_pairs`) est réutilisé, mais avec les colonnes
`PC1..PC15` à la place des gènes bruts comme candidats.

### Résultat surprenant : l'ACP n'améliore pas systématiquement la séparation

| Comparaison | Variables brutes | ACP (15 PC) | Delta |
|---|---|---|---|
| Toutes classes | 16.72 | 16.27 | -0.44 (léger mieux) |
| BRCA vs COAD | 7.76 | 2.45 | **-5.31 (bien mieux)** |
| BRCA vs KIRC | 1.60 | 2.81 | +1.21 (pire) |
| BRCA vs LUAD | 10.64 | 10.52 | -0.12 (quasi identique) |
| BRCA vs PRAD | 2.07 | 2.26 | +0.20 (pire) |
| COAD vs KIRC | 0.83 | 1.51 | +0.68 (pire, passe sous le seuil de "bien separe") |
| COAD vs LUAD | 2.01 | 3.13 | +1.12 (pire) |
| COAD vs PRAD | 0.93 | 0.80 | -0.13 (mieux) |
| KIRC vs LUAD | 11.20 | 3.92 | **-7.28 (bien mieux)** |
| KIRC vs PRAD | 1.15 | 1.52 | +0.36 (pire) |
| LUAD vs PRAD | 1.57 | 1.25 | -0.32 (mieux) |

**6 paires sur 10 s'améliorent, 4 empirent** — un vrai mélange, pas une amélioration
systématique. Deux explications complémentaires (vérifiées, pas juste supposées) :

1. **L'ACP est non supervisée** : elle maximise la variance totale sans utiliser les
   labels de classe. Rien ne garantit qu'une direction de forte variance corresponde à
   une direction qui sépare les classes. Une hypothèse initiale ("l'outlier de
   `gene_14110` domine une composante") a été testée et **infirmée** : ses loadings
   sont faibles sur toutes les composantes (max 0.088, contre 0.15-0.22 pour les gènes
   qui dominent réellement PC1-PC3). Ce n'est donc pas la concentration d'un outlier
   qui explique les cas dégradés, mais bien le fait que l'ACP mélange des gènes
   discriminants (ex. `gene_11393`, `gene_9067`, très bons pour COAD vs KIRC) avec
   d'autres gènes qui portent de la variance sans pouvoir séparatoire, diluant le
   signal utile.
2. **La sensibilité aux outliers de `Overlap()` persiste dans l'espace ACP** : vérifié
   visuellement sur COAD vs KIRC (voir `figures/method2_pca/COAD_vs_KIRC.png`) — les
   deux nuages restent visuellement bien distincts, mais un seul patient KIRC isolé
   (PC2 ≈ 13.3, loin du reste de son groupe) suffit à gonfler `dist_intra(KIRC)` et
   donc l'Overlap, alors que la séparation visuelle globale est bonne. Le problème
   n'est donc pas propre aux données RNA-seq brutes : la définition même de `Overlap()`
   (max/min) reste fragile à quelques points extrêmes, peu importe la transformation.

**Point clé pour le rapport** : "ACP améliore la séparabilité" n'est pas une règle
générale à énoncer sans nuance — ça dépend de la paire de classes, et ce TP en donne
une démonstration empirique claire à discuter (ACP = variance-driven et non
supervisée, donc pas alignée par construction avec la tâche de séparation de classes).

## 5ter. Méthode 2 — t-SNE (`src/run_method2_tsne.py`)

### Ce qui a été fait

`sklearn.manifold.TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")`.
Contrairement à l'ACP, pas de sélection de paire à faire (embedding direct en 2D) —
mais comme discuté, t-SNE optimise localement, donc on doit **relancer l'algorithme
séparément** :
- une fois sur les 801 patients (5 classes) → l'embedding "global"
- une fois par paire de classes (sur le sous-ensemble des 2 classes seulement, ex. 300
  BRCA + 146 KIRC = 446 patients) → 10 embeddings "locaux" différents

Chaque run utilise les 200 gènes standardisés comme entrée (pas seulement 2 variables).
`Overlap()` est ensuite calculé sur l'embedding 2D résultant, uniquement pour pouvoir
comparer avec les deux autres approches — l'énoncé ne l'exige pas explicitement pour
la Méthode 2, mais c'est nécessaire pour l'analyse comparative demandée en section 2.2
point 3.

### Résultats : nette amélioration, mais pas uniforme

| Comparaison | Variables brutes | ACP (15 PC) | t-SNE |
|---|---|---|---|
| Toutes classes | 16.72 | 16.27 | **2.60** |
| BRCA vs COAD | 7.76 | 2.45 | 3.64 |
| BRCA vs KIRC | 1.60 | 2.81 | 4.55 |
| BRCA vs LUAD | 10.64 | 10.52 | 2.90 |
| BRCA vs PRAD | 2.07 | 2.26 | 1.47 |
| COAD vs KIRC | 0.83 | 1.51 | **0.33** |
| COAD vs LUAD | 2.01 | 3.13 | 2.41 |
| COAD vs PRAD | 0.93 | 0.80 | **0.28** |
| KIRC vs LUAD | 11.20 | 3.92 | **0.66** |
| KIRC vs PRAD | 1.15 | 1.52 | 1.77 |
| LUAD vs PRAD | 1.57 | 1.25 | 1.31 |
| **Nb paires < 1** | 2/10 | 1/10 | **3/10** |

Vérifié visuellement (`figures/method2_tsne/KIRC_vs_LUAD.png`) : les deux nuages sont
nettement séparés spatialement, cohérent avec Overlap=0.66.

**Interprétation** : t-SNE exploite les 200 gènes simultanément et de façon
non-linéaire (contrairement à ACP qui est linéaire, et à la sélection de 2 gènes bruts
qui n'en utilise que 2), ce qui explique la nette amélioration pour l'embedding global
et pour les cas auparavant les plus difficiles (KIRC vs LUAD : 11.20 → 0.66). Mais
**ce n'est pas universel** : BRCA vs KIRC est en fait moins bon avec t-SNE (4.55) qu'avec
la simple paire de gènes bruts (1.60). t-SNE n'est pas non plus supervisé — il préserve
les voisinages locaux dans l'espace à 200 dimensions, sans notion de "classe" — donc il
n'y a aucune garantie qu'il améliore chaque paire individuellement, malgré sa nette
supériorité en moyenne. **Aucune des trois méthodes ne domine sur toutes les paires** :
c'est le message central à faire ressortir dans l'analyse comparative du rapport.

Rappel important pour la discussion : contrairement à l'ACP, les distances dans un
embedding t-SNE ne sont pas directement comparables à des distances euclidiennes
classiques (t-SNE préserve des voisinages, pas des distances globales), donc les
valeurs d'Overlap y sont indicatives mais moins rigoureusement interprétables qu'avec
Euclidienne/Mahalanobis sur données brutes ou ACP.

## 6. Fichiers du projet à ce stade

```
IFT799-SD/TPs/TP1/
├── TCGA-PANCAN-HiSeq-801x20531/   # données brutes extraites (ne pas soumettre)
├── data_cache/                    # sous-ensemble reduit + standardise (ne pas soumettre)
│   ├── X_reduced.csv
│   ├── y.csv
│   └── selected_genes.txt         # <- ceci, par contre, va dans le rapport/remise
├── figures/
│   ├── method2_original_vars/     # scatter plots + summary.csv (Methode 2, vars brutes)
│   ├── method2_pca/                # scatter plots + summary.csv (Methode 2, ACP 15 PC)
│   ├── method2_tsne/               # scatter plots + summary.csv (Methode 2, t-SNE)
│   └── comparison_3_methods.csv    # tableau recapitulatif brut/ACP/t-SNE
├── src/
│   ├── data_loading.py            # chargement + reduction de dimension
│   ├── build_dataset.py           # pipeline complet, genere data_cache/
│   ├── distances.py               # dist_intra, dist_inter, Overlap (Methode 1)
│   ├── run_method1.py             # calcule et affiche les tableaux de Methode 1
│   ├── variable_selection.py      # SFS + criteres global/local (Methode 2)
│   ├── plotting.py                # scatter plots colores par classe (palette fixe)
│   ├── run_method2_original_vars.py  # selection + figures, variables originales
│   ├── run_method2_pca.py         # ACP (15 composantes) + selection + figures
│   └── run_method2_tsne.py        # t-SNE global + par paire de classes + figures
├── requirements.txt
├── .venv/                          # environnement virtuel Python (ne pas soumettre)
└── NOTES_TP1.md                    # ce fichier
```

## 7. Ce qu'il reste à faire

Le cœur technique (Méthode 1 + les 3 approches de Méthode 2) est complet et vérifié.

**Histogrammes optionnels (fait)** : plutôt que de produire des figures séparées,
`plotting.scatter_with_marginals` combine le nuage de points obligatoire avec les
histogrammes 1D par classe en marge (haut = distribution de var1, droite = distribution
de var2), dans le style d'un `seaborn.jointplot`. Utilisé par défaut dans les 3 scripts
`run_method2_*.py` — toutes les figures dans `figures/` ont donc déjà les histogrammes.
Bonus pédagogique observé sur `figures/method2_tsne/COAD_vs_PRAD.png` : la séparation y
est presque entièrement portée par l'axe tSNE1 (histogramme du haut quasi disjoint),
alors que tSNE2 seul (histogramme de droite) chevauche beaucoup — bon exemple concret
pour le rapport de pourquoi la 2e dimension reste nécessaire même quand une seule
variable domine.

Reste à faire pour finaliser la remise :

1. **Rédaction du rapport** : objectif + démarche de chaque méthode, tableaux (Méthode
   1), figures (Méthode 2), et surtout la section d'analyse comparative — la matière
   est déjà largement présente dans ce fichier (sections 4, 5, 5bis, 5ter) :
   - Transformation vs non-transformation : voir tableau comparatif section 5ter.
   - Euclidienne vs Mahalanobis : voir section 4 (résultats proches, mêmes
     conclusions qualitatives, mais classements légèrement différents).
   - ACP vs t-SNE : voir section 5ter (t-SNE bien meilleur en moyenne et sur
     l'embedding global, mais aucune méthode ne domine sur toutes les paires de
     classes individuellement).
2. **Citations des sources** : `sklearn.decomposition.PCA` et `sklearn.manifold.TSNE`
   utilisés tels quels (scikit-learn, https://scikit-learn.org) — à citer dans le
   rapport et/ou en commentaire dans le code, conformément à la consigne anti-plagiat.
3. **Identification de l'équipe** (noms, CIPs/matricules, classe IFT599-1/IFT799-51/
   IFT799-52) à ajouter dans le rapport et dans les fichiers de code.
