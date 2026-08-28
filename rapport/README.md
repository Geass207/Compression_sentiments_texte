# Template LaTeX - Rapport de stage

## Table des matières

- [Enregistrer le template](#enregistrer-le-template)
- [Motivations](#motivations)
- [Fonctionnalités](#fonctionnalités)
  - [Classe `ensiie-report`](#classe-ensiie-report)
    - [Options](#options)
    - [Macros](#macros)
  - [Paquet `preamble`](#paquet-preamble)
    - [Commande `fileInput`](#commande-fileinput)
    - [Commande `tcbFileInput`](#commande-tcbfileinput)
    - [Environnement `terminalcode`](#environnement-terminalcode)
    - [Commande `inlinegraphics`](#commande-inlinegraphics)
- [Rédaction du rapport](#rédaction-du-rapport)
- [Compatibilité](#compatibilité)
- [Développement](#développement)
- [FAQ](#faq)



## Enregistrer le template

- Aller sur **https://git.iiens.net/lucas2023/iiepp-latex**.
- Sélectionner le dossier `template`.
- Sélectionner le dossier `internship`.
- Cliquer sur le bouton `Code`.
- Dans la partie *Download directory*, choisir le format de l'archive à télécharger.



## Motivations

Il s'agit d'une proposition de template, inspiré du template de 3A disponible sur le site https://stages.ensiie.fr, pour vous permettre de réaliser simplement votre rapport sans perdre du temps à comprendre certains détails de LaTeX.



## Fonctionnalités

### Classe `ensiie-report`

*Inspiration et reprise de la page de garde du template d'Antoine '**Phoko**' Barbier, disponible à cette adresse : https://gitlab.com/Phokopi/rapport_template.*

Cette classe définit essentiellement des macros, des commandes personnalisées, et le style des pages pour faire un rapport de stage.
La classe de base est `article`, donc les parties du document sont des sections.

#### Options

Des options sont applicables à la classe.

- `language=[fr|en]` | Définition de la langue de votre rapport. Cela ne changera que la langue de la page de garde. Le paquet `babel` n'étant pas utilisé, cela ne changera pas les conventions du document (qui varient surtout pour le français). Langue par défaut : `fr`.
- `year=[1A|2A|3A]` | Définition de l'année de votre stage. Cela changera que l'année indiquée sur la page de garde. Année par défaut : `1A`.
- `enterprise=[|ensIIE|CEA]` | Définition de l'entreprise dans laquelle vous faites votre stage. Cela est purement cosmétique, et cela ajoutera seulement des illustrations sur la page de garde. Option facultative.

**Exemples d'application :**

```latex
\documentclass[language=fr, year=2A, enterprise=CEA]{ensiie-report} % Rapport d'un stage au CEA de 2A en français
\documentclass[language=en, year=3A]{ensiie-report} % Rapport de stage de 3A en anglais
```

#### Macros

Comme indiqués par la présence de `TODO`s dans `main.tex`, vous devez spécifier certaines informations pour afficher proprement la page de garde, l'entête et le pied de page.
Elles sont répertoriées dans la partie `MÉTADONNÉES` du fichier `main.tex`.

Les macros standards sont nécessaires, peu importe le choix de la classe.

```latex
\title{Sujet de stage} % Sujet du stage
\author{Prénom NOM} % Prénom NOM de l'étudiant(e)
\date{Jour1 AA mois1 20XX au jour2 BB mois2 20YY} % Dates du stage
```

Les autres macros sont internes à la classe `ensiie-report` et doivent alors être supprimées (ou commentées) si vous utilisez une autre classe.

```latex
\years{20XX-20YY} % Année scolaire en cours
\logoSchool{img/logo_ensiie.png} % Chemin vers le logo de l'ensIIE (ex : img/logo_ensiie.png)
\logoEnterprise{example-image-a} % Chemin vers le logo de l'entreprise du stage (ex : img/logo_entreprise.png)
\academicSupervisor{Prénom NOM} % Tuteur de stage
\internshipSupervisor{Prénom NOM} % Maître de stage
\internshipEnterprise{Entreprise (Ville, département)} % Lieu du stage
```

### Paquet `preamble`

**Ce paquet est compatible avec n'importe quelle classe.**

Elle (re)définit plusieurs paramètres pour sublimer votre rapport.

L'intérêt du paquet réside essentiellement sur les choix cosmétiques pour assurer la propreté du rapport, ainsi que sur les quelques commandes personnalisées présentes.

#### Commande `fileInput`

> **/!\ La commande fonctionne avec la version 2025 de TeX Live. Si vous avez une erreur en tentant d'utiliser cette commande, référez-vous à la commande [`tcbFileInput`](#commande-tcbfileinput).**

Insertion d'un fichier de code. Il prend en entrée le chemin relatif vers le fichier ainsi que 4 options.
- `language` | Choix du langage du fichier pour avoir la coloration syntaxique associée. Langage par défaut : `text` (aucune coloration syntaxique).
- `logo` | Choix d'une illustration dans la légende. Référez-vous à la liste d'illustrations du paquet [awesome5](https://texlive.mycozy.space/fonts/fontawesome5/doc/fontawesome5.pdf). Logo par défaut : `\faFile`.
- `caption` | Ajout d'une légende du fichier. Légende par défaut : `Légende`.
- `label` | Ajout d'une étiquette pour faire référence au code dans le document avec les commandes `\ref` ou `\autoref`, par exemple. **Convention :** préfixe `lst:` pour faciliter la recherche de l'étiquette. Label par défaut : `lst:todo`.
- `fontsize` | Choix de la police du texte. Police par défaut : `\small`

**Exemple d'utilisation :**

```latex
\fileInput[%
    language=python,
    logo=\faPython,
    caption={Exemple de script Python},
    label=lst:dummy,
    fontsize=\small
]{lst/dummy.py}
```

#### Commande `tcbFileInput`

Alternative à la commande [`fileInput`](#commande-fileinput), le principe est le même. Possibilité d'ajouter des options.

**Exemple d'utilisation :**

```latex
\tcbFileInput{python}{\faPython}{lst/dummy.py}{Exemple de script Python}{lst:dummy}
```

#### Environnement `terminalcode`

Environnement pour illustrer un terminal shell. Possibilité d'ajouter des options.

**Exemple d'utilisation :**

```text
\begin{terminalcode}[%
    minted options={breaklines, autogobble, numbersep=8mm, fontsize=\normalsize},
]
$ echo "Hello world!"
Hello world!
$ sleep 1
$
\end{terminalcode}
```

### Commande `inlinegraphics`

Insertion d'une image à la suite d'un texte. *L'idée est la même que pour les bulles de texte de la série de jeux **Super Mario Galaxy**.*

**Exemple d'utilisation :**

```latex
Ceci est une introduction à \texttt{mpi4py} \inlinegraphics{img/logo_mpi4py.pdf} pour tous.
```



## Rédaction du rapport

Vous devez insérer des métadonnées au début du fichier `main.tex` comme indiqué précédemment.

Le fichier `glossaries.tex` vous permet d'insérer votre glossaire, des mots de vocabulaires, ou des symboles. Le template présente des exemples d'application.

Le fichier `references.bib` vous permet d'ajouter votre bibliographie. Le template présente des exemples d'application.

Je vous suggère la convention suivante pour avoir un code source de votre rapport organisée et propre :
- Le dossier `img` contient toutes les images.
- Le dossier `lst` contient vos listings, c'est-à-dire vos morceaux de code.
- Le dossier `sub` contient des sous-fichiers pour les différentes parties de votre rapport. Libre à vous de les organiser comme vous le souhaitez.
- Le dossier `tikz` contient vos illustrations faites à la main avec TikZ.



## Compatibilité

Ce template est adaptée à la rédaction sur **VSCode**/**VSCodium** (je vous recommande notamment de copier le dossier `.vscode` pour profiter de mes configurations).

Le fichier `.latexmkrc` assure également sa compatibilité sur **Overleaf** (en plus d'être nécessaire sur **VSCode**/**VSCodium** pour gérer proprement le glossaire).

**L'externalisation TikZ ne fonctionne pas sur Overleaf.** Par conséquent, les lignes permettant de l'activer sont commentées dans `main.tex`. Vous pouvez les décommenter si vous travaillez en local et si vous souhaitez profiter de cette fonctionnalité pour gagner du temps de compilation.

Un `Makefile` est également présent si vous développez sur un éditeur de texte depuis le terminal. `make` assure la compilation avec l'outil `latexmk`.



## Développement

Ce template a été réalisé de manière à ce que vous n'ayez pas besoin de toucher aux fichiers `ensiie-report.cls` et `preamble.sty`.

Vous pouvez emprunter le fichier `preamble.sty` tel que conçu pour vos rapports personnels, puisqu'il ne dépend pas la classe `ensiie-report.cls`. **Toutefois, ce paquet n'est pas compatible avec la classe `beamer`.**

Si certains choix cosmétiques vous déplaisent, je vous laisse les modifier si besoin.

Toujours dans un souci de propreté, n'hésitez pas à ajouter vos paquets et commandes personnalisées dans `preamble.sty`.

N'hésitez pas à faire un fork de ce dépôt si vous souhaitez effectuer des pull requests pour améliorer ce template.



## FAQ

> **Comment utiliser ce template ?**

Vous pouvez enregistrer le template dans une archive en suivant les étapes données au début de ce README.

Ensuite, vous pouvez :
- créer votre propre dépôt Git et y déposer l'archive pour réutiliser le template. Utilisez soit le [Gitlab de l'école](https://gitlab.pedago.ensiie.fr), soit le Gitlab de votre boîte si votre rapport est confidentiel, pour vous assurer de ne pas perdre votre progression,
- déposer l'archive dans un nouveau projet Ovealeaf. Utilisez, de préférence, soit [l'Overleaf d'ARISE](https://overleaf.iiens.net/) (New project -> Upload Project, archive au format `zip` requise), soit l'Overleaf de votre entreprise s'il en existe un si votre rapport est confidentiel. Je ne recommande pas [l'Overleaf du CNRS](https://plmlatex.math.cnrs.fr/), car il semble manquer des paquets.


> **Où ajouter des paquets ?**

Dans un souci de propreté/clarté, je vous recommande d'ajouter des paquets supplémentaires dont vous auriez besoin dans le **fichier `preamble.sty`**.
Plus particulièrement, un espace à la ligne 84 de ce même fichier précise que vous pouvez les ajouter à un endroit précis.


> **C'est quoi la différence entre `usepackage` et `RequirePackage` ?**

La différence remarquable entre ces deux commandes est qu'on peut utiliser `RequirePackage` avant la déclaration d'une classe, ce qui n'est pas possible avec `usepackage`.

D'autre part, l'usage est d'utiliser la commande `RequirePackage` quand on souhaite utiliser des paquets existants dans nos propres paquets (fichier `preamble.sty`) ou nos propres classes (fichier `ensiie-report.cls`).

De préférence, utilisez `RequirePackage` dans les fichiers au format `sty` et `cls`, et utilisez `usepackage` dans les fichiers au format `tex`.

Vous pouvez utiliser `RequirePackage` de la même manière qu'on utilise `usepackage` (`\RequirePackage{tikz}` fait la même chose que `\usepackage{tikz}`).


> **Qui dois-je remercier dans la partie dédiée aux remerciements ?**

**Qui vous voulez.**

Cette partie est facultative.


> **Dois-je nécessairement avoir les cinq sous-parties présentes dans l'introduction ?**

**Je ne sais pas.**

Ces cinq sous-parties sont présentes dans le template de l'école, donc je les ai ajouté dans le template pour cette raison.

Je rappelle tout de même que le template de l'école est à destination des 3As.
Par conséquent, pour un stage de 1A ou de 2A, nous pouvons supposer que la notation soit plus indulgente.


> **Je ne sais pas quoi mettre dans les résultats obtenus. Cette partie est-elle obligatoire ?**

**Non**, cette partie n'est pas obligatoire.
J'ai personnellement pris l'initiative de l'ajouter.

Le template de l'école n'a qu'une seule partie sur les travaux réalisés pour ce qui est du *corps du rapport*.

Par exemple, si vous avez développé un site web, vous n'aurez pas obtenu de *résultats* à proprement parlé, puisque votre travail était de développer le site.
Par conséquent, seule la partie sur les travaux réalisés peut suffire.

Je rappelle que vous êtes libres d'organiser comme bon vous semble votre rapport : ce template vous sert de base, mais ne constitue nullement un standard.


> **J'ai complété mes fichiers `references.bib` et `glossary.tex` mais ils ne s'affichent pas tous, pourquoi ?**

Les éléments de votre bibliographie/glossaire qui apparaissent dans le document compilé sont ceux auxquels vous y avez fait référence **au moins une fois**.

En d'autres termes, si vous déclarez un élément de bibliographie avec comme clé `dummy`, vous devez au moins y faire référence une fois dans votre document avec `\cite{dummy}`. Il en est de même pour les éléments du glossaire (commande `\gls` ou `\acs`, par exemple).


> **Ma bibliographie/mon glossaire ne se met pas à jour, pourquoi ?**

Avant tout, avez-vous bien copié le fichier `.latexmkrc` ?

Ensuite, notez que que votre bibliographie et votre glossaire se mettent à jour **à chaque compilation** (du moins, c'est ce qu'assure latexmk).

De ce fait, si vous utilisez un éditeur de texte comme TeXstudio, Texmaker, ou TeXShop, il est possible que ces derniers utilisent le compilateur `pdflatex` seulement.
Or, la bibliographie se met à jour grâce à la commande `bibtex` (plus d'infos [ici](https://www.overleaf.com/learn/latex/Bibliography_management_with_bibtex)), et le glossaire se met à jour grâce à la commande (plus d'infos [ici](https://fr.overleaf.com/learn/latex/Glossaries)).

C'est pour cela que je vous recommande d'utiliser soit le Makefile fourni, soit l'outil latexmk directement.
