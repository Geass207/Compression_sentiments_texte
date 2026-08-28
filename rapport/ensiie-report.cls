% -*- coding: utf-8 -*-

%  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *  %
%                                                     d8b d8b   %
%  Author      : Lorenzo 'Lorocas' LUCAS -- ROBLOT   888888888  %
%  Description : ensIIE 2025 report internship class `Y88888P'  %
%  Version     : 1.0                                  `Y888P'   %
%  Date        : 15/07/2025                            `Y8P'    %
%                                                       `Y'     %
%  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *  %

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% DÉCLARATION DU PAQUET %%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ensiie-report}[2025/07/15 ensIIE report internship class]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% CHARGEMENT DE LA CLASSE DE BASE %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\LoadClass[12pt, a4paper, titlepage]{article}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% DÉCLARATION DES SOUS-PAQUETS NÉCESSAIRES %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\typeout{[ensiie-report] Début de l'importation des paquets}

% Pour tout ce qu'est encodage, entre autre
\RequirePackage[T1]{fontenc} % Primordial pour l'encodage des caractères
\RequirePackage[utf8]{inputenc} % Pour bien gérer l'encodage UTF-8

% Pour soigner la mise en forme des pages
\RequirePackage[a4paper]{geometry} % Modification de la taille des bordures
\RequirePackage{fancyhdr} % Pour modifier l'entête et le bas de page
\RequirePackage{eso-pic} % Pour ajouter des images en fond des pages

% Pour améliorer la présentation du texte de manière générale
\RequirePackage{xcolor} % red green blue cyan magenta yellow black gray white darkgray lightgray brown lime olive orange pink purple teal violet

% Pour ajouter des images
\RequirePackage{graphicx, calc} % Insérer des images

% Pour utiliser TikZ
\RequirePackage{tikz}
\RequirePackage{standalone} % Insertion des illustrations TikZ optimisée

% Autre
\RequirePackage{indentfirst} % Pour indenter la première ligne de chaque section
\RequirePackage{afterpage} % Pour insérer une page vide après la 1ère page
\RequirePackage{lastpage} % Pour obtenir la macro qui correspond au nombre de pages
\RequirePackage{titling} % Pour avoir le nom de l'auteur qui apparaît dans l'entête
\RequirePackage{etoolbox} % Pour modifier la macro \sloppy
\RequirePackage{pgfkeys} % Pour créer des couples clé-valeur
\RequirePackage{kvoptions, ifthen} % Pour gérer proprement les options du paquet

\typeout{[ensiie-report] Fin de l'importation des paquets}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% COULEURS PERSONNALISÉES %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\definecolor{bronze_ensiie}{HTML}{c38c6d}
\definecolor{argent_ensiie}{HTML}{aeb6bc}
\definecolor{or_ensiie}{HTML}{ebce9b}
\definecolor{bleu_ensiie}{HTML}{5777a1}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% DÉFINITION DE MACROS %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Affichage de la nature du rapport
\newcommand{\reportYear}{}
\newcommand{\titleUnAFR}{Rapport de stage -- Première année}
\newcommand{\titleUnAEN}{First Year Internship Report}
\newcommand{\titleDeuxAFR}{Rapport de stage -- Deuxième année}
\newcommand{\titleDeuxAEN}{Second Year Internship Report}
\newcommand{\titleTroisAFR}{Rapport de stage -- Troisième année}
\newcommand{\titleTroisAEN}{Third Year Internship Report}

%% Sous-titre
\newcommand{\subtitle}[1]{%
  \def\storedSubtitle{#1}%
}
\newcommand{\storedSubtitle}{}

% Affichage de l'année scolaire en cours
\newcommand{\years}[1]{%
  \def\storedYears{#1}%
}
\newcommand{\storedYears}{20XX-20XX}

% Affichage du logo de l'école
\newcommand{\logoSchool}[1]{%
  \def\storedLogoSchool{#1}%
}
\newcommand{\storedLogoSchool}{}
\newcommand{\insertLogoSchool}{%
  \ifx\storedLogoSchool\empty
    % Undefined logo
    \ClassWarning{ensiie-report}{Undefined logo \string\logoSchool}%
    \fbox{\parbox[c][0.14\textheight][c]{0.14\textheight}{\centering\textcolor{red}{\texttt{UNDEFINED LOGO}}}}%
  \else
    % File exists
    \IfFileExists{\storedLogoSchool}{%
      \includegraphics[height=0.14\textheight]{\storedLogoSchool}%
    }{%
      % File not found
      % \ClassWarning{ensiie-report}{File '\storedLogoSchool' not found}%
      \includegraphics[draft, height=0.14\textheight]{\storedLogoSchool}%
    }%
  \fi
}

% Affichage du logo de l'entreprise
\newcommand{\logoEnterprise}[1]{%
  \def\storedLogoEnterprise{#1}%
}
\newcommand{\storedLogoEnterprise}{}
\newcommand{\insertLogoEnterprise}{%
  \ifx\storedLogoEnterprise\empty
    % Undefined logo
    \ClassWarning{ensiie-report}{Undefined logo \string\logoEnterprise}%
    \fbox{\parbox[c][0.14\textheight][c]{0.14\textheight}{\centering\textcolor{red}{\texttt{UNDEFINED LOGO}}}}%
  \else
    % File exists
    \IfFileExists{\storedLogoEnterprise}{%
      \includegraphics[height=0.14\textheight]{\storedLogoEnterprise}%
    }{%
      % File not found
      % \ClassWarning{ensiie-report}{File '\storedLogoEnterprise' not found}%
      \includegraphics[draft, height=0.14\textheight]{\storedLogoEnterprise}%
    }%
  \fi
}

% Affichage de l'étudiant
\newcommand{\studentMacro}{}
\newcommand{\studentMacroFR}{Étudiant }
\newcommand{\studentMacroEN}{Student}


% Affichage du nom du tuteur de stage
\newcommand{\academicSupervisorMacro}{}
\newcommand{\academicSupervisorMacroFR}{Tuteur académique à l'ensIIE }
\newcommand{\academicSupervisorMacroEN}{Academic supervisor at ensIIE}

\newcommand{\academicSupervisor}[1]{%
  \def\storedAcademicSupervisor{#1}%
}
\newcommand{\storedAcademicSupervisor}{Prénom NOM}


% Affichage du nom de l'entreprise et son adresse
\newcommand{\internshipEnterpriseMacro}{}
\newcommand{\internshipEnterpriseMacroFR}{Entreprise }
\newcommand{\internshipEnterpriseMacroEN}{Company}

\newcommand{\internshipEnterprise}[1]{%
  \def\storedInternshipEnterprise{#1}%
}
\newcommand{\storedInternshipEnterprise}{Nom (adresse)}


% Affichage du nom du maître de stage
\newcommand{\internshipSupervisorMacro}{}
\newcommand{\internshipSupervisorMacroFR}{Maître de stage }
\newcommand{\internshipSupervisorMacroEN}{Internship supervisor}

\newcommand{\internshipSupervisor}[1]{%
  \def\storedInternshipSupervisor{#1}%
}
\newcommand{\storedInternshipSupervisor}{Prénom NOM}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% MISE EN FORME DES PAGES %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Changement des bordures / Entête et bas de page
\geometry{hmargin=2.5cm,vmargin=2.5cm} % Base 3.6cm/4.5cm
\pagestyle{fancy}

% Définition du style des pages préliminaires
% Ajout de l'anglais
\newcommand{\pageMacro}{}
\newcommand{\pageMacroFR}{sur}
\newcommand{\pageMacroEN}{of}

\fancypagestyle{FirstPages}
{
  \lhead{{\thetitle}}
  % \chead{haut de page centre} % Facultatif
  \rhead{\storedYears}
  \lfoot{\textsc{\theauthor}} % Facultatif
  \cfoot{}
  \rfoot{Page \thepage{} \pageMacro{} \pageref*{EndFirstPages}} % Facultatif
  % \renewcommand{\headrulewidth}{0.4pt}
  \renewcommand{\footrulewidth}{0.4pt} % Facultatif % Pour ajouter un trait en bas de page
}

% Définition du style du corps du rapport
\fancypagestyle{MainPages}
{
  \lhead{{\thetitle}}
  % \chead{haut de page centre} % Facultatif
  \rhead{\storedYears}
  \lfoot{\textsc{\theauthor}} % Facultatif
  \cfoot{}
  \rfoot{Page \thepage{} \pageMacro{} \pageref*{LastPage}} % Facultatif
  % \renewcommand{\headrulewidth}{0.4pt}
  \renewcommand{\footrulewidth}{0.4pt} % Facultatif % Pour ajouter un trait en bas de page
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% DÉFINITION DE LA PAGE DE GARDE %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\renewcommand{\maketitle}{%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DÉBUT DÉFINITION TITLEPAGE %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\begin{titlepage}
\centering

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% LOGO ÉCOLE %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\vspace*{1.5\baselineskip}
\insertLogoSchool
\vspace{15pt}

\textsc{\large École Nationale Supérieure d'Informatique pour l'Industrie et l'Entreprise}

\vfill

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% TITRE %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Au dessus du titre
\rule{\textwidth}{1.6pt}\vspace*{-\baselineskip}\vspace*{2pt}
\rule{\textwidth}{0.4pt}

\vspace{0.75\baselineskip}

% Titre
{\LARGE
\textsc{\textbf{\reportYear}}

\vspace{10pt}

\begin{spacing}{1.1}
    {\scshape\thetitle}
\end{spacing}
}

% Sous-titre
\ifthenelse{\equal{\storedSubtitle}{}}
{
  \vspace{0.75\baselineskip}
}
% --- %
{
  \vspace{10pt}

  {
    \small
    \begin{spacing}{1.1}
      {\scshape\storedSubtitle}
    \end{spacing}
  }

  \vspace{0.7\baselineskip}
}

% En dessous du titre
\rule{\textwidth}{0.4pt}\vspace*{-\baselineskip}\vspace{3.2pt}
\rule{\textwidth}{1.6pt}

\vfill

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% AUTEURS %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

{\scshape\Large\textbf{\studentMacro :} \theauthor}

\vspace{0.5\baselineskip}

{\scshape\Large\textbf{\academicSupervisorMacro :} \storedAcademicSupervisor}

\vspace{1.0\baselineskip}

\insertLogoEnterprise

\vspace{1.0\baselineskip}

{\scshape\Large\textbf{\internshipEnterpriseMacro :} \storedInternshipEnterprise}

\vspace{0.5\baselineskip}

{\scshape\Large\textbf{\internshipSupervisorMacro :} \storedInternshipSupervisor}

\vfill

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% ANNÉE ET PROJET %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

{\large\scshape {\thedate}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% FIN DÉFINITION TITLEPAGE %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\end{titlepage}
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% BACKGROUND DE LA PAGE DE GARDE %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcommand{\BackgroundPicFirstPage}{}
\newcommand{\CEABackgroundFirstPage}
{
  % Image de la vague en haut sur toute la longueur
	% \put(-0.02\paperwidth,0.91\paperheight){%%
	% 	\includegraphics[width=1.1\paperwidth,keepaspectratio]{img/cube_cea2.png}
	% }%%

    % Image des cubes seulement en haut à gauche
    \put(-0.23\paperwidth,0.91\paperheight){%%
		\includegraphics[width=0.4\paperwidth,keepaspectratio]{img/cube_cea.png}
	}%%

    % Image des cubes seulement en bas à droite
    \put(0.85\paperwidth,-0.03\paperheight){%%
		\includegraphics[width=0.4\paperwidth,keepaspectratio]{img/cube_cea.png}
	}%%
}
\newcommand{\ensIIEBackgroundFirstPage}
{\tikzexternaldisable
\usetikzlibrary{shadings}

% Des cercles en haut à gauche
\put(-0.25\paperwidth,0.83\paperheight){%%
\begin{tikzpicture}[fill opacity=0.3]
  \fill[shading=radial, inner color=bleu_ensiie!0, outer color=bleu_ensiie!80] (-1,-2) circle (4cm);
  \fill[shading=radial, inner color=argent_ensiie!0, outer color=argent_ensiie!80] (3,1) circle (4cm);
\end{tikzpicture}
}%%

% Des cercles en bas à droite
\put(0.75\paperwidth,-0.19\paperheight){%%
\begin{tikzpicture}[fill opacity=0.3]
  \fill[shading=radial, inner color=bronze_ensiie!0, outer color=bronze_ensiie!80] (2,3) circle (4cm);
  \fill[shading=radial, inner color=or_ensiie!0, outer color=or_ensiie!80] (0,0) circle (4cm);
\end{tikzpicture}
}%%
\tikzexternalenable}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% BACKGROUND DE TOUTES LES PAGES %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% \newcommand\BackgroundPicAllPages
%   {%
%     % Image en haut à droite
% 	\put(\LenToUnit{\paperwidth-0.143\paperwidth},\LenToUnit{\paperheight-0.14\paperheight}){%%
% 		\includegraphics[width=0.2\paperwidth,angle=90,keepaspectratio]{img/cube_cea.png}
% 	}%%

% 	% Image en haut à gauche
% 	\put(-0.4,\LenToUnit{\paperheight+0.001\paperheight}){%%
%         \includegraphics[width=0.2\paperwidth,angle=90,keepaspectratio]{img/cube_cea.png}
% 	}%%

% 	% Image en bas à gauche
% 	\put(0,0){%%
%     \includegraphics[width=0.2\paperwidth,angle=90,keepaspectratio]{img/cube_cea.png}
% 	}%%

% 	% Image en bas à droite
% 	\put(\LenToUnit{\paperwidth-0.1\paperwidth},0){%%
%     \includegraphics[width=0.2\paperwidth,angle=90,keepaspectratio]{img/cube_cea.png}
% 	}%%

% }%

% \newcommand\BackgroundPages
% {%
% \tikzexternaldisable
% \begin{tikzpicture}[remember picture, overlay]
% 	\draw[black, line width=3pt]
% 		(4pt,4pt) rectangle (\paperwidth - 4pt, \paperheight - 4pt);
% \end{tikzpicture}%
% \tikzexternalenable
% }

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% MODIFICATIONS DE LA BIBLIOGRAPHIE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Pour limiter les warnings dans la bibliographie
\apptocmd{\sloppy}{\hbadness 10000\relax}{}{}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% AUTRES COMMANDES PERSONNALISÉES %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Pour ajouter une page vierge
\newcommand{\blankpage}{%
  \ClearShipoutPictureBG
  \null
  \thispagestyle{empty}%
  \addtocounter{page}{-1}%
  \newpage
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% DÉCLARATION DES OPTIONS DU PAQUET %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Setup des options clé-valeur avec kvoptions
\SetupKeyvalOptions{%
  family=preamble, % nom de famille pour préfixer les options internes
  prefix=preamble@ % préfixe utilisé pour les commandes internes
}

% Déclaration des options avec valeurs par défaut
% Langue (défaut : fr)
\DeclareStringOption[fr]{language}

% Année
\DeclareStringOption[]{year}

% Entreprise (défaut : vide/null)
\DeclareStringOption[]{enterprise}

% Traitement des options inconnues
\DeclareVoidOption*{%
  \ClassWarning{ensiie-report}{Unknown '\CurrentOption'}%
}

% Traitement des options passées
\ProcessKeyvalOptions*

% Débug dans le fichier de log
\makeatletter
\typeout{[ensiie-report] language=\preamble@language, year=\preamble@year, enterprise=\preamble@enterprise}
\makeatother

%%%%%%%%%%%%%
\makeatletter
%%%%%%%%%%%%%

% Vérification de la langue
\ifthenelse{\equal{\preamble@language}{fr} \OR \equal{\preamble@language}{en}}
{} % nothing to do
{ % unknown language, warning
    \ClassWarning{ensiie-report}{Unknown language: '\preamble@language'. Set to default: 'fr'}
}

% Vérification de l'année
\ifthenelse{\equal{\preamble@year}{}}
{
    \ClassWarning{ensiie-report}{Year not specified. Set to default: '1A'}
}
{} % nothing to do

% Initialisation de la macro \reportYear
\newcommand{\preamble@setReportYear}{%
  \ifthenelse{\equal{\preamble@language}{en}}{%
    \ifthenelse{\equal{\preamble@year}{1A}}{\renewcommand{\reportYear}{\titleUnAEN}}{%
    \ifthenelse{\equal{\preamble@year}{2A}}{\renewcommand{\reportYear}{\titleDeuxAEN}}{%
    \ifthenelse{\equal{\preamble@year}{3A}}{\renewcommand{\reportYear}{\titleTroisAEN}}{%
      \ClassWarning{ensiie-report}{Unknown year: '\preamble@year'}%
      \renewcommand{\reportYear}{\textbf{\color{red}{Unknown Year}}}%
    }}}%
  }{%
    \ifthenelse{\equal{\preamble@year}{1A}}{\renewcommand{\reportYear}{\titleUnAFR}}{%
    \ifthenelse{\equal{\preamble@year}{2A}}{\renewcommand{\reportYear}{\titleDeuxAFR}}{%
    \ifthenelse{\equal{\preamble@year}{3A}}{\renewcommand{\reportYear}{\titleTroisAFR}}{%
      \ClassWarning{ensiie-report}{Unknown year: '\preamble@year'}%
      \renewcommand{\reportYear}{\textbf{\color{red}{Année inconnue}}}%
    }}}%
  }%
}

% Initialisation de la macro \academicSupervisorMacro
\newcommand{\preamble@setAcademicSupervisorMacro}{%
  \ifthenelse{\equal{\preamble@language}{en}}{%
    \renewcommand{\academicSupervisorMacro}{\academicSupervisorMacroEN}
  }
  {%
    \ifthenelse{\equal{\preamble@language}{fr}}{%
      \renewcommand{\academicSupervisorMacro}{\academicSupervisorMacroFR}

    }{%
      \ClassWarning{ensiie-report}{Unknown language: '\preamble@language'. Set to default: 'fr'}%
      \renewcommand{\academicSupervisorMacro}{\academicSupervisorMacroFR}
    }

  }
}

% Initialisation de la macro \internshipSupervisorMacro
\newcommand{\preamble@setInternshipSupervisorMacro}{%
  \ifthenelse{\equal{\preamble@language}{en}}{%
    \renewcommand{\internshipSupervisorMacro}{\internshipSupervisorMacroEN}
  }
  {%
    \ifthenelse{\equal{\preamble@language}{fr}}{%
      \renewcommand{\internshipSupervisorMacro}{\internshipSupervisorMacroFR}

    }{%
      \ClassWarning{ensiie-report}{Unknown language: '\preamble@language'. Set to default: 'fr'}%
      \renewcommand{\internshipSupervisorMacro}{\internshipSupervisorMacroFR}
    }

  }
}

% Initialisation de la macro \studentMacro
\newcommand{\preamble@setStudentMacro}{%
  \ifthenelse{\equal{\preamble@language}{en}}{%
    \renewcommand{\studentMacro}{\studentMacroEN}
  }
  {%
    \ifthenelse{\equal{\preamble@language}{fr}}{%
      \renewcommand{\studentMacro}{\studentMacroFR}

    }{%
      \ClassWarning{ensiie-report}{Unknown language: '\preamble@language'. Set to default: 'fr'}%
      \renewcommand{\studentMacro}{\studentMacroFR}
    }

  }
}

% Initialisation de la macro \internshipEnterpriseMacro
\newcommand{\preamble@setInternshipEnterpriseMacro}{%
  \ifthenelse{\equal{\preamble@language}{en}}{%
    \renewcommand{\internshipEnterpriseMacro}{\internshipEnterpriseMacroEN}
  }
  {%
    \ifthenelse{\equal{\preamble@language}{fr}}{%
      \renewcommand{\internshipEnterpriseMacro}{\internshipEnterpriseMacroFR}

    }{%
      \ClassWarning{ensiie-report}{Unknown language: '\preamble@language'. Set to default: 'fr'}%
      \renewcommand{\internshipEnterpriseMacro}{\internshipEnterpriseMacroFR}
    }

  }
}
% Choix de l'entreprise
\newcommand{\preamble@setEnterpriseSettings}{%
    \ifthenelse{\equal{\preamble@enterprise}{}}{%
    % enterprise not specified
    \def\BackgroundPicFirstPage{}%
    \ClassInfo{ensiie-report}{Enterprise not specified. Set to default: '\null'}%
    }{%
        \ifthenelse{\equal{\preamble@enterprise}{ensIIE}}{%
            % ensIIE Enterprise
            \def\BackgroundPicFirstPage{\ensIIEBackgroundFirstPage}%
        }{%
            \ifthenelse{\equal{\preamble@enterprise}{CEA}}{%
                % CEA Enterprise
                \def\BackgroundPicFirstPage{\CEABackgroundFirstPage}
            }{%
            % unknown enterprise
                \def\BackgroundPicFirstPage{}%
                \ClassWarning{ensiie-report}{Unknown enterprise: '\preamble@enterprise'. Set to default: '\null'}%
            }%
        }%
    }%
}

% Initialisation de la macro \pageMacro
\newcommand{\preamble@setPageMacro}{%
  \ifthenelse{\equal{\preamble@language}{en}}{%
    \renewcommand{\pageMacro}{\pageMacroEN}
  }
  {%
    \ifthenelse{\equal{\preamble@language}{fr}}{%
      \renewcommand{\pageMacro}{\pageMacroFR}

    }{%
      \ClassWarning{ensiie-report}{Unknown language: '\preamble@language'. Set to default: 'fr'}%
      % default to french
      \renewcommand{\pageMacro}{\pageMacroFR}%
    }

  }
}


% Setup des variables
\AtBeginDocument{%
  \preamble@setReportYear
  \preamble@setEnterpriseSettings
  \preamble@setAcademicSupervisorMacro
  \preamble@setInternshipSupervisorMacro
  \preamble@setStudentMacro
  \preamble@setInternshipEnterpriseMacro
  \preamble@setPageMacro
}
\makeatother

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% FIN DU FICHIER %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\endinput
