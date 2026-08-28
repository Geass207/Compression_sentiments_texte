# Variables
TEX_MAIN = main.tex
PDF_MAIN = $(TEX_MAIN:.tex=.pdf)
OPTS = -pdf -shell-escape -synctex=1 -file-line-error -interaction=nonstopmode

# Couleurs
GREEN=\033[1;32m
YELLOW=\033[1;33m
RESET=\033[0m

# Cibles principales
all:
	@echo "$(YELLOW)Compilation...$(RESET)"
	@latexmk $(OPTS) $(TEX_MAIN)
	@echo "$(GREEN)Compilation terminée.$(RESET)"

# Nettoyage des fichiers PDF
clean_pdf:
	@echo "$(YELLOW)Suppression des fichiers PDF...$(RESET)"
	@rm -f *.pdf
	@echo "$(GREEN)Suppression des fichiers PDF terminée.$(RESET)"

# Nettoyage des fichiers auxiliaires de compilation
clean_aux:
	@echo "$(YELLOW)Suppression des fichiers auxiliaires de compilation...$(RESET)"
	@rm -f *.aux *.log *.synctex.gz *.bbl *.blg *.out *.glg *.glo *.gls *.xdy *.ist \
		*.toc *.listing *.loa *.lof *.lot *.lol *.fls *.fdb_latexmk *.auxlock \
		*.acn *.acr *.alg *.sbl \
		sub/*.aux sub/*.fdb_latexmk sub/*.fls sub/*.log \
		*.dvi *.nav *.snm *.vrb *.slg *.sym \
		tikz/*.dpth tikz/*.log tikz/*.md5 tikz/*.listing tikz/*.acn tikz/*.glo tikz/*.ist tikz/*.sbl tikz/*.pdf
	@rm -fr _minted*
	@rm -fr tikz/build/*
	@echo "$(GREEN)Suppression des fichiers auxiliaires terminée.$(RESET)"

# Nettoyage complet
clean: clean_pdf clean_aux
	@echo "$(GREEN)Nettoyage complet terminé.$(RESET)"

.PHONY: all clean clean_pdf clean_aux
