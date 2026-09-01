# Variables
TEX_MAIN = main.tex
PDF_MAIN = $(TEX_MAIN:.tex=.pdf)
BUILD_DIR = build
OPTS = -pdf -shell-escape -synctex=1 -file-line-error -interaction=nonstopmode
DIR_OPTS = -outdir=$(BUILD_DIR)
LEGACY_AUX = main.acn main.acr main.alg main.aux main.bbl main.blg \
	main.fdb_latexmk main.fls main.glg main.glo main.gls main.ist \
	main.listing main.loa main.lof main.log main.lot main.out main.run.xml \
	main.sbl main.slg main.sym main.synctex.gz main.toc

# Couleurs
GREEN=\033[1;32m
YELLOW=\033[1;33m
RESET=\033[0m

# Cibles principales
all: $(BUILD_DIR)
	@echo "$(YELLOW)Compilation...$(RESET)"
	@latexmk $(OPTS) $(DIR_OPTS) $(TEX_MAIN)
	@cp $(BUILD_DIR)/$(PDF_MAIN) $(PDF_MAIN)
	@echo "$(GREEN)Compilation terminée.$(RESET)"

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# Nettoyage des fichiers PDF
clean_pdf:
	@echo "$(YELLOW)Suppression des fichiers PDF...$(RESET)"
	@rm -f $(PDF_MAIN) $(BUILD_DIR)/$(PDF_MAIN)
	@echo "$(GREEN)Suppression des fichiers PDF terminée.$(RESET)"

# Nettoyage des fichiers auxiliaires de compilation
clean_aux:
	@echo "$(YELLOW)Suppression des fichiers auxiliaires de compilation...$(RESET)"
	@latexmk -c $(TEX_MAIN)
	@latexmk -c $(DIR_OPTS) $(TEX_MAIN)
	@rm -f $(LEGACY_AUX)
	@rm -rf $(BUILD_DIR)
	@rm -fr _minted*
	@rm -fr tikz/build/*
	@echo "$(GREEN)Suppression des fichiers auxiliaires terminée.$(RESET)"

# Nettoyage complet
clean: clean_pdf clean_aux
	@echo "$(GREEN)Nettoyage complet terminé.$(RESET)"

.PHONY: all clean clean_pdf clean_aux
