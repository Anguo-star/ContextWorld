LATEXMK ?= latexmk

PDFLATEX_FLAGS = -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build

.PHONY: all zh en full clean

all: zh

zh:
	$(LATEXMK) -xelatex -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build main.tex

en:
	# Two rounds resolve references between the standalone main and supplement.
	$(LATEXMK) $(PDFLATEX_FLAGS) main_en.tex
	$(LATEXMK) $(PDFLATEX_FLAGS) supplementary.tex
	$(LATEXMK) $(PDFLATEX_FLAGS) main_en.tex
	$(LATEXMK) $(PDFLATEX_FLAGS) supplementary.tex
	$(MAKE) full

full:
	$(LATEXMK) $(PDFLATEX_FLAGS) full.tex

clean:
	$(LATEXMK) -c -outdir=build main.tex main_en.tex supplementary.tex full.tex
