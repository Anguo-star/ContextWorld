LATEXMK ?= latexmk

.PHONY: all zh en clean

all: zh

zh:
	$(LATEXMK) -xelatex -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build main.tex

en:
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build main_en.tex

clean:
	$(LATEXMK) -c -outdir=build main.tex main_en.tex
