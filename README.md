# ZipTok3D Technical Report

This directory contains the ZipTok3D manuscript migrated to the ZIP Lab
technical-report template. Set `main.tex` as the main file on Overleaf.

## Active document structure

- `main.tex`: report metadata, shared macros, title-page teaser, and section
  assembly.
- `sections/00_abstract.tex`: abstract.
- `sections/01_introduction.tex`: introduction.
- `sections/02_related_works.tex`: related work.
- `sections/03_method.tex`: method.
- `sections/04_experiments.tex`: experiments.
- `sections/05_conclusion.tex`: conclusion.
- `sections/06_appendix.tex`: the migrated supplementary material.
- `figures/`: main-paper figures and institutional logos.
- `figures/supplementary/`: supplementary figures and complete heatmaps.
- `supplementary_assets/`: supplementary plotting and diagnostic scripts,
  numerical diagnostics, and the TRELLIS evaluation split manifest.
- `reference.bib`: bibliography used by both the main paper and appendix.
- `ziplab-tech-report.sty`: ZIP Lab report style.

## Preserved source material

- `template_original/` is an untouched copy of the template files that were in
  this directory before migration.
- `source_original/` contains untouched copies of the AAAI manuscript,
  supplementary source, and bibliography used for the migration.
- The original template `main.tex` is also retained in a disabled `\iffalse`
  block at the beginning of the active `main.tex` for immediate comparison.

## Compilation

Compile `main.tex` with PDFLaTeX and BibTeX. A typical local sequence is:

```text
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The manuscript text, equations, tables, captions, and reported values are
carried over from the current AAAI source. Adaptations are limited to the
technical-report entry point, section assembly, appendix numbering, and asset
paths required by this directory structure.

## Template credit

ZIP Lab technical-report template and style by
[Weijie Wang](http://lhmd.top). The style file is distributed under the LPPL
terms stated in `ziplab-tech-report.sty`.
