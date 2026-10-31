# ContextWorld figure design

The overview explains the benchmark; the training figure presents the experimental finding. Training curves are not repeated in the overview.

- `benchmark_overview.pdf` / `.svg`: two-panel overview at full paper width. Panel A uses an existing paired PushT trajectory. Panel B shows separate calls to a frozen model and the row/column scoring comparisons.
- `training_two_panels.pdf` / `.svg`: prediction loss and training-query NRE at single-column width. Color denotes task; marker and line style denote encoder training. NRE contains only the two measured epochs. Development controls remain part of the full experimental comparison.
- `figure_design.pdf`: both figures shown at paper size, with captions and adjacent experimental text. This is a layout preview, not a second manuscript.
- `assets/pusher_pair/provenance.json`: source pair, image hashes, action blocks, and illustrative-sample selection. Original JPEGs are retained; the diagram applies the same crop to every frame and adds displacement annotations.

The selected pair illustrates the construction; it is not a new evaluation sample or a change to benchmark sampling. Condition labels and gains are figure annotations, not model inputs. Prediction outputs are latent states, not decoded images. The error matrix illustrates future discrimination and history benefit; full metric definitions remain in the paper.

Regenerate from the repository root:

```bash
python paper/design/draw_overview.py
python paper/design/draw_training.py
cd paper
PATH=/opt/texlive/2025/bin/x86_64-linux:$PATH pdflatex -interaction=nonstopmode -halt-on-error -output-directory=design -jobname=figure_design design/preview.tex
```

The plots use the checked-in `native_training_loss_v1` and `native_epoch_checkpoints_v1` CSV files. No new training, interpolation of NRE measurements, or synthetic experimental scores are used.
