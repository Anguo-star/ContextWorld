# ContextWorld figures

The overview explains the paired evaluation. The training figure presents one measured example; it is not repeated in the overview.

- `benchmark_overview.pdf` / `.svg`: the reference design's rounded panels, image cards, blue/orange condition colors and error matrix, using a real Cube pair. History actions are drawn between observations as five-step vertical-command traces. The current observation and all action blocks are identical across conditions.
- `training_two_panels.pdf` / `.svg`: PLDM contact friction, Original → Mixed (Joint). The left plot contains six recorded prediction-loss means; the right contains the two measured training-query NRE values. No intermediate NRE values are interpolated.
- `figure_design.pdf`: both figures at paper size with captions.
- `assets/cube_pair/provenance.json`: source pair, action blocks, image hashes and sample selection. The same crop is applied to each unmodified source image. The pair was selected for legibility, without changing benchmark sampling or scores.
- `assets/environments/provenance.json`: sources for the environment thumbnails. The PushT thumbnail comes from `assets/pusher_pair/provenance.json`.
- `assets/fonts/`: embedded Roboto fonts, distributed under the accompanying SIL Open Font License. The font source is [Google Fonts](https://github.com/google/fonts/tree/main/ofl/roboto).

The layout follows the [supplied design reference](https://chatgpt.com/share/6ae598f7-561c-83e9-a081-a60ac913b7d8). Scene images and plot values come from ContextWorld records. The model outputs are shown as latent states; the diagram does not imply pixel decoding. Rows of the error matrix compare candidate futures, while columns compare input histories for a fixed target.

## Regeneration

The SVG drawings use Python's standard library. PDF/PNG export requires Node.js, Playwright and Chromium:

```bash
python paper/design/draw_overview.py
python paper/design/draw_training.py
node paper/design/render_figures.cjs
```

Set `CW_FIGURE_BROWSER` to an existing Chromium executable if needed. Set `NODE_PATH` if Playwright is installed outside the repository.

To regenerate all manuscript figures, including the full training controls retained in the appendix:

```bash
python paper/generate_figures.py
```

To compile the figure preview with TeX Live:

```bash
cd paper
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=design -jobname=figure_design design/preview.tex
```

Training plots use the checked-in `native_training_loss_v1` and `native_epoch_checkpoints_v1` CSV files. The selected values and run identifier are also recorded in `training_two_panels.json`. Compiling the paper uses the saved figure PDFs and does not require regenerating them or accessing the original datasets.
