# ContextWorld figures

The overview explains the paired evaluation. The training figure presents one measured example; it is not repeated in the overview.

- `benchmark_overview.pdf` / `.svg`: the reference design's rounded panels, image cards, blue/orange condition colors and error matrix, using high-resolution renderings of a real Cube pair. Lift and reverse commands label transitions between observations; the shared query action also appears between the current and future states. Dashed height references and Lifts/Stays labels distinguish the outcomes. The current observation and all action blocks are identical across conditions.
- `training_two_panels.pdf` / `.svg`: PLDM contact friction, Original → Mixed (Joint). The left plot contains six recorded prediction-loss means; the right contains the two measured training-query NRE values. No intermediate NRE values are interpolated.
- `figure_design.pdf`: both figures at paper size with captions.
- `assets/cube_pair/provenance.json`: source pair, action blocks, image hashes and sample selection. The original 224-pixel observations are retained for provenance. The pair was selected for legibility, without changing benchmark sampling or scores.
- `assets/cube_pair_hd/`: 1600 × 900 native simulator renderings, the saved replay states and rendering provenance. A shared camera, palette and lighting make the cube visible at paper size. The recorded cube positions are preserved; the evaluated models still receive the original observations.
- `assets/environments/provenance.json`: sources for the environment thumbnails. The PushT thumbnail comes from `assets/pusher_pair/provenance.json`.
- `assets/fonts/`: embedded Roboto fonts, distributed under the accompanying SIL Open Font License. The font source is [Google Fonts](https://github.com/google/fonts/tree/main/ofl/roboto).

The layout follows the [supplied design reference](https://chatgpt.com/share/6ae598f7-561c-83e9-a081-a60ac913b7d8). Scene states and plot values come from ContextWorld records. The lift/reverse sequence illustrates the Cube carry rule; it is not a claim that every benchmark task uses these commands. The model outputs are shown as latent states; the diagram does not imply pixel decoding. Rows of the error matrix compare candidate futures, while columns compare input histories for a fixed target.

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

The high-resolution scene PNGs are included. Re-rendering them is optional and requires the ContextWorld Cube simulation dependencies (MuJoCo, OGBench, NumPy and Pillow), plus OSMesa for headless rendering:

```bash
python paper/design/render_cube_assets.py
```

The saved states avoid needing the source dataset for this display-only rendering. The source-state comparison and original observation hashes are documented in the two Cube provenance files.

Training plots use the checked-in `native_training_loss_v1` and `native_epoch_checkpoints_v1` CSV files. The selected values and run identifier are also recorded in `training_two_panels.json`. Compiling the paper uses the saved figure PDFs and does not require regenerating them or accessing the original datasets.
