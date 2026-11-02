# ContextWorld figures

The overview explains the controlled task design, paired evaluation and task coverage. The training figure presents a measured example; it is not repeated in the overview.

- `benchmark_overview.pdf` / `.svg`: an editable reconstruction of the supplied 1448 × 1086 design. Panel A shows a recorded Cube carry-rule pair; panel B shows separate frozen-model calls and the four error comparisons; panel C maps nine tasks to four environments. Commands label transitions, and both conditions use exactly the same current-observation rendering. Text, arrows, matrix cells and panel frames remain vector graphics. Cube images are high-resolution native renderings of recorded states. Other environment views are saved dataset screenshots. The design reference supplies the layout and visual styling, not scene pixels.
- `benchmark_overview_pusher.pdf` / `.svg`: alternative main-figure preview using a real PushT pusher-displacement pair. The original scene screenshots share one crop. All three transitions are labeled Move; red arrows summarize the shared action directions. Dashed arrows show measured endpoint displacement, including the return to the shared current observation, not an interpolated path. A faint translucent silhouette with a dashed outline in the lower future panel marks the T block's current pose; its silhouette is traced from the shared original observation and saved in `assets/pusher_pair/current_t_outline.json`. These annotations are not part of model inputs. Sources and arrow semantics are recorded in `overview_pusher.json`.
- `overview_comparison.pdf`: PushT and Cube alternatives at manuscript width on separate pages. The manuscript continues to use the Cube version pending selection.
- `benchmark_overview_pusher_response.pdf` / `.svg`: PushT alternative with near-square error cells and detached row/column headers beside a Response fidelity panel. Prediction is a shared vertical row-axis title; True future is the shared horizontal column title, with A/B labels aligned separately to the cells. A light card groups the title, axes, matrix, and legends, matching the adjacent fidelity card. A compact footer legend uses blue and orange line swatches for future discrimination and history benefit, matching the comparison arrows. The heading defines d as latent prediction error and states that arrows point to the desired lower error. Four comparison arrows sit 8 px outside the cells, between the headers and cells, with only the outward-facing wing of each arrowhead drawn: row A points left, row B right, column A up, and column B down. Arrowheads indicate desired lower errors, not measured rankings. Row comparisons measure future discrimination, column comparisons measure history benefit, and NRE compares the predicted and true differences between conditions. The Response fidelity card is balanced with the adjacent matrix: their headings and explanatory lines share a baseline, and the card extends upward while retaining compact side margins. Its vectors are enlarged uniformly. The two arrows are conceptual latent response vectors, not measured trajectories or a projection of model outputs.
- `prediction_image_preview.pdf`: layout alternative with true-future thumbnails labeled Reference beside the latent prediction symbols. These are visual references, not decoded predictions. Generate it with `--example pusher --response-fidelity --prediction-images` and export `benchmark_overview_pusher_response_images`.
- `response_fidelity_preview.pdf`: the response-fidelity alternative at manuscript width, with its explanatory caption. This preview preserves the real PushT images and does not replace the current manuscript figure.
- `assets/reference/main_figure_design.png`: the complete supplied design, retained at its original resolution. It is used only as a visual layout reference and is not embedded in the figure. `overview_crop.json` records the native data sources, shared crop and state/action checks.
- `training_two_panels.pdf` / `.svg`: PLDM contact friction, Original → Mixed (Joint). The left plot contains six recorded prediction-loss means; the right contains the two measured training-query NRE values. No intermediate NRE values are interpolated.
- `training_full.pdf` / `.svg`: supplementary friction/damping controls using the same axes, fonts and metric colors as Figure 4. Shape and line style distinguish Joint from Frozen Encoder; filled/open NRE markers distinguish Training from Development.
- `figure_design.pdf`: the overview and training figure at paper size with captions.
- `assets/cube_pair/provenance.json`: source pair, action blocks, image hashes and sample selection for the displayed pair. The original 224-pixel observations are retained for provenance. The pair was selected for legibility, without changing benchmark sampling or scores.
- `assets/cube_pair_hd/`: 1600 × 900 native simulator renderings, the saved replay states and rendering provenance. A shared camera, palette and lighting make the cube visible at paper size. The recorded cube positions are preserved; the evaluated models still receive the original observations.
- `assets/environments/provenance.json`: sources for the environment thumbnails. The PushT thumbnail comes from `assets/pusher_pair/provenance.json`.
- `assets/fonts/`: embedded Roboto fonts, distributed under the accompanying SIL Open Font License. The font sources are [Roboto](https://github.com/google/fonts/tree/main/ofl/roboto) and [Roboto Condensed](https://github.com/google/fonts/tree/main/ofl/robotocondensed). Roboto Condensed is used only in the two narrow brace labels.

The layout follows the [new supplied design reference](https://chatgpt.com/share/6ae80941-7830-83ea-a2af-476c8f212ea8). The overview combines native scene images with a schematic evaluation flow; the training plots use measured values. Lift/reverse illustrates the Cube carry rule, not the actions used in every task. Predictions are latent states, so the diagram does not imply pixel decoding. Rows compare candidate futures; columns compare histories for a fixed target. In the original layout, checkmarks illustrate successful comparisons, not measured outcomes. The response-fidelity alternative preserves the matrix colors and formulas, separates headers from the error cells, and replaces the surrounding comparison boxes with open half-arrowheads; its NRE formula uses the same ratio of sums over pairs as the manuscript. Panel A and the caption state that future actions are shared; panel B focuses on the two history inputs. Input and output arrows align with the centers of their rows and are symmetric about the taller model box. Lowercase latent notation follows the manuscript.

## Regeneration

The SVG drawings use Python's standard library. PDF/PNG export requires Node.js, Playwright and Chromium:

```bash
python paper/design/draw_overview.py
python paper/design/draw_training.py
python paper/design/draw_training_full.py
node paper/design/render_figures.cjs
# To export only the overview: node paper/design/render_figures.cjs benchmark_overview
```

To build the PushT alternative and its comparison PDF without changing the manuscript figure:

```bash
python paper/design/draw_overview.py --example pusher
node paper/design/render_figures.cjs benchmark_overview_pusher
cd paper
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=design -jobname=overview_comparison design/overview_comparison.tex
```

The [response-fidelity design discussion](https://chatgpt.com/share/6ae83146-0a04-83ea-86d4-c335690cff4c) motivates the optional third comparison in panel B. Build it with:

```bash
python paper/design/draw_overview.py --example pusher --response-fidelity
node paper/design/render_figures.cjs benchmark_overview_pusher_response
cd paper
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=design -jobname=response_fidelity_preview design/response_fidelity_preview.tex
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

The high-resolution native scene PNGs used in the overview are included. Re-rendering them is optional and requires the ContextWorld Cube simulation dependencies (MuJoCo, OGBench, NumPy and Pillow), plus OSMesa for headless rendering:

```bash
python paper/design/render_cube_assets.py
```

The saved states avoid needing the source dataset for this display-only rendering. The source-state comparison and original observation hashes are documented in the two Cube provenance files.

Training plots use the checked-in `native_training_loss_v1` and `native_epoch_checkpoints_v1` CSV files. The selected values and run identifier are also recorded in `training_two_panels.json`. Compiling the paper uses the saved figure PDFs and does not require regenerating them or accessing the original datasets.
