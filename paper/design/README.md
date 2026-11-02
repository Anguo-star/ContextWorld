# ContextWorld figures

The manuscript uses `benchmark_overview_pusher_response.pdf`, copied to `paper/figures/protocol_overview.pdf` by `paper/generate_figures.py`. The overview explains the controlled task design, paired evaluation and task coverage. The training figure presents a measured example; it is not repeated in the overview.

- `benchmark_overview.pdf` / `.svg`: an editable reconstruction of the supplied 1448 × 1086 design. Panel A shows a recorded Cube carry-rule pair; panel B shows separate frozen-model calls and the four error comparisons; panel C maps nine tasks to four environments. Commands label transitions, and both conditions use exactly the same current-observation rendering. Text, arrows, matrix cells and panel frames remain vector graphics. Cube images are high-resolution native renderings of recorded states. Other environment views are saved dataset screenshots. The design reference supplies the layout and visual styling, not scene pixels.
- `benchmark_overview_pusher.pdf` / `.svg`: alternative main-figure preview using a real PushT pusher-displacement pair. The original scene screenshots share one crop. All three transitions are labeled Move; red arrows summarize the shared action directions. Dashed arrows show measured endpoint displacement, including the return to the shared current observation, not an interpolated path. A faint translucent silhouette with a dashed outline in the lower future panel marks the T block's current pose; its silhouette is traced from the shared original observation and saved in `assets/pusher_pair/current_t_outline.json`. These annotations are not part of model inputs. Sources and arrow semantics are recorded in `overview_pusher.json`.
- `overview_comparison.pdf`: PushT and Cube alternatives at manuscript width on separate pages.
- `benchmark_overview_pusher_response.pdf` / `.svg`: PushT overview with three equal evaluation cards: Future discrimination, History benefit, and Response fidelity. The first two cards use exactly the same four schematic states. Filled circles denote predictions; open circles denote true futures. A halo and two emphasized error segments identify the fixed prediction or fixed true future. The NRE vectors are computed from the same schematic differences, with a common rotation and scale. These are conceptual geometry, not measured embeddings; source history images remain unchanged.
- `paired_evaluation_design.pdf` / `.svg` / `.png`: the same B panel exported separately for inspection at manuscript width. Its layout follows the latest supplied design, with three numbered pastel cards. Marker meanings are stated in the figure caption, without a separate key row. Its outer title and panel badge match panels A and C. `assets/reference/paired_evaluation_design.png` retains the original design image for comparison, not as paper content.
- `prediction_image_preview.pdf`: layout alternative with true-future thumbnails labeled Reference beside the latent prediction symbols. These are visual references, not decoded predictions. Generate it with `--example pusher --response-fidelity --prediction-images` and export `benchmark_overview_pusher_response_images`.
- `response_fidelity_preview.pdf`: the response-fidelity alternative at manuscript width, with its explanatory caption. This standalone preview uses the same real PushT figure as the manuscript.
- `assets/reference/main_figure_design.png`: the complete supplied design, retained at its original resolution. It is used only as a visual layout reference and is not embedded in the figure. `overview_crop.json` records the native data sources, shared crop and state/action checks.
- `training_two_panels.pdf` / `.svg`: PLDM contact friction, Original → Mixed (Joint). The left plot contains six recorded prediction-loss means; the right contains the two measured training-query NRE values. No intermediate NRE values are interpolated.
- `training_full.pdf` / `.svg`: supplementary friction/damping controls using the same axes, fonts and metric colors as Figure 4. Shape and line style distinguish Joint from Frozen Encoder; filled/open NRE markers distinguish Training from Development.
- `figure_design.pdf`: the overview and training figure at paper size with captions.
- `assets/cube_pair/provenance.json`: source pair, action blocks, image hashes and sample selection for the displayed pair. The original 224-pixel observations are retained for provenance. The pair was selected for legibility, without changing benchmark sampling or scores.
- `assets/cube_pair_hd/`: 1600 × 900 native simulator renderings, the saved replay states and rendering provenance. A shared camera, palette and lighting make the cube visible at paper size. The recorded cube positions are preserved; the evaluated models still receive the original observations.
- `assets/environments/provenance.json`: sources for the environment thumbnails. The PushT thumbnail comes from `assets/pusher_pair/provenance.json`.
- `assets/fonts/`: embedded Roboto fonts, distributed under the accompanying SIL Open Font License. The font sources are [Roboto](https://github.com/google/fonts/tree/main/ofl/roboto) and [Roboto Condensed](https://github.com/google/fonts/tree/main/ofl/robotocondensed). Roboto Condensed is used only in the two narrow brace labels.

The layout follows the [new supplied design reference](https://chatgpt.com/share/6ae80941-7830-83ea-a2af-476c8f212ea8). The overview combines native scene images with a schematic evaluation flow; the training plots use measured values. Lift/reverse illustrates the Cube carry rule, not the actions used in every task. Predictions are latent states, so the diagram does not imply pixel decoding. In the original matrix layout, rows compare candidate futures and columns compare histories for a fixed target; checkmarks illustrate successful comparisons, not measured outcomes. The response-fidelity alternative replaces the matrix with two views of the same four schematic states, followed by their response vectors. All four distances are shown; emphasis identifies which two errors are compared for condition A, and the equations also specify the symmetric comparison for B. Its NRE formula uses the same ratio of sums over pairs as the manuscript. Panel A and the caption state that future actions are shared; panel B focuses on the two history inputs. Input and output arrows align with the centers of their rows and are symmetric about the taller model box. Lowercase latent notation follows the manuscript.

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

The [three-subpanel design discussion](https://chatgpt.com/share/6ae8562d-f4ec-83ea-8185-eb0fb5418e9d) supplies the current B-panel layout. Real observation images are preserved; schematic point positions and response vectors are made geometrically consistent. Build it with:

```bash
python paper/design/draw_overview.py --example pusher --response-fidelity
node paper/design/render_figures.cjs benchmark_overview_pusher_response paired_evaluation_design
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
