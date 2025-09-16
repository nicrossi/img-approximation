# Genetic Image Compressor
An image “compressor” that approximates any input picture by layering semi-transparent triangles on a white canvas. 
A custom Genetic Algorithms engine evolves candidate artworks—each genome encodes triangle count, positions, colors, and alpha—optimizing a fitness function to minimize the visual error vs. the target image. 

The system supports pluggable selection/crossover/mutation strategies and YAML/JSON configs for reproducible runs. Outputs include the best-rendered image, evolution metrics (fitness over generations), 
and an image representation of the final triangle set—offering a compact, human-legible “compressed” form.

## Models Overview
- **Gene**: Each [triangle](./src/models/triangle.py) is represented by its 3 vertex points, an RGBA color, and a z-index. `(p1,p2,p3, R,G,B,A, z-index)` -> 11 alleles per gene `(7 floats, 4 ints)`.
    - Vertex points `p1, p2, p3` coordinates are normalized `floats` in `[0, 1]` relative to the canvas size.
    - Color channels `R, G, B, A` are `ints` in `[0, 255]`.
    - `z-index` is the gene's position in the genome list.


- **Genome**: Each [individual](./src/models/individual.py) encodes a fixed number of triangles. Sorted list of `num_triangles` genes, `z-index` order matters (higher index triangles overlay earlier ones, see [PillowRenderer](./src/engine/PillowRenderer.py)).

## Selection Strategies

- Parents selection: Configured under `selection.name` (optional parameters via `selection.params`)
- New generation selection: Configured under `survivor_selection.name` (optional parameters via `survivor_selection.params`)

| Strategy          | Description / Key Params                                                                     |
| ----------------- |----------------------------------------------------------------------------------------------|
| `elite`           | Picks the top‑`k` fitness values directly, no randomness                                     |
| `roulette`        | Weighted roulette wheel selection based on fitness values                                    |
| `universal`       | Stochastic universal sampling—equally spaced pointers reduce variance                        |
| `boltzmann`       | Temperature‑based scaling that decays each generation; params: `t_initial`, `t_final`, `decay` |
| `tournament`      | Chooses the best from random subsets; param: `tournament_size`                               |
| `prob_tournament` | Two-way tournament; best chosen with probability `threshold`                                 |
| `ranking`         | Ranks individuals and applies roulette to normalized ranks                                   |

## Crossover Strategies

Configured under crossover.name with (optional parameters via`crossover.params`)

| Strategy    | Description / Key Params                                                                      |
| ----------- |-----------------------------------------------------------------------------------------------|
| `one_point` | Cuts parent genomes at one point and swaps tails                                              |
| `two_point` | Swaps the segment between two crossover points                                                |
| `uniform`   | Each triangle inherits from either parent with probability `p`                                |
| `annular`   | Inserts a random ring (segment) of triangles from one parent into the other                   |


## Mutation Strategies

Configured under `mutation.name` with (optional parameters via `mutation.params`)

| Strategy     | Description / Key Params                                                                                                                                                                                       |
| ------------ |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `uniform`    | Mutates each vertex and color with independent rates; occasional triangle swap. Params: `point_rate`, `point_sigma`, `color_rate`, `color_sigma`, `swap_rate`                                                  |
| `gen`        | Small Gaussian jitter applied to all triangle genes; params: `point_sigma`, `color_sigma`                                                                                                                      |
| `multigen`   | Mutates a random subset of triangles intensively; params: `min_genes`, `max_genes`, `point_sigma`, `color_sigma`                                                                                               |
| `nonuniform` | Mutation range shrinks over time; mutates vertices or color channels based on probabilities. Params include `b`, `p_mutate_vertices`, `p_vertex_component`, `p_color_component`                                |


## Configuration

Experiment parameters live in `configs/config.yaml`. The loader merges a base file with optional profiles and command‑line
overrides, expands `${...}` placeholders, and appends a timestamp to `experiment.output_dir`. A minimal example:

```yaml
experiment:
  name: test
  output_dir: output/${experiment.name}
data:
  image_path: assets/argentina-flag.png
  canvas_size: [300, 200]
renderer:
  backend: pillow
fitness:
  name: pixel_mse
selection:
  name: roulette
survivor_selection:
  name: elite
crossover:
  name: one_point
mutation:
  name: simple
ga:
  pop_size: 50
genome:
  num_triangles: 30
seed: 42
```

**Top‑level keys**

- **experiment**: Run metadata. `name` is the run label; `output_dir` is the results folder (a timestamp is appended).
- **data**: Target image settings. `image_path` is the source file; `canvas_size` is the render/eval width and height.
- **renderer**: Drawing backend. Currently supports `pillow`.
- **fitness**: GA scoring function. `name` selects the method; `params` holds optional options.
- **selection**, **crossover**, **mutation**: GA operators. Each defines a `name` and optional `params` (e.g., rates).
- **ga**: Core GA settings such as `pop_size`, `generations`, `elitism`, and whether to maximize or minimize.
- **genome**: Individual layout. `num_triangles` is the number of encoded triangles.
- **seed**: Random seed for reproducibility.

Values can be overridden from the CLI using `--profile` to load `configs/profiles/<profile>.yaml` and `--set key=value` for
ad‑hoc changes, e.g.:

```bash
  python -m src.main --config configs/config.yaml --set ga.generations=10
```

