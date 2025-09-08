# Genetic Image Compressor
An image “compressor” that approximates any input picture by layering semi-transparent triangles on a white canvas. 
A custom Genetic Algorithms engine evolves candidate artworks—each genome encodes triangle count, positions, colors, and alpha—optimizing a fitness function to minimize the visual error vs. the target image. 

The system supports pluggable selection/crossover/mutation strategies and YAML/JSON configs for reproducible runs. Outputs include the best-rendered image, evolution metrics (fitness over generations), 
and an image representation of the final triangle set—offering a compact, human-legible “compressed” form.

## Models Overview
- **Gene**: Each [triangle](./src/models/triangle.py) is represented by its 3 vertex points and a RGBA color. `(p1,p2,p3, R,G,B,A)` -> 10 alleles per gene `(6 floats, 4 ints)`.
    - Vertex points `p1, p2, p3` coordinates are normalized `floats` in `[0, 1]` relative to the canvas size.
    - Color channels `R, G, B, A` are `ints` in `[0, 255]`. 


- **Genome**: Each [individual](./src/models/individual.py) encodes a fixed number of triangles. Sorted list of `num_triangles` genes, `z-order` matters (later triangles overlay earlier ones, see [PillowRenderer](./src/engine/PillowRenderer.py)).

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

