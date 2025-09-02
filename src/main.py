import argparse
import json
import numpy as np
from PIL import Image
from pathlib import Path



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, args.profile, args.overrides)
    out = Path(cfg.experiment.output_dir);
    out.mkdir(parents=True, exist_ok=True)

    target = np.array(Image.open(cfg.data.image_path).convert("RGBA").resize(cfg.data.canvas_size))
    renderer = PillowRenderer(*cfg.data.canvas_size) if cfg.renderer.backend == "pillow" else OpenCVRenderer(*cfg.data.canvas_size)
    fit = None # Fitness function TBD
    select = build_selection(cfg.selection.name, cfg.selection.params)
    xover = build_crossover(cfg.crossover.name, cfg.crossover.params)
    mutate = TriangleMutation(**cfg.mutation.params)

    eng = # GAEngine()

    pop = init_population(cfg.ga.pop_size, cfg.genome.num_triangles, cfg.data.canvas_size)
    best = eng.run(pop)
    (out / "best.json").write_text(json.dumps(best, default=lambda o: o.__dict__, indent=2))

if __name__ == "__main__": main()