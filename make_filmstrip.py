"""
make_filmstrip.py

Thin command-line wrapper around visualize_force_rollout() so you can export
filmstrip frames without editing constants at the bottom of a file.

Pick a trajectory with pick_trajectory.py first, then:

    # wind figure: model and data must BOTH be the 20-wind set
    python make_filmstrip.py \
        --model models/FIN_W20_LEARNK_1 \
        --data  data/mojoco_paper_replica_20_wind \
        --traj  543 --panels 6 --no-gif

    # force-decomposition figure on the 0-wind set
    python make_filmstrip.py \
        --model models/FIN_W0_1 \
        --data  data/mojoco_paper_replica_0_wind \
        --traj  529 --panels 6 --tangent-gain 12 --no-gif

    # choose the frames yourself instead of letting it pick
    python make_filmstrip.py --model ... --data ... --traj 543 \
        --frames 14,19,23,31,48,90

THE MODEL AND THE DATA MUST MATCH. A model trained on 20-wind rolled out on
0-wind data is a different experiment, and the forces it draws are not the ones
in your results table.

Frames land in  <model>/frames_traj<N>/frame_XXX.pdf  unless you pass --outdir.
"""

import argparse
import os
import sys

from visualize_force_model import visualize_force_rollout


def parse_frames(text):
    if text is None:
        return "auto"
    return [int(v) for v in text.replace(" ", "").split(",") if v]


def main():
    p = argparse.ArgumentParser(
        description="Export filmstrip frames from one rollout.")
    p.add_argument("--model", required=True,
                   help="folder holding *_norms.pt and the checkpoint")
    p.add_argument("--data", required=True, help="trajectory folder")
    p.add_argument("--traj", type=int, required=True, help="trajectory number")

    p.add_argument("--frames", default=None,
                   help="comma-separated frame numbers; omit to auto-select")
    p.add_argument("--panels", type=int, default=6,
                   help="how many frames to auto-select (default 6)")
    p.add_argument("--outdir", default=None, help="where the frames go")
    p.add_argument("--format", default="pdf", choices=("pdf", "svg", "png"))

    p.add_argument("--no-gif", action="store_true",
                   help="skip the animation and only write frames")
    p.add_argument("--keep-labels", action="store_true",
                   help="keep the HUD, title, legend and axes on each frame")

    p.add_argument("--tangent-gain", type=float, default=8.0,
                   help="friction arrows are small; raise this to see them")
    p.add_argument("--normal-gain", type=float, default=1.0)
    p.add_argument("--fluid-gain", type=float, default=1.0)
    p.add_argument("--mg-widths", type=float, default=1.0,
                   help="a force of m*g draws this many block-widths long")

    p.add_argument("--arrow-mode", default="log", choices=("log", "clip", "linear"),
                   help="log: compresses big forces so impacts and friction "
                        "are both legible (default). clip: linear then capped. "
                        "linear: physically proportional, impacts leave frame.")
    p.add_argument("--max-arrow", type=float, default=2.2,
                   help="cap in block-widths, used by --arrow-mode clip")
    p.add_argument("--zoom", type=float, default=1.13,
                   help=">1 pulls the camera in; 1.13 is ~13%% closer")
    p.add_argument("--slide-nodes", type=int, default=4,
                   help="loaded nodes required in the sliding frame")
    p.add_argument("--elev", type=float, default=18.0)
    p.add_argument("--azim", type=float, default=-62.0)
    p.add_argument("--checkpoint", default="best", choices=("best", "final"))
    p.add_argument("--prefix", default=None,
                   help="model prefix; omit to auto-detect the single *_norms.pt")
    p.add_argument("--weights-only", action="store_true")
    p.add_argument("--unscale", action="store_true")
    p.add_argument("--no-ground-truth", action="store_true",
                   help="hide the blue ground-truth cube")
    a = p.parse_args()

    for path, what in ((a.model, "model folder"), (a.data, "data folder")):
        if not os.path.isdir(path):
            sys.exit(f"{what} does not exist: {path}")

    gif_path = os.path.join(a.model, f"force_rollout_{a.traj}.gif")

    out = visualize_force_rollout(
        a.model, a.data, a.traj,
        model_prefix=a.prefix,
        save_path=gif_path,
        show=False,
        weights_only=a.weights_only,
        unscale=a.unscale,
        checkpoint=a.checkpoint,
        draw_ground_truth=not a.no_ground_truth,
        mg_arrow_widths=a.mg_widths,
        normal_gain=a.normal_gain,
        tangent_gain=a.tangent_gain,
        fluid_gain=a.fluid_gain,
        save_frames=parse_frames(a.frames),
        n_panels=a.panels,
        frame_dir=a.outdir,
        frame_format=a.format,
        frame_clean=not a.keep_labels,
        elev=a.elev,
        azim=a.azim,
        arrow_mode=a.arrow_mode,
        max_arrow_widths=a.max_arrow,
        zoom=a.zoom,
        slide_min_nodes=a.slide_nodes,
        make_gif=not a.no_gif,
    )

    print("\n  frames:")
    for f in out["frames"]:
        print(f"    {f}")
    print(f"\n  contact at frame {out['t_contact']}, "
          f"settled at {out['t_settle']}, {out['n_frames']} frames total")
    if out["gif"]:
        print(f"  gif: {out['gif']}")


if __name__ == "__main__":
    main()
