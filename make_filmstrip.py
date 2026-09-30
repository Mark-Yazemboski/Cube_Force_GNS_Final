"""Command-line wrapper for exporting force-model rollout filmstrips.

Uses ``visualize_force_rollout`` to save selected frames and optional GIFs.
The model and data directories must represent the same wind condition.
Creates an entire filmstrip of key frames from the force-model rollout.
"""

import argparse
import os
import sys

from visualize_force_model import visualize_force_rollout

# Parses command-line arguments and invokes the visualization function.
def parse_frames(text):
    if text is None:
        return "auto"
    return [int(v) for v in text.replace(" ", "").split(",") if v]


def main():

    # Parse command-line arguments.
    p = argparse.ArgumentParser(
        description="Export filmstrip frames from one rollout.")

    # Define the command-line arguments.
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

    # Verify that the specified model and data directories exist.
    for path, what in ((a.model, "model folder"), (a.data, "data folder")):
        if not os.path.isdir(path):
            sys.exit(f"{what} does not exist: {path}")

    # Construct the path for the output GIF based on the model directory and trajectory.
    gif_path = os.path.join(a.model, f"force_rollout_{a.traj}.gif")

    # Invoke the visualization function to generate the filmstrip and optionally the GIF.
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
