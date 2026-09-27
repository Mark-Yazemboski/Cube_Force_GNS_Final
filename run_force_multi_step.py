"""
run_force_multi_step.py

Main file for running the force GNS architecture
"""

import os
import torch
import wall
from train_force_gns import train_force_gnn
from evaluate_force_model import evaluate_force_model
from visualize_force_model import visualize_force_rollout
from run_report import save_run_report
from run_diagnostics import collect_run_diagnostics
from generate_node_states import BLOCK_HALF_WIDTH
from physics_losses import summarize_diagnostics, reset_diagnostics


torch.set_float32_matmul_precision('high')

script_dir = os.path.dirname(os.path.abspath(__file__))

# Sets where our floor/wall is located. This is so the model knows hwo far each node is form the floor
# And also what the normal vector of the floor is.
Floor = wall.wall(center_position=(0, 0, 0), size=(2, 2), normal=(0, 0, 1))

# Points to the trajectory folder you want to train on
trajectory_folder = os.path.join(script_dir, "data/mojoco_paper_replica_20_wind")

# Settings for loading and preprocessing trajectory data
# Keeps both False when loading raw MuJoCo-generated trajectory data
# Keep  both True when loading the papers trajectory data "tosses_processed"
weights_only_load = False                        
unscale_trajectory_data = False

# Defines the total number of trajectories you want to use from the trajectory folder, and what percentage
# of the trajectories you want to use for training and validation. The rest will be used for testing
Num_total_trajectories = 569
training_percentage = 0.5
validation_percentage = 0.3

Num_train = int(training_percentage * Num_total_trajectories)
Num_val = int(validation_percentage * Num_total_trajectories)

#This is the number of training trajectories to actually use.
# Override for experiments with smaller training sets
Used_Num_train_trajectories = 256

#Used to tell the program where to start the training trajectories from
#Used in tests where using less than the full set of training trajectories
#This allows for flexibility in selecting subsets of the training data for experiments.
train_start = 0         

#Sets the ranges of the training, validation, and test trajectories
train_range = range(train_start, train_start + Used_Num_train_trajectories)
val_range = range(Num_train, Num_train + Num_val)
test_range = range(Num_train + Num_val, Num_total_trajectories - 1)

# Print out the ranges for verification
print(f"Training range: {train_range}")
print(f"Validation range: {val_range}")
print(f"Test range: {test_range}")

# ----------------------------------------------------------------------
# Architecture / recipe (matches the current best acceleration recipe)
# ----------------------------------------------------------------------
nodes_per_edge = 2
K_nearest_neighbors = 3
message_passing_layers = 5
repeat_blocks = 1
Latent_dimension = 128
pos_history = 3
batch_size = 512
learning_rate = 1e-4
noise_scale = 3e-4 * BLOCK_HALF_WIDTH            # meters/step, same as accel runs

# Training length, in epochs, the SAME for every configuration.
# At K=1 this is the budget that has been used all along (validation flattens
# around 6k, so 10k is headroom). Holding it fixed across K also keeps the
# number of OPTIMIZER STEPS roughly constant - samples per trajectory are
# T - h - K, so K barely changes the steps per epoch - which makes a K
# comparison a comparison of gradient updates, not of compute. The cost is
# wall clock: a K-step unroll backprops through K predictions, so K=4 runs
# about 4x longer per epoch.
epochs = 10000



# Multistep Settings --------------------------------------------------------------------------------------------
multistep = 4                                    # K >= 2 required for w_fluid_smooth
curriculum_epochs = 50                           # epochs per phase of [1,2,4,8]
curriculum_schedule = None                       # None -> powers of 2 up to multistep

#Sets the learning rate scheduler for the training process, 3 options "decay", "cosine", or None
Learning_Rate_Scheduler = None                

#Accumulation settings will modify how many contact frames are added to the batch during training.
accumulation_steps = 1

# Validation and checkpoint intervals
# How often to run validation and save checkpoints (in epochs)
validation_check_interval = 10
epoch_checkpoint_interval = 100

# Keeps the last N checkpoints, deleting older ones to save storage.
# Set to 0 to keep every checkpoint.
keep_last_n_checkpoints = 2




# ----------------------------------------------------------------------
# CONSTANT SETTINGS
# ----------------------------------------------------------------------
DT = 1.0 / 148.0        # replica record rate. NOTE: generate_node_states.DT_RECORD
                        # is 0.00674 (0.0001348*50) while the replica records at
                        # 1/148 = 0.006757 - the known small mismatch. The wind
                        # FEATURE (imported builder) keeps the old constant for
                        # parity with existing runs; the DYNAMICS here uses the
                        # correct 1/148.

GRAVITY = None          # None -> read from replica_physics in the data (9.615)


MASS = 0.37             # Mass of the cube

contact_d0 = 0.02               # soft geometric contact gate center (m)
contact_tau = 0.005             # gate width (m)



# ----------------------------------------------------------------------
# LEARNED CONSTANT SETTINGS
# ----------------------------------------------------------------------


K_OVER_M = 0.0285      #Initilized value for the drag coefficient at the cube's center of mass

LEARN_K = True        # Setting to determine if k/m should be optimized for during training
                      # (False holds k/m fixed at K_OVER_M)

MU_INIT = 0.3          # friction coefficient init
LEARN_MU = True        # recover mu from data (the drag-coefficient story)
                       # (False holds mu fixed at MU_INIT)



use_wind_feature = True        # Stage 1: off. Stage 2+: on for wind datasets.
use_drag_baseline = True        # analytic drag at COM (calibrated k/m); the
                                # anchor term assumes this is the fluid center

# Loss: "accel" = per-node acceleration MSE, the SAME objective as the
# acceleration model's _unroll_chain_loss_accel, so the parity comparison is
# exact and the logged loss number is directly comparable. "position" = the
# block-width position MSE. Keep "accel" until parity is established.
loss_mode = "accel"

# ----------------------------------------------------------------------
# PHYSICS-INFORMED LOSS (proposal Eq. 5-6; one function per term in
# physics_losses.py). Raw magnitudes print every epoch - calibrate each
# gamma so (gamma * raw) is ~1-10%% of the position loss after epoch 1.
#
# What the wrench labels showed and which term answers it:
#   fluid channel carried ~0.2 mg of friction (= mu m g), at every wind level
#     -> w_fluid_anchor pins fluid to the analytic drag law
#     -> w_fluid_smooth forbids the jumpy, contact-synchronized compensation
#        (the chaotic pink arrow) - NEEDS multistep >= 2
#     -> w_fric_dir / w_fric_mag give the displaced friction a
#        correctly-structured home: anti-parallel to slip, proportional to
#        the local normal force, one global mu. mu is LEARNABLE by default,
#        so the model recovers the friction coefficient the same way it
#        recovered the drag coefficient (replica ground truth: mu = 0.198).
#   h_pen needs no weight: normal forces are >= 0 by construction (softplus).
# --- CONTACT / friction ---------------------------------------------
# Coulomb friction is enforced as two separate halves (direction, magnitude)
# rather than one joint residual || phi_t + mu phi_n vhat ||^2: with the joint
# form a single global mu absorbs any directional error (mu -> mu_true * <cos>;
# measured here: mu_param 0.156 vs mu_implied 0.216, ~44 deg of misalignment).
w_fric_dir = .3       # gamma_1a : direction half - fixes crossing arrows
w_fric_mag = .1       # gamma_1b : magnitude half - mu's ONLY gradient path
w_fric_cone= 1.5       # gamma_1c : ||phi_t|| <= mu phi_n, STATIC regime too

# --- FLUID -----------------------------------------------------------
# The anchor ladder (Aug 28-29, 3 seeds/cell) was monotonic with no motion
# cost: fluid_err_contact 0.1368 -> 0.0479 mg from w=0 to w=1e-1, 65% down,
# center error flat within the baseline spread. It had NOT plateaued at 1e-1.
w_fluid_anchor = 3e-2  # gamma_3a: fluid FORCE == analytic drag law
w_fluid_smooth = 3e-2  # gamma_3b: fluid force smooth in time (K >= 2 only)



# ----------------------------------------------------------------------
# Naming / paths
# ----------------------------------------------------------------------
extra_name = "force_phys_loss_zero_Weight"      # CHANGE PER EXPERIMENT
model_folder_path = os.path.join(script_dir, "models", extra_name)
os.makedirs(model_folder_path, exist_ok=True)
save_model_path = os.path.join(
    model_folder_path, f"{Used_Num_train_trajectories}_force_gns_model.pt")

# Everything runs from this one file so a single batch job on ROAR trains,
# evaluates, and renders the GIFs without a second submission.
Train_model = True
Evaluate_model = True
Visualize_model = True

# Which test trajectories to render as GIFs. Keep this short - each one is a
# full rollout plus a matplotlib animation, so ~10-30 s apiece.
VISUALIZE_TRAJECTORIES = [test_range[0], test_range[len(test_range) // 2]]
VISUALIZE_SHOW = False          # False on a compute node (no display)

# One canonical row per force run, kept in its OWN master file so the force
# architecture's numbers never get mixed into the acceleration model's
# all_runs_master.csv. Opens directly in Excel.
Save_run_report = True
FORCE_MASTER_CSV = os.path.join(script_dir, "models", "all_force_runs_master.csv")

# ----------------------------------------------------------------------
if Train_model:
    # Clear the diagnostic buffers in case several trainings share a process.
    reset_diagnostics()
    train_force_gnn(
        Wall=Floor,
        train_range=train_range,
        val_range=val_range,
        save_model_path=save_model_path,
        trajectory_folder=trajectory_folder,
        epochs=epochs,
        batch_size=batch_size,
        accumulation_steps=accumulation_steps,
        lr=learning_rate,
        nodes_per_edge=nodes_per_edge,
        nearest_neighbors=K_nearest_neighbors,
        h=pos_history,
        message_passing_layers=message_passing_layers,
        repeat_blocks=repeat_blocks,
        latent_dim=Latent_dimension,
        weights_only=weights_only_load,
        unscale_data=unscale_trajectory_data,
        noise_scale=noise_scale,
        multistep=multistep,
        curriculum_epochs=curriculum_epochs,
        curriculum_schedule=curriculum_schedule,
        Learning_Rate_Scheduler=Learning_Rate_Scheduler,
        use_wind=use_wind_feature,
        dt=DT,
        gravity=GRAVITY,
        mass=MASS,
        use_drag_baseline=use_drag_baseline,
        k_over_m=K_OVER_M, learn_k=LEARN_K,
        contact_d0=contact_d0,
        contact_tau=contact_tau,
        loss_mode=loss_mode,
        w_fric_dir=w_fric_dir, w_fric_mag=w_fric_mag, w_fric_cone=w_fric_cone,
        w_fluid_anchor=w_fluid_anchor,
        w_fluid_smooth=w_fluid_smooth,
        mu_init=MU_INIT, learn_mu=LEARN_MU,
        validation_check_interval=validation_check_interval,
        epoch_checkpoint_interval=epoch_checkpoint_interval,
        keep_last_n_checkpoints=keep_last_n_checkpoints,
    )

# ----------------------------------------------------------------------
if Evaluate_model:
    print("\n" + "#" * 70)
    print("# EVALUATION")
    print("#" * 70)
    metrics = evaluate_force_model(
        model_folder=model_folder_path,
        data_folder=trajectory_folder,
        test_indices=test_range,
        weights_only=weights_only_load,
        unscale=unscale_trajectory_data,
    )

    # Mean over the last 20 epochs of the per-epoch diagnostics that
    # slip_gate_report() recorded during training (one per epoch, from that
    # epoch's first batch): friction alignment, mu_implied, slip-gate
    # occupancy. These are the metrics the friction sweep is ranked on, so
    # they belong in the same CSV row as force_contact_err_contact instead of
    # only in the log. Averaged, not final-value: one batch is noisy.
    # Empty when this process did not train (Train_model = False).
    diagnostics = summarize_diagnostics(last_n=20)
    metrics.update({k: float(v) for k, v in diagnostics.items()})

    # Recovered mu / k/m, their traces, and the converged training loss, read
    # from the checkpoints training wrote. Never fail a finished run over
    # reporting.
    try:
        metrics.update(collect_run_diagnostics(save_model_path))
    except Exception as e:
        print(f"  checkpoint diagnostics unavailable: {e}")

    print("\nSummary:", {k: (round(v, 4) if isinstance(v, float) else v)
                         for k, v in metrics.items()})

    if Save_run_report:
        print("\nEnd-of-run diagnostics (mean of last "
              f"{diagnostics.get('diag_n_epochs', 0)} epochs):")
        for k, v in sorted(diagnostics.items()):
            print(f"    {k:<22} {v:.6g}")

        settings = dict(
            architecture="force",           # distinguishes these rows at a glance
            dataset=trajectory_folder,
            n_train=Used_Num_train_trajectories,
            train_range=f"{train_range.start}-{train_range.stop}",
            val_range=f"{val_range.start}-{val_range.stop}",
            test_range=f"{test_range.start}-{test_range.stop}",
            nodes_per_edge=nodes_per_edge,
            nearest_neighbors=K_nearest_neighbors,
            message_passing_layers=message_passing_layers,
            repeat_blocks=repeat_blocks,
            latent_dim=Latent_dimension,
            pos_history=pos_history,
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs,
            noise_scale=noise_scale,
            multistep=multistep,
            curriculum_epochs=curriculum_epochs,
            scheduler=Learning_Rate_Scheduler,
            loss_mode=loss_mode,
            use_wind=use_wind_feature,
            use_drag_baseline=use_drag_baseline,
            k_over_m=K_OVER_M, learn_k=LEARN_K,
            contact_d0=contact_d0,
            contact_tau=contact_tau,
            dt=DT, gravity=GRAVITY, mass=MASS,
            w_fric_dir=w_fric_dir, w_fric_mag=w_fric_mag,
            w_fric_cone=w_fric_cone,
            w_fluid_anchor=w_fluid_anchor,
            w_fluid_smooth=w_fluid_smooth,
            learn_mu=LEARN_MU,
        )
        save_run_report(model_folder_path, settings, metrics,
                        run_name=extra_name, master_csv=FORCE_MASTER_CSV)

# ----------------------------------------------------------------------
if Visualize_model:
    print("\n" + "#" * 70)
    print("# VISUALIZATION")
    print("#" * 70)
    for traj_idx in VISUALIZE_TRAJECTORIES:
        try:
            out = visualize_force_rollout(
                model_folder=model_folder_path,
                data_folder=trajectory_folder,
                trajectory=int(traj_idx),
                show=VISUALIZE_SHOW,
                weights_only=weights_only_load,
                unscale=unscale_trajectory_data,
            )
            print(f"  wrote {out}")
        except Exception as e:
            # A failed GIF must never take down a finished training run.
            print(f"  visualization of traj {traj_idx} FAILED: {type(e).__name__}: {e}")

print("\nAll done.")
