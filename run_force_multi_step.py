"""Configure and execute a complete force GNS experiment. This is the main script
for choosing trajectory folders and dataset splits, model architecture,
training duration, multistep settings, physical constants, and physics-loss
weights. It uses those settings to call the trainer, evaluate saved models,
collect training and checkpoint diagnostics, write individual and master run
reports, and generate rollout visualizations. Flags control which stages run,
making this the place to change experiment settings while the imported
modules implement training, evaluation, reporting, and plotting.
"""

import os
import torch
import wall
from train_force_gns import train_force_gnn, epochs_for_optimizer_steps
from evaluate_force_model import evaluate_force_model
from visualize_force_model import visualize_force_rollout
from run_report import save_run_report, collect_run_diagnostics
from force_data import BLOCK_HALF_WIDTH
from physics_losses import summarize_diagnostics, reset_diagnostics


#Setting that just ensures high precision for matrix multiplications in float32
torch.set_float32_matmul_precision('high')

#Gets the current script directory
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
# Architecture / recipe
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

# Set an optimizer-step target (e.g. 1_000_000) to derive the epoch budget.
# None uses epochs below. Conversion counts actual trajectory windows, partial
# batches, gradient accumulation, and curriculum phases, then rounds UP to a
# whole epoch. Early stopping can end the run before reaching this target.
target_optimizer_steps = None
epochs = 10000




# Stop after this many epochs without a lower validation center error.
# None disables early stopping. Patience begins at the final curriculum K;
# validation also runs at the deadline before deciding whether to stop.
early_stopping_patience = 2000

# Multistep Settings --------------------------------------------------------------------------------------------
multistep = 4                                    # K >= 2 required for w_fluid_smooth
curriculum_epochs = 50                           # epochs per phase of [1,2,4,8]
curriculum_schedule = None                       # None -> powers of 2 up to multistep

#Sets the learning rate scheduler for the training process, 3 options "decay", "cosine", or None
Learning_Rate_Scheduler = None                

# Number of minibatches whose gradients are accumulated per optimizer update.
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
DT = 1.0 / 148.0        # replica record rate. NOTE: force_data.DT_RECORD
                        # is 0.00674 (0.0001348*50) while the replica records at
                        # 1/148 = 0.006757 - the known small mismatch. The wind
                        # FEATURE (imported builder) keeps the old constant for
                        # parity with existing runs; the DYNAMICS here uses the
                        # correct 1/148.

GRAVITY = None          # None -> read from replica_physics in the data (9.615)


MASS = 0.37             # Mass of the cube

# Soft geometric contact gate applied independently to each cube node.
# contact_d0 is the distance where the contact weight is 0.5; contact_tau
# controls how gradually the weight changes with distance. The resulting
# weight scales the predicted contact force and friction-loss contributions.
# These parameters detect proximity to the surface, not static vs. sliding;
# that distinction uses the separate slip-speed gate in train_force_gns.py.
contact_d0 = 0.02               # 50% contact weight at this distance (m)
contact_tau = 0.005             # distance softness of the transition (m)



# ----------------------------------------------------------------------
# LEARNED CONSTANT SETTINGS
# ----------------------------------------------------------------------


K_OVER_M = 0.0285      #Initilized value for the drag coefficient at the cube's center of mass

LEARN_K = True        # Setting to determine if k/m should be optimized for during training
                      # (False holds k/m fixed at K_OVER_M)

MU_INIT = 0.3          # friction coefficient init
LEARN_MU = True        # recover mu from data (the drag-coefficient story)
                       # (False holds mu fixed at MU_INIT)


# Wind feature turns on the wind related information in the node states and allows for the model to predict wind effects.
use_wind_feature = True        

# Drag baseline makes the model already know that the aerodynamic drag at the cube's center of mass follows the analytic drag law.
# The model will then only try and learn the residual drag effects beyond the analytic baseline.
# Having this inabled makes the models prediction of k/m worse.
use_drag_baseline = False        
                                

# Loss mode determines the main loss type that really enforces the main motion of the cube.
# The "accel" loss focuses on matching the predicted accelerations to the ground truth, while the 
# "position" loss focuses on the cube's positional accuracy.
# Accel performes better for capturing the dynamic response of the cube
loss_mode = "accel"

# ----------------------------------------------------------------------
# PHYSICS-INFORMED LOSS 

# --- FRICTION -----------------------------------------------------------

w_fric_dir = 6       #direction: Enforces the correct orientation of the friction force relative to the slip direction
w_fric_mag = 1       #magnitude: Enforces the friction forces to follow Coulomb's law
w_fric_cone= 1.5       #Cone: Enforces the friction force to lie within the Coulomb friction cone

# --- FLUID -----------------------------------------------------------

w_fluid_anchor = 2e-1  #Fluid anchor: Enforces the fluid force to stay near the analytic drag law
w_fluid_smooth = 3e-2  #Fluid smooth: Enforces the fluid force to vary smoothly in time (K >= 2 only)



#Where you set the run name and model folder paths for what and where you want the model to be saved.
run_name = "force_phys_loss_zero_Weight"      
model_folder_path = os.path.join(script_dir, "models", run_name)
os.makedirs(model_folder_path, exist_ok=True)
save_model_path = os.path.join(
    model_folder_path, f"{Used_Num_train_trajectories}_force_gns_model.pt")


# Flags to control the training, evaluation, and visualization of the model.
Train_model = True
Evaluate_model = True
Visualize_model = True


# Trajectory numbers that will be visualized into GIFsduring the evaluation.
VISUALIZE_TRAJECTORIES = [test_range[0], test_range[len(test_range) // 2]]

#Flag to actually show the GIFS after they are generated (Gifs are always saved, this just controls display)
VISUALIZE_SHOW = False

# Flag to save a run report and specify the master CSV file for all runs.
Save_run_report = True
master_excel_file_name = "master_test_tracker.csv"
FORCE_MASTER_CSV = os.path.join(script_dir, "models", master_excel_file_name)

# ----------------------------------------------------------------------

#If the training flag is set, train the model.
planned_optimizer_steps = None
if Train_model:

    # Clear the diagnostic buffers in case several trainings share a process.
    reset_diagnostics()

    # Inspect actual trajectory lengths so the step budget matches the trainer.
    if target_optimizer_steps is not None:
        trajectory_lengths = [
            int(torch.load(os.path.join(trajectory_folder, f"{idx}.pt"),
                           weights_only=weights_only_load)[0].shape[0])
            for idx in train_range
        ]
        epochs, planned_optimizer_steps = epochs_for_optimizer_steps(
            trajectory_lengths, target_optimizer_steps, batch_size,
            pos_history, multistep, accumulation_steps,
            curriculum_epochs, curriculum_schedule)
        print(f"Optimizer-step target: {target_optimizer_steps:,} -> {epochs:,} epochs "
              f"({planned_optimizer_steps:,} planned updates before early stopping)")

    #Train the force GNN model with the specified parameters.
    train_force_gnn(
        Wall=Floor,
        train_range=train_range,
        val_range=val_range,
        save_model_path=save_model_path,
        trajectory_folder=trajectory_folder,
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
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
#If the evaluation flag is set, evaluate the model.
if Evaluate_model:
    print("\n" + "#" * 70)
    print("# EVALUATION")
    print("#" * 70)

    # Evaluate the trained force GNN model on the test set.
    # Always returned: center_error, angle_error_deg, floor_penetration,
    # center_error_std, angle_error_std, floor_penetration_std,
    # phase_center [airborne, contact, settled],
    # phase_angle [airborne, contact, settled], n_test, have_wrench_labels.
    # With wrench labels, also returned:
    #   impulse_timing_fraction[_std], impulse_E_frame_over_signal[_std],
    #   impulse_E_total_over_signal[_std], impulse_n_traj (when available);
    #   force_contact_err_{airborne,contact,settled},
    #   force_contact_true_{airborne,contact,settled},
    #   force_fluid_err_{airborne,contact,settled},
    #   force_fluid_true_{airborne,contact,settled} (for phases with samples);
    #   force_{contact,fluid}_{rms_true,rms_err,err_over_signal}, and
    #   force_{contact,fluid}_r2 (when the true signal has nonzero variance).
    metrics = evaluate_force_model(
        model_folder=model_folder_path,
        data_folder=trajectory_folder,
        test_indices=test_range,
        weights_only=weights_only_load,
        unscale=unscale_trajectory_data,
    )

    # summarize_diagnostics returns the following keys from the last 20
    # slip_gate_report records (one record per epoch, from its first batch):
    #   diag_align, diag_align_std: mean friction/slip anti-alignment cosine
    #     and its standard deviation; +1 means friction opposes slip.
    #   diag_misalign_deg: angle from perfect anti-alignment, computed from
    #     the mean cosine (not the mean of per-epoch angles).
    #   diag_mu_implied, diag_mu_implied_std: implied Coulomb mu from predicted
    #     forces, mean and standard deviation.
    #   diag_gate_frac: mean contact-weight fraction admitted by the slip gate.
    #   diag_cancel_slide, diag_cancel_static: mean friction-force
    #     cancellation fraction in sliding and static branches.
    #   diag_n_epochs: number of records included in the tail.
    # These values are merged into metrics
    # and therefore included in the run report when one is saved.
    diagnostics = summarize_diagnostics(last_n=20)
    metrics.update({k: float(v) for k, v in diagnostics.items()})


    # collect_run_diagnostics returns optional keys from the saved checkpoints:
    #   recovered_mu, recovered_k_over_m: final values from the physics file.
    #   final_train_loss, final_train_loss_std, final_train_loss_n: mean,
    #     standard deviation, and finite-value count over the last 20 losses.
    #   epochs_completed, stopped_early: actual duration and stopping status.
    #   best_val_loss, best_val_epoch, total_optimizer_steps: values saved in
    #     the loss-history file, when present.
    #   mu_init / k_init, mu_drift / k_drift, and
    #     mu_tail_slope_per_1k_ep / k_tail_slope_per_1k_ep: initial trace value,
    #     end-minus-start change, and slope over the last 10% of each trace.
    metrics.update(collect_run_diagnostics(save_model_path))

    # print a summary of the collected metrics
    print("\nSummary:", {k: (round(v, 4) if isinstance(v, float) else v)
                         for k, v in metrics.items()})


    if Save_run_report:
        print("\nEnd-of-run diagnostics (mean of last "
              f"{diagnostics.get('diag_n_epochs', 0)} epochs):")
        for k, v in sorted(diagnostics.items()):
            print(f"    {k:<22} {v:.6g}")

        # saves the runs settings into a dictionary
        settings = dict(
            architecture="force",           
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
            target_optimizer_steps=target_optimizer_steps,
            planned_optimizer_steps=planned_optimizer_steps,
            accumulation_steps=accumulation_steps,
            early_stopping_patience=early_stopping_patience,
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

        # saves the run report to excel
        save_run_report(model_folder_path, settings, metrics,
                        run_name=run_name, master_csv=FORCE_MASTER_CSV)

# ----------------------------------------------------------------------
#Visualizes the trajectories you specify in VISUALIZE_TRAJECTORIES, and the modeles predictions of them
if Visualize_model:
    print("\n" + "#" * 70)
    print("# VISUALIZATION")
    print("#" * 70)
    for traj_idx in VISUALIZE_TRAJECTORIES:
        try:
            # visualize the rollout for the current trajectory index
            # Saving and showing the visualization happens in the function itself
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
