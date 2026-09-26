import torch
import numpy as np
import matplotlib.pyplot as plt
from display_results import rollout_trajectory_feedback_shape_match

#This file is used to compute all of the different metrics that are used to compare the truth cube toss, to the
# predicted cube toss from the GNN. The metrics we compute are:
# 1. Relative center position error: the distance between the predicted and true center of mass, normalized
#    by the block width.
# 2. Absolute angle difference: the angle difference between the predicted and true rotation matrices,
#    converted to degrees.
# 3. Floor penetration: the maximum depth that any part of the cube goes below the
#    floor (z=0), normalized by the block width.

BLOCK_HALF_WIDTH = 0.0524
BLOCK_WIDTH = 2 * BLOCK_HALF_WIDTH

# --- thresholds for phase segmentation ---
CONTACT_Z_THRESH    = 0.2 * BLOCK_HALF_WIDTH   # lowest node within ~1cm of floor => in contact
SETTLE_SPEED_THRESH = 0.01 * BLOCK_WIDTH       # per-step COM displacement below this => "stopped"
SETTLE_RUN          = 5                          # consecutive sub-threshold frames => settled


def plot_phase_error_curves(trajectory_folder, model, Wall, test_trajectory_indices,
                            nodes_per_edge, nearest_neighbors, rest_positions,
                            accel_std, accel_mean, x_mean, x_std, e_mean, e_std,
                            weights_only_load, unscale_trajectory_data, h, use_wind=False,
                            zero_at_phase_start=True, min_frac=0.5, save_path=None):
    """
    Per-frame error vs time, split into airborne / contact / settled, averaged over
    the test set (mean +/- 1 std band). Re-indexes each phase to 'frames since phase
    start' so trajectories with different contact times align.

    zero_at_phase_start: subtract each trajectory's error at its phase start, so each
        panel shows GROWTH within the phase (y can go negative => error shrank).
    min_frac: only plot frames where >= this fraction of trajectories still contribute.
    """
    phases = ['airborne', 'contact', 'settled']
    center_segs = {p: [] for p in phases}
    angle_segs  = {p: [] for p in phases}

    slope_rows = []

    N = len(test_trajectory_indices)
    print(f"[phase curves] rolling out {N} test trajectories "
          f"(use_wind={use_wind})...", flush=True)

    for i, throw_number in enumerate(test_trajectory_indices, 1):
        pred_positions, true_positions, _ = rollout_trajectory_feedback_shape_match(
            trajectory_folder, model, Wall,
            throw_number=throw_number,
            nodes_per_edge=nodes_per_edge, nearest_neighbors=nearest_neighbors,
            rest_positions=rest_positions, accel_std=accel_std, accel_mean=accel_mean,
            x_mean=x_mean, x_std=x_std, e_mean=e_mean, e_std=e_std,
            do_shape_match=True, shape_alpha=1.0, return_edge_info=True,
            weights_only_load=weights_only_load,
            unscale_trajectory_data=unscale_trajectory_data, h=h, use_wind=use_wind,
        )
        m = compute_metrics(pred_positions, true_positions, rest_positions)
        ce = m['center_error_t'].numpy()
        ae = m['angle_error_t_deg'].numpy()
        tc, ts = compute_phase_boundaries(true_positions)
        bounds = {'airborne': (0, tc), 'contact': (tc, ts), 'settled': (ts, len(ce))}
        for p in phases:
            a, b = bounds[p]
            if b - a < 1:
                continue
            cseg, aseg = ce[a:b].copy(), ae[a:b].copy()
            if zero_at_phase_start:
                cseg = cseg - cseg[0]
                aseg = aseg - aseg[0]
            center_segs[p].append(cseg)
            angle_segs[p].append(aseg)

        print(f"  [{i:>3}/{N}] traj {throw_number}: "
              f"airborne {tc:>3}  contact {ts - tc:>3}  settled {len(ce) - ts:>3} frames",
              flush=True)
        
        

    print(f"[phase curves] rollouts done, building figure...", flush=True)

    def stack_stats(seg_list):
        if not seg_list:
            return None
        maxlen = max(len(s) for s in seg_list)
        arr = np.full((len(seg_list), maxlen), np.nan)
        for i, s in enumerate(seg_list):
            arr[i, :len(s)] = s
        return (np.nanmean(arr, axis=0), np.nanstd(arr, axis=0),
                np.sum(~np.isnan(arr), axis=0))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    row_info = [(center_segs, 'center error (/width)'), (angle_segs, 'angle error (deg)')]
    for row, (segdict, ylabel) in enumerate(row_info):
        for col, p in enumerate(phases):
            ax = axes[row, col]
            stats = stack_stats(segdict[p])
            if stats is None:
                ax.set_title(f"{p}: no data"); ax.grid(alpha=0.3); continue
            mean, std, n = stats
            keep = n >= max(3, int(min_frac * n.max()))
            x = np.arange(len(mean))[keep]
            y = mean[keep]

            # Compute slope of the mean curve
            if len(x) >= 2:
                slope, intercept = np.polyfit(x, y, 1)
            else:
                slope = np.nan
                intercept = np.nan

            med_len = int(np.median([len(s) for s in segdict[p]]))
            max_mean_error = np.nanmax(y) if len(y) > 0 else np.nan

            metric_name = "Center Error" if row == 0 else "Angle Error"

            print(
                f"{metric_name:12s} | {p:9s} | "
                f"Median Frames: {med_len:3d} | "
                f"Slope: {slope:8.5f} | "
                f"Max Mean Error: {max_mean_error:8.5f}"
            )

            slope_rows.append(dict(metric=metric_name, phase=p,
                                   median_frames=med_len, slope=float(slope),
                                   max_mean_error=float(max_mean_error)))

            ax.plot(x, y, color='C0', lw=2)
            ax.fill_between(x, (mean - std)[keep], (mean + std)[keep], alpha=0.25, color='C0')
            if len(x) >= 2:
                # Optional: show linear fit
                ax.plot(x, slope * x + intercept,
                        color='C3', ls='--', lw=1.5)

                # Show slope on the subplot
                ax.text(
                    0.03, 0.97,
                    f"Slope = {slope:.4f}",
                    transform=ax.transAxes,
                    ha='left',
                    va='top',
                    fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
                )
            if zero_at_phase_start:
                ax.axhline(0.0, color='k', lw=0.6, ls='--', alpha=0.5)
            med_len = int(np.median([len(s) for s in segdict[p]]))
            if row == 0:
                ax.set_title(f"{p}  (n={len(segdict[p])}, median {med_len} frames)")
            if row == 1:
                ax.set_xlabel("frames since phase start")
            if col == 0:
                ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)

    ttl = "Per-phase error growth" + ("  (zeroed at phase start)" if zero_at_phase_start else "  (absolute)")
    fig.suptitle(ttl, fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved phase error curves to {save_path}")
    plt.show()

    return slope_rows


def compute_phase_boundaries(true_positions,
                             contact_z_thresh=CONTACT_Z_THRESH,
                             settle_speed_thresh=SETTLE_SPEED_THRESH,
                             settle_run=SETTLE_RUN):
    """
    Returns (t_contact, t_settle) splitting a trajectory into:
        airborne : [0, t_contact)
        contact  : [t_contact, t_settle)   (impact + bounce/slide transient)
        settled  : [t_settle, T)
    Defined on ground truth so boundaries are identical across models/ablations.
    """
    T = true_positions.shape[0]
    min_z = true_positions[..., 2].min(dim=1).values        # (T,) lowest node each frame
    cm    = true_positions.mean(dim=1)                       # (T,3)
    speed = torch.norm(cm[1:] - cm[:-1], dim=-1)             # (T-1,) per-step COM displacement

    contact_mask = min_z <= contact_z_thresh
    nz = torch.nonzero(contact_mask, as_tuple=False)
    t_contact = int(nz[0].item()) if nz.numel() > 0 else T   # T => never contacts (all airborne)

    t_settle = T
    below = speed < settle_speed_thresh
    start = min(t_contact, len(speed))
    for t in range(start, len(speed) - settle_run + 1):
        if (bool(below[t:t + settle_run].all())
                and bool((speed[t:] < 2.0 * settle_speed_thresh).all())):   # and STAYS slow
            t_settle = t
            break
    return t_contact, t_settle

#Computes the angle difference between two rotation matrices.
def angle_between_rotations(R_pred, R_true):

    R_rel = R_pred.T @ R_true
    trace = torch.trace(R_rel)
    # Clamp to valid range for arccos
    cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
    return torch.arccos(cos_angle)


#Computes the three metrics for a single trajectory, given the predicted and true positions over time, 
#as well as the rest positions of the cube's nodes.
def compute_metrics(pred_positions, true_positions, rest_positions):

    #Makes sure everything is on the same device for computation.
    device = pred_positions.device
    rest_positions = rest_positions.to(device)

    T, N, _ = pred_positions.shape

    # Centers of mass per timestep
    pred_cm = pred_positions.mean(dim=1)  # (T,3)
    true_cm = true_positions.mean(dim=1)  # (T,3)

    # Rest relative positions (constant)
    q = rest_positions - rest_positions.mean(dim=0)  # (N,3)

    # Pred/true relative positions per timestep
    p_pred = pred_positions - pred_cm[:, None, :]  # (T,N,3)
    p_true = true_positions - true_cm[:, None, :]  # (T,N,3)

    # Apq per timestep: (T,3,3)
    # Apq[t] = sum_n p[t,n]^T outer q[n]
    Apq_pred = torch.einsum('tni,nj->tij', p_pred, q)  # (T,3,3)
    Apq_true = torch.einsum('tni,nj->tij', p_true, q)  # (T,3,3)

    # Batched SVD
    U_p, S_p, Vh_p = torch.linalg.svd(Apq_pred)  # U:(T,3,3), Vh:(T,3,3)
    U_t, S_t, Vh_t = torch.linalg.svd(Apq_true)

    # Rotation fix to ensure a proper rotation (det = +1)
    # d = det(U @ Vh) for each timestep
    d_p = torch.linalg.det(U_p @ Vh_p)  # (T,)
    d_t = torch.linalg.det(U_t @ Vh_t)  # (T,)

    # Build D matrices batched: (T,3,3)
    D_p = torch.eye(3, device=device).expand(T, 3, 3).clone()
    D_t = torch.eye(3, device=device).expand(T, 3, 3).clone()
    D_p[:, 2, 2] = d_p
    D_t[:, 2, 2] = d_t

    R_pred = U_p @ D_p @ Vh_p  # (T,3,3)
    R_true = U_t @ D_t @ Vh_t  # (T,3,3)

    # --- Metric 1: Relative center position error
    center_errors = torch.norm(pred_cm - true_cm, dim=1) / BLOCK_WIDTH  # (T,)

    # --- Metric 2: Absolute angle difference ---
    R_rel = R_pred.transpose(-1, -2) @ R_true  # (T,3,3)
    trace = R_rel.diagonal(dim1=-2, dim2=-1).sum(-1)  # (T,)
    cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
    angle_errors = torch.arccos(cos_angle)  # (T,)

    # --- Metric 3: Floor penetration  ---
    z = pred_positions[..., 2]  # (T,N)
    floor_penetrations = torch.clamp(-z, min=0.0).amax(dim=1) / BLOCK_WIDTH  # (T,)

    t_contact, t_settle = compute_phase_boundaries(true_positions)

    def _phase_avg(vec):
        T_ = vec.shape[0]
        out = {}
        for name, sl in (('airborne', slice(0, t_contact)),
                         ('contact',  slice(t_contact, t_settle)),
                         ('settled',  slice(t_settle, T_))):
            seg = vec[sl]
            out[name] = seg.mean().item() if seg.numel() > 0 else float('nan')
        return out

    center_phase = _phase_avg(center_errors)
    angle_phase  = _phase_avg(torch.rad2deg(angle_errors))

    #Returns all of the metrics averaged across the trajectory, 
    #as well as the per-timestep values for each metric for further analysis if desired.
    # print("Center error for whole trajectory:")
    # print(center_errors)
    # print("Angle error (degrees) for whole trajectory:")
    # print(torch.rad2deg(angle_errors))
    # print("Floor penetration for whole trajectory:")
    # print(floor_penetrations)
    return {
        'center_error':            center_errors.mean().item(),
        'angle_error_deg':         torch.rad2deg(angle_errors).mean().item(),
        'floor_penetration':       floor_penetrations.mean().item(),
        'center_error_t':          center_errors.detach().cpu(),
        'angle_error_t_deg':       torch.rad2deg(angle_errors).detach().cpu(),
        'floor_penetration_t':     floor_penetrations.detach().cpu(),
        'center_error_airborne': center_phase['airborne'],
        'center_error_contact':  center_phase['contact'],
        'center_error_settled':  center_phase['settled'],
        'angle_error_airborne':  angle_phase['airborne'],
        'angle_error_contact':   angle_phase['contact'],
        'angle_error_settled':   angle_phase['settled'],
        't_contact': t_contact, 't_settle': t_settle,
    }


#This function runs a rollout of the GNN model on the specified test trajectories, 
#computes the metrics for each trajectory, and averages them across all trajectories to get an overall 
#performance evaluation of the model. It prints out the average and standard deviation for each metric across the test set.
def evaluate_model(trajectory_folder, model, Wall, test_trajectory_indices, nodes_per_edge,
                   nearest_neighbors,
                   rest_positions, accel_std, accel_mean,
                   x_mean, x_std, e_mean, e_std, weights_only_load, unscale_trajectory_data,h, use_wind = False):
    

    all_center_errors = []
    all_angle_errors = []
    all_floor_penetrations = []
    phase_keys = ['center_error_airborne','center_error_contact','center_error_settled',
                  'angle_error_airborne','angle_error_contact','angle_error_settled']
    phase_acc = {k: [] for k in phase_keys}

    #runs through each trajectory in the test set
    for throw_number in test_trajectory_indices:
        print(f"Evaluating trajectory {throw_number}...")

        #Simulates a rollout of the GNN model on the current trajectory, getting the predicted and true positions over time.
        pred_positions, true_positions, _ = rollout_trajectory_feedback_shape_match(
            trajectory_folder,
            model, Wall,
            throw_number=throw_number,
            nodes_per_edge=nodes_per_edge,
            nearest_neighbors=nearest_neighbors,
            rest_positions=rest_positions,
            accel_std=accel_std,
            accel_mean=accel_mean,
            x_mean=x_mean,
            x_std=x_std,
            e_mean=e_mean,
            e_std=e_std,
            do_shape_match=True,
            shape_alpha=1.0,
            return_edge_info=True,
            weights_only_load=weights_only_load,
            unscale_trajectory_data=unscale_trajectory_data,
            h = h,
            use_wind = use_wind,
        )

        #Computes the metrics for the current trajectory and appends them to the lists for averaging later.
        metrics = compute_metrics(pred_positions, true_positions, rest_positions)
        all_center_errors.append(metrics['center_error'])
        all_angle_errors.append(metrics['angle_error_deg'])
        all_floor_penetrations.append(metrics['floor_penetration'])

        for k in phase_keys:
            phase_acc[k].append(metrics[k])

    
    

    #Prints the average and standard deviation of each metric across the test set.
    print("\n--- Test Set Metrics ---")
    print(f"Center Error ( / width):   {np.mean(all_center_errors):.4f} ± {np.std(all_center_errors):.4f}")
    print(f"Angle Error (degrees):     {np.mean(all_angle_errors):.4f} ± {np.std(all_angle_errors):.4f}")
    print(f"Floor Penetration (/ width):         {np.mean(all_floor_penetrations):.4f} ± {np.std(all_floor_penetrations):.4f}")

    # ...after the loop, with the other prints:
    print("\n--- Phase breakdown (airborne / contact / settled) ---")
    print(f"Center (/width): "
        f"{np.nanmean(phase_acc['center_error_airborne']):.4f} / "
        f"{np.nanmean(phase_acc['center_error_contact']):.4f} / "
        f"{np.nanmean(phase_acc['center_error_settled']):.4f}")
    print(f"Angle (deg):     "
        f"{np.nanmean(phase_acc['angle_error_airborne']):.4f} / "
        f"{np.nanmean(phase_acc['angle_error_contact']):.4f} / "
        f"{np.nanmean(phase_acc['angle_error_settled']):.4f}")

    return {
        'center_error':      np.mean(all_center_errors),
        'angle_error_deg':   np.mean(all_angle_errors),
        'floor_penetration': np.mean(all_floor_penetrations),
        'center_error_std':      np.std(all_center_errors),
        'angle_error_std':       np.std(all_angle_errors),
        'floor_penetration_std': np.std(all_floor_penetrations),
        'phase_center': [np.nanmean(phase_acc['center_error_airborne']),
                         np.nanmean(phase_acc['center_error_contact']),
                         np.nanmean(phase_acc['center_error_settled'])],
        'phase_angle':  [np.nanmean(phase_acc['angle_error_airborne']),
                         np.nanmean(phase_acc['angle_error_contact']),
                         np.nanmean(phase_acc['angle_error_settled'])],
        'n_test': len(all_center_errors),
    }


