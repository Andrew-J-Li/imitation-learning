import numpy as np
from robosuite.controllers import load_composite_controller_config
from robots import MyCustomEnv
from sift import video_to_motions
import matplotlib.pyplot as plt


def run_with_offset(offset: int, video_path: str = 'yaw.mov', verbose: bool = False):
    """
    Run the robot simulation with a given target index offset.
    
    Args:
        offset: How many frames ahead to look for the target (1 = current, 2 = one ahead, etc.)
        video_path: Path to the video file
        verbose: Whether to print step-by-step info
        
    Returns:
        dict with mean errors for yaw, roll, pitch
    """
    controller_config = load_composite_controller_config(controller="BASIC")

    env = MyCustomEnv(
        robots="GR1",
        controller_configs=controller_config,
        has_renderer=False,  # Disable rendering for speed
        render_camera="frontview",
    )
    obs = env.reset()

    # Get relative motions from video
    motions = video_to_motions(path=video_path)
    
    # Accumulate motions to get absolute target at each frame
    cumulative_yaw = 0.0
    cumulative_roll = 0.0
    cumulative_pitch = 0.0

    targets = []
    for motion in motions:
        if motion is not None:
            cumulative_yaw += motion[0]
            cumulative_roll += motion[1]
            cumulative_pitch += motion[2]
        # Cap at joint limits
        capped_yaw = np.clip(cumulative_yaw, -90.0, 90.0)
        capped_roll = np.clip(cumulative_roll, -15.0, 15.0)
        capped_pitch = np.clip(cumulative_pitch, -15.0, 15.0)
        targets.append((capped_yaw, capped_roll, capped_pitch))

    errors = []

    for step in range(len(targets) + 50):
        # Get current head position
        head_yaw_pos = env.sim.data.qpos[env.sim.model.joint_name2id('robot0_head_yaw')]
        head_roll_pos = env.sim.data.qpos[env.sim.model.joint_name2id('robot0_head_roll')]
        head_pitch_pos = env.sim.data.qpos[env.sim.model.joint_name2id('robot0_head_pitch')]
        
        head_yaw_deg = np.degrees(head_yaw_pos)
        head_roll_deg = np.degrees(head_roll_pos)
        head_pitch_deg = np.degrees(head_pitch_pos)

        # Use target with offset
        target_idx = min(step + offset - 1, len(targets) - 1)
        if target_idx >= 0 and target_idx < len(targets):
            target_yaw, target_roll, target_pitch = targets[target_idx]
        else:
            target_yaw, target_roll, target_pitch = head_yaw_deg, head_roll_deg, head_pitch_deg
        
        # Compute error
        yaw_error = target_yaw - head_yaw_deg
        roll_error = target_roll - head_roll_deg
        pitch_error = target_pitch - head_pitch_deg
        
        # Proportional control
        action = np.zeros(env.action_dim)
        action[15] = np.clip(yaw_error / 90.0, -1.0, 1.0)
        action[16] = np.clip(roll_error / 15.0, -1.0, 1.0)
        action[17] = np.clip(pitch_error / 15.0, -1.0, 1.0)
        
        obs, reward, done, info = env.step(action)

        # Examine accumulated errors (current position vs previous target)
        if step > 0 and step <= len(targets):
            prev_yaw, prev_roll, prev_pitch = targets[step - 1]
            errors.append((
                abs(head_yaw_deg - prev_yaw),
                abs(head_roll_deg - prev_roll),
                abs(head_pitch_deg - prev_pitch),
            ))

        if verbose and step % 50 == 0:
            print(f"Step {step:3d}: yaw={head_yaw_deg:6.2f}° | err: y={yaw_error:.1f}")

    env.close()

    if not errors:
        return {'yaw': float('inf'), 'roll': float('inf'), 'pitch': float('inf'), 'total': float('inf')}

    errors_arr = np.array(errors)
    median_yaw = np.median(errors_arr[:, 0])
    median_roll = np.median(errors_arr[:, 1])
    median_pitch = np.median(errors_arr[:, 2])
    median_total = median_yaw + median_roll + median_pitch

    return {
        'yaw': median_yaw,
        'roll': median_roll,
        'pitch': median_pitch,
        'total': median_total
    }


if __name__ == "__main__":
    
    # Test different offsets
    offsets_to_test = range(1, 11)  # Test offsets 1 through 10
    results = []
    
    print("Testing different target index offsets...")
    print("=" * 60)
    
    for offset in offsets_to_test:
        print(f"\nRunning with offset = {offset}...")
        result = run_with_offset(offset, video_path='interpolation.mov', verbose=False)
        results.append(result)
        print(f"  Median Yaw Error:   {result['yaw']:.4f}°")
        print(f"  Median Roll Error:  {result['roll']:.4f}°")
        print(f"  Median Pitch Error: {result['pitch']:.4f}°")
        print(f"  Total Error:        {result['total']:.4f}°")
    
    # Find best offset
    total_errors = [r['total'] for r in results]
    best_idx = np.argmin(total_errors)
    best_offset = list(offsets_to_test)[best_idx]
    
    print("\n" + "=" * 60)
    print(f"BEST OFFSET: {best_offset}")
    print(f"  Median Yaw Error:   {results[best_idx]['yaw']:.4f}°")
    print(f"  Median Roll Error:  {results[best_idx]['roll']:.4f}°")
    print(f"  Median Pitch Error: {results[best_idx]['pitch']:.4f}°")
    print(f"  Total Error:        {results[best_idx]['total']:.4f}°")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    offsets_list = list(offsets_to_test)
    
    axes[0, 0].plot(offsets_list, [r['yaw'] for r in results], 'b-o')
    axes[0, 0].set_xlabel('Offset')
    axes[0, 0].set_ylabel('Median Yaw Error (°)')
    axes[0, 0].set_title('Yaw Error vs Offset')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(x=best_offset, color='r', linestyle='--', alpha=0.5, label=f'Best: {best_offset}')
    axes[0, 0].legend()
    
    axes[0, 1].plot(offsets_list, [r['roll'] for r in results], 'g-o')
    axes[0, 1].set_xlabel('Offset')
    axes[0, 1].set_ylabel('Median Roll Error (°)')
    axes[0, 1].set_title('Roll Error vs Offset')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axvline(x=best_offset, color='r', linestyle='--', alpha=0.5, label=f'Best: {best_offset}')
    axes[0, 1].legend()
    
    axes[1, 0].plot(offsets_list, [r['pitch'] for r in results], 'r-o')
    axes[1, 0].set_xlabel('Offset')
    axes[1, 0].set_ylabel('Median Pitch Error (°)')
    axes[1, 0].set_title('Pitch Error vs Offset')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axvline(x=best_offset, color='r', linestyle='--', alpha=0.5, label=f'Best: {best_offset}')
    axes[1, 0].legend()
    
    axes[1, 1].plot(offsets_list, total_errors, 'k-o')
    axes[1, 1].set_xlabel('Offset')
    axes[1, 1].set_ylabel('Total Median Error (°)')
    axes[1, 1].set_title('Total Error vs Offset')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axvline(x=best_offset, color='r', linestyle='--', alpha=0.5, label=f'Best: {best_offset}')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('output/offset_comparison.png', dpi=150)
    plt.show()
