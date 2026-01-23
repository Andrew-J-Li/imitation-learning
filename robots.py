import robosuite as suite
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import EmptyArena
from robosuite.models.tasks import ManipulationTask
from robosuite.controllers import load_composite_controller_config
from sift import video_to_motions
import numpy as np
import matplotlib.pyplot as plt

class MyCustomEnv(ManipulationEnv):
    """Custom environment with GR1 robot."""

    def __init__(
        self,
        robots="GR1",
        env_configuration="default",
        controller_configs=None,
        base_types="default",
        gripper_types="default",
        initialization_noise=None,
        use_camera_obs=False,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
        **kwargs
    ):

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types=base_types,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
            **kwargs
        )

    def _load_model(self):
        """Load robot and arena models."""
        super()._load_model()

        # Adjust base pose for empty arena
        xpos = self.robots[0].robot_model.base_xpos_offset["empty"]
        self.robots[0].robot_model.set_base_xpos(xpos)

        # Create arena
        mujoco_arena = EmptyArena()
        mujoco_arena.set_origin([0, 0, 0])

        # Create task (combines arena and robots)
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
        )

    def _setup_references(self):
        """Setup references to important objects."""
        super()._setup_references()

    def _setup_observables(self):
        """Setup observation functions."""
        observables = super()._setup_observables()
        return observables

    def _reset_internal(self):
        """Reset simulation internals."""
        super()._reset_internal()

    def reward(self, action=None):
        return False

# Create and run the environment
if __name__ == "__main__":
    # Hyperparameters
    OFFSET = 8  # Lookahead frames for target
    
    controller_config = load_composite_controller_config(controller="BASIC")

    env = MyCustomEnv(
        robots="GR1",
        controller_configs=controller_config,
        has_renderer=True,
        render_camera="frontview",
    )
    obs = env.reset()

    # Get relative motions from video
    motions = video_to_motions(path='interpolation.mov')
    
    # Accumulate motions to get absolute target at each frame
    cumulative_yaw = 0.0
    cumulative_roll = 0.0
    cumulative_pitch = 0.0

    # Naively accumulate all frame-to-frame changes 
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
    
    print(f"Total accumulated rotation: yaw={cumulative_yaw:.1f}°, roll={cumulative_roll:.1f}°, pitch={cumulative_pitch:.1f}°")

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
        target_idx = min(step + OFFSET, len(targets) - 1)
        if target_idx >= 0 and target_idx < len(targets):
            target_yaw, target_roll, target_pitch = targets[target_idx]
        else:
            target_yaw, target_roll, target_pitch = head_yaw_deg, head_roll_deg, head_pitch_deg

        # Compute error
        yaw_error = target_yaw - head_yaw_deg
        roll_error = target_roll - head_roll_deg
        pitch_error = target_pitch - head_pitch_deg
        
        # Proportional control: action proportional to error
        # Scale error to action space (error in degrees / max degrees)
        action = np.zeros(env.action_dim)
        action[15] = np.clip(yaw_error / 90.0, -1.0, 1.0)
        action[16] = np.clip(roll_error / 15.0, -1.0, 1.0)
        action[17] = np.clip(pitch_error / 15.0, -1.0, 1.0)
        
        obs, reward, done, info = env.step(action)
        env.render()

        # Examine accumulated errors
        if step > 0 and step <= len(targets):
            prev_yaw, prev_roll, prev_pitch = targets[step - 1]
            errors.append((
                abs(head_yaw_deg - prev_yaw),
                abs(head_roll_deg - prev_roll),
                abs(head_pitch_deg - prev_pitch),
                abs(prev_yaw),
                abs(prev_roll),
                abs(prev_pitch)
            ))

        # Print every 50 steps
        if step % 50 == 0:
            print(f"Step {step:3d}: head_yaw={head_yaw_deg:6.2f}°, head_roll={head_roll_deg:6.2f}°, head_pitch={head_pitch_deg:6.2f}° | err: y={yaw_error:.1f} r={roll_error:.1f} p={pitch_error:.1f}")

    env.close()

    # Plot errors over frames
    if errors:
        errors_arr = np.array(errors)
        frames = np.arange(len(errors))
        
        # Calculate percentage errors relative to target magnitude
        eps = 0.1
        pct_errors = np.column_stack([
            np.minimum(errors_arr[:, 0] / np.maximum(errors_arr[:, 3], eps) * 100, 100),
            np.minimum(errors_arr[:, 1] / np.maximum(errors_arr[:, 4], eps) * 100, 100),
            np.minimum(errors_arr[:, 2] / np.maximum(errors_arr[:, 5], eps) * 100, 100)
        ])
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        
        # Yaw
        ax1 = axes[0]
        ax1_pct = ax1.twinx()
        line1, = ax1.plot(frames, errors_arr[:, 0], 'b-', linewidth=0.8, label='Degrees')
        line1_pct, = ax1_pct.plot(frames, pct_errors[:, 0], 'b--', linewidth=0.8, alpha=0.5, label='Percent')
        ax1.set_ylabel('Yaw Error (°)', color='b')
        ax1_pct.set_ylabel('Yaw Error (%)', color='b')
        ax1.set_title('Tracking Errors Over Time')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax1.legend(handles=[line1, line1_pct], loc='upper right')
        
        # Roll
        ax2 = axes[1]
        ax2_pct = ax2.twinx()
        line2, = ax2.plot(frames, errors_arr[:, 1], 'g-', linewidth=0.8, label='Degrees')
        line2_pct, = ax2_pct.plot(frames, pct_errors[:, 1], 'g--', linewidth=0.8, alpha=0.5, label='Percent')
        ax2.set_ylabel('Roll Error (°)', color='g')
        ax2_pct.set_ylabel('Roll Error (%)', color='g')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.legend(handles=[line2, line2_pct], loc='upper right')
        
        # Pitch
        ax3 = axes[2]
        ax3_pct = ax3.twinx()
        line3, = ax3.plot(frames, errors_arr[:, 2], 'r-', linewidth=0.8, label='Degrees')
        line3_pct, = ax3_pct.plot(frames, pct_errors[:, 2], 'r--', linewidth=0.8, alpha=0.5, label='Percent')
        ax3.set_ylabel('Pitch Error (°)', color='r')
        ax3_pct.set_ylabel('Pitch Error (%)', color='r')
        ax3.set_xlabel('Frame')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax3.legend(handles=[line3, line3_pct], loc='upper right')
        
        plt.tight_layout()
        plt.savefig('output/tracking_errors.png', dpi=150)
        plt.show()
        
        # Print summary stats
        print(f"\nError Statistics:")
        print(f"  Yaw   - Mean: {np.mean(errors_arr[:, 0]):6.2f}° ({np.mean(pct_errors[:, 0]):5.1f}%), Std: {np.std(errors_arr[:, 0]):6.2f}°, Max: {np.max(errors_arr[:, 0]):6.2f}°")
        print(f"  Roll  - Mean: {np.mean(errors_arr[:, 1]):6.2f}° ({np.mean(pct_errors[:, 1]):5.1f}%), Std: {np.std(errors_arr[:, 1]):6.2f}°, Max: {np.max(errors_arr[:, 1]):6.2f}°")
        print(f"  Pitch - Mean: {np.mean(errors_arr[:, 2]):6.2f}° ({np.mean(pct_errors[:, 2]):5.1f}%), Std: {np.std(errors_arr[:, 2]):6.2f}°, Max: {np.max(errors_arr[:, 2]):6.2f}°")