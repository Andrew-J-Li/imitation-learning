import robosuite as suite
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import EmptyArena
from robosuite.models.tasks import ManipulationTask
from sift import produce_motions

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
    print(produce_motions())
    # controller_config = load_composite_controller_config(controller="BASIC")

    # env = MyCustomEnv(
    #     robots="GR1",
    #     controller_configs=controller_config,
    #     has_renderer=True,
    #     render_camera="frontview",
    # )

    # obs = env.reset()

    # # Print action dimension info
    # print(f"Action dimension: {env.action_dim}")
    
    # # Get the robot to understand action structure
    # robot = env.robots[0]
    
    # for step in range(150):
    #     action = np.zeros(env.action_dim)
        
    #     # Head is at indices 15-17 (yaw, roll, pitch)
    #     action[15] = -1.0  # Head yaw = rotate toward +90 degrees
        
    #     obs, reward, done, info = env.step(action)
    #     env.render()
        
    #     # Print head orientation every 50 steps
    #     if step % 50 == 0:
    #         # Get head joint positions from simulation
    #         head_yaw_pos = env.sim.data.qpos[env.sim.model.joint_name2id('robot0_head_yaw')]
    #         head_roll_pos = env.sim.data.qpos[env.sim.model.joint_name2id('robot0_head_roll')]
    #         head_pitch_pos = env.sim.data.qpos[env.sim.model.joint_name2id('robot0_head_pitch')]
            
    #         # Convert to degrees
    #         head_yaw_deg = np.degrees(head_yaw_pos)
    #         head_roll_deg = np.degrees(head_roll_pos)
    #         head_pitch_deg = np.degrees(head_pitch_pos)
            
    #         print(f"Step {step:3d}: head_yaw={head_yaw_deg:6.2f}°, head_roll={head_roll_deg:6.2f}°, head_pitch={head_pitch_deg:6.2f}°")

    # env.close()