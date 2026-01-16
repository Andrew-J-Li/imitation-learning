import robosuite as suite
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import EmptyArena
from robosuite.models.tasks import ManipulationTask
from robosuite.controllers import load_composite_controller_config
import numpy as np

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

    controller_config = load_composite_controller_config(controller="BASIC")

    env = MyCustomEnv(
        robots="GR1",
        controller_configs=controller_config,
        has_renderer=True,
        render_camera="frontview",
    )

    obs = env.reset()

    for _ in range(1000):
        action = np.zeros(env.action_dim)
        obs, reward, done, info = env.step(action)
        env.render()

    env.close()