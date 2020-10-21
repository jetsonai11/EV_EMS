from gym.envs.registration import register

register(
    id='FASTSim-v0',
    entry_point='FASTSim_Package.envs:FASTSimEnvironment',
    max_episode_steps=2000,
)
