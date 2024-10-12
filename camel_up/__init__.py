from gymnasium.envs.registration import register

register(
    id="camel_up",
    entry_point="camel_up.gym_env:CamelUpEnv",
    max_episode_steps=300,
)
