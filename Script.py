from aquar_env import AquarEnv

env = AquarEnv(headless=False)
# Gymnasium reset(): (obs, info)
obs, info = env.reset()
print("Obs shape:", obs.shape)           # (4, 84, 84)

# Gymnasium step(): (obs, reward, terminated, truncated, info)
obs, r, terminated, truncated, info = env.step(env.action_space.sample())
done = terminated or truncated
print("Reward:", r, "Done:", done)

env.close()
