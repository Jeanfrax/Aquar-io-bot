from aquar_env import AquarEnv
from stable_baselines3 import DQN

# 1) Crear el env en modo visible
env = AquarEnv(headless=False)

# 2) Cargar el mejor modelo
model = DQN.load("best_model/best_model.zip", env=env)

# 3) Corre un episodio
obs, info = env.reset()
done = False
total_reward = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    # desempacamos los 5 valores de step()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward

print("Recompensa total del episodio:", total_reward)
env.close()
