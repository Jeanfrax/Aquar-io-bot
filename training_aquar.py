import os
import time
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from aquar_env import AquarEnv, _browser, _playwright

def main():
    # Directorios para logs y para guardar el mejor modelo
    log_dir = "logs"
    best_dir = "best_model"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)

    # Crear entornos de entrenamiento y evaluación
    train_env = Monitor(AquarEnv(headless=True), log_dir)
    eval_env  = Monitor(AquarEnv(headless=True), log_dir)

    # Callback para evaluar periódicamente y guardar el mejor modelo
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_dir,
        log_path=log_dir,
        eval_freq=50_000,
        deterministic=True,
        render=False
    )

    # Callback opcional para detener si alcanza un umbral de recompensa
    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=500,  # ajusta según tu necesidad
        verbose=1
    )

    # Crear y configurar el agente DQN
    model = DQN(
        "CnnPolicy",
        train_env,
        verbose=1,
        buffer_size=100_000,
        learning_starts=5_000,
        batch_size=32,
        gamma=0.99,
        tensorboard_log="tensorboard"
    )

    # Entrenar el modelo
    model.learn(
        total_timesteps=1_000_000,
        callback=[eval_callback, stop_callback]
    )

    # Guardar el modelo final
    model.save(os.path.join(best_dir, "final_model"))

    # Cerrar entornos
    train_env.close()
    eval_env.close()

    # Cerrar navegador Playwright global
    _browser.close()
    _playwright.stop()

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = (time.time() - start_time) / 60
    print(f"Training completed in {elapsed:.2f} minutes")
