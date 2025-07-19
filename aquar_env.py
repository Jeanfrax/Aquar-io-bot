import time
import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces
from playwright.sync_api import sync_playwright

# Inicializa Playwright solo una vez
_playwright = sync_playwright().start()
_browser    = _playwright.chromium.launch(headless=True)

class AquarEnv(gym.Env):
    """
    Entorno Gymnasium para Aquar.io usando Playwright y OpenCV.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, headless=True, frame_shape=(84,84), frame_stack=4):
        super().__init__()
        # Espacios de observación y acción
        self.frame_shape      = frame_shape
        self.frame_stack      = frame_stack
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(frame_stack, *frame_shape),
            dtype=np.uint8
        )
        self.action_space = spaces.Discrete(6)  # arriba, abajo, izquierda, derecha, click, no‑op

        # Cada env usa su propio contexto
        self.context = _browser.new_context(viewport={"width":640,"height":480})
        self.page    = self.context.new_page()
        self.page.goto("https://aquar.io")
        self.page.wait_for_timeout(2000)

        self.frames     = []
        self.last_score = 0

    def _grab_frame(self):
        png = self.page.screenshot()
        img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
        return cv2.resize(img, self.frame_shape)

    def reset(self, **kwargs):
        # 1) Quita el overlay de carga si existe
        self.page.evaluate("() => document.getElementById('layer-loading')?.remove()")
        # 2) Click en “Play as Guest”
        self.page.click("#play")
        # 3) Espera a que aparezca el HUD de puntuación
        self.page.wait_for_selector("#score", timeout=5000)

        # 4) Inicializa el buffer de frames
        first = self._grab_frame()
        self.frames = [first] * self.frame_stack
        self.last_score = 0

        obs = np.stack(self.frames, axis=0)
        return obs, {}

    def step(self, action):
        # Mapea la acción a teclado o ratón
        if   action == 0: self.page.keyboard.press("ArrowUp")
        elif action == 1: self.page.keyboard.press("ArrowDown")
        elif action == 2: self.page.keyboard.press("ArrowLeft")
        elif action == 3: self.page.keyboard.press("ArrowRight")
        elif action == 4: self.page.mouse.click(320, 240)
        # acción 5: no-op

        self.page.wait_for_timeout(100)  # delay por frame

        # Captura el siguiente frame
        frame = self._grab_frame()
        self.frames.pop(0)
        self.frames.append(frame)
        obs = np.stack(self.frames, axis=0)

        # Extrae la puntuación del DOM
        score_txt = self.page.evaluate(
            "() => document.querySelector('#score')?.innerText || '0'"
        )
        score = int(score_txt) if score_txt.isdigit() else 0
        reward = score - self.last_score - 1
        self.last_score = score

        # Comprueba fin de partida
        terminated = "Game Over" in self.page.content()
        truncated  = False

        return obs, reward, terminated, truncated, {}

    def render(self, mode="human"):
        if mode == "human":
            cv2.imshow("Aquar.io", self.frames[-1])
            cv2.waitKey(1)

    def close(self):
        # Cierra solo este contexto
        self.context.close()

# Al final de tu script de entrenamiento (training_aquar.py) añade:
#   _browser.close()
#   _playwright.stop()
