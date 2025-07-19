import time
import random
import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces
from playwright.sync_api import sync_playwright

# ------------------------------------------------------
# 1) Definición del entorno AquarEnv
# ------------------------------------------------------
_playwright = sync_playwright().start()

_browser = _playwright.chromium.launch(
    headless=True,
    args=["--no-sandbox", "--disable-setuid-sandbox"]
)


class AquarEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self,
                 width: int = 84,
                 height: int = 84,
                 actions: int = 5,
                 video_dir: str = r'C:\Users\micha\Desktop\BotNeuronal\Aquar.io\videos'):
        super().__init__()
        self.width, self.height, self.actions = width, height, actions
        self.action_space = spaces.Discrete(actions)
        self.observation_space = spaces.Box(0, 255, (4, height, width), np.uint8)

        # Creo contexto con grabación de video
        self.context = _browser.new_context(
            record_video_dir=video_dir,
            record_video_size={"width": 800, "height": 600}
        )
        self.page = self.context.new_page()
        self.page.goto("https://aquar.io", timeout=60000)
        # Quitar loader inicial si lo hay
        self.page.evaluate("() => document.getElementById('layer-loading')?.remove()")
        self.frames = []

    def reset(self, **kwargs):
        # *** Login automático paso a paso ***
        self.page.evaluate("() => document.getElementById('layer-loading')?.remove()")
        time.sleep(0.3)

        # 1) Nick aleatorio
        nick = f"GARAY{random.randint(1,20)}"
        self.page.fill("input[type='text']", nick)
        print("1) Nickname →", nick)
        time.sleep(0.3)

        # 2) Región “Europe”
        self.page.select_option("select[name='server']", label="Europe")
        print("2) Región → Europe")
        time.sleep(0.3)

        # 3) Modo “Teams”
        self.page.select_option("select[name='mode']", label="Teams")
        print("3) Modo → Teams")
        time.sleep(0.3)

        # 4) Equipo/Clan: buscar cualquier opción con “GARAY CLAN”
        clan_value = self.page.evaluate("""
            () => {
                const sel = document.querySelector('select[name="team"]');
                const opt = Array.from(sel.options)
                    .find(o => o.text.includes('GARAY CLAN'));
                return opt ? opt.value : null;
            }
        """)
        if clan_value is None:
            raise RuntimeError("No se encontró 'GARAY CLAN' en el desplegable")
        self.page.select_option("select[name='team']", value=clan_value)
        print("4) Equipo → GARAY CLAN")
        time.sleep(0.3)

        # 5) Play as Guest
        self.page.click("text=Play as Guest")
        print("5) ▶ Play as Guest")
        self.page.wait_for_selector("#score", timeout=20000)
        print("6) ✅ Partida iniciada")
        
        # 6) Capturo los primeros 4 frames
        self.frames = []
        for _ in range(4):
            buf = self.page.screenshot(type="png")
            img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
            self.frames.append(img)
        obs = np.stack(self.frames, axis=0)
        self.last_score = int(self.page.evaluate("() => parseInt(document.getElementById('score').textContent)"))
        return obs, {}

    def step(self, action):
        mapping = {0: [], 1: ['w'], 2: ['d'], 3: ['s'], 4: ['a']}
        for k in mapping[action]:
            self.page.keyboard.down(k)
        time.sleep(0.05)
        for k in mapping[action]:
            self.page.keyboard.up(k)

        buf = self.page.screenshot(type="png")
        img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
        self.frames.pop(0); self.frames.append(img)
        obs = np.stack(self.frames, axis=0)

        score = int(self.page.evaluate("() => parseInt(document.getElementById('score').textContent)"))
        reward = score - self.last_score - 1
        self.last_score = score
        done = bool(self.page.query_selector('#game-over'))
        return obs, reward, done, False, {"score": score}

    def render(self, mode="human"):
        if mode == "human":
            cv2.imshow("Aquar.io", self.frames[-1])
            cv2.waitKey(1)

    def close(self):
        self.context.close()

# Al final de login_bot.py
def do_login(seed=None):
    """
    Arranca AquarEnv, hace login, cierra y devuelve True/False.
    `seed` solo sirve si quieres variar el nick.
    """
    try:
        if seed is not None:
            import random
            random.seed(seed)
        env = AquarEnv()
        obs, info = env.reset()   # aquí ocurre el login
        # env.close()
        return True
    except Exception as e:
        print("ERROR en do_login:", e)
        return False

if __name__ == "__main__":
    # Test manual de un solo login
    ok = do_login()
    print("Login único:", "OK" if ok else "FAIL")

