# run_login_pool.py

from concurrent.futures import ProcessPoolExecutor, as_completed
from login_bot import do_login

def main():
    n_bots = 20
    with ProcessPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(do_login, seed=i) for i in range(n_bots)]
        for future in as_completed(futures):
            print("Login", "OK" if future.result() else "FAIL")
    print("\nTodos los logins han finalizado. ¡Navegadores abiertos!")
