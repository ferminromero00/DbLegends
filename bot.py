# Bot de farmeo del modo historia de Dragon Ball Legends (BlueStacks + ADB + OpenCV)
# Modo aprendizaje: al encontrar una pantalla desconocida guarda la captura en
# capturas/desconocidas/ y se detiene para que se defina la nueva pantalla.
import json
import os
import subprocess
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import requests

BASE = os.path.dirname(os.path.abspath(__file__))
UNKNOWN_DIR = os.path.join(BASE, "capturas", "desconocidas")

os.system("")  # activa los codigos de color ANSI en la consola de Windows
C_RESET, C_DIM = "\033[0m", "\033[90m"
C_CYAN, C_GREEN, C_YELLOW, C_RED, C_MAGENTA = "\033[96m", "\033[92m", "\033[93m", "\033[91m", "\033[95m"


def log(msg, color=C_RESET):
    print(f"{color}{msg}{C_RESET}")


def load_config():
    with open(os.path.join(BASE, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    for s in cfg["screens"]:
        if s.get("detector"):
            continue
        tpl = cv2.imread(os.path.join(BASE, s["template"]))
        if tpl is None:
            sys.exit(f"No se pudo cargar la plantilla {s['template']}")
        s["_tpl"] = tpl
    cfg["_recovery_tpl"] = cv2.imread(os.path.join(BASE, "screens", "menu_boton.png"))
    # el token de Telegram vive en telegram.json (fuera del repo git)
    tg_path = os.path.join(BASE, "telegram.json")
    if os.path.exists(tg_path):
        with open(tg_path, encoding="utf-8") as f:
            cfg["telegram"] = json.load(f)
    return cfg


def detect_arrow(img):
    """Flecha verde de tutorial: color lima saturado + geometria.
    Devuelve el punto donde tocar (delante de la punta), o None.
    Distingue si apunta hacia abajo o hacia arriba por el reparto de masa."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (28, 250, 185), (56, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cnts:
        a = cv2.contourArea(c)
        if not (2300 < a < 3600):
            continue
        x, y, w, h = cv2.boundingRect(c)
        if not (65 <= w <= 110 and 45 <= h <= 85):
            continue
        if not (0.9 < w / h < 1.9):
            continue
        if best is None or a > best[0]:
            best = (a, x, y, w, h)
    if best is None:
        return None
    _, x, y, w, h = best
    roi = mask[y:y + h, x:x + w]
    top = int(np.count_nonzero(roi[: h // 2]))
    bottom = int(np.count_nonzero(roi[h // 2:]))
    cx = x + w // 2
    if top >= bottom:
        return (cx, y + h + 45)      # apunta hacia abajo: tocar debajo
    return (cx, max(20, y - 55))     # apunta hacia arriba: tocar encima de la punta


def notify_telegram(cfg, texto, imagen=None):
    """Avisa por Telegram cuando el bot se detiene; adjunta la captura si la hay."""
    tg = cfg.get("telegram")
    if not tg or not tg.get("token"):
        return
    try:
        if imagen and os.path.exists(imagen):
            with open(imagen, "rb") as f:
                requests.post(
                    f"https://api.telegram.org/bot{tg['token']}/sendPhoto",
                    data={"chat_id": tg["chat_id"], "caption": texto},
                    files={"photo": f}, timeout=15)
        else:
            requests.post(
                f"https://api.telegram.org/bot{tg['token']}/sendMessage",
                data={"chat_id": tg["chat_id"], "text": texto}, timeout=15)
        print("Aviso de Telegram enviado.")
    except Exception as e:
        print(f"No se pudo enviar el aviso de Telegram: {e}")


def adb(cfg, *args, binary=False):
    r = subprocess.run([cfg["adb"], "-s", cfg["device"], *args],
                       capture_output=True, timeout=30)
    return r.stdout if binary else r.stdout.decode(errors="replace")


def capture(cfg):
    raw = adb(cfg, "exec-out", "screencap", "-p", binary=True)
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    return img


def tap(cfg, x, y):
    adb(cfg, "shell", "input", "tap", str(x), str(y))


def match_screen(cfg, img):
    """Devuelve la primera pantalla cuya plantilla aparece en la captura,
    junto con el centro (x, y) de la coincidencia en coordenadas de pantalla."""
    for s in cfg["screens"]:
        if s.get("detector") == "flecha_verde":
            pt = detect_arrow(img)
            if pt:
                return s, 1.0, pt
            continue
        region = s.get("region")
        if region:
            x1, y1, x2, y2 = region
            haystack = img[y1:y2, x1:x2]
        else:
            x1, y1 = 0, 0
            haystack = img
        res = cv2.matchTemplate(haystack, s["_tpl"], cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        if score >= s.get("threshold", 0.87):
            px = s.get("pixel")
            if px and not pixel_matches(img, px["x"], px["y"], px["bgr"], px.get("tol", 40)):
                continue
            th, tw = s["_tpl"].shape[:2]
            center = (x1 + loc[0] + tw // 2, y1 + loc[1] + th // 2)
            return s, float(score), center
    return None, 0.0, None


def pixel_matches(img, x, y, bgr, tol):
    px = img[y, x].astype(int)
    return all(abs(int(px[i]) - bgr[i]) <= tol for i in range(3))


def run_actions(cfg, screen, img, center):
    for a in screen["actions"]:
        t = a["type"]
        if t == "tap":
            log(f"    tap ({a['x']},{a['y']}) {a.get('comment','')}", C_DIM)
            tap(cfg, a["x"], a["y"])
        elif t == "tap_match":
            x = center[0] + a.get("dx", 0)
            y = center[1] + a.get("dy", 0)
            log(f"    tap_match ({x},{y}) {a.get('comment','')}", C_DIM)
            tap(cfg, x, y)
        elif t == "sleep":
            time.sleep(a["s"])
        elif t == "tap_if_pixel":
            fresh = capture(cfg)
            if fresh is not None and pixel_matches(fresh, a["x"], a["y"], a["bgr"], a.get("tol", 40)):
                log(f"    tap_if_pixel ({a['x']},{a['y']}) -> pixel coincide, tap. {a.get('comment','')}", C_DIM)
                tap(cfg, a["x"], a["y"])
            else:
                log(f"    tap_if_pixel ({a['x']},{a['y']}) -> no coincide, se omite. {a.get('comment','')}", C_DIM)
        elif t == "swipe":
            adb(cfg, "shell", "input", "swipe", str(a["x1"]), str(a["y1"]),
                str(a["x2"]), str(a["y2"]), str(a.get("ms", 300)))
        elif t == "back":
            adb(cfg, "shell", "input", "keyevent", "4")
        else:
            print(f"    accion desconocida: {t}")


def main():
    cfg = load_config()
    os.makedirs(UNKNOWN_DIR, exist_ok=True)
    subprocess.run([cfg["adb"], "connect", cfg["device"]], capture_output=True, timeout=30)
    log("=" * 52, C_CYAN)
    log(f"  Bot iniciado - {len(cfg['screens'])} pantallas conocidas", C_CYAN)
    log("  Ctrl+C para parar", C_CYAN)
    log("=" * 52, C_CYAN)
    inicio = time.time()
    misiones = 0

    last_run = {}          # nombre de pantalla -> timestamp de la ultima ejecucion
    unknown_since = None   # primer instante en que dejamos de reconocer la pantalla
    last_screen_name = None
    same_count = 0         # ejecuciones consecutivas de la misma pantalla (anti-atasco)
    recoveries = 0         # rescates via menu seguidos, para no ciclar

    while True:
        img = capture(cfg)
        if img is None:
            print("Captura fallida, reintentando...")
            time.sleep(2)
            continue

        screen, score, center = match_screen(cfg, img)
        now = time.time()

        if screen:
            unknown_since = None
            recoveries = 0
            cooldown = screen.get("cooldown", 5)
            if now - last_run.get(screen["name"], 0) >= cooldown:
                if screen["name"] == last_screen_name:
                    same_count += 1
                else:
                    same_count = 1
                    last_screen_name = screen["name"]
                if screen["actions"] and same_count > screen.get("max_repeats", cfg.get("max_repeats", 6)):
                    path = os.path.join(UNKNOWN_DIR, f"{datetime.now():%Y%m%d_%H%M%S}_atasco.png")
                    cv2.imwrite(path, img)
                    log(f"[{datetime.now():%H:%M:%S}] ATASCO: '{screen['name']}' ejecutada "
                        f"{same_count} veces seguidas sin avanzar. Captura:\n  {path}", C_RED)
                    notify_telegram(cfg, f"Bot DB Legends parado: atasco en la pantalla "
                                         f"'{screen['name']}' (se repitio {same_count} veces).", path)
                    return
                color = C_MAGENTA if screen["name"].startswith("combate") else C_CYAN
                log(f"[{datetime.now():%H:%M:%S}] pantalla: {screen['name']} (score {score:.2f})", color)
                last_run[screen["name"]] = now
                run_actions(cfg, screen, img, center)
                if screen["name"] == "recap_desafios":
                    misiones += 1
                    mins = (now - inicio) / 60
                    ritmo = f"  ({mins:.0f} min, {misiones / mins * 60:.1f}/hora)" if mins >= 1 else ""
                    log(f"  >> MISION COMPLETADA #{misiones}{ritmo}", C_GREEN)
        else:
            if unknown_since is None:
                unknown_since = now
            elif now - unknown_since >= cfg.get("unknown_timeout", 10):
                # rescate: si se ve el boton de menu (≡), abrirlo; la regla del
                # menu llevara de vuelta a Historia
                if recoveries < cfg.get("recovery_max", 3):
                    res = cv2.matchTemplate(img, cfg["_recovery_tpl"], cv2.TM_CCOEFF_NORMED)
                    _, s_, _, l_ = cv2.minMaxLoc(res)
                    if s_ >= 0.9:
                        th, tw = cfg["_recovery_tpl"].shape[:2]
                        recoveries += 1
                        log(f"[{datetime.now():%H:%M:%S}] RESCATE {recoveries}: pantalla "
                            f"perdida, abro el menu para volver a Historia", C_YELLOW)
                        tap(cfg, l_[0] + tw // 2, l_[1] + th // 2)
                        unknown_since = None
                        time.sleep(1.5)
                        continue
                path = os.path.join(UNKNOWN_DIR, f"{datetime.now():%Y%m%d_%H%M%S}.png")
                cv2.imwrite(path, img)
                log(f"[{datetime.now():%H:%M:%S}] PANTALLA DESCONOCIDA durante "
                    f"{cfg.get('unknown_timeout', 10)}s. Captura guardada en:\n  {path}", C_RED)
                log("Definela en config.json y vuelve a lanzar el bot.", C_RED)
                notify_telegram(cfg, "Bot DB Legends parado: pantalla desconocida y sin "
                                     "boton de menu para rescatarse.", path)
                return

        time.sleep(cfg.get("poll_interval", 0.7))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBot detenido por el usuario.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            with open(os.path.join(BASE, "telegram.json"), encoding="utf-8") as f:
                _cfg = {"telegram": json.load(f)}
            notify_telegram(_cfg, f"Bot DB Legends parado: error inesperado: {e}")
        except Exception:
            pass
        raise
