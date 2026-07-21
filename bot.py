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
    for s in cfg["screens"]:
        if s.get("detector") in ("libro_pendiente", "historia_completa"):
            s["_tpl_chevron"] = cv2.imread(os.path.join(BASE, "screens", "historia_chevron.png"))
            s["_tpl_sello"] = cv2.imread(os.path.join(BASE, "screens", "todo_listo.png"))
            s["_tpl_indice"] = cv2.imread(os.path.join(BASE, "screens", "indice_historias.png"))
        if s.get("detector") == "historia_completa":
            s["_tpl_parte1"] = cv2.imread(os.path.join(BASE, "screens", "parte1_diamante.png"))
        if s.get("detector") == "evento_zenkai":
            s["_tpl_tab_on"] = cv2.imread(os.path.join(BASE, "screens", "tab_historia_orig_on.png"))
            s["_tpl_estrella"] = cv2.imread(os.path.join(BASE, "screens", "estrella_evento_gris.png"))
            s["_tpl_limitado"] = cv2.imread(os.path.join(BASE, "screens", "limitado.png"))
        if s.get("detector") == "flecha_naranja":
            s["_tpl_flecha"] = cv2.imread(os.path.join(BASE, "screens", "flecha_naranja.png"))
            s["_tpl_listo"] = cv2.imread(os.path.join(BASE, "screens", "nodo_listo.png"))
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


def detect_libro_pendiente(img, s):
    """En la lista de Historia: busca el libro visible MAS ABAJO (el mas
    antiguo) sin el sello ¡TODO LISTO! y devuelve su chevron para expandirlo."""
    # solo actuar si de verdad estamos en la pantalla de Historia
    if cv2.matchTemplate(img, s["_tpl_indice"], cv2.TM_CCOEFF_NORMED).max() < 0.85:
        return None
    res = cv2.matchTemplate(img, s["_tpl_chevron"], cv2.TM_CCOEFF_NORMED)
    th, tw = s["_tpl_chevron"].shape[:2]
    sh, sw = s["_tpl_sello"].shape[:2]
    ys, xs = np.where(res >= 0.85)
    vistos = []
    # de abajo hacia arriba: se abre primero el libro pendiente mas antiguo
    for y, x in sorted(zip(ys.tolist(), xs.tolist()), reverse=True):
        if x < 750:
            continue
        if any(abs(y - v) < 40 for v in vistos):
            continue
        vistos.append(y)
        # zona donde estaria el sello TODO LISTO de este banner (arriba a la derecha)
        y1, y2 = y - 215, y - 80
        if y1 < 0:
            continue  # banner cortado por arriba: no fiable, que lo resuelva el scroll
        roi = img[y1:y2, 600:900]
        if roi.shape[0] >= sh and roi.shape[1] >= sw:
            if cv2.matchTemplate(roi, s["_tpl_sello"], cv2.TM_CCOEFF_NORMED).max() >= 0.8:
                continue  # este libro ya esta TODO LISTO
        return (x + tw // 2, y + th // 2)
    return None


def detect_historia_completa(img, s):
    """Lista de Historia con TODOS los libros visibles sellados y la parte 1
    a la vista (fondo alcanzado): devuelve el punto del boton EVENTO."""
    if cv2.matchTemplate(img, s["_tpl_indice"], cv2.TM_CCOEFF_NORMED).max() < 0.85:
        return None
    if cv2.matchTemplate(img, s["_tpl_parte1"], cv2.TM_CCOEFF_NORMED).max() < 0.85:
        return None
    res = cv2.matchTemplate(img, s["_tpl_chevron"], cv2.TM_CCOEFF_NORMED)
    sh, sw = s["_tpl_sello"].shape[:2]
    ys, xs = np.where(res >= 0.85)
    vistos = []
    encontrados = 0
    for y, x in sorted(zip(ys.tolist(), xs.tolist())):
        if x < 750 or any(abs(y - v) < 40 for v in vistos):
            continue
        vistos.append(y)
        y1, y2 = y - 215, y - 80
        if y1 < 0:
            continue
        roi = img[y1:y2, 600:900]
        if roi.shape[0] < sh or roi.shape[1] < sw:
            continue
        encontrados += 1
        if cv2.matchTemplate(roi, s["_tpl_sello"], cv2.TM_CCOEFF_NORMED).max() < 0.8:
            return None  # hay un libro pendiente: aun no hemos terminado
    if encontrados == 0:
        return None
    return (277, 1450)  # todo sellado y estamos al fondo: boton EVENTO


def detect_flecha_naranja(img, s):
    """Flecha naranja del mapa de evento. Solo devuelve el punto de la etapa
    si el nodo bajo la flecha NO tiene ya el cartel ¡LISTO!"""
    res = cv2.matchTemplate(img, s["_tpl_flecha"], cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    if score < 0.85:
        return None
    x, y = loc
    th, tw = s["_tpl_flecha"].shape[:2]
    lh, lw = s["_tpl_listo"].shape[:2]
    y1, y2 = y + 190, min(img.shape[0], y + 320)
    x1, x2 = max(0, x - 80), min(img.shape[1], x + tw + 160)
    roi = img[y1:y2, x1:x2]
    if roi.shape[0] >= lh and roi.shape[1] >= lw:
        if cv2.matchTemplate(roi, s["_tpl_listo"], cv2.TM_CCOEFF_NORMED).max() >= 0.8:
            return None  # etapa ya completada: no reentrar
    return (x + tw // 2, y + th // 2 + 100)


def detect_evento_zenkai(img, s):
    """En Eventos > Historia original: busca la primera fila con estrella gris
    que NO sea un evento Limitado y devuelve el punto para abrirla."""
    if cv2.matchTemplate(img, s["_tpl_tab_on"], cv2.TM_CCOEFF_NORMED).max() < 0.85:
        return None
    res = cv2.matchTemplate(img, s["_tpl_estrella"], cv2.TM_CCOEFF_NORMED)
    lh, lw = s["_tpl_limitado"].shape[:2]
    ys, xs = np.where(res >= 0.85)
    vistos = []
    for y, x in sorted(zip(ys.tolist(), xs.tolist())):
        if x < 780 or any(abs(y - v) < 40 for v in vistos):
            continue
        vistos.append(y)
        roi = img[y + 20:y + 130, 600:880]
        if roi.shape[0] >= lh and roi.shape[1] >= lw:
            if cv2.matchTemplate(roi, s["_tpl_limitado"], cv2.TM_CCOEFF_NORMED).max() >= 0.8:
                continue  # evento Limitado: no nos interesa
        # eventos ya visitados esta sesion sin nada que hacer hoy: saltarlos
        patch = img[max(0, y - 120):y + 40, 60:560]
        zona = img[max(0, y - 160):y + 80, 40:600]
        agotado = False
        for p in s.get("_agotados", []):
            if zona.shape[0] >= p.shape[0] and zona.shape[1] >= p.shape[1]:
                if cv2.matchTemplate(zona, p, cv2.TM_CCOEFF_NORMED).max() >= 0.9:
                    agotado = True
                    break
        if agotado:
            continue
        s["_ultimo_patch"] = patch.copy()
        return (450, y + 22)
    return None


def match_screen(cfg, img):
    """Devuelve la primera pantalla cuya plantilla aparece en la captura,
    junto con el centro (x, y) de la coincidencia en coordenadas de pantalla."""
    for s in cfg["screens"]:
        if s.get("detector") == "flecha_verde":
            pt = detect_arrow(img)
            if pt:
                return s, 1.0, pt
            continue
        if s.get("detector") == "libro_pendiente":
            pt = detect_libro_pendiente(img, s)
            if pt:
                return s, 1.0, pt
            continue
        if s.get("detector") == "historia_completa":
            pt = detect_historia_completa(img, s)
            if pt:
                return s, 1.0, pt
            continue
        if s.get("detector") == "evento_zenkai":
            pt = detect_evento_zenkai(img, s)
            if pt:
                return s, 1.0, pt
            continue
        if s.get("detector") == "flecha_naranja":
            pt = detect_flecha_naranja(img, s)
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
            log(f"    swipe ({a['x1']},{a['y1']}) -> ({a['x2']},{a['y2']}) {a.get('comment','')}", C_DIM)
            adb(cfg, "shell", "input", "swipe", str(a["x1"]), str(a["y1"]),
                str(a["x2"]), str(a["y2"]), str(a.get("ms", 300)))
        elif t == "scroll_hasta_fondo":
            # baja la lista hasta que deje de moverse (fondo del todo)
            for i in range(a.get("max", 12)):
                antes = capture(cfg)
                adb(cfg, "shell", "input", "swipe", "450", "1000", "450", "350", "400")
                time.sleep(a.get("espera", 1.0))
                despues = capture(cfg)
                if antes is None or despues is None:
                    break
                diff = float(np.mean(cv2.absdiff(antes[200:1150, 40:870],
                                                 despues[200:1150, 40:870])))
                log(f"    scroll al fondo (pasada {i + 1}, movimiento {diff:.1f})", C_DIM)
                if diff < 1.5:
                    log("    fondo de la lista alcanzado", C_DIM)
                    break
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
    pingpong = 0           # ciclos abrir libro -> colapsarlo (nada que hacer dentro)
    pingpong_scroll = False  # ya bajamos al fondo una vez por este bucle
    mapa_bucle = 0         # ciclos evento_mapa_flecha -> preparacion_ya_completada

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
                # anti ping-pong: abrir un libro y que la unica accion posible sea
                # volver a colapsarlo significa que dentro no hay nada pendiente
                if screen["name"] == "historia_libro_completo" and last_screen_name == "historia_abrir_pendiente":
                    pingpong += 1
                elif screen["name"] not in ("historia_libro_completo", "historia_abrir_pendiente"):
                    pingpong = 0
                    pingpong_scroll = False
                # anti-bucle: flecha -> ya completada -> flecha (evento sin mas etapas)
                if screen["name"] == "preparacion_ya_completada" and last_screen_name == "evento_mapa_flecha":
                    mapa_bucle += 1
                elif screen["name"] not in ("evento_mapa_flecha", "preparacion_ya_completada"):
                    mapa_bucle = 0
                if screen["name"] == "evento_mapa_flecha" and mapa_bucle >= 3:
                    log(f"[{datetime.now():%H:%M:%S}] BUCLE flecha/completado {mapa_bucle}v: "
                        f"marco evento como agotado y salgo del mapa", C_YELLOW)
                    ez = next((s for s in cfg["screens"] if s.get("detector") == "evento_zenkai"), None)
                    if ez is not None and ez.get("_ultimo_patch") is not None:
                        ez.setdefault("_agotados", []).append(ez["_ultimo_patch"])
                        ez["_ultimo_patch"] = None
                    run_actions(cfg, {"actions": [{"type": "tap", "x": 60, "y": 1543, "comment": "volver del mapa"},
                                                  {"type": "sleep", "s": 1.5}]}, img, center)
                    mapa_bucle = 0
                    time.sleep(1.0)
                    continue
                if screen["name"] == "historia_abrir_pendiente" and pingpong >= 3:
                    if not pingpong_scroll:
                        log(f"[{datetime.now():%H:%M:%S}] BUCLE abrir/colapsar libro: dentro no hay "
                            f"nada pendiente, bajo al fondo a mirar los libros de abajo", C_YELLOW)
                        run_actions(cfg, {"actions": [{"type": "scroll_hasta_fondo"}]}, img, center)
                        pingpong = 0
                        pingpong_scroll = True
                        last_run[screen["name"]] = now
                        last_screen_name = screen["name"]
                        same_count = 1
                        time.sleep(1.0)
                        continue
                    path = os.path.join(UNKNOWN_DIR, f"{datetime.now():%Y%m%d_%H%M%S}_atasco.png")
                    cv2.imwrite(path, img)
                    log(f"[{datetime.now():%H:%M:%S}] ATASCO: el bucle abrir/colapsar libro persiste "
                        f"tras bajar al fondo. Captura:\n  {path}", C_RED)
                    notify_telegram(cfg, "Bot DB Legends parado: bucle abrir/colapsar un libro de "
                                         "historia sin nada pendiente dentro.", path)
                    return
                if screen["name"] == last_screen_name:
                    same_count += 1
                else:
                    same_count = 1
                    last_screen_name = screen["name"]
                if screen["actions"] and same_count > screen.get("max_repeats", cfg.get("max_repeats", 6)):
                    rescue = screen.get("rescue_actions")
                    if rescue:
                        log(f"[{datetime.now():%H:%M:%S}] '{screen['name']}' repetida "
                            f"{same_count} veces -> intento rescate", C_YELLOW)
                        same_count = 0
                        run_actions(cfg, {"actions": rescue}, img, center)
                        time.sleep(1.0)
                        continue
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
                if screen.get("stop"):
                    msg = screen.get("stop_msg", f"Bot parado por la pantalla '{screen['name']}'.")
                    log(f"[{datetime.now():%H:%M:%S}] FIN: {msg}", C_GREEN)
                    notify_telegram(cfg, msg)
                    return
                if screen["name"] in ("evento_mapa_volver", "jefe_raid_ok"):
                    # el mapa no tenia nada que hacer: recordar el evento y no reentrar hoy
                    ez = next((s for s in cfg["screens"] if s.get("detector") == "evento_zenkai"), None)
                    if ez is not None and ez.get("_ultimo_patch") is not None:
                        ez.setdefault("_agotados", []).append(ez["_ultimo_patch"])
                        ez["_ultimo_patch"] = None
                        log(f"  >> evento sin nada pendiente hoy: lo salto el resto de la sesion "
                            f"({len(ez['_agotados'])} saltados)", C_YELLOW)
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
