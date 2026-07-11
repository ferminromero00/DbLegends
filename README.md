# Bot de farmeo — Dragon Ball Legends

Bot que farmea el modo historia de Dragon Ball Legends automáticamente en BlueStacks,
usando ADB para leer la pantalla y tocar, y OpenCV para reconocer en qué pantalla está.

## Cómo funciona

Un bucle simple: **capturar pantalla → reconocer → actuar → repetir**.

- Cada pantalla conocida se define en [`config.json`](config.json): una plantilla de imagen
  (recorte en `screens/`) o un detector especial, más la lista de acciones (toques, esperas).
- El orden de las pantallas en `config.json` es la **prioridad**: el combate va primero
  para que ninguna otra regla toque nada en mitad de una pelea.
- Tipos de acción: `tap` (coordenadas fijas), `tap_match` (relativo a donde se encontró la
  plantilla), `tap_if_pixel` (solo si un píxel tiene cierto color, p. ej. un checkbox marcado),
  `sleep`, `swipe`, `back`.
- Detector especial `flecha_verde`: encuentra la flecha de los tutoriales por color
  (verde lima puro) y geometría, distingue si apunta arriba o abajo, y toca el botón señalado.

## Robustez

- **Anti-atasco**: si una pantalla se ejecuta muchas veces seguidas sin avanzar, para.
- **Rescate**: si lleva 60s sin reconocer nada, abre el menú del juego (≡) y vuelve a Historia.
- **Aviso por Telegram**: si se detiene por cualquier fallo, envía mensaje con la captura.
- **Modo aprendizaje**: ante una pantalla desconocida guarda la captura en
  `capturas/desconocidas/` para definirla y relanzar.

## Requisitos

1. **BlueStacks** con ADB activado (Ajustes → Avanzado → Android Debug Bridge).
2. **Python 3.12 embebido** en `tools/python/` con `opencv-python-headless`, `numpy` y
   `requests` (no va en el repo; descargar el [embeddable package](https://www.python.org/downloads/windows/),
   instalar pip con get-pip.py y `pip install opencv-python-headless numpy requests`).
3. (Opcional) `telegram.json` en la raíz para los avisos:
   ```json
   { "token": "TOKEN_DE_BOTFATHER", "chat_id": "TU_CHAT_ID" }
   ```

## Uso

Doble click en **`IniciarBot.bat`** (o `tools\python\python.exe -u bot.py`).
La consola muestra cada pantalla reconocida, cada toque, y un contador de misiones
completadas con el ritmo por hora. `Ctrl+C` para parar.
