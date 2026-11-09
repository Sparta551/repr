"""
magic_chess_gui.py
==================

Interfaz gráfica (pygame) para visualizar y jugar el ambiente *Magic Chess*.

La GUI no modifica las reglas: solo llama a los métodos públicos de
``MagicChessEnv`` (``step``, ``acciones_legales``, ``snapshot``...) y dibuja
los snapshots guardados en un ``Historial``. Así se puede retroceder y
avanzar sin tocar el ambiente real.

Modos
-----
* **Automático**: los dos jugadores son agentes.
* **Manual**: el jugador 0 (abajo) lo controla el usuario con el mouse y el
  jugador 1 (arriba) es un agente.

Controles
---------
Mouse (modo manual, jugador de abajo):
    clic en una carta de la tienda ..... comprar
    clic en una unidad de la banca ..... desplegar en el tablero
    clic en una unidad del tablero ..... devolver a la banca
    clic derecho en una unidad ......... vender
    botones Renovar / Combinar / Terminar turno

Teclado:
    Espacio  reproducir / pausar        Flechas  retroceder / avanzar un paso
    Inicio / Fin  ir al primer / último estado
    + / -    velocidad                  M  cambiar modo     D  dificultad
    N        nueva partida              Esc  salir

Para conectar sus propios algoritmos, agréguenlos al diccionario ``AGENTES``.
Un agente solo necesita el método ``actuar(env, jugador) -> Accion``.

Ejecutar:  python magic_chess_gui.py
"""

from __future__ import annotations

import math
import sys
from typing import Callable, Dict, List, Optional, Tuple

import pygame

from magic_chess_env import (
    CATALOGO,
    EMPATE,
    Accion,
    ConfiguracionJuego,
    MagicChessEnv,
    TipoAccion,
)
from agentes import FABRICAS
from utils import (
    COLOR_ELEMENTO,
    COLORES,
    AgenteAleatorio,
    Elemento,
    Historial,
    aclarar,
    formatear_poder,
    oscurecer,
)

# ---------------------------------------------------------------------------
# Configuración visual
# ---------------------------------------------------------------------------
ANCHO, ALTO = 1280, 800
FPS = 60
DIFICULTADES = ["facil", "normal", "dificil"]
VELOCIDADES = [1.0, 2.0, 4.0, 8.0, 16.0]   # acciones por segundo
DURACION_COMBATE = 1.6                     # segundos de la animación de combate

#: Agentes disponibles (botones "Agente J0/J1" los recorren en este orden).
#: Los algoritmos están en agentes.py (FABRICAS).
AGENTES: Dict[str, Callable[[], object]] = dict(FABRICAS)
AGENTES["Aleatorio"] = lambda: AgenteAleatorio()

# Geometría de los paneles
PANEL_X, PANEL_W, PANEL_H = 20, 860, 285
PANEL_Y = {1: 58, 0: 355}          # jugador 1 arriba (rival), jugador 0 abajo
CARTA_TAB = (96, 100)
CARTA_BANCA = (80, 84)
SEP = 8
TIENDA_X = PANEL_X + 660
SIDEBAR_X, SIDEBAR_W = 900, 360


# ---------------------------------------------------------------------------
# Widgets sencillos
# ---------------------------------------------------------------------------
class Boton:
    """Botón rectangular con texto y callback."""

    def __init__(self, rect: Tuple[int, int, int, int], texto: str,
                 accion: Callable[[], None]) -> None:
        self.rect = pygame.Rect(rect)
        self.texto = texto
        self.accion = accion
        self.activo = True

    def dibujar(self, sup: pygame.Surface, fuente: pygame.font.Font,
                mouse: Tuple[int, int]) -> None:
        if not self.activo:
            color = COLORES["boton_inactivo"]
        elif self.rect.collidepoint(mouse):
            color = COLORES["boton_hover"]
        else:
            color = COLORES["boton"]
        pygame.draw.rect(sup, color, self.rect, border_radius=6)
        txt = fuente.render(self.texto, True,
                            COLORES["texto"] if self.activo else COLORES["texto_suave"])
        sup.blit(txt, txt.get_rect(center=self.rect.center))

    def clic(self, pos: Tuple[int, int]) -> bool:
        if self.activo and self.rect.collidepoint(pos):
            self.accion()
            return True
        return False


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def dibujar_estrella(sup, centro, radio, color) -> None:
    cx, cy = centro
    puntos = []
    for i in range(10):
        r = radio if i % 2 == 0 else radio * 0.45
        ang = -math.pi / 2 + i * math.pi / 5
        puntos.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
    pygame.draw.polygon(sup, color, puntos)


def dibujar_corazon(sup, centro, tam, color) -> None:
    cx, cy = centro
    r = tam // 4
    pygame.draw.circle(sup, color, (cx - r, cy - r // 2), r)
    pygame.draw.circle(sup, color, (cx + r, cy - r // 2), r)
    pygame.draw.polygon(sup, color, [(cx - 2 * r, cy - r // 3),
                                     (cx + 2 * r, cy - r // 3),
                                     (cx, cy + int(1.6 * r))])


def dibujar_moneda(sup, centro, radio) -> None:
    pygame.draw.circle(sup, COLORES["oro"], centro, radio)
    pygame.draw.circle(sup, oscurecer(COLORES["oro"], 0.35), centro, radio, 2)


def dibujar_icono_elemento(sup, elemento: Elemento, centro, tam) -> None:
    """Gota (agua), llama (fuego) u hoja (planta)."""
    cx, cy = centro
    color = aclarar(COLOR_ELEMENTO[elemento], 0.55)
    if elemento == Elemento.AGUA:
        pygame.draw.circle(sup, color, (cx, cy + tam // 5), tam // 3)
        pygame.draw.polygon(sup, color, [(cx, cy - tam // 2),
                                         (cx - tam // 3, cy + tam // 6),
                                         (cx + tam // 3, cy + tam // 6)])
    elif elemento == Elemento.FUEGO:
        pygame.draw.polygon(sup, color, [(cx, cy - tam // 2), (cx + tam // 3, cy),
                                         (cx + tam // 5, cy + tam // 2),
                                         (cx - tam // 5, cy + tam // 2),
                                         (cx - tam // 3, cy)])
    else:
        pygame.draw.ellipse(sup, color, (cx - tam // 4, cy - tam // 2,
                                         tam // 2, tam))
        pygame.draw.line(sup, oscurecer(color, 0.4), (cx, cy - tam // 2 + 3),
                         (cx, cy + tam // 2 - 3), 2)


# ---------------------------------------------------------------------------
# GUI principal
# ---------------------------------------------------------------------------
class MagicChessGUI:
    """Ventana principal: dibuja el snapshot actual del historial y gestiona
    la interacción, la reproducción y el modo de juego."""

    def __init__(self, semilla: Optional[int] = 7) -> None:
        pygame.init()
        pygame.display.set_caption("Magic Chess - Representación y solución automática")
        self.pantalla = pygame.display.set_mode((ANCHO, ALTO))
        self.reloj = pygame.time.Clock()
        self.f_titulo = pygame.font.SysFont("arial", 22, bold=True)
        self.f_normal = pygame.font.SysFont("arial", 16)
        self.f_negrita = pygame.font.SysFont("arial", 16, bold=True)
        self.f_peque = pygame.font.SysFont("arial", 13)
        self.f_grande = pygame.font.SysFont("arial", 34, bold=True)

        self.semilla = semilla
        self.dificultad = "normal"
        self.modo = "auto"                       # "auto" | "manual"
        self.nombres_agentes = ["Reglas", "A*h2"]
        self.reproduciendo = False
        self.indice_velocidad = 1
        self.acumulador = 0.0
        self.turno_rr = 0                        # alterna qué jugador actúa
        self.mensaje = ""
        self.combate_t: Optional[float] = None   # tiempo de animación de combate
        self.combate_mostrado: Optional[int] = None
        self.zonas_click: List[Tuple[pygame.Rect, str, int, int]] = []

        self.historial = Historial()
        self._crear_botones()
        self.nueva_partida()

    # ------------------------------------------------------------------
    # Partida
    # ------------------------------------------------------------------
    def nueva_partida(self) -> None:
        cfg = ConfiguracionJuego.desde_dificultad(self.dificultad)
        self.env = MagicChessEnv(cfg, semilla=self.semilla)
        self.agentes = [AGENTES[n]() for n in self.nombres_agentes]
        self.historial.limpiar()
        self.historial.agregar(self.env.snapshot())
        self.combate_t = None
        self.combate_mostrado = None
        self.reproduciendo = False
        self.mensaje = f"Nueva partida ({self.dificultad}, semilla {self.semilla})."
        if self.semilla is not None:
            self.semilla += 1

    def _humano_juega(self) -> bool:
        return self.modo == "manual"

    def _ejecutar(self, jugador: int, accion: Accion) -> None:
        """Aplica una acción al ambiente real y guarda el snapshot."""
        if not self.historial.en_vivo():
            self.historial.ir_final()
        try:
            _, _, _, info = self.env.step(jugador, accion)
        except ValueError as e:
            self.mensaje = str(e)
            return
        self.historial.agregar(self.env.snapshot())
        if "combate" in info:
            self.combate_t = 0.0

    def paso_agente(self) -> bool:
        """Un paso de simulación: actúa un agente. Devuelve False si no hay
        nada que hacer (p. ej. esperando al humano o partida terminada)."""
        if self.env.es_terminal():
            return False
        pendientes = self.env.jugadores_pendientes()
        if self._humano_juega():
            pendientes = [j for j in pendientes if j != 0]
        if not pendientes:
            return False
        jugador = pendientes[self.turno_rr % len(pendientes)]
        self.turno_rr += 1
        accion = self.agentes[jugador].actuar(self.env, jugador)
        self._ejecutar(jugador, accion)
        return True

    def avanzar(self) -> None:
        """Botón '>': si se está revisando el pasado, avanza el cursor; si
        se está en vivo, ejecuta un paso real."""
        if not self.historial.en_vivo():
            self.historial.adelante()
        elif not self.paso_agente():
            self.mensaje = ("Es tu turno (jugador de abajo)."
                            if self._humano_juega() and not self.env.es_terminal()
                            else "No hay más pasos.")

    # ------------------------------------------------------------------
    # Acciones del humano (jugador 0)
    # ------------------------------------------------------------------
    def _accion_humana(self, accion: Accion) -> None:
        if not self._humano_juega() or not self.historial.en_vivo():
            self.mensaje = "Cambia a modo manual y ve al estado actual para jugar."
            return
        if self.env.es_legal(0, accion):
            self._ejecutar(0, accion)
            if accion.es_terminar():
                self.reproduciendo = True   # deja que el agente rival juegue
        else:
            self.mensaje = f"Acción no válida: {accion}"

    def _renovar(self) -> None:
        self._accion_humana(Accion(TipoAccion.RENOVAR))

    def _combinar(self) -> None:
        for a in self.env.acciones_legales(0):
            if a.tipo == TipoAccion.COMBINAR:
                self._accion_humana(a)
                return
        self.mensaje = "No tienes 3 copias iguales para combinar."

    def _terminar(self) -> None:
        self._accion_humana(Accion(TipoAccion.TERMINAR))

    # ------------------------------------------------------------------
    # Botones
    # ------------------------------------------------------------------
    def _crear_botones(self) -> None:
        y1, y2 = 668, 722
        def h():
            self.historial.ir_inicio()
            self.reproduciendo = False
        def a():
            self.historial.atras()
            self.reproduciendo = False
        def f():
            self.historial.ir_final()
        self.btn_play = Boton((150, y1, 90, 36), "Play", self._alternar_play)
        self.botones = [
            Boton((20, y1, 50, 36), "|<", h),
            Boton((78, y1, 64, 36), "< Paso", a),
            self.btn_play,
            Boton((248, y1, 64, 36), "Paso >", self.avanzar),
            Boton((320, y1, 50, 36), ">|", f),
            Boton((390, y1, 36, 36), "-", lambda: self._velocidad(-1)),
            Boton((480, y1, 36, 36), "+", lambda: self._velocidad(1)),
            Boton((20, y2, 150, 36), "Modo", self._cambiar_modo),
            Boton((178, y2, 150, 36), "Dificultad", self._cambiar_dificultad),
            Boton((336, y2, 150, 36), "Nueva partida", self.nueva_partida),
            Boton((494, y2, 150, 36), "Agente J1", lambda: self._cambiar_agente(1)),
            Boton((652, y2, 150, 36), "Agente J0", lambda: self._cambiar_agente(0)),
        ]
        self.btn_renovar = Boton((TIENDA_X, PANEL_Y[0] + 250, 190, 28),
                                 "Renovar", self._renovar)
        self.btn_combinar = Boton((560, y1, 130, 36), "Combinar", self._combinar)
        self.btn_terminar = Boton((698, y1, 150, 36), "Terminar turno", self._terminar)
        self.botones += [self.btn_renovar, self.btn_combinar, self.btn_terminar]

    def _alternar_play(self) -> None:
        self.reproduciendo = not self.reproduciendo
        if self.reproduciendo:
            self.historial.ir_final()

    def _velocidad(self, delta: int) -> None:
        self.indice_velocidad = max(0, min(len(VELOCIDADES) - 1,
                                           self.indice_velocidad + delta))

    def _cambiar_modo(self) -> None:
        self.modo = "manual" if self.modo == "auto" else "auto"
        self.mensaje = ("Modo manual: controlas al jugador de abajo (J0)."
                        if self.modo == "manual" else "Modo automático: agente vs agente.")

    def _cambiar_dificultad(self) -> None:
        i = DIFICULTADES.index(self.dificultad)
        self.dificultad = DIFICULTADES[(i + 1) % len(DIFICULTADES)]
        self.nueva_partida()

    def _cambiar_agente(self, jugador: int) -> None:
        nombres = list(AGENTES)
        i = nombres.index(self.nombres_agentes[jugador])
        self.nombres_agentes[jugador] = nombres[(i + 1) % len(nombres)]
        self.agentes[jugador] = AGENTES[self.nombres_agentes[jugador]]()
        self.mensaje = f"J{jugador} ahora usa: {self.nombres_agentes[jugador]}"

    # ------------------------------------------------------------------
    # Eventos
    # ------------------------------------------------------------------
    def _manejar_eventos(self) -> bool:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    return False
                teclas = {
                    pygame.K_SPACE: self._alternar_play,
                    pygame.K_RIGHT: self.avanzar,
                    pygame.K_LEFT: self.historial.atras,
                    pygame.K_HOME: self.historial.ir_inicio,
                    pygame.K_END: self.historial.ir_final,
                    pygame.K_m: self._cambiar_modo,
                    pygame.K_d: self._cambiar_dificultad,
                    pygame.K_n: self.nueva_partida,
                    pygame.K_PLUS: lambda: self._velocidad(1),
                    pygame.K_KP_PLUS: lambda: self._velocidad(1),
                    pygame.K_MINUS: lambda: self._velocidad(-1),
                    pygame.K_KP_MINUS: lambda: self._velocidad(-1),
                }
                if ev.key in teclas:
                    teclas[ev.key]()
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button in (1, 3):
                self._clic(ev.pos, ev.button)
        return True

    def _clic(self, pos, boton_mouse: int) -> None:
        if boton_mouse == 1:
            for b in self.botones:
                if b.clic(pos):
                    return
        for rect, zona, jugador, idx in self.zonas_click:
            if jugador != 0 or not rect.collidepoint(pos):
                continue
            if boton_mouse == 3 and zona in ("banca", "tablero"):
                self._accion_humana(Accion(TipoAccion.VENDER, (zona, idx)))
            elif zona == "tienda":
                self._accion_humana(Accion(TipoAccion.COMPRAR, idx))
            elif zona == "banca":
                self._accion_humana(Accion(TipoAccion.A_TABLERO, idx))
            elif zona == "tablero":
                self._accion_humana(Accion(TipoAccion.A_BANCA, idx))
            return

    # ------------------------------------------------------------------
    # Actualización
    # ------------------------------------------------------------------
    def _actualizar(self, dt: float) -> None:
        humano = self._humano_juega() and self.historial.en_vivo()
        turno_humano = (humano and not self.env.jugadores[0].listo
                        and not self.env.es_terminal())
        self.btn_renovar.activo = turno_humano
        self.btn_combinar.activo = turno_humano
        self.btn_terminar.activo = turno_humano

        if self.combate_t is not None:
            self.combate_t += dt
            if self.combate_t >= DURACION_COMBATE:
                self.combate_t = None
            return  # pausa la simulación mientras se anima el combate
        if not self.reproduciendo or not self.historial.en_vivo():
            return
        self.acumulador += dt * VELOCIDADES[self.indice_velocidad]
        while self.acumulador >= 1.0:
            self.acumulador -= 1.0
            if not self.paso_agente():
                if self.env.es_terminal() or self._humano_juega():
                    self.reproduciendo = False
                self.acumulador = 0.0
                break
            if self.combate_t is not None:
                break


    # ------------------------------------------------------------------
    # Dibujo
    # ------------------------------------------------------------------
    def _texto(self, txt, pos, fuente=None, color=None, centro=False) -> pygame.Rect:
        fuente = fuente or self.f_normal
        img = fuente.render(str(txt), True, color or COLORES["texto"])
        rect = img.get_rect(center=pos) if centro else img.get_rect(topleft=pos)
        self.pantalla.blit(img, rect)
        return rect

    def _dibujar(self) -> None:
        sup = self.pantalla
        sup.fill(COLORES["fondo"])
        snap = self.historial.actual()
        self.zonas_click = []
        if snap is None:
            return
        self._dibujar_encabezado(snap)
        for j in (1, 0):
            self._dibujar_panel_jugador(snap, j)
        self._dibujar_combate(snap)
        self._dibujar_registro(snap)
        self._dibujar_controles()
        if self.combate_t is not None and snap.get("ultimo_combate"):
            self._dibujar_animacion_combate(snap["ultimo_combate"], self.combate_t)
        if snap["fase"] == "terminado":
            self._dibujar_fin(snap)
        self._dibujar_tooltip(snap)
        pygame.display.flip()

    def _dibujar_encabezado(self, snap) -> None:
        self._texto("MAGIC CHESS", (20, 14), self.f_titulo)
        info = (f"Ronda {snap['ronda']}/{snap['config']['max_rondas']}   |   "
                f"Fase: {snap['fase']}   |   Tablero: {snap['capacidad_tablero']} casillas   |   "
                f"Pasos: {snap['pasos']}   |   Dificultad: {self.dificultad}   |   "
                f"Modo: {'manual' if self.modo == 'manual' else 'automático'}")
        self._texto(info, (200, 19), self.f_normal, COLORES["texto_suave"])

    def _dibujar_panel_jugador(self, snap, j: int) -> None:
        sup = self.pantalla
        y0 = PANEL_Y[j]
        p = snap["jugadores"][j]
        color_j = COLORES[f"jugador{j}"]
        panel = pygame.Rect(PANEL_X, y0, PANEL_W, PANEL_H)
        pygame.draw.rect(sup, COLORES["panel"], panel, border_radius=10)
        pygame.draw.rect(sup, color_j if not p["listo"] else COLORES["panel_borde"],
                         panel, 2, border_radius=10)

        # Encabezado del jugador
        if j == 0 and self.modo == "manual":
            quien = "Humano"
        else:
            quien = self.nombres_agentes[j]
        self._texto(f"J{j} - {quien}", (PANEL_X + 14, y0 + 10), self.f_negrita, color_j)
        x = PANEL_X + 170
        for v in range(snap["config"]["vidas_iniciales"]):
            col = COLORES["vida"] if v < p["vidas"] else COLORES["casilla"]
            dibujar_corazon(sup, (x + v * 22, y0 + 21), 18, col)
        x += snap["config"]["vidas_iniciales"] * 22 + 16
        dibujar_moneda(sup, (x, y0 + 20), 9)
        self._texto(f"{p['oro']}  (+{snap['interes'][j]} interés)", (x + 14, y0 + 11))
        poder = snap["poder"][j]
        self._texto(f"Poder: {formatear_poder(poder['total'])}", (x + 170, y0 + 11),
                    self.f_negrita)
        self._texto(f"V{p['victorias']} D{p['derrotas']} E{p['empates']}",
                    (x + 290, y0 + 11), color=COLORES["texto_suave"])
        if p["listo"]:
            self._texto("LISTO", (PANEL_X + PANEL_W - 70, y0 + 11), self.f_negrita,
                        COLORES["victoria"])

        combinables = {tuple(c) for c in snap["combinables"][j]}

        # Tablero
        self._texto("Tablero", (PANEL_X + 16, y0 + 38), self.f_peque, COLORES["texto_suave"])
        cap = snap["capacidad_tablero"]
        for i in range(6):
            rect = pygame.Rect(PANEL_X + 16 + i * (CARTA_TAB[0] + SEP), y0 + 56, *CARTA_TAB)
            if i < len(p["tablero"]):
                u = p["tablero"][i]
                self._dibujar_unidad(rect, u, (u["nombre"], u["estrellas"]) in combinables)
                self.zonas_click.append((rect, "tablero", j, i))
            else:
                col = COLORES["casilla"] if i < cap else COLORES["casilla_bloqueada"]
                pygame.draw.rect(sup, col, rect, border_radius=8)
                if i >= cap:
                    self._texto("bloq.", rect.center, self.f_peque,
                                COLORES["texto_suave"], centro=True)

        # Banca
        self._texto("Banca", (PANEL_X + 16, y0 + 164), self.f_peque, COLORES["texto_suave"])
        for i in range(snap["config"]["tamano_banca"]):
            rect = pygame.Rect(PANEL_X + 16 + i * (CARTA_BANCA[0] + SEP), y0 + 182,
                               *CARTA_BANCA)
            if i < len(p["banca"]):
                u = p["banca"][i]
                self._dibujar_unidad(rect, u, (u["nombre"], u["estrellas"]) in combinables)
                self.zonas_click.append((rect, "banca", j, i))
            else:
                pygame.draw.rect(sup, COLORES["casilla"], rect, border_radius=8)

        # Sinergias activas
        sx = PANEL_X + 16 + 6 * (CARTA_BANCA[0] + SEP) + 8
        self._texto("Sinergias", (sx, y0 + 164), self.f_peque, COLORES["texto_suave"])
        for k, (elem, n) in enumerate(poder["sinergias_activas"].items()):
            e = Elemento(elem)
            pygame.draw.rect(sup, oscurecer(COLOR_ELEMENTO[e], 0.2),
                             (sx, y0 + 184 + k * 28, 120, 24), border_radius=5)
            dibujar_icono_elemento(sup, e, (sx + 14, y0 + 196 + k * 28), 16)
            self._texto(f"{elem} x{n}", (sx + 28, y0 + 188 + k * 28), self.f_peque)

        # Tienda
        self._texto("Tienda", (TIENDA_X, y0 + 38), self.f_peque, COLORES["texto_suave"])
        for i, nombre in enumerate(p["tienda"]):
            rect = pygame.Rect(TIENDA_X, y0 + 56 + i * 64, 190, 56)
            if nombre is None:
                pygame.draw.rect(sup, COLORES["casilla_bloqueada"], rect, border_radius=8)
                self._texto("vendida", rect.center, self.f_peque,
                            COLORES["texto_suave"], centro=True)
                continue
            plantilla = CATALOGO[nombre]
            alcanza = plantilla.costo <= p["oro"]
            base = COLOR_ELEMENTO[plantilla.elemento]
            pygame.draw.rect(sup, base if alcanza else oscurecer(base, 0.55), rect,
                             border_radius=8)
            dibujar_icono_elemento(sup, plantilla.elemento, (rect.x + 20, rect.centery), 22)
            self._texto(nombre, (rect.x + 40, rect.y + 8), self.f_negrita)
            self._texto(f"poder {plantilla.poder_base:.0f}", (rect.x + 40, rect.y + 30),
                        self.f_peque)
            dibujar_moneda(sup, (rect.right - 34, rect.centery), 8)
            self._texto(str(plantilla.costo), (rect.right - 22, rect.centery - 9),
                        self.f_negrita)
            self.zonas_click.append((rect, "tienda", j, i))
        if j == 1:
            self._texto(f"Renovar: {snap['config']['costo_renovar']} oro",
                        (TIENDA_X, y0 + 254), self.f_peque, COLORES["texto_suave"])
        else:
            self.btn_renovar.texto = f"Renovar ({snap['config']['costo_renovar']} oro)"

    def _dibujar_unidad(self, rect: pygame.Rect, u: dict, combinable: bool) -> None:
        sup = self.pantalla
        e = Elemento(u["elemento"])
        base = COLOR_ELEMENTO[e]
        pygame.draw.rect(sup, oscurecer(base, 0.15), rect, border_radius=8)
        pygame.draw.rect(sup, base, rect.inflate(-6, -6), border_radius=6)
        if combinable:
            pygame.draw.rect(sup, COLORES["oro"], rect, 3, border_radius=8)
        dibujar_icono_elemento(sup, e, (rect.centerx, rect.y + rect.h * 0.36),
                               int(rect.h * 0.32))
        for s in range(u["estrellas"]):
            dibujar_estrella(sup, (rect.x + 14 + s * 15, rect.y + 13), 6, COLORES["oro"])
        self._texto(u["nombre"], (rect.centerx, rect.bottom - 28), self.f_peque,
                    centro=True)
        self._texto(formatear_poder(u["poder"]), (rect.centerx, rect.bottom - 12),
                    self.f_peque, aclarar(base, 0.8), centro=True)

    def _dibujar_combate(self, snap) -> None:
        sup = self.pantalla
        caja = pygame.Rect(SIDEBAR_X, 58, SIDEBAR_W, 300)
        pygame.draw.rect(sup, COLORES["panel"], caja, border_radius=10)
        self._texto("Último combate", (caja.x + 14, caja.y + 10), self.f_negrita)
        comb = snap.get("ultimo_combate")
        if not comb:
            self._texto("Aún no hay combates.", (caja.x + 14, caja.y + 40),
                        color=COLORES["texto_suave"])
            return
        g = comb["ganador_ronda"]
        res = "Empate" if g == EMPATE else f"Gana J{g}"
        self._texto(f"Ronda {comb['ronda']}: {res}", (caja.x + 14, caja.y + 36),
                    color=COLORES["empate"] if g == EMPATE else COLORES[f"jugador{g}"])
        maximo = max(1e-6, *(d["total"] for d in comb["desglose"]))
        for k, j in enumerate((1, 0)):
            d = comb["desglose"][j]
            y = caja.y + 70 + k * 110
            self._texto(f"J{j}", (caja.x + 14, y), self.f_negrita, COLORES[f"jugador{j}"])
            ancho = int((caja.w - 80) * d["total"] / maximo)
            pygame.draw.rect(sup, COLORES["casilla"], (caja.x + 50, y, caja.w - 80, 18),
                             border_radius=4)
            pygame.draw.rect(sup, COLORES[f"jugador{j}"], (caja.x + 50, y, ancho, 18),
                             border_radius=4)
            self._texto(formatear_poder(d["total"]), (caja.right - 26, y + 9),
                        self.f_peque, centro=True)
            lineas = [
                f"base {d['base']:.1f}  +  sinergia {d['sinergia']:.1f}",
                f"x ventaja {d['factor_ventaja']:.3f}   x ruido {d['ruido']:.3f}",
            ]
            for n, linea in enumerate(lineas):
                self._texto(linea, (caja.x + 50, y + 26 + n * 18), self.f_peque,
                            COLORES["texto_suave"])
            for n, u in enumerate(comb["tableros"][j][:8]):
                col = COLOR_ELEMENTO[Elemento(u["elemento"])]
                pygame.draw.rect(sup, col, (caja.x + 50 + n * 30, y + 66, 24, 18),
                                 border_radius=4)
                self._texto(str(u["estrellas"]), (caja.x + 62 + n * 30, y + 75),
                            self.f_peque, centro=True)

    def _dibujar_registro(self, snap) -> None:
        # Estado interno de los agentes (en vivo)
        caja = pygame.Rect(SIDEBAR_X, 368, SIDEBAR_W, 150)
        pygame.draw.rect(self.pantalla, COLORES["panel"], caja, border_radius=10)
        self._texto("Estado interno de los agentes", (caja.x + 14, caja.y + 8),
                    self.f_negrita)
        for col, j in enumerate((1, 0)):
            x = caja.x + 14 + col * 175
            agente = self.agentes[j]
            if j == 0 and self.modo == "manual":
                lineas = ["Humano"]
            elif hasattr(agente, "lineas_estado"):
                lineas = agente.lineas_estado()
            else:
                lineas = [getattr(agente, "nombre", type(agente).__name__)]
            for n, linea in enumerate(lineas[:7]):
                color = COLORES[f"jugador{j}"] if n == 0 else COLORES["texto_suave"]
                self._texto(f"J{j}: {linea}" if n == 0 else linea[:27],
                            (x, caja.y + 32 + n * 16), self.f_peque, color)

        caja = pygame.Rect(SIDEBAR_X, 526, SIDEBAR_W, 114)
        pygame.draw.rect(self.pantalla, COLORES["panel"], caja, border_radius=10)
        self._texto("Registro de eventos", (caja.x + 14, caja.y + 8), self.f_negrita)
        for n, linea in enumerate(snap["registro"][-5:]):
            if len(linea) > 52:
                linea = linea[:50] + "..."
            col = COLORES["texto"] if "Combate" in linea or "Fin" in linea else COLORES["texto_suave"]
            self._texto(linea, (caja.x + 14, caja.y + 30 + n * 16), self.f_peque, col)

    def _dibujar_controles(self) -> None:
        mouse = pygame.mouse.get_pos()
        self.btn_play.texto = "Pausa" if self.reproduciendo else "Play"
        for b in self.botones:
            b.dibujar(self.pantalla, self.f_normal, mouse)
        self._texto(f"{VELOCIDADES[self.indice_velocidad]:g}/s", (452, 677),
                    self.f_normal, centro=True)
        h = self.historial
        estado = "en vivo" if h.en_vivo() else "revisando"
        self._texto(f"Historial {h.cursor + 1}/{len(h)} ({estado})", (870, 672),
                    self.f_normal, COLORES["texto_suave"])
        self._texto(f"J1: {self.nombres_agentes[1]}   J0: "
                    f"{'Humano' if self.modo == 'manual' else self.nombres_agentes[0]}",
                    (870, 696), self.f_normal, COLORES["texto_suave"])
        if self.mensaje:
            self._texto(self.mensaje, (870, 730), self.f_normal, COLORES["oro"])
        self._texto("Espacio: play/pausa  <- ->: paso  M: modo  D: dificultad  N: nueva",
                    (870, 760), self.f_peque, COLORES["texto_suave"])

    def _dibujar_animacion_combate(self, comb, t: float) -> None:
        sup = self.pantalla
        velo = pygame.Surface((PANEL_W, 582), pygame.SRCALPHA)
        velo.fill((0, 0, 0, 150))
        sup.blit(velo, (PANEL_X, 58))
        caja = pygame.Rect(0, 0, 520, 200)
        caja.center = (PANEL_X + PANEL_W // 2, 58 + 291)
        pygame.draw.rect(sup, COLORES["panel"], caja, border_radius=12)
        pygame.draw.rect(sup, COLORES["oro"], caja, 2, border_radius=12)
        self._texto(f"COMBATE - Ronda {comb['ronda']}", (caja.centerx, caja.y + 24),
                    self.f_titulo, centro=True)
        progreso = min(1.0, t / (DURACION_COMBATE * 0.6))
        maximo = max(1e-6, *(d["total"] for d in comb["desglose"]))
        for k, j in enumerate((1, 0)):
            total = comb["desglose"][j]["total"]
            y = caja.y + 60 + k * 50
            self._texto(f"J{j}", (caja.x + 20, y), self.f_negrita, COLORES[f"jugador{j}"])
            ancho = int(400 * progreso * total / maximo)
            pygame.draw.rect(sup, COLORES["casilla"], (caja.x + 60, y, 400, 24), border_radius=5)
            pygame.draw.rect(sup, COLORES[f"jugador{j}"], (caja.x + 60, y, ancho, 24),
                             border_radius=5)
            self._texto(formatear_poder(total * progreso), (caja.x + 490, y + 12),
                        self.f_negrita, centro=True)
        if progreso >= 1.0:
            g = comb["ganador_ronda"]
            txt = "¡Empate!" if g == EMPATE else f"¡Gana J{g}!"
            col = COLORES["empate"] if g == EMPATE else COLORES[f"jugador{g}"]
            self._texto(txt, (caja.centerx, caja.bottom - 26), self.f_titulo, col, centro=True)

    def _dibujar_fin(self, snap) -> None:
        if self.combate_t is not None:
            return
        g = snap["ganador"]
        txt = "EMPATE" if g == EMPATE else f"GANA J{g}"
        col = COLORES["empate"] if g == EMPATE else COLORES[f"jugador{g}"]
        caja = pygame.Rect(0, 0, 360, 90)
        caja.center = (PANEL_X + PANEL_W // 2, 58 + 291)
        pygame.draw.rect(self.pantalla, COLORES["panel"], caja, border_radius=12)
        pygame.draw.rect(self.pantalla, col, caja, 3, border_radius=12)
        self._texto(txt, (caja.centerx, caja.centery - 10), self.f_grande, col, centro=True)
        self._texto("N = nueva partida", (caja.centerx, caja.bottom - 16), self.f_peque,
                    COLORES["texto_suave"], centro=True)

    def _dibujar_tooltip(self, snap) -> None:
        mouse = pygame.mouse.get_pos()
        for rect, zona, j, idx in self.zonas_click:
            if not rect.collidepoint(mouse):
                continue
            p = snap["jugadores"][j]
            if zona == "tienda":
                pl = CATALOGO[p["tienda"][idx]]
                lineas = [pl.nombre, f"Elemento: {pl.elemento.value}",
                          f"Costo: {pl.costo}  Poder: {pl.poder_base:.0f}"]
            else:
                u = p[zona][idx]
                lineas = [f"{u['nombre']} {'*' * u['estrellas']}",
                          f"Elemento: {u['elemento']}",
                          f"Poder: {u['poder']:.1f}  Costo: {u['costo']}"]
            w = 190
            x = min(mouse[0] + 14, ANCHO - w - 4)
            y = min(mouse[1] + 14, ALTO - 70)
            pygame.draw.rect(self.pantalla, (15, 16, 20), (x, y, w, 64), border_radius=6)
            for n, l in enumerate(lineas):
                self._texto(l, (x + 8, y + 6 + n * 18), self.f_peque)
            return

    # ------------------------------------------------------------------
    # Bucle principal
    # ------------------------------------------------------------------
    def ejecutar(self) -> None:
        corriendo = True
        while corriendo:
            dt = self.reloj.tick(FPS) / 1000.0
            corriendo = self._manejar_eventos()
            self._actualizar(dt)
            self._dibujar()
        pygame.quit()


def main() -> None:
    semilla = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    MagicChessGUI(semilla=semilla).ejecutar()


if __name__ == "__main__":
    main()
