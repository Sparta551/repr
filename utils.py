"""
utils.py
========

Utilidades compartidas del proyecto *Magic Chess*.

Este módulo NO depende de ninguna librería gráfica: lo pueden importar tanto
el ambiente (``magic_chess_env.py``) como la interfaz (``magic_chess_gui.py``),
los algoritmos de búsqueda y el notebook.

Contenido
---------
1. Elementos y tabla de ventajas (agua > fuego > planta > agua).
2. Constantes de juego que usa el cálculo de poder (estrellas, sinergias).
3. Funciones económicas (interés / bonificación por ahorro).
4. Generador de números aleatorios reproducible.
5. Herramientas de medición para los algoritmos (cronómetro y métricas).
6. Historial de estados para reproducir / retroceder una partida.
7. Paleta de colores y funciones de formato usadas por la GUI.
8. Política aleatoria de referencia (sirve para probar el ambiente).
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# 1. Elementos
# ---------------------------------------------------------------------------
class Elemento(str, Enum):
    """Los tres elementos a los que puede pertenecer una unidad.

    Hereda de ``str`` para que se pueda imprimir / serializar directamente
    (por ejemplo ``Elemento.AGUA == "agua"``).
    """

    AGUA = "agua"
    FUEGO = "fuego"
    PLANTA = "planta"


#: Ciclo de ventajas: la clave le gana al valor.
#: agua apaga el fuego, el fuego quema la planta, la planta absorbe el agua.
VENCE_A: Dict[Elemento, Elemento] = {
    Elemento.AGUA: Elemento.FUEGO,
    Elemento.FUEGO: Elemento.PLANTA,
    Elemento.PLANTA: Elemento.AGUA,
}


def ventaja_elemental(atacante: Elemento, defensor: Elemento) -> int:
    """Devuelve +1 si ``atacante`` vence a ``defensor``, -1 si pierde y 0 si
    son del mismo elemento.

    Se usa para calcular el factor de ventaja de un equipo frente a otro.
    """
    if VENCE_A[atacante] == defensor:
        return 1
    if VENCE_A[defensor] == atacante:
        return -1
    return 0


# ---------------------------------------------------------------------------
# 2. Constantes de combate
# ---------------------------------------------------------------------------
#: Multiplicador de poder según el nivel (estrellas) de la unidad.
#: Tres unidades iguales de nivel n se combinan en una de nivel n+1.
MULT_ESTRELLAS: Dict[int, float] = {1: 1.0, 2: 2.2, 3: 4.5}
ESTRELLAS_MAX: int = 3
COPIAS_PARA_COMBINAR: int = 3

#: Sinergias elementales: (mínimo de unidades del mismo elemento, bonificación).
#: La bonificación se aplica sobre el poder de las unidades de ese elemento.
#: Se toma el umbral más alto alcanzado.
SINERGIAS: List[tuple] = [(2, 0.10), (3, 0.20), (4, 0.35)]

#: Peso del factor de ventaja elemental en el poder total.
#: factor = 1 + PESO_VENTAJA * ventaja_normalizada, con ventaja en [-1, 1].
PESO_VENTAJA: float = 0.25


def bono_sinergia(cantidad: int) -> float:
    """Bonificación (fracción) que otorga tener ``cantidad`` unidades de un
    mismo elemento desplegadas en el tablero."""
    bono = 0.0
    for minimo, valor in SINERGIAS:
        if cantidad >= minimo:
            bono = valor
    return bono


# ---------------------------------------------------------------------------
# 3. Economía
# ---------------------------------------------------------------------------
def calcular_interes(oro: int, paso: int = 10, tope: int = 3) -> int:
    """Bonificación por ahorro: +1 de oro por cada ``paso`` de oro guardado,
    hasta un máximo de ``tope``.

    Ejemplo: con 27 de oro y paso=10 se obtienen 2 de interés.
    """
    if paso <= 0:
        return 0
    return min(tope, max(0, oro) // paso)


# ---------------------------------------------------------------------------
# 4. Aleatoriedad reproducible
# ---------------------------------------------------------------------------
def crear_rng(semilla: Optional[int] = None) -> random.Random:
    """Crea un generador aleatorio independiente.

    Usar un ``random.Random`` propio (y no el módulo ``random`` global) hace
    que cada instancia del ambiente sea reproducible y que clonar el ambiente
    para buscar no altere la partida original.
    """
    return random.Random(semilla)


# ---------------------------------------------------------------------------
# 5. Métricas para los algoritmos
# ---------------------------------------------------------------------------
@dataclass
class MetricasBusqueda:
    """Contenedor de las métricas pedidas en el enunciado (sección 2.2).

    Los algoritmos deben actualizar estos contadores mientras buscan; la GUI y
    la ventana de comparación los leen.
    """

    algoritmo: str = ""
    tiempo_s: float = 0.0
    nodos_generados: int = 0
    nodos_expandidos: int = 0
    max_frontera: int = 0
    longitud_solucion: int = 0
    costo_solucion: float = 0.0
    encontro_solucion: bool = False
    es_optima: Optional[bool] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def registrar_frontera(self, tamano: int) -> None:
        """Actualiza el tamaño máximo de frontera (memoria aproximada)."""
        if tamano > self.max_frontera:
            self.max_frontera = tamano

    def como_dict(self) -> Dict[str, Any]:
        """Representación plana (útil para tablas o ``pandas.DataFrame``)."""
        return asdict(self)


class Cronometro:
    """Gestor de contexto para medir tiempo de ejecución.

    Ejemplo::

        with Cronometro() as c:
            algoritmo(env)
        print(c.segundos)
    """

    def __init__(self) -> None:
        self.inicio = 0.0
        self.segundos = 0.0

    def __enter__(self) -> "Cronometro":
        self.inicio = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.segundos = time.perf_counter() - self.inicio


# ---------------------------------------------------------------------------
# 6. Historial (reproducción paso a paso)
# ---------------------------------------------------------------------------
class Historial:
    """Lista de *snapshots* del ambiente con un cursor.

    Permite los controles de reproducción que pide el enunciado: avanzar,
    retroceder, ir al inicio o al final. Cada snapshot es un diccionario
    independiente (ver ``MagicChessEnv.snapshot``), así que retroceder no
    modifica el ambiente real.
    """

    def __init__(self) -> None:
        self.estados: List[Dict[str, Any]] = []
        self.cursor: int = -1

    def __len__(self) -> int:
        return len(self.estados)

    def agregar(self, snapshot: Dict[str, Any]) -> None:
        """Agrega un snapshot al final y mueve el cursor allí."""
        self.estados.append(snapshot)
        self.cursor = len(self.estados) - 1

    def actual(self) -> Optional[Dict[str, Any]]:
        if 0 <= self.cursor < len(self.estados):
            return self.estados[self.cursor]
        return None

    def en_vivo(self) -> bool:
        """True si el cursor está en el último estado (la partida real)."""
        return self.cursor == len(self.estados) - 1

    def atras(self) -> None:
        self.cursor = max(0, self.cursor - 1)

    def adelante(self) -> None:
        self.cursor = min(len(self.estados) - 1, self.cursor + 1)

    def ir_inicio(self) -> None:
        self.cursor = 0 if self.estados else -1

    def ir_final(self) -> None:
        self.cursor = len(self.estados) - 1

    def limpiar(self) -> None:
        self.estados.clear()
        self.cursor = -1


# ---------------------------------------------------------------------------
# 7. Colores y formato (usados por la GUI, sin depender de pygame)
# ---------------------------------------------------------------------------
COLORES = {
    "fondo": (24, 26, 33),
    "panel": (36, 39, 50),
    "panel_borde": (60, 64, 80),
    "casilla": (48, 52, 66),
    "casilla_bloqueada": (32, 34, 42),
    "texto": (230, 232, 240),
    "texto_suave": (150, 155, 170),
    "oro": (240, 196, 64),
    "vida": (226, 74, 86),
    "boton": (70, 92, 160),
    "boton_hover": (92, 118, 200),
    "boton_inactivo": (60, 62, 72),
    "victoria": (88, 190, 120),
    "derrota": (210, 80, 80),
    "empate": (170, 170, 170),
    "jugador0": (90, 160, 255),
    "jugador1": (255, 140, 90),
    "resaltado": (255, 255, 255),
}

COLOR_ELEMENTO = {
    Elemento.AGUA: (64, 140, 230),
    Elemento.FUEGO: (232, 96, 56),
    Elemento.PLANTA: (82, 176, 92),
}


def aclarar(color: tuple, factor: float = 0.3) -> tuple:
    """Mezcla un color RGB con blanco (factor 0 = igual, 1 = blanco)."""
    return tuple(int(c + (255 - c) * factor) for c in color)


def oscurecer(color: tuple, factor: float = 0.3) -> tuple:
    """Mezcla un color RGB con negro (factor 0 = igual, 1 = negro)."""
    return tuple(int(c * (1 - factor)) for c in color)


def formatear_poder(valor: float) -> str:
    """Formato corto para mostrar poder en pantalla."""
    return f"{valor:.1f}"


# ---------------------------------------------------------------------------
# 8. Política aleatoria de referencia
# ---------------------------------------------------------------------------
class AgenteAleatorio:
    """Agente que elige una acción legal al azar.

    No es uno de los algoritmos del proyecto; sirve para probar el ambiente y
    la GUI, y como línea base en los experimentos. Cualquier agente que se
    conecte a la GUI solo necesita un método ``actuar(env, jugador)``.

    ``prob_terminar`` controla qué tan rápido termina su turno, para que no
    pase la ronda entera moviendo fichas.
    """

    nombre = "Aleatorio"

    def __init__(self, semilla: Optional[int] = None,
                 prob_terminar: float = 0.25) -> None:
        self.rng = crear_rng(semilla)
        self.prob_terminar = prob_terminar

    def actuar(self, env, jugador: int):
        acciones = env.acciones_legales(jugador)
        terminar = [a for a in acciones if a.es_terminar()]
        otras = [a for a in acciones if not a.es_terminar()]
        if not otras or self.rng.random() < self.prob_terminar:
            return terminar[0]
        return self.rng.choice(otras)
