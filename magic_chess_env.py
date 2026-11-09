"""
magic_chess_env.py
==================

Ambiente de *Magic Chess* simplificado para dos jugadores (agentes),
implementado según el documento de reglas del proyecto.

Resumen de las reglas
---------------------
* Tres elementos: agua vence a fuego, fuego a planta y planta a agua.
* Catálogo de unidades de coste 1 y coste 2 (4 por elemento).
* Economía: 2 de oro al inicio; en cada ronda nueva +3 de oro pasivo y +1 de
  bonificación si, tras sumar el ingreso, se tienen al menos 6
  (G_turno = G_actual + 3 + B). El oro no gastado se conserva.
* Tienda: 3 fichas aleatorias del catálogo; renovarla cuesta 2 de oro.
* Banca de 3 espacios. Tablero con límite creciente:
  ronda 1 -> 1, ronda 2 -> 2, rondas 3-4 -> 3, rondas 5-6 -> 4, ronda 7+ -> 5.
* Mejora: 2 fichas iguales (mismo nombre y nivel) EN LA BANCA se combinan en
  una del nivel siguiente (máximo nivel 3).
* Poder de una ficha: tabla por coste y nivel (1 / 2.25 / 4 y 2 / 3.5 / 5.5).
* Poder del equipo: P = (sum P_i * S_i) * E
    - S_i: 1.2 si su elemento tiene 2 fichas DIFERENTES en el tablero,
      1.5 si tiene 3 o más, 1 en otro caso.
    - E = 2 si el tipo del equipo vence al tipo del equipo rival, si no 1.
* Combate: gana el mayor poder; el perdedor pierde 1 vida; empate = nadie
  pierde. Cada jugador empieza con 5 vidas; quien llega a 0 pierde.

Decisiones tomadas sobre los puntos pendientes (sección 12 del documento)
------------------------------------------------------------------------
1. Catálogo: 2 fichas de coste 1 y 2 de coste 2 por elemento (``CATALOGO``).
2. Ciclo elemental: agua > fuego > planta > agua.
3. Ahorro: se evalúa después de sumar el ingreso pasivo, una vez por ronda.
   En la ronda 1 solo se tiene el oro inicial.
4. Rondas intermedias: se mantiene el último límite alcanzado (4 -> 3,
   6 -> 4, 8+ -> 5).
5. Coste 2: misma regla que coste 1 (2 copias iguales).
6. Banca llena: no se puede comprar (la compra no está entre las acciones
   legales) hasta liberar espacio moviendo o combinando fichas.
7. Empate: ronda sin pérdidas de vidas.
8. Orden del turno: ingreso -> bonificación -> compras / renovaciones /
   mejoras / colocación (en el orden que el jugador quiera) -> combate.

Además, el *tipo del equipo* (necesario para la ventaja elemental, que se
aplica entre equipos) se define como el elemento con más fichas en el
tablero; si hay empate en cantidad, el de mayor poder base sumado; si sigue
empatado, el equipo no tiene tipo y no recibe ni otorga ventaja.
``max_rondas`` es un límite de seguridad (no está en el documento): si se
alcanza, gana quien tenga más vidas.

Formalización (para el informe)
-------------------------------
* **Estado (S)**: ronda, fase y, por jugador: oro, vidas, banca, tablero
  (fichas con nivel), tienda y si ya terminó su turno. Discreto y finito.
* **Acciones (A)**: discretas -> COMPRAR(i), RENOVAR, COMBINAR(nombre,
  nivel), A_TABLERO(i), A_BANCA(i), TERMINAR.
* **Transición (T)**: determinista salvo la generación de la tienda
  (aleatoria). Dinámica conocida: *model-based*.
* **Recompensa (R)**: +1 / -1 por ganar / perder una ronda, 0 en empate;
  +10 / -10 al ganar / perder la partida.
* **Objetivo**: dejar al rival en 0 vidas.

Interfaz principal (estilo *gym*, para dos jugadores)
-----------------------------------------------------
    env = MagicChessEnv(ConfiguracionJuego(), semilla=7)
    env.reset()
    while not env.es_terminal():
        for j in env.jugadores_pendientes():
            obs, recompensa, terminado, info = env.step(j, agente[j].actuar(env, j))
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from utils import (
    COPIAS_PARA_COMBINAR,
    MULT_VENTAJA,
    NIVEL_MAX,
    PODER_POR_NIVEL,
    Elemento,
    calcular_bono_ahorro,
    crear_rng,
    multiplicador_sinergia,
    ventaja_elemental,
)


# ===========================================================================
# 1. Catálogo de unidades
# ===========================================================================
@dataclass(frozen=True)
class PlantillaUnidad:
    """Definición fija de un tipo de ficha (lo que ofrece la tienda).

    Atributos
    ---------
    nombre : identificador único del tipo de ficha.
    elemento : agua, fuego o planta (sinergias y ventaja elemental).
    coste : oro para comprarla (1 o 2). Determina su tabla de poder.

    Es inmutable porque varias fichas comparten la misma plantilla.
    """

    nombre: str
    elemento: Elemento
    coste: int

    def poder(self, nivel: int) -> float:
        """Poder base de esta ficha en el ``nivel`` dado (tablas 7.1 y 7.2)."""
        return PODER_POR_NIVEL[self.coste][nivel]


#: Catálogo: por elemento, 2 fichas de coste 1 y 2 de coste 2. Tener al
#: menos 3 fichas distintas por elemento permite activar la sinergia de 1.5.
CATALOGO: Dict[str, PlantillaUnidad] = {
    p.nombre: p
    for p in [
        PlantillaUnidad("Gota", Elemento.AGUA, 1),
        PlantillaUnidad("Burbuja", Elemento.AGUA, 1),
        PlantillaUnidad("Marea", Elemento.AGUA, 2),
        PlantillaUnidad("Leviatan", Elemento.AGUA, 2),
        PlantillaUnidad("Chispa", Elemento.FUEGO, 1),
        PlantillaUnidad("Ceniza", Elemento.FUEGO, 1),
        PlantillaUnidad("Brasa", Elemento.FUEGO, 2),
        PlantillaUnidad("Fenix", Elemento.FUEGO, 2),
        PlantillaUnidad("Brote", Elemento.PLANTA, 1),
        PlantillaUnidad("Musgo", Elemento.PLANTA, 1),
        PlantillaUnidad("Espino", Elemento.PLANTA, 2),
        PlantillaUnidad("Ent", Elemento.PLANTA, 2),
    ]
}


@dataclass
class Unidad:
    """Una ficha concreta que posee un jugador.

    Atributos
    ---------
    uid : identificador único en la partida (evita que la misma ficha esté
        en dos lugares y permite a la GUI seguirla).
    nombre : nombre de su plantilla en ``CATALOGO``.
    nivel : 1 a 3; sube al combinar 2 copias iguales.
    """

    uid: int
    nombre: str
    nivel: int = 1

    @property
    def plantilla(self) -> PlantillaUnidad:
        return CATALOGO[self.nombre]

    @property
    def elemento(self) -> Elemento:
        return self.plantilla.elemento

    @property
    def coste(self) -> int:
        return self.plantilla.coste

    @property
    def poder(self) -> float:
        """Poder base P_i según coste y nivel."""
        return self.plantilla.poder(self.nivel)

    def clave(self) -> Tuple[str, int]:
        """(nombre, nivel): dos fichas con la misma clave son intercambiables
        y son las que se pueden combinar."""
        return (self.nombre, self.nivel)

    def como_dict(self) -> Dict[str, Any]:
        return {"uid": self.uid, "nombre": self.nombre, "nivel": self.nivel,
                "elemento": self.elemento.value, "coste": self.coste,
                "poder": self.poder}


# ===========================================================================
# 2. Configuración
# ===========================================================================
@dataclass
class ConfiguracionJuego:
    """Parámetros de una instancia. Los valores por defecto son los del
    documento de reglas; ``desde_dificultad`` crea variantes."""

    vidas_iniciales: int = 5          # vidas de cada jugador
    oro_inicial: int = 2              # oro en la ronda 1
    ingreso_pasivo: int = 3           # oro que se recibe en cada ronda nueva
    umbral_ahorro: int = 6            # oro mínimo para la bonificación
    bono_ahorro: int = 1              # bonificación por ahorro
    coste_renovar: int = 2            # coste de renovar la tienda
    tamano_tienda: int = 3            # fichas ofrecidas por la tienda
    tamano_banca: int = 3             # espacios de banca
    #: (ronda desde la que aplica, máximo de fichas en el tablero)
    limites_tablero: List[Tuple[int, int]] = field(
        default_factory=lambda: [(1, 1), (2, 2), (3, 3), (5, 4), (7, 5)])
    danio_por_derrota: int = 1        # vidas que pierde el perdedor
    max_rondas: int = 50              # límite de seguridad

    @classmethod
    def desde_dificultad(cls, nivel: str) -> "ConfiguracionJuego":
        """* ``facil``: más oro inicial y renovar más barato.
        * ``normal``: reglas del documento.
        * ``dificil``: menos vidas y renovar más caro: los errores pesan más.
        """
        nivel = nivel.lower()
        if nivel == "facil":
            return cls(oro_inicial=4, coste_renovar=1)
        if nivel == "dificil":
            return cls(vidas_iniciales=3, coste_renovar=3)
        return cls()

    def capacidad_tablero(self, ronda: int) -> int:
        """Máximo de fichas en combate en ``ronda`` (se mantiene el último
        límite alcanzado en las rondas intermedias)."""
        limite = self.limites_tablero[0][1]
        for desde, maximo in self.limites_tablero:
            if ronda >= desde:
                limite = maximo
        return limite

    @property
    def tablero_maximo(self) -> int:
        return max(m for _, m in self.limites_tablero)


# ===========================================================================
# 3. Estado de cada jugador
# ===========================================================================
@dataclass
class EstadoJugador:
    """Lo que describe a un jugador en un instante (sección 11).

    oro, vidas, banca (no combate), tablero (sí combate), tienda (nombres;
    ``None`` = ya comprada), listo (terminó su preparación) y estadísticas
    (victorias, derrotas, empates, oro gastado y ganado).
    """

    oro: int
    vidas: int
    banca: List[Unidad] = field(default_factory=list)
    tablero: List[Unidad] = field(default_factory=list)
    tienda: List[Optional[str]] = field(default_factory=list)
    listo: bool = False
    victorias: int = 0
    derrotas: int = 0
    empates: int = 0
    oro_gastado: int = 0
    oro_ganado: int = 0

    def combinables(self) -> List[Tuple[str, int]]:
        """Claves (nombre, nivel) con al menos 2 copias en la BANCA y nivel
        menor al máximo."""
        conteo: Dict[Tuple[str, int], int] = {}
        for u in self.banca:
            conteo[u.clave()] = conteo.get(u.clave(), 0) + 1
        return sorted(k for k, n in conteo.items()
                      if n >= COPIAS_PARA_COMBINAR and k[1] < NIVEL_MAX)

    def como_dict(self) -> Dict[str, Any]:
        return {"oro": self.oro, "vidas": self.vidas,
                "banca": [u.como_dict() for u in self.banca],
                "tablero": [u.como_dict() for u in self.tablero],
                "tienda": list(self.tienda), "listo": self.listo,
                "victorias": self.victorias, "derrotas": self.derrotas,
                "empates": self.empates, "oro_gastado": self.oro_gastado,
                "oro_ganado": self.oro_ganado}


# ===========================================================================
# 4. Acciones
# ===========================================================================
class TipoAccion(str, Enum):
    """Decisiones posibles durante la preparación."""

    COMPRAR = "comprar"        # arg: índice en la tienda
    RENOVAR = "renovar"        # sin argumento
    COMBINAR = "combinar"      # arg: (nombre, nivel)
    A_TABLERO = "a_tablero"    # arg: índice en la banca
    A_BANCA = "a_banca"        # arg: índice en el tablero
    TERMINAR = "terminar"      # listo para el combate


@dataclass(frozen=True)
class Accion:
    """Acción = tipo + argumento. Inmutable y *hashable*."""

    tipo: TipoAccion
    arg: Any = None

    def es_terminar(self) -> bool:
        return self.tipo == TipoAccion.TERMINAR

    def __str__(self) -> str:
        return self.tipo.value if self.arg is None else f"{self.tipo.value}({self.arg})"


# ===========================================================================
# 5. Poder de un equipo
# ===========================================================================
@dataclass
class DesglosePoder:
    """Cálculo del poder P = (sum P_i * S_i) * E, paso a paso.

    base : sum P_i (sin modificadores).
    con_sinergia : sum P_i * S_i.
    mult_ventaja : E (2 o 1).
    tipo : tipo del equipo (elemento dominante) o None.
    sinergias : {elemento: (fichas diferentes, S)} de los elementos con S > 1.
    """

    base: float = 0.0
    con_sinergia: float = 0.0
    mult_ventaja: float = 1.0
    tipo: Optional[str] = None
    sinergias: Dict[str, Tuple[int, float]] = field(default_factory=dict)

    @property
    def total(self) -> float:
        return self.con_sinergia * self.mult_ventaja

    def como_dict(self) -> Dict[str, Any]:
        return {"base": self.base, "con_sinergia": self.con_sinergia,
                "mult_ventaja": self.mult_ventaja, "tipo": self.tipo,
                "sinergias": {k: list(v) for k, v in self.sinergias.items()},
                "total": self.total}


def tipo_equipo(tablero: List[Unidad]) -> Optional[Elemento]:
    """Elemento dominante del tablero: el de más fichas; en empate, el de
    mayor poder base; si sigue empatado (o el tablero está vacío), None."""
    if not tablero:
        return None
    puntaje: Dict[Elemento, Tuple[int, float]] = {}
    for u in tablero:
        n, p = puntaje.get(u.elemento, (0, 0.0))
        puntaje[u.elemento] = (n + 1, p + u.poder)
    orden = sorted(puntaje.items(), key=lambda kv: kv[1], reverse=True)
    if len(orden) > 1 and orden[0][1] == orden[1][1]:
        return None
    return orden[0][0]


def calcular_poder(tablero: List[Unidad], tablero_rival: List[Unidad]
                   ) -> DesglosePoder:
    """Poder de ``tablero`` frente a ``tablero_rival`` (sección 9):

    1. P_i según coste y nivel.
    2. Para cada elemento se cuentan las fichas DIFERENTES (por nombre) del
       tablero; S_i = 1.2 con 2, 1.5 con 3 o más. Las copias no cuentan.
    3. E = 2 si el tipo del equipo vence al tipo del rival.
    """
    d = DesglosePoder()
    if not tablero:
        return d
    diferentes: Dict[Elemento, set] = {}
    for u in tablero:
        diferentes.setdefault(u.elemento, set()).add(u.nombre)
    for u in tablero:
        s = multiplicador_sinergia(len(diferentes[u.elemento]))
        d.base += u.poder
        d.con_sinergia += u.poder * s
    for elem, nombres in diferentes.items():
        s = multiplicador_sinergia(len(nombres))
        if s > 1.0:
            d.sinergias[elem.value] = (len(nombres), s)
    mio, suyo = tipo_equipo(tablero), tipo_equipo(tablero_rival)
    d.tipo = mio.value if mio else None
    if ventaja_elemental(mio, suyo) > 0:
        d.mult_ventaja = MULT_VENTAJA
    return d


# ===========================================================================
# 6. Ambiente
# ===========================================================================
EMPATE = -1  # valor de ``ganador`` cuando la partida termina igualada


class MagicChessEnv:
    """Ambiente de dos jugadores (índices 0 y 1).

    Atributos públicos: ``config``, ``jugadores``, ``ronda``, ``fase``
    ("preparacion" | "terminado"), ``ganador`` (None, 0, 1 o EMPATE),
    ``ultimo_combate``, ``registro`` (mensajes) y ``pasos``.
    """

    RECOMPENSA_RONDA = 1.0
    RECOMPENSA_PARTIDA = 10.0

    def __init__(self, config: Optional[ConfiguracionJuego] = None,
                 semilla: Optional[int] = None) -> None:
        self.config = config or ConfiguracionJuego()
        self.semilla = semilla
        self.reset(semilla)

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------
    def reset(self, semilla: Optional[int] = None) -> Dict[str, Any]:
        """Reinicia a la ronda 1: 2 de oro, 5 vidas y tienda nueva."""
        if semilla is not None:
            self.semilla = semilla
        self.rng = crear_rng(self.semilla)
        self._siguiente_uid = 0
        self.ronda = 1
        self.fase = "preparacion"
        self.ganador: Optional[int] = None
        self.ultimo_combate: Optional[Dict[str, Any]] = None
        self.registro: List[str] = []
        self.pasos = 0
        cfg = self.config
        self.jugadores = [EstadoJugador(oro=cfg.oro_inicial, vidas=cfg.vidas_iniciales)
                          for _ in range(2)]
        for j in range(2):
            self.jugadores[j].oro_ganado = cfg.oro_inicial
            self._generar_tienda(j)
        self._log(f"Ronda 1: {cfg.oro_inicial} de oro c/u, tablero de "
                  f"{self.capacidad_tablero()} ficha(s).")
        return self.observacion(0)

    def clonar(self) -> "MagicChessEnv":
        """Copia profunda e independiente (para simular sin alterar la
        partida real)."""
        return copy.deepcopy(self)

    # ------------------------------------------------------------------
    # Consultas
    # ------------------------------------------------------------------
    def es_terminal(self) -> bool:
        return self.fase == "terminado"

    def rival(self, jugador: int) -> int:
        return 1 - jugador

    def capacidad_tablero(self) -> int:
        return self.config.capacidad_tablero(self.ronda)

    def jugadores_pendientes(self) -> List[int]:
        if self.es_terminal():
            return []
        return [j for j in range(2) if not self.jugadores[j].listo]

    def acciones_legales(self, jugador: int,
                         sin_duplicados: bool = True) -> List[Accion]:
        """Acciones válidas (se comprueban oro, espacio y nivel):

        * COMPRAR i: hay ficha en la casilla i, alcanza el oro y hay espacio
          en la banca.
        * RENOVAR: alcanza el oro.
        * COMBINAR (n, k): 2 copias de (n, k) en la banca y k < 3.
        * A_TABLERO i: el tablero no ha llegado a su límite.
        * A_BANCA i: hay espacio en la banca.
        * TERMINAR: siempre.

        Con ``sin_duplicados`` las acciones sobre fichas idénticas se cuentan
        una vez (llevan a estados equivalentes).
        """
        if self.es_terminal() or self.jugadores[jugador].listo:
            return []
        j = self.jugadores[jugador]
        cfg = self.config
        acciones: List[Accion] = []

        if len(j.banca) < cfg.tamano_banca:
            vistos = set()
            for i, nombre in enumerate(j.tienda):
                if nombre is None or CATALOGO[nombre].coste > j.oro:
                    continue
                if sin_duplicados and nombre in vistos:
                    continue
                vistos.add(nombre)
                acciones.append(Accion(TipoAccion.COMPRAR, i))

        if j.oro >= cfg.coste_renovar:
            acciones.append(Accion(TipoAccion.RENOVAR))

        for clave in j.combinables():
            acciones.append(Accion(TipoAccion.COMBINAR, clave))

        if len(j.tablero) < self.capacidad_tablero():
            vistos = set()
            for i, u in enumerate(j.banca):
                if sin_duplicados and u.clave() in vistos:
                    continue
                vistos.add(u.clave())
                acciones.append(Accion(TipoAccion.A_TABLERO, i))

        if len(j.banca) < cfg.tamano_banca:
            vistos = set()
            for i, u in enumerate(j.tablero):
                if sin_duplicados and u.clave() in vistos:
                    continue
                vistos.add(u.clave())
                acciones.append(Accion(TipoAccion.A_BANCA, i))

        acciones.append(Accion(TipoAccion.TERMINAR))
        return acciones

    def es_legal(self, jugador: int, accion: Accion) -> bool:
        return accion in self.acciones_legales(jugador, sin_duplicados=False)

    # ------------------------------------------------------------------
    # Transición
    # ------------------------------------------------------------------
    def step(self, jugador: int, accion: Accion
             ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Aplica ``accion`` de ``jugador``. Devuelve (observación,
        recompensa del jugador, terminado, info). ``info["recompensas"]``
        trae las de ambos y ``info["combate"]`` el resultado si hubo combate.
        Lanza ``ValueError`` si la acción no es legal."""
        if self.es_terminal():
            raise ValueError("La partida ya terminó.")
        if not self.es_legal(jugador, accion):
            raise ValueError(f"Acción ilegal para J{jugador}: {accion}")

        self.pasos += 1
        j = self.jugadores[jugador]
        cfg = self.config
        t = accion.tipo
        info: Dict[str, Any] = {"accion": str(accion), "jugador": jugador}
        recompensas = [0.0, 0.0]

        if t == TipoAccion.COMPRAR:
            nombre = j.tienda[accion.arg]
            coste = CATALOGO[nombre].coste
            j.oro -= coste
            j.oro_gastado += coste
            j.tienda[accion.arg] = None          # no se puede volver a comprar
            j.banca.append(self._nueva_unidad(nombre))
            self._log(f"J{jugador} compra {nombre} (-{coste} oro).")

        elif t == TipoAccion.RENOVAR:
            j.oro -= cfg.coste_renovar           # cobro separado de la generación
            j.oro_gastado += cfg.coste_renovar
            self._generar_tienda(jugador)
            self._log(f"J{jugador} renueva la tienda: {', '.join(j.tienda)}.")

        elif t == TipoAccion.COMBINAR:
            nombre, nivel = accion.arg
            self._combinar(j, nombre, nivel)
            self._log(f"J{jugador} combina 2x {nombre} nv{nivel} -> nv{nivel + 1}.")

        elif t == TipoAccion.A_TABLERO:
            u = j.banca.pop(accion.arg)
            j.tablero.append(u)
            self._log(f"J{jugador} despliega {u.nombre} nv{u.nivel}.")

        elif t == TipoAccion.A_BANCA:
            u = j.tablero.pop(accion.arg)
            j.banca.append(u)
            self._log(f"J{jugador} retira {u.nombre} nv{u.nivel} a la banca.")

        elif t == TipoAccion.TERMINAR:
            j.listo = True
            self._log(f"J{jugador} termina su preparación.")
            if all(p.listo for p in self.jugadores):
                info["combate"] = self._resolver_combate(recompensas)

        info["recompensas"] = recompensas
        return self.observacion(jugador), recompensas[jugador], self.es_terminal(), info

    # ------------------------------------------------------------------
    # Poder y evaluación
    # ------------------------------------------------------------------
    def poder_equipo(self, jugador: int,
                     tablero_rival: Optional[List[Unidad]] = None,
                     tablero_propio: Optional[List[Unidad]] = None) -> DesglosePoder:
        """Poder del tablero de ``jugador`` contra el del rival. Se pueden
        pasar tableros hipotéticos (útil para heurísticas y búsqueda)."""
        propio = self.jugadores[jugador].tablero if tablero_propio is None else tablero_propio
        rival = (self.jugadores[self.rival(jugador)].tablero
                 if tablero_rival is None else tablero_rival)
        return calcular_poder(propio, rival)

    def evaluar_estado(self, jugador: int) -> float:
        """Valor heurístico simple del estado para ``jugador``."""
        if self.es_terminal():
            if self.ganador == jugador:
                return 1e6
            return 0.0 if self.ganador == EMPATE else -1e6
        yo, otro = self.jugadores[jugador], self.jugadores[self.rival(jugador)]
        return (100.0 * (yo.vidas - otro.vidas)
                + self.poder_equipo(jugador).total
                - self.poder_equipo(self.rival(jugador)).total
                + 0.3 * yo.oro)

    # ------------------------------------------------------------------
    # Observaciones y snapshot
    # ------------------------------------------------------------------
    def observacion(self, jugador: int) -> Dict[str, Any]:
        """Estado propio completo + parte pública del rival (vidas y
        tablero)."""
        rival = self.jugadores[self.rival(jugador)]
        return {"jugador": jugador, "ronda": self.ronda, "fase": self.fase,
                "capacidad_tablero": self.capacidad_tablero(),
                "propio": self.jugadores[jugador].como_dict(),
                "rival": {"vidas": rival.vidas,
                          "tablero": [u.como_dict() for u in rival.tablero]}}

    def snapshot(self) -> Dict[str, Any]:
        """Estado completo como diccionario de tipos básicos (para la GUI y el
        historial)."""
        cfg = self.config
        return {
            "ronda": self.ronda,
            "fase": self.fase,
            "ganador": self.ganador,
            "pasos": self.pasos,
            "capacidad_tablero": self.capacidad_tablero(),
            "jugadores": [p.como_dict() for p in self.jugadores],
            "poder": [self.poder_equipo(j).como_dict() for j in range(2)],
            # bonificación que recibiría en la próxima ronda si no gasta más
            "bono_proximo": [calcular_bono_ahorro(p.oro + cfg.ingreso_pasivo,
                                                  cfg.umbral_ahorro, cfg.bono_ahorro)
                             for p in self.jugadores],
            "combinables": [[list(k) for k in p.combinables()] for p in self.jugadores],
            "ultimo_combate": copy.deepcopy(self.ultimo_combate),
            "registro": list(self.registro[-12:]),
            "config": {"vidas_iniciales": cfg.vidas_iniciales,
                       "tamano_banca": cfg.tamano_banca,
                       "tamano_tienda": cfg.tamano_tienda,
                       "coste_renovar": cfg.coste_renovar,
                       "tablero_maximo": cfg.tablero_maximo,
                       "ingreso_pasivo": cfg.ingreso_pasivo,
                       "umbral_ahorro": cfg.umbral_ahorro,
                       "max_rondas": cfg.max_rondas},
        }

    # ------------------------------------------------------------------
    # Editor de instancias
    # ------------------------------------------------------------------
    def fijar_oro(self, jugador: int, oro: int) -> None:
        self.jugadores[jugador].oro = max(0, int(oro))

    def fijar_vidas(self, jugador: int, vidas: int) -> None:
        self.jugadores[jugador].vidas = max(1, int(vidas))

    def fijar_ronda(self, ronda: int) -> None:
        """Cambia la ronda (afecta el límite del tablero)."""
        self.ronda = max(1, min(int(ronda), self.config.max_rondas))

    def agregar_unidad(self, jugador: int, nombre: str, nivel: int = 1,
                       zona: str = "banca") -> bool:
        """Agrega una ficha sin coste. Devuelve False si no hay espacio."""
        if nombre not in CATALOGO or not 1 <= nivel <= NIVEL_MAX:
            raise ValueError(f"Unidad inválida: {nombre} nv{nivel}")
        j = self.jugadores[jugador]
        if zona == "tablero":
            if len(j.tablero) >= self.capacidad_tablero():
                return False
            j.tablero.append(self._nueva_unidad(nombre, nivel))
        else:
            if len(j.banca) >= self.config.tamano_banca:
                return False
            j.banca.append(self._nueva_unidad(nombre, nivel))
        return True

    def fijar_tienda(self, jugador: int, nombres: List[Optional[str]]) -> None:
        for n in nombres:
            if n is not None and n not in CATALOGO:
                raise ValueError(f"Unidad desconocida: {n}")
        tam = self.config.tamano_tienda
        self.jugadores[jugador].tienda = (list(nombres) + [None] * tam)[:tam]

    def vaciar_jugador(self, jugador: int) -> None:
        self.jugadores[jugador].banca.clear()
        self.jugadores[jugador].tablero.clear()

    # ------------------------------------------------------------------
    # Lógica interna
    # ------------------------------------------------------------------
    def _log(self, mensaje: str) -> None:
        self.registro.append(mensaje)

    def _nueva_unidad(self, nombre: str, nivel: int = 1) -> Unidad:
        u = Unidad(uid=self._siguiente_uid, nombre=nombre, nivel=nivel)
        self._siguiente_uid += 1
        return u

    def _generar_tienda(self, jugador: int) -> None:
        """Tienda: ``tamano_tienda`` fichas elegidas al azar (uniforme) del
        catálogo. Solo genera; el cobro de la renovación se hace en
        ``step`` (lógicas separadas, como recomienda el documento)."""
        nombres = list(CATALOGO)
        self.jugadores[jugador].tienda = [self.rng.choice(nombres)
                                          for _ in range(self.config.tamano_tienda)]

    def _combinar(self, j: EstadoJugador, nombre: str, nivel: int) -> None:
        """Consume 2 copias de (nombre, nivel) de la banca y deja en la banca
        una ficha de nivel + 1 (operación controlada: se verifica antes en
        ``acciones_legales``)."""
        restantes = COPIAS_PARA_COMBINAR
        for i in range(len(j.banca) - 1, -1, -1):
            if restantes and j.banca[i].clave() == (nombre, nivel):
                j.banca.pop(i)
                restantes -= 1
        j.banca.append(self._nueva_unidad(nombre, nivel + 1))

    def _resolver_combate(self, recompensas: List[float]) -> Dict[str, Any]:
        """Calcula los poderes, resta vidas, reparte recompensas y avanza la
        ronda o termina la partida."""
        cfg = self.config
        desgloses = [self.poder_equipo(j) for j in range(2)]
        p0, p1 = desgloses[0].total, desgloses[1].total
        if abs(p0 - p1) < 1e-9:
            ganador_ronda = EMPATE
            for p in self.jugadores:
                p.empates += 1
            self._log(f"Combate ronda {self.ronda}: empate ({p0:g} vs {p1:g}).")
        else:
            ganador_ronda = 0 if p0 > p1 else 1
            perdedor = 1 - ganador_ronda
            self.jugadores[ganador_ronda].victorias += 1
            self.jugadores[perdedor].derrotas += 1
            self.jugadores[perdedor].vidas -= cfg.danio_por_derrota
            recompensas[ganador_ronda] += self.RECOMPENSA_RONDA
            recompensas[perdedor] -= self.RECOMPENSA_RONDA
            self._log(f"Combate ronda {self.ronda}: gana J{ganador_ronda} "
                      f"({p0:g} vs {p1:g}). J{perdedor} pierde una vida.")
        resultado = {"ronda": self.ronda, "ganador_ronda": ganador_ronda,
                     "desglose": [d.como_dict() for d in desgloses],
                     "tableros": [[u.como_dict() for u in p.tablero]
                                  for p in self.jugadores]}
        self.ultimo_combate = resultado
        if any(p.vidas <= 0 for p in self.jugadores) or self.ronda >= cfg.max_rondas:
            self._terminar_partida(recompensas)
        else:
            self._iniciar_ronda()
        return resultado

    def _terminar_partida(self, recompensas: List[float]) -> None:
        v0, v1 = self.jugadores[0].vidas, self.jugadores[1].vidas
        if v0 == v1:
            self.ganador = EMPATE
            self._log("Fin de la partida: empate.")
        else:
            self.ganador = 0 if v0 > v1 else 1
            recompensas[self.ganador] += self.RECOMPENSA_PARTIDA
            recompensas[1 - self.ganador] -= self.RECOMPENSA_PARTIDA
            self._log(f"Fin de la partida: gana J{self.ganador}.")
        self.fase = "terminado"

    def _iniciar_ronda(self) -> None:
        """Nueva ronda: G = G_actual + 3 + B (B se evalúa después de sumar el
        ingreso pasivo), tienda nueva y vuelta a la preparación."""
        cfg = self.config
        self.ronda += 1
        for idx, p in enumerate(self.jugadores):
            p.oro += cfg.ingreso_pasivo
            bono = calcular_bono_ahorro(p.oro, cfg.umbral_ahorro, cfg.bono_ahorro)
            p.oro += bono
            p.oro_ganado += cfg.ingreso_pasivo + bono
            p.listo = False
            self._generar_tienda(idx)
        self._log(f"Ronda {self.ronda}: +{cfg.ingreso_pasivo} oro (+bono si >= "
                  f"{cfg.umbral_ahorro}). Tablero: {self.capacidad_tablero()}.")


# ===========================================================================
# Prueba rápida: python magic_chess_env.py
# ===========================================================================
if __name__ == "__main__":
    from utils import AgenteAleatorio

    env = MagicChessEnv(ConfiguracionJuego(), semilla=1)
    agentes = [AgenteAleatorio(semilla=10), AgenteAleatorio(semilla=20)]
    while not env.es_terminal():
        for jug in env.jugadores_pendientes():
            while not env.jugadores[jug].listo and not env.es_terminal():
                env.step(jug, agentes[jug].actuar(env, jug))
    print("\n".join(env.registro[-6:]))
    print(f"Ganador: {env.ganador} | rondas: {env.ronda} | pasos: {env.pasos}")
