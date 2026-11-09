"""
magic_chess_env.py
==================

Ambiente de *Magic Chess* simplificado para dos jugadores (agentes).

Resumen del juego
-----------------
Cada ronda tiene dos fases:

1. **Preparación**: cada jugador, con su oro, puede comprar unidades de su
   tienda (3 opciones), renovar la tienda, combinar 3 copias iguales en una
   unidad de mayor nivel, mover fichas entre la banca y el tablero, vender, y
   finalmente terminar su turno.
2. **Combate**: cuando ambos terminan, se calcula el poder de los dos equipos
   desplegados en el tablero. El de menor poder pierde una vida. Luego empieza
   una nueva ronda: ambos reciben oro (ingreso base + interés por ahorro +
   bono por victoria) y una tienda nueva.

La partida termina cuando un jugador se queda sin vidas o se alcanza el máximo
de rondas (gana quien tenga más vidas; si hay igualdad, empate).

Formalización (para el informe)
-------------------------------
* **Estado (S)**: ronda, fase y, por cada jugador: oro, vidas, unidades en
  banca, unidades en tablero (con su nivel), opciones de la tienda y si ya
  terminó su turno. Espacio discreto y finito (acotado por max_rondas, el
  tamaño de banca/tablero y el oro máximo alcanzable).
* **Acciones (A)**: discretas -> COMPRAR(i), RENOVAR, COMBINAR(nombre, nivel),
  A_TABLERO(i), A_BANCA(i), VENDER(zona, i), TERMINAR.
* **Transición (T)**: determinista para las acciones de preparación; es
  estocástica en la generación de la tienda (RENOVAR y cada ronda nueva) y,
  opcionalmente, en el combate (``ruido_combate``). La dinámica es conocida
  (el agente puede clonar el ambiente y simular -> *model-based*).
* **Recompensa (R)**: +1 / -1 al ganar / perder una ronda, 0 en empate, y
  +10 / -10 al ganar / perder la partida. Las acciones de preparación dan 0.
* **Objetivo**: dejar al rival en 0 vidas (o tener más vidas al final).
  Estados terminales: alguna vida en 0 o ronda > max_rondas.

Interfaz principal (estilo *gym*, pero para dos jugadores)
----------------------------------------------------------
    env = MagicChessEnv(ConfiguracionJuego(), semilla=7)
    obs = env.reset()
    while not env.es_terminal():
        for j in env.jugadores_pendientes():
            accion = agente[j].actuar(env, j)
            obs, recompensa, terminado, info = env.step(j, accion)

Métodos útiles para los algoritmos: ``acciones_legales``, ``clonar``,
``poder_equipo``, ``evaluar_estado``. Para la GUI: ``snapshot``. Para el
editor de instancias: ``fijar_oro``, ``fijar_vidas``, ``agregar_unidad``,
``fijar_tienda``, ``vaciar_jugador``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from utils import (
    COPIAS_PARA_COMBINAR,
    ESTRELLAS_MAX,
    MULT_ESTRELLAS,
    PESO_VENTAJA,
    Elemento,
    bono_sinergia,
    calcular_interes,
    crear_rng,
    ventaja_elemental,
)


# ===========================================================================
# 1. Catálogo de unidades
# ===========================================================================
@dataclass(frozen=True)
class PlantillaUnidad:
    """Definición *fija* de un tipo de unidad (lo que aparece en la tienda).

    Atributos
    ---------
    nombre : identificador único del tipo de unidad.
    elemento : agua, fuego o planta; determina sinergias y ventajas.
    costo : oro que cuesta comprarla (1, 2 o 3). También define su "tier":
        las unidades caras aparecen con menor probabilidad al inicio.
    poder_base : poder de la unidad con 1 estrella.

    Es ``frozen`` (inmutable) porque varias unidades comparten la misma
    plantilla y nunca debe cambiar durante la partida.
    """

    nombre: str
    elemento: Elemento
    costo: int
    poder_base: float


#: Catálogo reducido: 3 unidades por elemento, una por cada costo.
#: Las unidades de un mismo costo son simétricas entre elementos para que
#: ninguna composición sea dominante por sí sola: lo que decide es la
#: economía, las combinaciones y la ventaja frente al rival.
CATALOGO: Dict[str, PlantillaUnidad] = {
    p.nombre: p
    for p in [
        PlantillaUnidad("Gota", Elemento.AGUA, 1, 3.0),
        PlantillaUnidad("Marea", Elemento.AGUA, 2, 5.0),
        PlantillaUnidad("Leviatan", Elemento.AGUA, 3, 8.0),
        PlantillaUnidad("Chispa", Elemento.FUEGO, 1, 3.0),
        PlantillaUnidad("Brasa", Elemento.FUEGO, 2, 5.0),
        PlantillaUnidad("Fenix", Elemento.FUEGO, 3, 8.0),
        PlantillaUnidad("Brote", Elemento.PLANTA, 1, 3.0),
        PlantillaUnidad("Espino", Elemento.PLANTA, 2, 5.0),
        PlantillaUnidad("Ent", Elemento.PLANTA, 3, 8.0),
    ]
}


@dataclass
class Unidad:
    """Una ficha concreta que posee un jugador (instancia de una plantilla).

    Atributos
    ---------
    uid : identificador único dentro de la partida. Sirve a la GUI para
        seguir la misma ficha cuando se mueve entre banca y tablero.
    nombre : nombre de su plantilla en ``CATALOGO``.
    estrellas : nivel (1 a 3). Sube al combinar 3 copias iguales.
    """

    uid: int
    nombre: str
    estrellas: int = 1

    @property
    def plantilla(self) -> PlantillaUnidad:
        return CATALOGO[self.nombre]

    @property
    def elemento(self) -> Elemento:
        return self.plantilla.elemento

    @property
    def costo(self) -> int:
        return self.plantilla.costo

    @property
    def poder(self) -> float:
        """Poder individual = poder_base x multiplicador por estrellas."""
        return self.plantilla.poder_base * MULT_ESTRELLAS[self.estrellas]

    @property
    def valor_venta(self) -> int:
        """Oro que devuelve al venderla: el costo de todas las copias de
        1 estrella que se usaron para formarla (3^(estrellas-1))."""
        return self.costo * COPIAS_PARA_COMBINAR ** (self.estrellas - 1)

    def clave(self) -> Tuple[str, int]:
        """(nombre, estrellas): dos unidades con la misma clave son
        intercambiables. Se usa para combinar y para eliminar acciones
        duplicadas."""
        return (self.nombre, self.estrellas)

    def como_dict(self) -> Dict[str, Any]:
        return {
            "uid": self.uid,
            "nombre": self.nombre,
            "estrellas": self.estrellas,
            "elemento": self.elemento.value,
            "costo": self.costo,
            "poder": self.poder,
        }


# ===========================================================================
# 2. Configuración (parámetros de la instancia y dificultad)
# ===========================================================================
@dataclass
class ConfiguracionJuego:
    """Parámetros que definen una instancia del juego.

    Todos se pueden modificar desde la GUI (generador / editor de
    instancias). ``desde_dificultad`` crea configuraciones predefinidas.
    """

    vidas_iniciales: int = 5          # vidas de cada jugador
    oro_inicial: int = 5              # oro en la ronda 1
    ingreso_base: int = 5             # oro que se recibe al inicio de cada ronda
    paso_interes: int = 10            # +1 de oro por cada `paso_interes` ahorrado
    tope_interes: int = 3             # interés máximo por ronda
    bono_victoria: int = 1            # oro extra por ganar la ronda anterior
    costo_renovar: int = 2            # costo de renovar la tienda
    tamano_tienda: int = 3            # unidades ofrecidas por la tienda
    tamano_banca: int = 6             # capacidad de la banca
    tablero_inicial: int = 2          # casillas de tablero en la ronda 1
    tablero_crece_cada: int = 2       # +1 casilla cada N rondas
    tablero_maximo: int = 6           # tope de casillas del tablero
    max_rondas: int = 30              # límite de rondas (evita partidas infinitas)
    danio_por_derrota: int = 1        # vidas que pierde quien pierde la ronda
    ruido_combate: float = 0.0        # 0 = combate determinista; 0.1 = +-10 %
    #: Probabilidad de que la tienda ofrezca una unidad de costo 1, 2 o 3,
    #: según la ronda. Cada entrada: (ronda_desde, (p1, p2, p3)).
    probabilidades_tienda: List[Tuple[int, Tuple[float, float, float]]] = field(
        default_factory=lambda: [
            (1, (0.75, 0.25, 0.00)),
            (4, (0.50, 0.35, 0.15)),
            (8, (0.30, 0.40, 0.30)),
        ]
    )

    @classmethod
    def desde_dificultad(cls, nivel: str) -> "ConfiguracionJuego":
        """Configuraciones predefinidas.

        * ``facil``: más oro y combate determinista (las decisiones se notan
          claramente).
        * ``normal``: valores por defecto con un poco de ruido.
        * ``dificil``: menos oro, renovar más caro, menos vidas y más ruido:
          equivocarse en la economía se castiga más.
        """
        nivel = nivel.lower()
        if nivel == "facil":
            return cls(oro_inicial=8, ingreso_base=6, ruido_combate=0.0)
        if nivel == "dificil":
            return cls(vidas_iniciales=3, oro_inicial=4, ingreso_base=4,
                       costo_renovar=3, ruido_combate=0.15)
        return cls(ruido_combate=0.05)

    def capacidad_tablero(self, ronda: int) -> int:
        """Casillas de tablero disponibles en una ronda (crece con el tiempo)."""
        extra = (ronda - 1) // max(1, self.tablero_crece_cada)
        return min(self.tablero_maximo, self.tablero_inicial + extra)

    def probabilidades_por_costo(self, ronda: int) -> Tuple[float, float, float]:
        """Probabilidades de la tienda vigentes en ``ronda``."""
        vigente = self.probabilidades_tienda[0][1]
        for desde, probs in self.probabilidades_tienda:
            if ronda >= desde:
                vigente = probs
        return vigente


# ===========================================================================
# 3. Estado de cada jugador
# ===========================================================================
@dataclass
class EstadoJugador:
    """Todo lo que describe a un jugador en un instante.

    Atributos
    ---------
    oro : recurso para comprar y renovar.
    vidas : al llegar a 0 el jugador pierde la partida.
    banca : unidades compradas que NO combaten (almacén).
    tablero : unidades desplegadas; son las únicas que cuentan en combate.
    tienda : nombres de las unidades ofrecidas; ``None`` = casilla ya comprada.
    listo : True cuando el jugador terminó su fase de preparación.
    gano_ultima : resultado de la ronda anterior (para el bono de victoria).
    victorias / derrotas / empates : estadísticas de rondas.
    oro_gastado / oro_ganado : estadísticas para medir aprovechamiento del oro.
    """

    oro: int
    vidas: int
    banca: List[Unidad] = field(default_factory=list)
    tablero: List[Unidad] = field(default_factory=list)
    tienda: List[Optional[str]] = field(default_factory=list)
    listo: bool = False
    gano_ultima: bool = False
    victorias: int = 0
    derrotas: int = 0
    empates: int = 0
    oro_gastado: int = 0
    oro_ganado: int = 0

    def todas_las_unidades(self) -> List[Unidad]:
        return self.banca + self.tablero

    def contar_copias(self) -> Dict[Tuple[str, int], int]:
        """Cuántas unidades hay de cada (nombre, estrellas) entre banca y
        tablero. Determina qué combinaciones son posibles."""
        conteo: Dict[Tuple[str, int], int] = {}
        for u in self.todas_las_unidades():
            conteo[u.clave()] = conteo.get(u.clave(), 0) + 1
        return conteo

    def como_dict(self) -> Dict[str, Any]:
        return {
            "oro": self.oro,
            "vidas": self.vidas,
            "banca": [u.como_dict() for u in self.banca],
            "tablero": [u.como_dict() for u in self.tablero],
            "tienda": list(self.tienda),
            "listo": self.listo,
            "victorias": self.victorias,
            "derrotas": self.derrotas,
            "empates": self.empates,
            "oro_gastado": self.oro_gastado,
            "oro_ganado": self.oro_ganado,
        }


# ===========================================================================
# 4. Acciones
# ===========================================================================
class TipoAccion(str, Enum):
    """Tipos de decisión que puede tomar un jugador en la preparación."""

    COMPRAR = "comprar"        # arg: índice en la tienda
    RENOVAR = "renovar"        # sin argumento
    COMBINAR = "combinar"      # arg: (nombre, estrellas)
    A_TABLERO = "a_tablero"    # arg: índice en la banca
    A_BANCA = "a_banca"        # arg: índice en el tablero
    VENDER = "vender"          # arg: ("banca" | "tablero", índice)
    TERMINAR = "terminar"      # sin argumento: listo para el combate


@dataclass(frozen=True)
class Accion:
    """Una acción concreta = tipo + argumento.

    Es inmutable y *hashable* para poder guardarla en conjuntos/diccionarios
    (útil en búsqueda y en Q-learning).
    """

    tipo: TipoAccion
    arg: Any = None

    def es_terminar(self) -> bool:
        return self.tipo == TipoAccion.TERMINAR

    def __str__(self) -> str:
        return self.tipo.value if self.arg is None else f"{self.tipo.value}({self.arg})"


# ===========================================================================
# 5. Resultado de combate
# ===========================================================================
@dataclass
class DesglosePoder:
    """Cómo se obtuvo el poder de un equipo (para mostrarlo en la GUI y para
    las heurísticas).

    poder = (base + sinergia) * factor_ventaja * ruido
    """

    base: float = 0.0
    sinergia: float = 0.0
    factor_ventaja: float = 1.0
    ruido: float = 1.0
    sinergias_activas: Dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> float:
        return (self.base + self.sinergia) * self.factor_ventaja * self.ruido

    def como_dict(self) -> Dict[str, Any]:
        return {
            "base": self.base,
            "sinergia": self.sinergia,
            "factor_ventaja": self.factor_ventaja,
            "ruido": self.ruido,
            "sinergias_activas": dict(self.sinergias_activas),
            "total": self.total,
        }


# ===========================================================================
# 6. Ambiente
# ===========================================================================
EMPATE = -1  # valor de ``ganador`` cuando la partida termina igualada


class MagicChessEnv:
    """Ambiente de dos jugadores (índices 0 y 1).

    Atributos públicos
    ------------------
    config : ConfiguracionJuego con los parámetros de la instancia.
    jugadores : lista con los dos ``EstadoJugador``.
    ronda : número de ronda actual (empieza en 1).
    fase : "preparacion" o "terminado".
    ganador : None mientras se juega; 0, 1 o EMPATE al terminar.
    ultimo_combate : diccionario con el desglose del último combate.
    registro : lista de mensajes legibles (lo que pasó en la partida).
    pasos : número de acciones ejecutadas (contador global).
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
        """Reinicia la partida a la ronda 1 y devuelve la observación del
        jugador 0. Con la misma semilla se obtiene la misma partida (si los
        agentes también son deterministas)."""
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
        self.jugadores = [
            EstadoJugador(oro=cfg.oro_inicial, vidas=cfg.vidas_iniciales)
            for _ in range(2)
        ]
        for j in range(2):
            self.jugadores[j].oro_ganado = cfg.oro_inicial
            self._generar_tienda(j)
        self._log(f"Ronda 1: comienza la partida ({cfg.oro_inicial} de oro c/u).")
        return self.observacion(0)

    def clonar(self) -> "MagicChessEnv":
        """Copia profunda e independiente del ambiente.

        Es la base de los algoritmos de búsqueda: se clona, se simulan
        acciones sobre la copia y el ambiente real no cambia. El generador
        aleatorio también se copia, así la copia reproduce las mismas tiendas
        que vería el original.
        """
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
        """Jugadores que aún no han terminado su preparación."""
        if self.es_terminal():
            return []
        return [j for j in range(2) if not self.jugadores[j].listo]

    def acciones_legales(self, jugador: int,
                         sin_duplicados: bool = True) -> List[Accion]:
        """Lista de acciones válidas para ``jugador`` en el estado actual.

        Restricciones aplicadas (oro, espacio y nivel):
        * COMPRAR i: la casilla i tiene unidad, alcanza el oro y hay espacio
          en la banca.
        * RENOVAR: alcanza el oro.
        * COMBINAR (n, e): hay >= 3 copias de (n, e) y e < 3.
        * A_TABLERO i: hay casillas libres en el tablero.
        * A_BANCA i: hay espacio libre en la banca.
        * VENDER: siempre que exista la unidad.
        * TERMINAR: siempre.

        Si ``sin_duplicados`` es True, las acciones sobre unidades idénticas
        (mismo nombre y nivel) se cuentan una sola vez. Esto reduce mucho el
        factor de ramificación sin perder soluciones, porque esas acciones
        llevan a estados equivalentes.
        """
        if self.es_terminal() or self.jugadores[jugador].listo:
            return []
        j = self.jugadores[jugador]
        cfg = self.config
        acciones: List[Accion] = []

        # Comprar
        vistos = set()
        if len(j.banca) < cfg.tamano_banca:
            for i, nombre in enumerate(j.tienda):
                if nombre is None or CATALOGO[nombre].costo > j.oro:
                    continue
                if sin_duplicados and nombre in vistos:
                    continue
                vistos.add(nombre)
                acciones.append(Accion(TipoAccion.COMPRAR, i))

        # Renovar
        if j.oro >= cfg.costo_renovar:
            acciones.append(Accion(TipoAccion.RENOVAR))

        # Combinar
        for (nombre, estrellas), n in sorted(j.contar_copias().items()):
            if n >= COPIAS_PARA_COMBINAR and estrellas < ESTRELLAS_MAX:
                acciones.append(Accion(TipoAccion.COMBINAR, (nombre, estrellas)))

        # Mover banca -> tablero
        if len(j.tablero) < self.capacidad_tablero():
            vistos = set()
            for i, u in enumerate(j.banca):
                if sin_duplicados and u.clave() in vistos:
                    continue
                vistos.add(u.clave())
                acciones.append(Accion(TipoAccion.A_TABLERO, i))

        # Mover tablero -> banca
        if len(j.banca) < cfg.tamano_banca:
            vistos = set()
            for i, u in enumerate(j.tablero):
                if sin_duplicados and u.clave() in vistos:
                    continue
                vistos.add(u.clave())
                acciones.append(Accion(TipoAccion.A_BANCA, i))

        # Vender
        for zona in ("banca", "tablero"):
            vistos = set()
            for i, u in enumerate(getattr(j, zona)):
                if sin_duplicados and u.clave() in vistos:
                    continue
                vistos.add(u.clave())
                acciones.append(Accion(TipoAccion.VENDER, (zona, i)))

        acciones.append(Accion(TipoAccion.TERMINAR))
        return acciones

    def es_legal(self, jugador: int, accion: Accion) -> bool:
        return accion in self.acciones_legales(jugador, sin_duplicados=False)

    # ------------------------------------------------------------------
    # Transición
    # ------------------------------------------------------------------
    def step(self, jugador: int, accion: Accion
             ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Aplica ``accion`` del ``jugador`` (función de transición).

        Devuelve ``(observacion, recompensa, terminado, info)``:
        * ``recompensa`` es la del jugador que actuó.
        * ``info["recompensas"]`` trae la recompensa de ambos jugadores (útil
          porque el combate se resuelve cuando actúa el segundo jugador, y
          el primero también debe recibir su recompensa).
        * ``info["combate"]`` trae el desglose si en este paso hubo combate.

        Lanza ``ValueError`` si la acción no es legal.
        """
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
            costo = CATALOGO[nombre].costo
            j.oro -= costo
            j.oro_gastado += costo
            j.tienda[accion.arg] = None
            j.banca.append(self._nueva_unidad(nombre))
            self._log(f"J{jugador} compra {nombre} (-{costo} oro).")

        elif t == TipoAccion.RENOVAR:
            j.oro -= cfg.costo_renovar
            j.oro_gastado += cfg.costo_renovar
            self._generar_tienda(jugador)
            self._log(f"J{jugador} renueva la tienda: {', '.join(j.tienda)}.")

        elif t == TipoAccion.COMBINAR:
            nombre, estrellas = accion.arg
            self._combinar(j, nombre, estrellas)
            self._log(f"J{jugador} combina 3x {nombre}{'*' * estrellas} "
                      f"-> {nombre}{'*' * (estrellas + 1)}.")

        elif t == TipoAccion.A_TABLERO:
            u = j.banca.pop(accion.arg)
            j.tablero.append(u)
            self._log(f"J{jugador} despliega {u.nombre}{'*' * u.estrellas}.")

        elif t == TipoAccion.A_BANCA:
            u = j.tablero.pop(accion.arg)
            j.banca.append(u)
            self._log(f"J{jugador} retira {u.nombre}{'*' * u.estrellas} a la banca.")

        elif t == TipoAccion.VENDER:
            zona, i = accion.arg
            u = getattr(j, zona).pop(i)
            j.oro += u.valor_venta
            j.oro_ganado += u.valor_venta
            self._log(f"J{jugador} vende {u.nombre}{'*' * u.estrellas} "
                      f"(+{u.valor_venta} oro).")

        elif t == TipoAccion.TERMINAR:
            j.listo = True
            self._log(f"J{jugador} termina su preparación.")
            if all(p.listo for p in self.jugadores):
                info["combate"] = self._resolver_combate(recompensas)

        info["recompensas"] = recompensas
        terminado = self.es_terminal()
        return self.observacion(jugador), recompensas[jugador], terminado, info

    # ------------------------------------------------------------------
    # Cálculo de poder (también sirve como heurística)
    # ------------------------------------------------------------------
    def poder_equipo(self, jugador: int,
                     tablero_rival: Optional[List[Unidad]] = None,
                     tablero_propio: Optional[List[Unidad]] = None
                     ) -> DesglosePoder:
        """Poder del tablero de ``jugador`` contra el tablero rival (sin
        ruido).

        Se pueden pasar tableros hipotéticos para evaluar estados sin
        modificar el ambiente (útil en heurísticas).

        1. base: suma del poder individual de cada unidad desplegada.
        2. sinergia: por cada elemento con >= 2 unidades, se suma un % del
           poder de esas unidades (ver ``utils.SINERGIAS``).
        3. factor de ventaja: promedio de ventaja elemental de cada par
           (unidad propia, unidad rival) en [-1, 1], escalado por
           ``utils.PESO_VENTAJA``.
        """
        propio = self.jugadores[jugador].tablero if tablero_propio is None else tablero_propio
        rival = (self.jugadores[self.rival(jugador)].tablero
                 if tablero_rival is None else tablero_rival)
        d = DesglosePoder()
        if not propio:
            return d

        d.base = sum(u.poder for u in propio)

        por_elemento: Dict[Elemento, List[Unidad]] = {}
        for u in propio:
            por_elemento.setdefault(u.elemento, []).append(u)
        for elem, unidades in por_elemento.items():
            bono = bono_sinergia(len(unidades))
            if bono > 0:
                d.sinergia += bono * sum(u.poder for u in unidades)
                d.sinergias_activas[elem.value] = len(unidades)

        if rival:
            suma = sum(ventaja_elemental(a.elemento, b.elemento)
                       for a in propio for b in rival)
            ventaja = suma / (len(propio) * len(rival))
            d.factor_ventaja = 1.0 + PESO_VENTAJA * ventaja
        return d

    def evaluar_estado(self, jugador: int) -> float:
        """Valor heurístico simple del estado para ``jugador`` (diferencia de
        vidas, poder relativo y oro). Es un punto de partida para que los
        algoritmos definan sus propias heurísticas."""
        if self.es_terminal():
            if self.ganador == jugador:
                return 1e6
            if self.ganador == EMPATE:
                return 0.0
            return -1e6
        yo, otro = self.jugadores[jugador], self.jugadores[self.rival(jugador)]
        poder_yo = self.poder_equipo(jugador).total
        poder_otro = self.poder_equipo(self.rival(jugador)).total
        return (100.0 * (yo.vidas - otro.vidas)
                + (poder_yo - poder_otro)
                + 0.3 * yo.oro)

    # ------------------------------------------------------------------
    # Observaciones y snapshot
    # ------------------------------------------------------------------
    def observacion(self, jugador: int) -> Dict[str, Any]:
        """Lo que percibe ``jugador``: su estado completo y la parte pública
        del rival (vidas, tablero y poder estimado). La banca, la tienda y el
        oro del rival se ocultan (observabilidad parcial)."""
        rival = self.jugadores[self.rival(jugador)]
        return {
            "jugador": jugador,
            "ronda": self.ronda,
            "fase": self.fase,
            "capacidad_tablero": self.capacidad_tablero(),
            "propio": self.jugadores[jugador].como_dict(),
            "rival": {
                "vidas": rival.vidas,
                "tablero": [u.como_dict() for u in rival.tablero],
                "poder_estimado": self.poder_equipo(self.rival(jugador)).base,
            },
        }

    def snapshot(self) -> Dict[str, Any]:
        """Estado COMPLETO como diccionario independiente (solo tipos
        básicos). La GUI dibuja a partir de esto y el ``Historial`` lo guarda
        para poder retroceder."""
        return {
            "ronda": self.ronda,
            "fase": self.fase,
            "ganador": self.ganador,
            "pasos": self.pasos,
            "capacidad_tablero": self.capacidad_tablero(),
            "jugadores": [p.como_dict() for p in self.jugadores],
            "poder": [self.poder_equipo(j).como_dict() for j in range(2)],
            "interes": [calcular_interes(p.oro, self.config.paso_interes,
                                         self.config.tope_interes)
                        for p in self.jugadores],
            "combinables": [
                [list(k) for k, n in p.contar_copias().items()
                 if n >= COPIAS_PARA_COMBINAR and k[1] < ESTRELLAS_MAX]
                for p in self.jugadores
            ],
            "ultimo_combate": copy.deepcopy(self.ultimo_combate),
            "registro": list(self.registro[-12:]),
            "config": {
                "vidas_iniciales": self.config.vidas_iniciales,
                "tamano_banca": self.config.tamano_banca,
                "tamano_tienda": self.config.tamano_tienda,
                "costo_renovar": self.config.costo_renovar,
                "max_rondas": self.config.max_rondas,
            },
        }

    # ------------------------------------------------------------------
    # Editor de instancias
    # ------------------------------------------------------------------
    def fijar_oro(self, jugador: int, oro: int) -> None:
        self.jugadores[jugador].oro = max(0, int(oro))

    def fijar_vidas(self, jugador: int, vidas: int) -> None:
        self.jugadores[jugador].vidas = max(1, int(vidas))

    def fijar_ronda(self, ronda: int) -> None:
        """Cambia la ronda (afecta capacidad del tablero y probabilidades de
        la tienda)."""
        self.ronda = max(1, min(int(ronda), self.config.max_rondas))

    def agregar_unidad(self, jugador: int, nombre: str, estrellas: int = 1,
                       zona: str = "banca") -> bool:
        """Agrega una unidad sin costo (para construir estados iniciales).
        Devuelve False si no hay espacio."""
        if nombre not in CATALOGO or not 1 <= estrellas <= ESTRELLAS_MAX:
            raise ValueError(f"Unidad inválida: {nombre} {estrellas}*")
        j = self.jugadores[jugador]
        if zona == "tablero":
            if len(j.tablero) >= self.capacidad_tablero():
                return False
            j.tablero.append(self._nueva_unidad(nombre, estrellas))
        else:
            if len(j.banca) >= self.config.tamano_banca:
                return False
            j.banca.append(self._nueva_unidad(nombre, estrellas))
        return True

    def fijar_tienda(self, jugador: int, nombres: List[Optional[str]]) -> None:
        """Define manualmente las opciones de la tienda."""
        for n in nombres:
            if n is not None and n not in CATALOGO:
                raise ValueError(f"Unidad desconocida: {n}")
        tam = self.config.tamano_tienda
        self.jugadores[jugador].tienda = (list(nombres) + [None] * tam)[:tam]

    def vaciar_jugador(self, jugador: int) -> None:
        """Quita todas las unidades de banca y tablero."""
        self.jugadores[jugador].banca.clear()
        self.jugadores[jugador].tablero.clear()

    # ------------------------------------------------------------------
    # Lógica interna
    # ------------------------------------------------------------------
    def _log(self, mensaje: str) -> None:
        self.registro.append(mensaje)

    def _nueva_unidad(self, nombre: str, estrellas: int = 1) -> Unidad:
        u = Unidad(uid=self._siguiente_uid, nombre=nombre, estrellas=estrellas)
        self._siguiente_uid += 1
        return u

    def _generar_tienda(self, jugador: int) -> None:
        """Llena la tienda: primero se sortea el costo según la ronda y luego
        una unidad al azar de ese costo."""
        probs = self.config.probabilidades_por_costo(self.ronda)
        por_costo = {c: [p.nombre for p in CATALOGO.values() if p.costo == c]
                     for c in (1, 2, 3)}
        tienda = []
        for _ in range(self.config.tamano_tienda):
            costo = self.rng.choices((1, 2, 3), weights=probs)[0]
            tienda.append(self.rng.choice(por_costo[costo]))
        self.jugadores[jugador].tienda = tienda

    def _combinar(self, j: EstadoJugador, nombre: str, estrellas: int) -> None:
        """Quita 3 copias de (nombre, estrellas) y crea una de nivel +1.

        Se consumen primero las copias de la banca. Si alguna copia estaba en
        el tablero, la unidad nueva ocupa ese lugar en el tablero (así no se
        pierde una casilla desplegada); si no, queda en la banca.
        """
        clave = (nombre, estrellas)
        restantes = COPIAS_PARA_COMBINAR
        indices_banca = [i for i, u in enumerate(j.banca) if u.clave() == clave]
        indices_tablero = [i for i, u in enumerate(j.tablero) if u.clave() == clave]

        quitar_banca = indices_banca[:restantes]
        restantes -= len(quitar_banca)
        quitar_tablero = indices_tablero[:restantes]

        for i in sorted(quitar_banca, reverse=True):
            j.banca.pop(i)
        posicion_tablero = None
        for i in sorted(quitar_tablero, reverse=True):
            j.tablero.pop(i)
            posicion_tablero = i

        nueva = self._nueva_unidad(nombre, estrellas + 1)
        if posicion_tablero is not None:
            j.tablero.insert(min(posicion_tablero, len(j.tablero)), nueva)
        else:
            j.banca.append(nueva)

    def _resolver_combate(self, recompensas: List[float]) -> Dict[str, Any]:
        """Fase de combate: calcula poderes, resta vidas, reparte recompensas
        y prepara la siguiente ronda (o termina la partida)."""
        cfg = self.config
        desgloses = []
        for j in range(2):
            d = self.poder_equipo(j)
            if cfg.ruido_combate > 0:
                d.ruido = self.rng.uniform(1 - cfg.ruido_combate,
                                           1 + cfg.ruido_combate)
            desgloses.append(d)
        p0, p1 = desgloses[0].total, desgloses[1].total

        if abs(p0 - p1) < 1e-9:
            ganador_ronda = EMPATE
            for p in self.jugadores:
                p.empates += 1
                p.gano_ultima = False
            self._log(f"Combate ronda {self.ronda}: empate ({p0:.1f} vs {p1:.1f}).")
        else:
            ganador_ronda = 0 if p0 > p1 else 1
            perdedor = 1 - ganador_ronda
            self.jugadores[ganador_ronda].victorias += 1
            self.jugadores[ganador_ronda].gano_ultima = True
            self.jugadores[perdedor].derrotas += 1
            self.jugadores[perdedor].gano_ultima = False
            self.jugadores[perdedor].vidas -= cfg.danio_por_derrota
            recompensas[ganador_ronda] += self.RECOMPENSA_RONDA
            recompensas[perdedor] -= self.RECOMPENSA_RONDA
            self._log(f"Combate ronda {self.ronda}: gana J{ganador_ronda} "
                      f"({p0:.1f} vs {p1:.1f}). J{perdedor} pierde "
                      f"{cfg.danio_por_derrota} vida(s).")

        resultado = {
            "ronda": self.ronda,
            "ganador_ronda": ganador_ronda,
            "desglose": [d.como_dict() for d in desgloses],
            "tableros": [[u.como_dict() for u in p.tablero] for p in self.jugadores],
        }
        self.ultimo_combate = resultado

        # ¿Terminó la partida?
        muertos = [j for j in range(2) if self.jugadores[j].vidas <= 0]
        if muertos or self.ronda >= cfg.max_rondas:
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
        """Nueva ronda: ingresos (base + interés + bono), tienda nueva y
        jugadores vuelven a la fase de preparación."""
        cfg = self.config
        self.ronda += 1
        for idx, p in enumerate(self.jugadores):
            interes = calcular_interes(p.oro, cfg.paso_interes, cfg.tope_interes)
            bono = cfg.bono_victoria if p.gano_ultima else 0
            ingreso = cfg.ingreso_base + interes + bono
            p.oro += ingreso
            p.oro_ganado += ingreso
            p.listo = False
            self._generar_tienda(idx)
        self._log(f"Ronda {self.ronda}: +{cfg.ingreso_base} oro base "
                  f"(+interés y bono). Tablero: {self.capacidad_tablero()} casillas.")


# ===========================================================================
# Prueba rápida desde consola: python magic_chess_env.py
# ===========================================================================
if __name__ == "__main__":
    from utils import AgenteAleatorio

    env = MagicChessEnv(ConfiguracionJuego.desde_dificultad("normal"), semilla=1)
    agentes = [AgenteAleatorio(semilla=10), AgenteAleatorio(semilla=20)]
    while not env.es_terminal():
        for jug in env.jugadores_pendientes():
            while not env.jugadores[jug].listo and not env.es_terminal():
                env.step(jug, agentes[jug].actuar(env, jug))
    print("\n".join(env.registro[-6:]))
    print(f"Ganador: {env.ganador} | rondas: {env.ronda} | pasos: {env.pasos}")
