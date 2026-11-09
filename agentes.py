"""
agentes.py
==========

Algoritmos y agentes para *Magic Chess*.

Idea central: el problema de planificación de un turno
------------------------------------------------------
Una partida completa es un juego estocástico y adversario: no existe una
"solución" fija. Lo que sí se puede resolver de forma ÓPTIMA es la fase de
preparación de cada ronda, modelada como un problema de búsqueda:

* **Estado** (``EstadoTurno``): oro, unidades en banca, unidades en tablero y
  opciones de la tienda del jugador.
* **Acciones**: comprar(nombre), combinar(nombre, nivel), a_tablero(unidad),
  a_banca(unidad) y terminar. Se excluyen RENOVAR (su resultado es aleatorio:
  el agente no puede saber qué saldrá) y VENDER (ver nota en
  ``ProblemaTurno``); esas decisiones las toma una política externa.
* **Transición**: determinista (mismas reglas del ambiente).
* **Objetivo**: terminar el turno en el estado ``s`` que maximiza la función
  de utilidad

      V(s) = P(s) + w_interes * interes(oro) + w_oro * oro + w_banca * B(s)

  donde P(s) es el poder del tablero contra el tablero actual del rival
  (``MagicChessEnv.poder_equipo``) y B(s) el poder guardado en la banca
  (potencial para rondas futuras). Los pesos están en ``PesosEvaluacion``.

Como cualquier estado puede ser final (siempre se puede "terminar"), la
solución óptima es el camino hacia el estado alcanzable de mayor V.

Algoritmos de búsqueda (todos devuelven ``ResultadoBusqueda`` con métricas)
--------------------------------------------------------------------------
1. ``busqueda_anchura`` (BFS) - no informada. Explora todos los estados
   alcanzables por niveles. Completa y óptima respecto a V (si no se corta
   por el límite de nodos). Tiempo y memoria O(b^d).
2. ``busqueda_profundidad_limitada`` (DLS) - no informada. Profundidad
   máxima L. Memoria O(b*L). Óptima solo si el mejor estado está a
   profundidad <= L.
3. ``busqueda_a_estrella`` (A* para maximización / Branch & Bound) -
   informada. f(n) = g(n) + h(n), con g(n) = V(n) y h(n) una estimación
   OPTIMISTA de cuánto puede mejorar V desde n. Como h nunca subestima
   (admisible), cuando la mayor f de la frontera es <= al mejor V encontrado,
   la solución es óptima y se puede parar sin explorar todo.
   Dos heurísticas:
     - ``h1`` "por unidades": supone para todo el equipo la mejor ventaja
       elemental posible y la sinergia máxima de cada elemento.
     - ``h2`` "relajada": resuelve exactamente un problema relajado (las
       fichas se pueden reubicar gratis) para cada combinación de compras.
       Más informada -> muchos menos nodos expandidos.
4. ``busqueda_voraz`` - informada, ascenso de colina con anticipación de 2
   acciones. Muy rápida, sin garantía de optimalidad.

Agentes (método ``actuar(env, jugador) -> Accion``)
---------------------------------------------------
* ``AgenteBusqueda``: planifica el turno con uno de los algoritmos anteriores
  y ejecuta el plan acción por acción.
* ``AgenteReglas``: estrategia heurística basada en reglas (primera etapa de
  la propuesta). Sirve como referencia.
* ``AgenteQLearning``: aprendizaje por refuerzo tabular. Con un estado
  simplificado elige, en cada ronda, un *perfil* de juego (agresivo,
  equilibrado, ahorro, buscar combinaciones) que ejecuta ``AgenteReglas``.

Experimentos
------------
* ``comparar_busquedas(env, jugador)``: corre todos los algoritmos sobre la
  MISMA instancia y devuelve sus métricas (incluye si fueron óptimos).
* ``torneo(fabrica_a, fabrica_b, partidas)``: partidas completas entre dos
  agentes (porcentaje de victorias, rondas, vidas, uso del oro).
* ``entrenar_qlearning(...)``: entrena y guarda la tabla Q.

Desde consola::

    python agentes.py comparar        # tabla de algoritmos en una instancia
    python agentes.py torneo          # todos contra AgenteReglas
    python agentes.py entrenar 3000   # entrena Q-learning y guarda qtabla.json
"""

from __future__ import annotations

import heapq
from itertools import combinations
import json
import os
import sys
from collections import Counter, deque
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Tuple

from magic_chess_env import (
    CATALOGO,
    EMPATE,
    Accion,
    ConfiguracionJuego,
    MagicChessEnv,
    TipoAccion,
    Unidad,
)
from utils import (
    COPIAS_PARA_COMBINAR,
    ESTRELLAS_MAX,
    MULT_ESTRELLAS,
    PESO_VENTAJA,
    AgenteAleatorio,
    Cronometro,
    MetricasBusqueda,
    bono_sinergia,
    calcular_interes,
    crear_rng,
    ventaja_elemental,
)

Clave = Tuple[str, int]          # (nombre, estrellas) identifica una unidad
AccionPlan = Tuple[str, Any]     # acción del modelo de planificación

MAX_TRAZA = 400                  # eventos de búsqueda guardados para la GUI


# ===========================================================================
# 1. Modelo de planificación de un turno
# ===========================================================================
@dataclass(frozen=True)
class PesosEvaluacion:
    """Pesos de la función de utilidad V(s).

    * interes: valor de cada punto de interés que generará el oro guardado.
    * oro: valor de cada moneda guardada (flexibilidad futura).
    * banca: fracción del poder de la banca que se valora (potencial futuro).
    """

    interes: float = 2.0
    oro: float = 0.15
    banca: float = 0.25


@dataclass(frozen=True)
class EstadoTurno:
    """Estado del problema de planificación (inmutable y *hashable*).

    banca y tablero se guardan ORDENADOS: dos estados con las mismas
    unidades en distinto orden son equivalentes, así se evita explorarlos
    dos veces.
    """

    oro: int
    banca: Tuple[Clave, ...]
    tablero: Tuple[Clave, ...]
    tienda: Tuple[Optional[str], ...]

    def clave(self) -> tuple:
        return (self.oro, self.banca, self.tablero,
                tuple(sorted(t for t in self.tienda if t)))

    def __str__(self) -> str:
        def f(us):
            return ",".join(f"{n}{'*' * e}" for n, e in us) or "-"
        return f"oro={self.oro} tab=[{f(self.tablero)}] banca=[{f(self.banca)}]"


def _quitar(tupla: Tuple, elem) -> Tuple:
    lista = list(tupla)
    lista.remove(elem)
    return tuple(lista)


def _agregar(tupla: Tuple, elem) -> Tuple:
    return tuple(sorted(tupla + (elem,)))


class ProblemaTurno:
    """Formulación del turno de ``jugador`` como problema de búsqueda.

    Se construye a partir del ambiente real pero NO lo modifica: las
    transiciones se calculan sobre ``EstadoTurno`` (mucho más liviano que
    clonar el ambiente entero) replicando sus reglas.

    Nota sobre VENDER: sin vender, el oro nunca aumenta durante el turno.
    Esa propiedad hace que la cota optimista del término económico sea
    simplemente el valor actual, lo que mantiene admisibles las heurísticas.
    """

    def __init__(self, env: MagicChessEnv, jugador: int,
                 pesos: Optional[PesosEvaluacion] = None) -> None:
        self.env = env
        self.jugador = jugador
        self.pesos = pesos or PesosEvaluacion()
        j = env.jugadores[jugador]
        cfg = env.config
        self.cap_tablero = env.capacidad_tablero()
        self.tam_banca = cfg.tamano_banca
        self.paso_interes = cfg.paso_interes
        self.tope_interes = cfg.tope_interes
        self.tablero_rival = [Unidad(0, u.nombre, u.estrellas)
                              for u in env.jugadores[env.rival(jugador)].tablero]
        self.estado_inicial = EstadoTurno(
            oro=j.oro,
            banca=tuple(sorted(u.clave() for u in j.banca)),
            tablero=tuple(sorted(u.clave() for u in j.tablero)),
            tienda=tuple(j.tienda),
        )
        self._cache_valor: Dict[tuple, float] = {}
        self._cache_poder: Dict[tuple, float] = {}
        self._cache_cota: Dict[tuple, float] = {}
        self._cache_reparto: Dict[tuple, float] = {}
        # ventaja promedio de cada elemento contra el tablero rival (para h2)
        self._ventaja_elem = {}
        for p in CATALOGO.values():
            if self.tablero_rival:
                self._ventaja_elem[p.elemento] = sum(
                    ventaja_elemental(p.elemento, r.elemento)
                    for r in self.tablero_rival) / len(self.tablero_rival)
            else:
                self._ventaja_elem[p.elemento] = 0.0

    # -------------------------- acciones -----------------------------
    def acciones(self, s: EstadoTurno) -> List[AccionPlan]:
        """Acciones aplicables en ``s`` (sin duplicados por unidades
        idénticas). TERMINAR no se incluye: todo estado es candidato final."""
        acc: List[AccionPlan] = []
        if len(s.banca) < self.tam_banca:
            for nombre in sorted({t for t in s.tienda if t}):
                if CATALOGO[nombre].costo <= s.oro:
                    acc.append(("comprar", nombre))
        conteo = Counter(s.banca + s.tablero)
        for clave, n in sorted(conteo.items()):
            if n >= COPIAS_PARA_COMBINAR and clave[1] < ESTRELLAS_MAX:
                acc.append(("combinar", clave))
        if len(s.tablero) < self.cap_tablero:
            for clave in sorted(set(s.banca)):
                acc.append(("a_tablero", clave))
        if len(s.banca) < self.tam_banca:
            for clave in sorted(set(s.tablero)):
                acc.append(("a_banca", clave))
        return acc

    def resultado(self, s: EstadoTurno, a: AccionPlan) -> EstadoTurno:
        """Función de transición del modelo (mismas reglas que el ambiente)."""
        tipo, arg = a
        if tipo == "comprar":
            tienda = list(s.tienda)
            tienda[tienda.index(arg)] = None
            return replace(s, oro=s.oro - CATALOGO[arg].costo,
                           banca=_agregar(s.banca, (arg, 1)), tienda=tuple(tienda))
        if tipo == "combinar":
            # Igual que MagicChessEnv._combinar: consume primero la banca; si
            # usa alguna copia del tablero, la unidad nueva queda en el tablero.
            banca, tablero = list(s.banca), list(s.tablero)
            restantes = COPIAS_PARA_COMBINAR
            while restantes and arg in banca:
                banca.remove(arg)
                restantes -= 1
            usa_tablero = restantes > 0
            while restantes:
                tablero.remove(arg)
                restantes -= 1
            nueva = (arg[0], arg[1] + 1)
            (tablero if usa_tablero else banca).append(nueva)
            return replace(s, banca=tuple(sorted(banca)), tablero=tuple(sorted(tablero)))
        if tipo == "a_tablero":
            return replace(s, banca=_quitar(s.banca, arg), tablero=_agregar(s.tablero, arg))
        if tipo == "a_banca":
            return replace(s, tablero=_quitar(s.tablero, arg), banca=_agregar(s.banca, arg))
        raise ValueError(a)

    # -------------------------- evaluación ---------------------------
    @staticmethod
    def _poder_individual(clave: Clave) -> float:
        return CATALOGO[clave[0]].poder_base * MULT_ESTRELLAS[clave[1]]

    def poder_tablero(self, tablero: Tuple[Clave, ...]) -> float:
        if tablero not in self._cache_poder:
            unidades = [Unidad(0, n, e) for n, e in tablero]
            self._cache_poder[tablero] = self.env.poder_equipo(
                self.jugador, tablero_rival=self.tablero_rival,
                tablero_propio=unidades).total
        return self._cache_poder[tablero]

    def termino_economico(self, oro: int) -> float:
        return (self.pesos.interes * calcular_interes(oro, self.paso_interes,
                                                      self.tope_interes)
                + self.pesos.oro * oro)

    def valor(self, s: EstadoTurno) -> float:
        """Utilidad V(s) de terminar el turno en ``s``."""
        k = (s.oro, s.banca, s.tablero)
        if k not in self._cache_valor:
            banca = sum(self._poder_individual(c) for c in s.banca)
            self._cache_valor[k] = (self.poder_tablero(s.tablero)
                                    + self.termino_economico(s.oro)
                                    + self.pesos.banca * banca)
        return self._cache_valor[k]

    # -------------------------- heurísticas --------------------------
    @staticmethod
    def _alternativas_combinacion(conteo: Counter, maximo: int = 64
                                  ) -> List[List[Clave]]:
        """Todas las formas de dejar las unidades de ``conteo``: para cada
        grupo de 3 copias iguales se considera combinarlo o no. Así la cota
        cubre tanto los estados donde se combinó como donde no."""
        resultados: List[List[Clave]] = []
        vistos = set()
        pila = [Counter(conteo)]
        while pila and len(resultados) < maximo:
            c = pila.pop()
            firma = tuple(sorted(c.items()))
            if firma in vistos:
                continue
            vistos.add(firma)
            resultados.append(list(c.elements()))
            for (n, e), k in c.items():
                if k >= COPIAS_PARA_COMBINAR and e < ESTRELLAS_MAX:
                    nuevo = Counter(c)
                    nuevo[(n, e)] -= COPIAS_PARA_COMBINAR
                    nuevo[(n, e + 1)] += 1
                    pila.append(+nuevo)
        return resultados

    def _cota_unidades(self, unidades: List[Clave]) -> float:
        """Cota rápida de P(tablero) + w_banca * B(banca) con ``unidades``
        (usada por h1).

        Cada unidad recibe la sinergia que alcanzaría su elemento si todas
        las unidades de ese elemento estuvieran desplegadas, y el equipo la
        mejor ventaja elemental de cualquiera de sus unidades frente al
        tablero rival. Luego se ponen en el tablero las ``cap_tablero`` de
        mayor poder potenciado y el resto en la banca. No calcula repartos
        exactos, por eso es más barata pero menos ajustada que h2.
        """
        por_elem = Counter(CATALOGO[n].elemento for n, _ in unidades)
        mult = {el: 1.0 + bono_sinergia(min(self.cap_tablero, k))
                for el, k in por_elem.items()}
        factor = 1.0 + PESO_VENTAJA * max(
            (self._ventaja_elem[CATALOGO[n].elemento] for n, _ in unidades),
            default=0.0)
        en_tablero = sorted(
            ((self._poder_individual(c) * mult[CATALOGO[c[0]].elemento] * factor,
              self._poder_individual(c)) for c in unidades), reverse=True)
        top = sum(p for p, _ in en_tablero[:self.cap_tablero])
        resto = sum(b for _, b in en_tablero[self.cap_tablero:])
        # una unidad nunca vale menos en el tablero que en la banca en la
        # cota, salvo que w_banca sea enorme: se cubre ese caso también
        top = max(top, self.pesos.banca * sum(b for _, b in en_tablero[:self.cap_tablero]))
        return top + self.pesos.banca * resto

    def _mejor_reparto(self, unidades: Tuple[Clave, ...]) -> float:
        """Máximo exacto de P(tablero) + w_banca * B(banca) repartiendo
        LIBREMENTE ``unidades`` entre tablero (<= cap_tablero) y banca.

        Es la solución de un problema *relajado*: se ignora que mover fichas
        requiere acciones y que la banca tiene capacidad. Toda solución del
        problema real es solución del relajado, por eso su valor es una cota
        superior (heurística admisible). Se guarda en caché: muchos estados
        comparten el mismo conjunto de unidades y solo difieren en dónde
        están ubicadas.
        """
        if unidades in self._cache_reparto:
            return self._cache_reparto[unidades]
        total_banca = sum(self._poder_individual(c) for c in unidades)
        mejor = self.pesos.banca * total_banca          # tablero vacío
        vistos = set()
        for k in range(1, min(self.cap_tablero, len(unidades)) + 1):
            for idx in combinations(range(len(unidades)), k):
                tablero = tuple(unidades[i] for i in idx)
                if tablero in vistos:
                    continue
                vistos.add(tablero)
                en_banca = total_banca - sum(self._poder_individual(c) for c in tablero)
                mejor = max(mejor, self.poder_tablero(tablero)
                            + self.pesos.banca * en_banca)
        self._cache_reparto[unidades] = mejor
        return mejor

    def cota(self, s: EstadoTurno, heuristica: str = "h2") -> float:
        """f(s) = V(s) + h(s): cota SUPERIOR (admisible) del valor de
        cualquier estado alcanzable desde ``s``, incluido ``s``.

        * ``h1`` (por unidades): para cada subconjunto de compras pagable
          descuenta el oro gastado y acota el poder con ``_cota_unidades``
          (sinergia posible por elemento y la mejor ventaja elemental para
          todo el equipo). Barata por nodo, pero holgada.
        * ``h2`` (relajada): recorre cada subconjunto de compras de la tienda
          que alcance a pagarse y cada forma de combinar; para cada uno
          descuenta el oro gastado y calcula el MEJOR reparto exacto de esas
          unidades entre tablero y banca (``_mejor_reparto``), ignorando que
          mover fichas cuesta acciones. Es admisible porque todo estado
          alcanzable corresponde a una de esas opciones, y es mucho más
          ajustada que h1 (poda más), a cambio de más cómputo por nodo.

        En ambas el oro no puede aumentar (no se vende), así que el término
        económico de ``s`` también es una cota.
        """
        clave = (s.clave(), heuristica)
        if clave in self._cache_cota:
            return self._cache_cota[clave]
        propias = Counter(s.banca + s.tablero)
        tienda = [t for t in s.tienda if t]
        cota = float("-inf")
        for mascara in range(1 << len(tienda)):
            compra = [t for i, t in enumerate(tienda) if mascara >> i & 1]
            costo = sum(CATALOGO[t].costo for t in compra)
            if costo > s.oro:
                continue
            conteo = Counter(propias)
            for t in compra:
                conteo[(t, 1)] += 1
            alternativas = self._alternativas_combinacion(conteo)
            if heuristica == "h1":
                unidades = max(self._cota_unidades(u) for u in alternativas)
            else:
                unidades = max(self._mejor_reparto(tuple(sorted(u)))
                               for u in alternativas)
            cota = max(cota, unidades + self.termino_economico(s.oro - costo))
        cota = max(cota, self.valor(s))
        self._cache_cota[clave] = cota
        return cota

    # -------------------------- traducción ---------------------------
    def a_accion_env(self, a: AccionPlan) -> Accion:
        """Convierte una acción del plan en una ``Accion`` del ambiente real
        (busca el índice correspondiente en la tienda / banca / tablero)."""
        tipo, arg = a
        j = self.env.jugadores[self.jugador]
        if tipo == "terminar":
            return Accion(TipoAccion.TERMINAR)
        if tipo == "comprar":
            return Accion(TipoAccion.COMPRAR, j.tienda.index(arg))
        if tipo == "combinar":
            return Accion(TipoAccion.COMBINAR, tuple(arg))
        if tipo == "a_tablero":
            i = next(i for i, u in enumerate(j.banca) if u.clave() == tuple(arg))
            return Accion(TipoAccion.A_TABLERO, i)
        if tipo == "a_banca":
            i = next(i for i, u in enumerate(j.tablero) if u.clave() == tuple(arg))
            return Accion(TipoAccion.A_BANCA, i)
        raise ValueError(a)


# ===========================================================================
# 2. Algoritmos de búsqueda
# ===========================================================================
@dataclass
class ResultadoBusqueda:
    """Salida común de todos los algoritmos.

    plan : acciones del modelo (termina siempre en ("terminar", None)).
    estado_final / valor : mejor estado encontrado y su utilidad V.
    metricas : tiempo, nodos generados/expandidos, frontera máxima, etc.
    traza : eventos del proceso de búsqueda para animarlo en la GUI. Cada
        evento: {"n": #expansión, "estado": str, "valor": V, "f": f o None,
        "frontera": tamaño, "mejor": mejor V hasta ahora}.
    """

    plan: List[AccionPlan]
    estado_final: EstadoTurno
    valor: float
    metricas: MetricasBusqueda
    traza: List[Dict[str, Any]] = field(default_factory=list)


def _reconstruir(padres: Dict[tuple, Tuple[Optional[tuple], Optional[AccionPlan]]],
                 clave: tuple) -> List[AccionPlan]:
    plan = []
    while padres[clave][0] is not None:
        clave_padre, accion = padres[clave]
        plan.append(accion)
        clave = clave_padre
    plan.reverse()
    return plan


def _registrar(traza, n, s, v, f, frontera, mejor) -> None:
    if len(traza) < MAX_TRAZA:
        traza.append({"n": n, "estado": str(s), "valor": round(v, 3),
                      "f": None if f is None else round(f, 3),
                      "frontera": frontera, "mejor": round(mejor, 3)})


def busqueda_anchura(problema: ProblemaTurno,
                     limite_nodos: int = 200_000) -> ResultadoBusqueda:
    """BFS sobre el espacio de estados del turno (no informada).

    Expande por niveles y evalúa V en cada estado. Con conjunto de visitados
    cada estado se expande una sola vez. Si recorre todo el espacio, el mejor
    estado encontrado es el óptimo global de V.
    """
    m = MetricasBusqueda(algoritmo="BFS")
    traza: List[Dict[str, Any]] = []
    with Cronometro() as c:
        s0 = problema.estado_inicial
        frontera = deque([s0])
        padres = {s0.clave(): (None, None)}
        mejor, mejor_v = s0, problema.valor(s0)
        m.nodos_generados = 1
        completo = True
        while frontera:
            if m.nodos_expandidos >= limite_nodos:
                completo = False
                break
            s = frontera.popleft()
            m.nodos_expandidos += 1
            v = problema.valor(s)
            if v > mejor_v:
                mejor, mejor_v = s, v
            for a in problema.acciones(s):
                hijo = problema.resultado(s, a)
                k = hijo.clave()
                if k in padres:
                    continue
                padres[k] = (s.clave(), a)
                frontera.append(hijo)
                m.nodos_generados += 1
            m.registrar_frontera(len(frontera))
            _registrar(traza, m.nodos_expandidos, s, v, None, len(frontera), mejor_v)
        plan = _reconstruir(padres, mejor.clave()) + [("terminar", None)]
    return _cerrar(m, c, plan, mejor, mejor_v, traza,
                   optima=True if completo else None, problema=problema)


def busqueda_profundidad_limitada(problema: ProblemaTurno, limite: int = 6
                                  ) -> ResultadoBusqueda:
    """Búsqueda en profundidad con límite ``limite`` (no informada).

    Usa una pila. Para no repetir trabajo guarda la menor profundidad con la
    que se llegó a cada estado y solo lo re-expande si llega más arriba
    (así no se pierden estados alcanzables dentro del límite).
    """
    m = MetricasBusqueda(algoritmo=f"DLS(L={limite})")
    traza: List[Dict[str, Any]] = []
    with Cronometro() as c:
        s0 = problema.estado_inicial
        pila: List[Tuple[EstadoTurno, int, List[AccionPlan]]] = [(s0, 0, [])]
        profundidad_vista = {s0.clave(): 0}
        mejor, mejor_v, mejor_plan = s0, problema.valor(s0), []
        m.nodos_generados = 1
        while pila:
            s, d, camino = pila.pop()
            m.nodos_expandidos += 1
            v = problema.valor(s)
            if v > mejor_v:
                mejor, mejor_v, mejor_plan = s, v, camino
            if d < limite:
                for a in reversed(problema.acciones(s)):
                    hijo = problema.resultado(s, a)
                    k = hijo.clave()
                    if profundidad_vista.get(k, limite + 1) <= d + 1:
                        continue
                    profundidad_vista[k] = d + 1
                    pila.append((hijo, d + 1, camino + [a]))
                    m.nodos_generados += 1
            m.registrar_frontera(len(pila))
            _registrar(traza, m.nodos_expandidos, s, v, None, len(pila), mejor_v)
        plan = mejor_plan + [("terminar", None)]
    m.extra["limite"] = limite
    return _cerrar(m, c, plan, mejor, mejor_v, traza, optima=None,
                   problema=problema)


def busqueda_a_estrella(problema: ProblemaTurno, heuristica: str = "h2",
                        limite_nodos: int = 200_000) -> ResultadoBusqueda:
    """A* para maximización (equivalente a Branch & Bound primero-el-mejor).

    La frontera es una cola de prioridad ordenada por f(n) = V(n) + h(n)
    (``ProblemaTurno.cota``). Se saca siempre el nodo más prometedor. Como f
    es una cota superior admisible, en cuanto la mejor f de la frontera no
    supera el mejor V ya encontrado, ningún estado sin explorar puede ser
    mejor: se detiene con la solución óptima.
    """
    m = MetricasBusqueda(algoritmo=f"A* ({heuristica})")
    traza: List[Dict[str, Any]] = []
    with Cronometro() as c:
        s0 = problema.estado_inicial
        contador = 0
        frontera = [(-problema.cota(s0, heuristica), 0.0, 0, contador, s0)]
        profundidad = {s0.clave(): 0}
        padres = {s0.clave(): (None, None)}
        cerrados = set()
        mejor, mejor_v = s0, problema.valor(s0)
        m.nodos_generados = 1
        completo = True
        while frontera:
            f_neg, _, _, _, s = heapq.heappop(frontera)
            if -f_neg <= mejor_v + 1e-9:
                break                         # prueba de optimalidad
            k = s.clave()
            if k in cerrados:
                continue
            if m.nodos_expandidos >= limite_nodos:
                completo = False
                break
            cerrados.add(k)
            m.nodos_expandidos += 1
            v = problema.valor(s)
            if v > mejor_v:
                mejor, mejor_v = s, v
            for a in problema.acciones(s):
                hijo = problema.resultado(s, a)
                kh = hijo.clave()
                if kh in padres:
                    # ya conocido: si este camino es más corto, se reengancha
                    # (el valor no depende del camino, pero un plan corto es
                    # mejor solución: menos acciones)
                    if profundidad[k] + 1 < profundidad[kh]:
                        padres[kh] = (k, a)
                        profundidad[kh] = profundidad[k] + 1
                    continue
                padres[kh] = (k, a)
                profundidad[kh] = profundidad[k] + 1
                m.nodos_generados += 1
                vh = problema.valor(hijo)
                if vh > mejor_v:              # el hijo ya es una solución mejor
                    mejor, mejor_v = hijo, vh
                f = problema.cota(hijo, heuristica)
                if f > mejor_v + 1e-9:        # si f <= mejor V, se poda
                    contador += 1
                    # desempate: a igual f, primero el de mayor V y luego el
                    # menos profundo
                    heapq.heappush(frontera, (-f, -vh, profundidad[kh], contador, hijo))
            m.registrar_frontera(len(frontera))
            _registrar(traza, m.nodos_expandidos, s, v, -f_neg, len(frontera), mejor_v)
        plan = _reconstruir(padres, mejor.clave()) + [("terminar", None)]
    m.extra["heuristica"] = heuristica
    return _cerrar(m, c, plan, mejor, mejor_v, traza,
                   optima=True if completo else None, problema=problema)


def busqueda_voraz(problema: ProblemaTurno, anticipacion: int = 2
                   ) -> ResultadoBusqueda:
    """Ascenso de colina (voraz) con anticipación.

    Desde el estado actual mira todas las secuencias de hasta
    ``anticipacion`` acciones, avanza al estado de mayor V y repite hasta que
    ninguna secuencia mejore. Es rápida pero puede quedarse en un óptimo
    local (p. ej. no ve que comprar 3 copias y combinar es mejor).
    """
    m = MetricasBusqueda(algoritmo=f"Voraz (k={anticipacion})")
    traza: List[Dict[str, Any]] = []
    with Cronometro() as c:
        s = problema.estado_inicial
        v = problema.valor(s)
        plan: List[AccionPlan] = []
        m.nodos_generados = 1
        while True:
            m.nodos_expandidos += 1
            mejor_sec, mejor_s, mejor_v = None, None, v
            nivel = [(s, [])]
            for _ in range(anticipacion):
                siguiente = []
                for estado, sec in nivel:
                    for a in problema.acciones(estado):
                        hijo = problema.resultado(estado, a)
                        m.nodos_generados += 1
                        vh = problema.valor(hijo)
                        if vh > mejor_v + 1e-9:
                            mejor_sec, mejor_s, mejor_v = sec + [a], hijo, vh
                        siguiente.append((hijo, sec + [a]))
                nivel = siguiente
                m.registrar_frontera(len(nivel))
            _registrar(traza, m.nodos_expandidos, s, v, None, len(nivel), v)
            if mejor_sec is None:
                break
            plan += mejor_sec
            s, v = mejor_s, mejor_v
        plan.append(("terminar", None))
    m.extra["anticipacion"] = anticipacion
    return _cerrar(m, c, plan, s, v, traza, optima=None, problema=problema)


def simplificar_plan(problema: ProblemaTurno, plan: List[AccionPlan],
                     valor_objetivo: float) -> List[AccionPlan]:
    """Elimina acciones sobrantes de un plan (p. ej. desplegar una ficha y
    luego retirarla) sin bajar su utilidad. Prueba quitar una acción o un par
    de acciones mientras el plan siga siendo aplicable y su estado final
    tenga el mismo V. No cambia la solución, solo la acorta."""
    def aplicar(p):
        s = problema.estado_inicial
        for a in p:
            if a not in problema.acciones(s):
                return None
            s = problema.resultado(s, a)
        return s

    pasos = [a for a in plan if a[0] != "terminar"]
    cambio = True
    while cambio:
        cambio = False
        n = len(pasos)
        candidatos = [(i,) for i in range(n)] + [(i, j) for i in range(n)
                                                 for j in range(i + 1, n)]
        for quitar in candidatos:
            nuevo = [a for k, a in enumerate(pasos) if k not in quitar]
            final = aplicar(nuevo)
            if final is not None and problema.valor(final) >= valor_objetivo - 1e-9:
                pasos, cambio = nuevo, True
                break
    return pasos + [("terminar", None)]


def _cerrar(m, c, plan, estado, valor, traza, optima,
            problema: Optional[ProblemaTurno] = None) -> ResultadoBusqueda:
    if problema is not None:
        m.extra["longitud_original"] = len(plan)
        plan = simplificar_plan(problema, plan, valor)
    m.tiempo_s = c.segundos
    m.longitud_solucion = len(plan)
    m.costo_solucion = round(valor, 4)        # aquí "costo" = utilidad V
    m.encontro_solucion = True
    m.es_optima = optima
    return ResultadoBusqueda(plan, estado, valor, m, traza)


#: Registro de algoritmos disponibles: nombre -> función(problema)
ALGORITMOS: Dict[str, Callable[[ProblemaTurno], ResultadoBusqueda]] = {
    "BFS": lambda p: busqueda_anchura(p),
    "DLS": lambda p: busqueda_profundidad_limitada(p, limite=6),
    "A*h1": lambda p: busqueda_a_estrella(p, "h1"),
    "A*h2": lambda p: busqueda_a_estrella(p, "h2"),
    "Voraz": lambda p: busqueda_voraz(p, anticipacion=2),
}


# ===========================================================================
# 3. Agentes
# ===========================================================================
class AgenteBusqueda:
    """Agente basado en modelo: al empezar su turno formula el
    ``ProblemaTurno``, lo resuelve con el algoritmo elegido y ejecuta el
    plan una acción por llamada (así la GUI puede animarlo).

    Política de renovación (fuera de la búsqueda porque es aleatoria): si el
    plan óptimo no compra nada y, después de pagar la renovación, el oro
    sigue >= ``umbral_renovar``, renueva (máximo ``max_renovaciones`` por
    ronda) y vuelve a planificar.
    """

    def __init__(self, algoritmo: str = "A*h2",
                 pesos: Optional[PesosEvaluacion] = None,
                 umbral_renovar: int = 10, max_renovaciones: int = 2) -> None:
        self.algoritmo = algoritmo
        self.nombre = algoritmo
        self.pesos = pesos or PesosEvaluacion()
        self.umbral_renovar = umbral_renovar
        self.max_renovaciones = max_renovaciones
        self.reiniciar()

    def reiniciar(self) -> None:
        self._plan: List[AccionPlan] = []
        self._ronda = -1
        self._renovaciones = 0
        self.ultimo_resultado: Optional[ResultadoBusqueda] = None
        self.historial_metricas: List[MetricasBusqueda] = []

    def actuar(self, env: MagicChessEnv, jugador: int) -> Accion:
        if env.ronda != self._ronda:
            self._ronda, self._renovaciones, self._plan = env.ronda, 0, []
        if not self._plan:
            problema = ProblemaTurno(env, jugador, self.pesos)
            res = ALGORITMOS[self.algoritmo](problema)
            self.ultimo_resultado = res
            self.historial_metricas.append(res.metricas)
            compra = any(a[0] == "comprar" for a in res.plan)
            j = env.jugadores[jugador]
            costo = env.config.costo_renovar
            if (not compra and self._renovaciones < self.max_renovaciones
                    and j.oro - costo >= self.umbral_renovar
                    and len(j.banca) < env.config.tamano_banca):
                self._renovaciones += 1
                return Accion(TipoAccion.RENOVAR)
            self._plan = list(res.plan)
            self._problema = problema
        a = self._plan.pop(0)
        try:
            accion = self._problema.a_accion_env(a)
        except (ValueError, StopIteration):
            accion = None
        if accion is None or not env.es_legal(jugador, accion):
            self._plan = []                      # algo cambió: replanificar
            return Accion(TipoAccion.TERMINAR)
        return accion

    def lineas_estado(self) -> List[str]:
        """Estado interno resumido para la GUI."""
        r = self.ultimo_resultado
        if r is None:
            return [self.nombre, "sin búsquedas aún"]
        m = r.metricas
        opt = {True: "sí", None: "?", False: "no"}[m.es_optima]
        prox = self._plan[0][0] if self._plan else "-"
        return [m.algoritmo,
                f"V* = {r.valor:.2f}  óptima: {opt}",
                f"expandidos: {m.nodos_expandidos}",
                f"generados: {m.nodos_generados}",
                f"frontera máx: {m.max_frontera}",
                f"tiempo: {m.tiempo_s * 1000:.1f} ms",
                f"plan: {len(r.plan)} acc. próx: {prox}"]

    def resumen(self) -> str:
        r = self.ultimo_resultado
        if r is None:
            return f"{self.nombre}: sin búsquedas aún"
        m = r.metricas
        opt = {True: "óptima", None: "?", False: "no óptima"}[m.es_optima]
        return (f"{m.algoritmo}: V={r.valor:.1f} ({opt})  exp={m.nodos_expandidos} "
                f"gen={m.nodos_generados} front={m.max_frontera} "
                f"{m.tiempo_s * 1000:.1f}ms")


@dataclass(frozen=True)
class PerfilReglas:
    """Parámetros de la estrategia por reglas.

    * reserva: oro mínimo que intenta conservar (bonificación por ahorro).
    * renovar_si_oro: renueva si después de pagar le queda al menos esto.
    * max_renovaciones: renovaciones por ronda.
    * prioridad_copias: bono de puntuación por comprar copias de unidades
      que ya tiene (busca combinaciones).
    """

    nombre: str = "equilibrado"
    reserva: int = 10
    renovar_si_oro: int = 12
    max_renovaciones: int = 1
    prioridad_copias: float = 3.0


PERFILES = {
    "agresivo": PerfilReglas("agresivo", reserva=0, renovar_si_oro=4,
                             max_renovaciones=2, prioridad_copias=2.0),
    "equilibrado": PerfilReglas(),
    "ahorro": PerfilReglas("ahorro", reserva=30, renovar_si_oro=40,
                           max_renovaciones=0, prioridad_copias=3.0),
    "combos": PerfilReglas("combos", reserva=4, renovar_si_oro=6,
                           max_renovaciones=3, prioridad_copias=8.0),
}


class AgenteReglas:
    """Estrategia heurística basada en reglas, evaluadas en orden de
    prioridad en cada llamada:

    1. Si hay 3 copias iguales -> COMBINAR.
    2. Si hay casilla libre en el tablero -> desplegar la mejor unidad de la
       banca (poder ajustado por ventaja contra el rival).
    3. Si el tablero está lleno y la banca tiene una unidad mejor que la peor
       del tablero -> retirar la peor (luego la regla 2 despliega la mejor).
    4. Comprar la mejor unidad de la tienda (poder + ventaja + bono por
       copias) si alcanza el oro y no baja de la reserva; se permite romper
       la reserva si la compra completa una combinación o si el tablero
       tiene casillas vacías sin nada que desplegar.
    5. Renovar si sobra oro (según el perfil).
    6. TERMINAR.
    """

    def __init__(self, perfil: str = "equilibrado") -> None:
        self.perfil = PERFILES[perfil]
        self.nombre = f"Reglas-{perfil}"
        self.reiniciar()

    def reiniciar(self) -> None:
        self._ronda = -1
        self._renovaciones = 0
        self._pasos_ronda = 0
        self.ultima_regla = ""

    def _puntaje(self, env, jugador, nombre, estrellas=1) -> float:
        p = CATALOGO[nombre]
        rival = env.jugadores[env.rival(jugador)].tablero
        ventaja = (sum(ventaja_elemental(p.elemento, r.elemento) for r in rival)
                   / len(rival)) if rival else 0.0
        return p.poder_base * MULT_ESTRELLAS[estrellas] * (1 + PESO_VENTAJA * ventaja)

    def actuar(self, env: MagicChessEnv, jugador: int) -> Accion:
        if env.ronda != self._ronda:
            self._ronda, self._renovaciones, self._pasos_ronda = env.ronda, 0, 0
        self._pasos_ronda += 1
        if self._pasos_ronda > 40:                      # salvaguarda anti-bucles
            return self._regla("limite", Accion(TipoAccion.TERMINAR))
        j = env.jugadores[jugador]
        cfg = env.config
        cap = env.capacidad_tablero()

        # 1. Combinar
        for (n, e), c in j.contar_copias().items():
            if c >= COPIAS_PARA_COMBINAR and e < ESTRELLAS_MAX:
                return self._regla("combinar", Accion(TipoAccion.COMBINAR, (n, e)))

        punt = lambda u: self._puntaje(env, jugador, u.nombre, u.estrellas)
        # 2. Desplegar
        if j.banca and len(j.tablero) < cap:
            i = max(range(len(j.banca)), key=lambda k: punt(j.banca[k]))
            return self._regla("desplegar", Accion(TipoAccion.A_TABLERO, i))
        # 3. Reemplazar la peor del tablero
        if j.banca and j.tablero and len(j.banca) < cfg.tamano_banca:
            peor = min(range(len(j.tablero)), key=lambda k: punt(j.tablero[k]))
            if max(punt(u) for u in j.banca) > punt(j.tablero[peor]) + 1e-9:
                return self._regla("reemplazar", Accion(TipoAccion.A_BANCA, peor))

        # 4. Comprar
        if len(j.banca) < cfg.tamano_banca:
            copias = Counter(u.nombre for u in j.todas_las_unidades() if u.estrellas == 1)
            mejor, mejor_p = None, -1.0
            for i, n in enumerate(j.tienda):
                if n is None or CATALOGO[n].costo > j.oro:
                    continue
                completa = copias[n] + 1 >= COPIAS_PARA_COMBINAR
                hay_hueco = len(j.tablero) < cap and not j.banca
                if (j.oro - CATALOGO[n].costo < self.perfil.reserva
                        and not completa and not hay_hueco):
                    continue
                p = self._puntaje(env, jugador, n) + self.perfil.prioridad_copias * copias[n]
                if p > mejor_p:
                    mejor, mejor_p = i, p
            if mejor is not None:
                return self._regla("comprar", Accion(TipoAccion.COMPRAR, mejor))

        # 5. Renovar
        if (self._renovaciones < self.perfil.max_renovaciones
                and j.oro - cfg.costo_renovar >= self.perfil.renovar_si_oro):
            self._renovaciones += 1
            return self._regla("renovar", Accion(TipoAccion.RENOVAR))

        # 6. Terminar
        return self._regla("terminar", Accion(TipoAccion.TERMINAR))

    def _regla(self, nombre: str, accion: Accion) -> Accion:
        self.ultima_regla = nombre
        return accion

    def lineas_estado(self) -> List[str]:
        return [self.nombre, f"regla: {self.ultima_regla or '-'}",
                f"reserva: {self.perfil.reserva} oro",
                f"renovaciones: {self._renovaciones}/{self.perfil.max_renovaciones}"]

    def resumen(self) -> str:
        return f"{self.nombre}: regla disparada = {self.ultima_regla}"


class AgenteQLearning:
    """Q-learning tabular sobre una representación simplificada.

    * Estado abstracto: (fase de la partida, vidas propias, diferencia de
      vidas, nivel de oro, ¿mi tablero es más fuerte que el rival?).
    * Acciones (macro): elegir un perfil de ``PERFILES`` para toda la ronda,
      que luego ejecuta un ``AgenteReglas``.
    * Recompensa: la del ambiente (+1/-1 por ronda, +10/-10 al final).
    * Actualización: Q(s,a) += alfa * (r + gamma * max_a' Q(s',a') - Q(s,a)).

    Con ``aprendiendo=True`` explora (epsilon-greedy) y actualiza la tabla;
    con False solo explota lo aprendido.
    """

    ACCIONES = list(PERFILES)

    def __init__(self, alfa: float = 0.1, gamma: float = 0.95,
                 epsilon: float = 0.1, aprendiendo: bool = False,
                 semilla: Optional[int] = None,
                 archivo: Optional[str] = None) -> None:
        self.alfa, self.gamma, self.epsilon = alfa, gamma, epsilon
        self.aprendiendo = aprendiendo
        self.rng = crear_rng(semilla)
        self.Q: Dict[str, List[float]] = {}
        self.nombre = "Q-learning"
        self.ejecutores = {p: AgenteReglas(p) for p in PERFILES}
        if archivo and os.path.exists(archivo):
            self.cargar(archivo)
        self.reiniciar()

    def reiniciar(self) -> None:
        self._ronda = -1
        self._s: Optional[str] = None
        self._a: Optional[int] = None
        self._marcador = (0, 0)
        for e in self.ejecutores.values():
            e.reiniciar()

    # ----- representación -----
    @staticmethod
    def estado_abstracto(env: MagicChessEnv, jugador: int) -> str:
        yo = env.jugadores[jugador]
        otro = env.jugadores[env.rival(jugador)]
        fase = min(env.ronda // 4, 3)
        dif = max(-2, min(2, yo.vidas - otro.vidas))
        oro = min(yo.oro // 10, 3)
        mio = env.poder_equipo(jugador).total
        suyo = env.poder_equipo(env.rival(jugador)).total
        fuerza = 1 if mio > suyo * 1.1 else (-1 if mio < suyo * 0.9 else 0)
        return f"{fase}|{min(yo.vidas, 5)}|{dif}|{oro}|{fuerza}"

    def _q(self, s: str) -> List[float]:
        return self.Q.setdefault(s, [0.0] * len(self.ACCIONES))

    def _elegir(self, s: str) -> int:
        if self.aprendiendo and self.rng.random() < self.epsilon:
            return self.rng.randrange(len(self.ACCIONES))
        q = self._q(s)
        mejor = max(q)
        return self.rng.choice([i for i, v in enumerate(q) if v == mejor])

    def _actualizar(self, r: float, s2: Optional[str]) -> None:
        if not self.aprendiendo or self._s is None:
            return
        q = self._q(self._s)
        objetivo = r + (self.gamma * max(self._q(s2)) if s2 is not None else 0.0)
        q[self._a] += self.alfa * (objetivo - q[self._a])

    def _recompensa_ronda(self, env, jugador) -> float:
        yo = env.jugadores[jugador]
        marcador = (yo.victorias, yo.derrotas)
        r = ((marcador[0] - self._marcador[0]) - (marcador[1] - self._marcador[1])) \
            * MagicChessEnv.RECOMPENSA_RONDA
        self._marcador = marcador
        return r

    # ----- interfaz de agente -----
    def actuar(self, env: MagicChessEnv, jugador: int) -> Accion:
        if env.ronda != self._ronda:
            s = self.estado_abstracto(env, jugador)
            self._actualizar(self._recompensa_ronda(env, jugador), s)
            self._ronda = env.ronda
            self._s, self._a = s, self._elegir(s)
        return self.ejecutores[self.ACCIONES[self._a]].actuar(env, jugador)

    def finalizar(self, env: MagicChessEnv, jugador: int) -> None:
        """Llamar al terminar la partida para la última actualización."""
        r = self._recompensa_ronda(env, jugador)
        if env.ganador == jugador:
            r += MagicChessEnv.RECOMPENSA_PARTIDA
        elif env.ganador not in (None, EMPATE):
            r -= MagicChessEnv.RECOMPENSA_PARTIDA
        self._actualizar(r, None)

    def lineas_estado(self) -> List[str]:
        perfil = self.ACCIONES[self._a] if self._a is not None else "-"
        lineas = ["Q-learning", f"s = {self._s or '-'}", f"perfil: {perfil}"]
        if self._s is not None:
            q = self._q(self._s)
            lineas += [f"Q[{a[:5]}] = {v:+.2f}" for a, v in zip(self.ACCIONES, q)]
        lineas.append(f"estados: {len(self.Q)}")
        return lineas

    def resumen(self) -> str:
        perfil = self.ACCIONES[self._a] if self._a is not None else "-"
        return f"Q-learning: estado {self._s} -> perfil '{perfil}' ({len(self.Q)} estados)"

    # ----- persistencia -----
    def guardar(self, archivo: str) -> None:
        with open(archivo, "w", encoding="utf-8") as f:
            json.dump({"acciones": self.ACCIONES, "Q": self.Q}, f, indent=1)

    def cargar(self, archivo: str) -> None:
        with open(archivo, encoding="utf-8") as f:
            self.Q = json.load(f)["Q"]


# ===========================================================================
# 4. Experimentos
# ===========================================================================
def jugar_partida(env: MagicChessEnv, agentes: List[Any]) -> MagicChessEnv:
    """Juega una partida completa (cada jugador termina su turno por
    completo, luego el otro) y devuelve el ambiente final."""
    for a in agentes:
        if hasattr(a, "reiniciar"):
            a.reiniciar()
    while not env.es_terminal():
        for j in env.jugadores_pendientes():
            while not env.jugadores[j].listo and not env.es_terminal():
                env.step(j, agentes[j].actuar(env, j))
    for j, a in enumerate(agentes):
        if hasattr(a, "finalizar"):
            a.finalizar(env, j)
    return env


def torneo(fabrica_a: Callable[[], Any], fabrica_b: Callable[[], Any],
           partidas: int = 50, dificultad: str = "normal",
           semilla: int = 0) -> Dict[str, Any]:
    """Enfrenta dos agentes ``partidas`` veces alternando el asiento.

    Devuelve métricas de A: % de victorias, % de empates, rondas promedio,
    vidas restantes promedio, aprovechamiento del oro (gastado / ganado) y
    tiempo promedio de decisión cuando el agente es de búsqueda.
    """
    a, b = fabrica_a(), fabrica_b()
    gan = emp = 0
    rondas = vidas = uso_oro = 0.0
    for k in range(partidas):
        asiento = k % 2
        agentes = [a, b] if asiento == 0 else [b, a]
        env = MagicChessEnv(ConfiguracionJuego.desde_dificultad(dificultad),
                            semilla=semilla + k)
        jugar_partida(env, agentes)
        yo = env.jugadores[asiento]
        gan += env.ganador == asiento
        emp += env.ganador == EMPATE
        rondas += env.ronda
        vidas += yo.vidas
        uso_oro += yo.oro_gastado / max(1, yo.oro_ganado)
    res = {
        "agente": getattr(a, "nombre", type(a).__name__),
        "rival": getattr(b, "nombre", type(b).__name__),
        "partidas": partidas,
        "victorias_%": 100 * gan / partidas,
        "empates_%": 100 * emp / partidas,
        "rondas_prom": rondas / partidas,
        "vidas_prom": vidas / partidas,
        "uso_oro_%": 100 * uso_oro / partidas,
    }
    hist = getattr(a, "historial_metricas", None)
    if hist:
        res["tiempo_decision_ms"] = 1000 * sum(m.tiempo_s for m in hist) / len(hist)
        res["expandidos_prom"] = sum(m.nodos_expandidos for m in hist) / len(hist)
    return res


def comparar_busquedas(env: MagicChessEnv, jugador: int,
                       pesos: Optional[PesosEvaluacion] = None,
                       algoritmos: Optional[List[str]] = None
                       ) -> List[Dict[str, Any]]:
    """Ejecuta los algoritmos sobre la MISMA instancia y compara.

    Toma como referencia el mejor V hallado por un algoritmo que haya
    certificado optimalidad (BFS completo o A*), y marca cada resultado como
    óptimo o no comparando su V contra esa referencia.
    """
    nombres = algoritmos or list(ALGORITMOS)
    resultados = {n: ALGORITMOS[n](ProblemaTurno(env, jugador, pesos)) for n in nombres}
    certificados = [r.valor for r in resultados.values() if r.metricas.es_optima]
    referencia = max(certificados) if certificados else max(r.valor for r in resultados.values())
    filas = []
    for n, r in resultados.items():
        fila = r.metricas.como_dict()
        fila.pop("extra", None)
        fila["algoritmo"] = n
        fila["valor_V"] = round(r.valor, 3)
        fila["es_optima"] = abs(r.valor - referencia) < 1e-6
        fila["plan"] = " > ".join(t if a is None else f"{t}({a})" for t, a in r.plan)
        filas.append(fila)
    return filas


def entrenar_qlearning(episodios: int = 2000, archivo: str = "qtabla.json",
                       dificultad: str = "normal", semilla: int = 0,
                       epsilon_inicial: float = 0.3, epsilon_final: float = 0.02,
                       cada: int = 500) -> AgenteQLearning:
    """Entrena Q-learning contra una mezcla de rivales (reglas con perfiles
    distintos y aleatorio), con epsilon decreciente. Guarda la tabla en
    ``archivo``."""
    agente = AgenteQLearning(aprendiendo=True, semilla=semilla)
    rivales = [AgenteReglas("equilibrado"), AgenteReglas("agresivo"),
               AgenteReglas("ahorro"), AgenteAleatorio(semilla=semilla)]
    rng = crear_rng(semilla)
    victorias = 0
    for ep in range(episodios):
        agente.epsilon = epsilon_inicial + (epsilon_final - epsilon_inicial) * ep / max(1, episodios - 1)
        rival = rng.choice(rivales)
        asiento = ep % 2
        agentes = [agente, rival] if asiento == 0 else [rival, agente]
        env = MagicChessEnv(ConfiguracionJuego.desde_dificultad(dificultad),
                            semilla=semilla + ep)
        jugar_partida(env, agentes)
        victorias += env.ganador == asiento
        if cada and (ep + 1) % cada == 0:
            print(f"  episodio {ep + 1}: {100 * victorias / cada:.1f}% victorias "
                  f"(eps={agente.epsilon:.3f}, estados={len(agente.Q)})")
            victorias = 0
    agente.aprendiendo = False
    agente.guardar(archivo)
    return agente


# ===========================================================================
# 5. Fábricas para la GUI
# ===========================================================================
RUTA_QTABLA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qtabla.json")

FABRICAS: Dict[str, Callable[[], Any]] = {
    "Reglas": lambda: AgenteReglas("equilibrado"),
    "BFS": lambda: AgenteBusqueda("BFS"),
    "DLS": lambda: AgenteBusqueda("DLS"),
    "A*h1": lambda: AgenteBusqueda("A*h1"),
    "A*h2": lambda: AgenteBusqueda("A*h2"),
    "Voraz": lambda: AgenteBusqueda("Voraz"),
    "Q-learning": lambda: AgenteQLearning(archivo=RUTA_QTABLA),
}


def _imprimir_tabla(filas: List[Dict[str, Any]], columnas: List[str]) -> None:
    anchos = {c: max(len(c), *(len(_fmt(f.get(c))) for f in filas)) for c in columnas}
    print("  ".join(c.ljust(anchos[c]) for c in columnas))
    for f in filas:
        print("  ".join(_fmt(f.get(c)).ljust(anchos[c]) for c in columnas))


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 1 else f"{v:.2f}"
    return str(v)


if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else "comparar"

    if modo == "comparar":
        # Instancia de ejemplo: ronda 6 con unidades y oro a mitad de partida.
        env = MagicChessEnv(ConfiguracionJuego(), semilla=3)
        env.fijar_ronda(6)
        env.fijar_oro(0, 14)
        for n, e, z in [("Gota", 1, "tablero"), ("Gota", 1, "banca"),
                        ("Chispa", 2, "tablero"), ("Brote", 1, "banca")]:
            env.agregar_unidad(0, n, e, z)
        for n, z in [("Brasa", "tablero"), ("Ent", "tablero"), ("Brote", "tablero")]:
            env.agregar_unidad(1, n, 1, z)
        env.fijar_tienda(0, ["Gota", "Marea", "Espino"])
        filas = comparar_busquedas(env, 0)
        _imprimir_tabla(filas, ["algoritmo", "valor_V", "es_optima", "tiempo_s",
                                "nodos_generados", "nodos_expandidos",
                                "max_frontera", "longitud_solucion"])
        print("\nPlan óptimo (A*h2):",
              next(f["plan"] for f in filas if f["algoritmo"] == "A*h2"))

    elif modo == "torneo":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        filas = [torneo(FABRICAS[k], FABRICAS["Reglas"], partidas=n)
                 for k in FABRICAS]
        _imprimir_tabla(filas, ["agente", "rival", "victorias_%", "empates_%",
                                "rondas_prom", "vidas_prom", "uso_oro_%",
                                "tiempo_decision_ms"])

    elif modo == "entrenar":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
        entrenar_qlearning(n, RUTA_QTABLA)
        print(f"Tabla Q guardada en {RUTA_QTABLA}")
