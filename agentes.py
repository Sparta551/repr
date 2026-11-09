"""
agentes.py
==========

Los tres algoritmos del proyecto *Magic Chess*:

1. **Heurística** (``AgenteHeuristico``) - informado. Estrategia basada en
   reglas ordenadas por prioridad, que usa el cálculo de poder como función
   heurística para elegir qué comprar, desplegar o combinar.
2. **Backtracking** (``AgenteBacktracking``) - no informado. Recorre en
   profundidad, haciendo y deshaciendo acciones, TODAS las secuencias de
   decisiones posibles del turno y se queda con la de mayor utilidad. Es
   completo y encuentra el plan óptimo del turno.
3. **Q-learning** (``AgenteQLearning``) - aprendizaje por refuerzo. Aprende,
   jugando partidas simuladas, qué estilo de juego conviene en cada
   situación (estado simplificado).

El problema de un turno
-----------------------
La partida completa es aleatoria (tienda) y adversaria, así que no tiene una
solución única. Lo que sí se puede resolver de forma óptima es la fase de
preparación de cada ronda:

* **Estado** (``ModeloTurno``): oro, fichas en banca, fichas en tablero y
  tienda del jugador.
* **Acciones**: comprar(nombre), combinar(nombre, nivel), a_tablero(ficha),
  a_banca(ficha). RENOVAR queda fuera de la búsqueda porque su resultado es
  aleatorio; se decide con una regla aparte.
* **Utilidad** del estado en que se termina el turno::

      V(s) = P(s) + w_bono * B(oro + 3) + w_oro * oro + w_banca * banca(s)

  P(s): poder del tablero contra el tablero actual del rival (reglas del
  ambiente). B: bonificación por ahorro que se recibirá la próxima ronda.
  banca(s): poder base guardado en la banca (potencial futuro).

Comparación y experimentos
--------------------------
* ``comparar_algoritmos(env, jugador)``: los tres algoritmos deciden el MISMO
  turno; se comparan utilidad alcanzada, optimalidad, tiempo, nodos, etc.
* ``torneo(fabrica_a, fabrica_b, partidas)``: partidas completas.
* ``entrenar_qlearning(...)``: entrena y guarda la tabla Q.

Desde consola::

    python agentes.py comparar
    python agentes.py torneo 50
    python agentes.py entrenar 3000
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from magic_chess_env import (
    CATALOGO,
    EMPATE,
    Accion,
    ConfiguracionJuego,
    MagicChessEnv,
    TipoAccion,
    Unidad,
    calcular_poder,
)
from utils import (
    COPIAS_PARA_COMBINAR,
    NIVEL_MAX,
    Cronometro,
    MetricasBusqueda,
    calcular_bono_ahorro,
    crear_rng,
)

Clave = Tuple[str, int]          # (nombre, nivel)
AccionPlan = Tuple[str, Any]     # acción del modelo del turno

MAX_TRAZA = 400                  # eventos guardados para animar la búsqueda


# ===========================================================================
# 1. Modelo del turno
# ===========================================================================
@dataclass(frozen=True)
class PesosEvaluacion:
    """Pesos de la utilidad V(s)."""

    bono: float = 1.0      # valor de asegurar la bonificación por ahorro
    oro: float = 0.3       # valor de cada moneda guardada
    banca: float = 0.3     # fracción del poder de la banca que se valora


class ModeloTurno:
    """Copia ligera y MUTABLE del turno de ``jugador``.

    Permite aplicar una acción y luego deshacerla (``aplicar`` devuelve la
    información necesaria para ``deshacer``), que es la operación básica del
    backtracking. No modifica el ambiente real.
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
        self.ingreso = cfg.ingreso_pasivo
        self.umbral = cfg.umbral_ahorro
        self.bono = cfg.bono_ahorro
        self.tablero_rival = [Unidad(0, u.nombre, u.nivel)
                              for u in env.jugadores[env.rival(jugador)].tablero]
        # estado mutable
        self.oro = j.oro
        self.banca: List[Clave] = [u.clave() for u in j.banca]
        self.tablero: List[Clave] = [u.clave() for u in j.tablero]
        self.tienda: List[Optional[str]] = list(j.tienda)
        self._cache_poder: Dict[tuple, float] = {}

    # ----------------------------- estado ------------------------------
    def clave(self) -> tuple:
        """Identificador canónico: el orden de las fichas no importa."""
        return (self.oro, tuple(sorted(self.banca)), tuple(sorted(self.tablero)),
                tuple(sorted(t for t in self.tienda if t)))

    def descripcion(self) -> str:
        def f(us):
            return ",".join(f"{n}{k}" for n, k in sorted(us)) or "-"
        return f"oro={self.oro} tab=[{f(self.tablero)}] banca=[{f(self.banca)}]"

    # ---------------------------- acciones -----------------------------
    def acciones(self) -> List[AccionPlan]:
        """Acciones aplicables (sin duplicados por fichas idénticas)."""
        acc: List[AccionPlan] = []
        if len(self.banca) < self.tam_banca:
            for nombre in sorted({t for t in self.tienda if t}):
                if CATALOGO[nombre].coste <= self.oro:
                    acc.append(("comprar", nombre))
        conteo = Counter(self.banca)
        for clave, n in sorted(conteo.items()):
            if n >= COPIAS_PARA_COMBINAR and clave[1] < NIVEL_MAX:
                acc.append(("combinar", clave))
        if len(self.tablero) < self.cap_tablero:
            for clave in sorted(set(self.banca)):
                acc.append(("a_tablero", clave))
        if len(self.banca) < self.tam_banca:
            for clave in sorted(set(self.tablero)):
                acc.append(("a_banca", clave))
        return acc

    def aplicar(self, a: AccionPlan) -> Any:
        """Ejecuta ``a`` sobre el modelo y devuelve lo necesario para
        deshacerla."""
        tipo, arg = a
        if tipo == "comprar":
            i = self.tienda.index(arg)
            self.tienda[i] = None
            self.oro -= CATALOGO[arg].coste
            self.banca.append((arg, 1))
            return i
        if tipo == "combinar":
            for _ in range(COPIAS_PARA_COMBINAR):
                self.banca.remove(arg)
            self.banca.append((arg[0], arg[1] + 1))
            return None
        if tipo == "a_tablero":
            self.banca.remove(arg)
            self.tablero.append(arg)
            return None
        if tipo == "a_banca":
            self.tablero.remove(arg)
            self.banca.append(arg)
            return None
        raise ValueError(a)

    def deshacer(self, a: AccionPlan, info: Any) -> None:
        """Revierte ``a`` (inversa exacta de ``aplicar``)."""
        tipo, arg = a
        if tipo == "comprar":
            self.banca.remove((arg, 1))
            self.oro += CATALOGO[arg].coste
            self.tienda[info] = arg
        elif tipo == "combinar":
            self.banca.remove((arg[0], arg[1] + 1))
            self.banca.extend([arg] * COPIAS_PARA_COMBINAR)
        elif tipo == "a_tablero":
            self.tablero.remove(arg)
            self.banca.append(arg)
        elif tipo == "a_banca":
            self.banca.remove(arg)
            self.tablero.append(arg)

    # ---------------------------- utilidad -----------------------------
    def poder_tablero(self) -> float:
        k = tuple(sorted(self.tablero))
        if k not in self._cache_poder:
            unidades = [Unidad(0, n, nv) for n, nv in k]
            self._cache_poder[k] = calcular_poder(unidades, self.tablero_rival).total
        return self._cache_poder[k]

    def valor(self) -> float:
        """Utilidad V del estado actual del modelo."""
        bono = calcular_bono_ahorro(self.oro + self.ingreso, self.umbral, self.bono)
        banca = sum(CATALOGO[n].poder(nv) for n, nv in self.banca)
        return (self.poder_tablero() + self.pesos.bono * bono
                + self.pesos.oro * self.oro + self.pesos.banca * banca)

    # --------------------------- traducción ----------------------------
    def a_accion_env(self, a: AccionPlan) -> Accion:
        """Convierte una acción del plan en una ``Accion`` del ambiente."""
        tipo, arg = a
        j = self.env.jugadores[self.jugador]
        if tipo == "terminar":
            return Accion(TipoAccion.TERMINAR)
        if tipo == "comprar":
            return Accion(TipoAccion.COMPRAR, j.tienda.index(arg))
        if tipo == "combinar":
            return Accion(TipoAccion.COMBINAR, tuple(arg))
        if tipo == "a_tablero":
            return Accion(TipoAccion.A_TABLERO,
                          next(i for i, u in enumerate(j.banca) if u.clave() == tuple(arg)))
        if tipo == "a_banca":
            return Accion(TipoAccion.A_BANCA,
                          next(i for i, u in enumerate(j.tablero) if u.clave() == tuple(arg)))
        raise ValueError(a)


def valor_estado_real(env: MagicChessEnv, jugador: int,
                      pesos: Optional[PesosEvaluacion] = None) -> float:
    """V del estado actual del jugador en el ambiente real (misma medida
    para comparar los tres algoritmos)."""
    return ModeloTurno(env, jugador, pesos).valor()


# ===========================================================================
# 2. Backtracking
# ===========================================================================
@dataclass
class ResultadoBusqueda:
    """plan (termina en ("terminar", None)), valor V alcanzado, métricas y
    traza del proceso para la GUI. Cada evento de la traza:
    {"n", "tipo", "accion", "profundidad", "estado", "valor", "mejor"}
    con tipo en {"expandir", "retroceder", "repetido"}."""

    plan: List[AccionPlan]
    valor: float
    metricas: MetricasBusqueda
    traza: List[Dict[str, Any]]


def backtracking(modelo: ModeloTurno, limite_profundidad: Optional[int] = None
                 ) -> ResultadoBusqueda:
    """Búsqueda exhaustiva en profundidad con retroceso.

    En cada nodo se evalúa V y se actualiza el mejor; luego, para cada
    acción aplicable, se APLICA, se explora recursivamente y se DESHACE
    (retroceso). Se guarda la menor profundidad con que se llegó a cada
    estado: si se vuelve a llegar con una profundidad igual o mayor se poda
    (evita ciclos como desplegar / retirar la misma ficha y repetir estados
    alcanzados en otro orden); si se llega por un camino más corto se vuelve
    a explorar, así el plan final es el más corto entre los de mayor V.

    Propiedades: completo y óptimo respecto a V (sin límite de profundidad),
    porque el espacio del turno es finito y se recorre entero. Tiempo
    O(estados alcanzables * b); memoria O(d) en la pila de recursión más
    O(estados) en la tabla de visitados.

    Métricas: nodos generados (acciones aplicadas), expandidos (estados
    explorados) y ``max_frontera`` = profundidad máxima de la pila.
    """
    m = MetricasBusqueda(algoritmo="Backtracking")
    traza: List[Dict[str, Any]] = []
    camino: List[AccionPlan] = []
    visitados = {modelo.clave(): 0}          # estado -> menor profundidad vista
    mejor = {"v": modelo.valor(), "plan": []}

    def evento(tipo: str, accion, valor) -> None:
        if len(traza) < MAX_TRAZA:
            traza.append({"n": m.nodos_expandidos, "tipo": tipo,
                          "accion": None if accion is None else f"{accion[0]}({accion[1]})",
                          "profundidad": len(camino), "estado": modelo.descripcion(),
                          "valor": round(valor, 3), "mejor": round(mejor["v"], 3)})

    def explorar() -> None:
        m.nodos_expandidos += 1
        m.registrar_frontera(len(camino))
        v = modelo.valor()
        if (v > mejor["v"] + 1e-9
                or (abs(v - mejor["v"]) <= 1e-9 and len(camino) < len(mejor["plan"]))):
            mejor["v"], mejor["plan"] = v, list(camino)   # mejor o igual y más corto
        evento("expandir", camino[-1] if camino else None, v)
        if limite_profundidad is not None and len(camino) >= limite_profundidad:
            return
        for a in modelo.acciones():
            m.nodos_generados += 1
            info = modelo.aplicar(a)
            k = modelo.clave()
            if visitados.get(k, float("inf")) <= len(camino) + 1:
                evento("repetido", a, modelo.valor())   # ya explorado (poda)
            else:
                visitados[k] = len(camino) + 1
                camino.append(a)
                explorar()
                camino.pop()
            modelo.deshacer(a, info)                 # retroceso
            evento("retroceder", a, v)

    with Cronometro() as c:
        explorar()
    plan = simplificar_plan(modelo, mejor["plan"], mejor["v"]) + [("terminar", None)]
    m.tiempo_s = c.segundos
    m.longitud_solucion = len(plan)
    m.costo_solucion = round(mejor["v"], 4)        # aquí "costo" = utilidad V
    m.encontro_solucion = True
    m.es_optima = True if limite_profundidad is None else None
    m.extra["estados_visitados"] = len(visitados)
    return ResultadoBusqueda(plan, mejor["v"], m, traza)


def simplificar_plan(modelo: ModeloTurno, plan: List[AccionPlan],
                     valor_objetivo: float) -> List[AccionPlan]:
    """Quita acciones sobrantes del plan (p. ej. desplegar y retirar la misma
    ficha) sin bajar su utilidad. El DFS suele encontrar primero caminos
    largos; esto deja el mismo resultado con menos acciones."""
    def valor_si_aplicable(p):
        hechas, ok = [], True
        for a in p:
            if a not in modelo.acciones():
                ok = False
                break
            hechas.append((a, modelo.aplicar(a)))
        v = modelo.valor() if ok else None
        for a, info in reversed(hechas):
            modelo.deshacer(a, info)
        return v

    pasos = list(plan)
    cambio = True
    while cambio:
        cambio = False
        n = len(pasos)
        for quitar in [(i,) for i in range(n)] + [(i, j) for i in range(n)
                                                  for j in range(i + 1, n)]:
            nuevo = [a for k, a in enumerate(pasos) if k not in quitar]
            v = valor_si_aplicable(nuevo)
            if v is not None and v >= valor_objetivo - 1e-9:
                pasos, cambio = nuevo, True
                break
    return pasos


# ===========================================================================
# 3. Agentes
# ===========================================================================
class AgenteBacktracking:
    """Agente basado en modelo: al empezar su turno lo resuelve con
    ``backtracking`` y ejecuta el plan óptimo una acción por llamada.

    Renovación (fuera de la búsqueda por ser aleatoria): si el plan óptimo
    no compra nada, hay espacio en la banca y tras pagar se conserva al
    menos ``reserva_renovar`` de oro, renueva (máximo ``max_renovaciones``
    por ronda) y vuelve a planificar.
    """

    nombre = "Backtracking"

    def __init__(self, pesos: Optional[PesosEvaluacion] = None,
                 reserva_renovar: int = 3, max_renovaciones: int = 1) -> None:
        self.pesos = pesos or PesosEvaluacion()
        self.reserva_renovar = reserva_renovar
        self.max_renovaciones = max_renovaciones
        self.reiniciar()

    def reiniciar(self) -> None:
        self._plan: List[AccionPlan] = []
        self._ronda = -1
        self._renovaciones = 0
        self._modelo: Optional[ModeloTurno] = None
        self.ultimo_resultado: Optional[ResultadoBusqueda] = None
        self.historial_metricas: List[MetricasBusqueda] = []

    def planificar(self, env: MagicChessEnv, jugador: int) -> ResultadoBusqueda:
        self._modelo = ModeloTurno(env, jugador, self.pesos)
        res = backtracking(self._modelo)
        self.ultimo_resultado = res
        self.historial_metricas.append(res.metricas)
        return res

    def actuar(self, env: MagicChessEnv, jugador: int) -> Accion:
        if env.ronda != self._ronda:
            self._ronda, self._renovaciones, self._plan = env.ronda, 0, []
        if not self._plan:
            res = self.planificar(env, jugador)
            j = env.jugadores[jugador]
            compra = any(a[0] == "comprar" for a in res.plan)
            if (not compra and self._renovaciones < self.max_renovaciones
                    and len(j.banca) < env.config.tamano_banca
                    and j.oro - env.config.coste_renovar >= self.reserva_renovar):
                self._renovaciones += 1
                return Accion(TipoAccion.RENOVAR)
            self._plan = list(res.plan)
        a = self._plan.pop(0)
        try:
            accion = self._modelo.a_accion_env(a)
        except (ValueError, StopIteration):
            accion = None
        if accion is None or not env.es_legal(jugador, accion):
            self._plan = []                          # algo cambió: replanificar
            return Accion(TipoAccion.TERMINAR)
        return accion

    def lineas_estado(self) -> List[str]:
        r = self.ultimo_resultado
        if r is None:
            return [self.nombre, "sin búsquedas aún"]
        m = r.metricas
        prox = self._plan[0][0] if self._plan else "-"
        return [self.nombre,
                f"V* = {r.valor:.2f} (óptimo)",
                f"expandidos: {m.nodos_expandidos}",
                f"generados: {m.nodos_generados}",
                f"prof. máx: {m.max_frontera}",
                f"tiempo: {m.tiempo_s * 1000:.1f} ms",
                f"plan: {len(r.plan)} acc. próx: {prox}"]


@dataclass(frozen=True)
class PerfilHeuristico:
    """Parámetros de la estrategia heurística.

    reserva : oro mínimo a conservar al comprar (3 asegura llegar a 6 tras
        el ingreso de +3 y cobrar la bonificación por ahorro).
    renovar_si_oro : renueva si tras pagar le queda al menos esto.
    max_renovaciones : renovaciones por ronda.
    peso_copias : bono por cada copia ya poseída en la banca (busca mejoras).
    peso_variedad : bono por una ficha DIFERENTE de un elemento que ya está
        en el tablero (busca sinergia).
    """

    nombre: str = "equilibrado"
    reserva: int = 3
    renovar_si_oro: int = 4
    max_renovaciones: int = 1
    peso_copias: float = 1.0
    peso_variedad: float = 0.8


PERFILES = {
    "agresivo": PerfilHeuristico("agresivo", reserva=0, renovar_si_oro=0,
                                 max_renovaciones=2),
    "equilibrado": PerfilHeuristico(),
    "ahorro": PerfilHeuristico("ahorro", reserva=6, renovar_si_oro=10,
                               max_renovaciones=0),
    "combos": PerfilHeuristico("combos", reserva=1, renovar_si_oro=2,
                               max_renovaciones=3, peso_copias=3.0,
                               peso_variedad=0.3),
}


class AgenteHeuristico:
    """Estrategia heurística basada en reglas. En cada llamada evalúa las
    reglas en orden y dispara la primera que se cumple:

    R1 Combinar: hay 2 copias iguales en la banca.
    R2 Desplegar: hay espacio en el tablero -> la ficha de la banca que más
       aumenta el poder del tablero (h = poder con sinergia y ventaja frente
       al rival actual).
    R3 Reemplazar: tablero lleno y cambiar una ficha del tablero por una de
       la banca aumenta el poder -> retirar la del tablero (R2 despliega la
       otra en la siguiente llamada).
    R4 Comprar: la ficha de la tienda con mejor puntaje (aumento de poder +
       bono por copias + bono por variedad), si alcanza el oro y no rompe la
       reserva (salvo que complete una mejora o el tablero tenga espacio sin
       nada para desplegar).
    R5 Renovar: si sobra oro según el perfil.
    R6 Terminar.

    Métricas tipo sistema experto: reglas evaluadas y reglas disparadas.
    """

    def __init__(self, perfil: str = "equilibrado") -> None:
        self.perfil = PERFILES[perfil]
        self.nombre = "Heurística" if perfil == "equilibrado" else f"Heurística-{perfil}"
        self.reiniciar()

    def reiniciar(self) -> None:
        self._ronda = -1
        self._renovaciones = 0
        self._pasos_ronda = 0
        self.ultima_regla = "-"
        self.reglas_evaluadas = 0
        self.reglas_disparadas = 0

    @staticmethod
    def _poder(env, jugador, tablero: List[Unidad]) -> float:
        """Función heurística: poder de un tablero hipotético frente al
        tablero actual del rival."""
        return calcular_poder(tablero, env.jugadores[env.rival(jugador)].tablero).total

    def _disparar(self, regla: str, accion: Accion) -> Accion:
        self.ultima_regla = regla
        self.reglas_disparadas += 1
        return accion

    def actuar(self, env: MagicChessEnv, jugador: int) -> Accion:
        if env.ronda != self._ronda:
            self._ronda, self._renovaciones, self._pasos_ronda = env.ronda, 0, 0
        self._pasos_ronda += 1
        j = env.jugadores[jugador]
        cfg = env.config
        cap = env.capacidad_tablero()
        if self._pasos_ronda > 30:                          # salvaguarda
            return self._disparar("R6 terminar", Accion(TipoAccion.TERMINAR))
        poder_actual = self._poder(env, jugador, j.tablero)

        # R1 combinar
        self.reglas_evaluadas += 1
        comb = j.combinables()
        if comb:
            return self._disparar("R1 combinar", Accion(TipoAccion.COMBINAR, comb[0]))

        # R2 desplegar
        self.reglas_evaluadas += 1
        if j.banca and len(j.tablero) < cap:
            ganancia = [self._poder(env, jugador, j.tablero + [u]) for u in j.banca]
            i = max(range(len(j.banca)), key=lambda k: ganancia[k])
            return self._disparar("R2 desplegar", Accion(TipoAccion.A_TABLERO, i))

        # R3 reemplazar
        self.reglas_evaluadas += 1
        if j.banca and j.tablero and len(j.tablero) >= cap and len(j.banca) < cfg.tamano_banca:
            mejor, mejor_p = None, poder_actual + 1e-9
            for t in range(len(j.tablero)):
                for u in j.banca:
                    p = self._poder(env, jugador, j.tablero[:t] + j.tablero[t + 1:] + [u])
                    if p > mejor_p:
                        mejor, mejor_p = t, p
            if mejor is not None:
                return self._disparar("R3 reemplazar", Accion(TipoAccion.A_BANCA, mejor))

        # R4 comprar
        self.reglas_evaluadas += 1
        if len(j.banca) < cfg.tamano_banca:
            copias = Counter(u.clave() for u in j.banca)
            nombres_tab: Dict[Any, set] = {}
            for u in j.tablero:
                nombres_tab.setdefault(u.elemento, set()).add(u.nombre)
            hay_hueco = len(j.tablero) < cap and not j.banca
            mejor, mejor_p = None, float("-inf")
            for i, n in enumerate(j.tienda):
                if n is None or CATALOGO[n].coste > j.oro:
                    continue
                completa = copias[(n, 1)] + 1 >= COPIAS_PARA_COMBINAR
                if (j.oro - CATALOGO[n].coste < self.perfil.reserva
                        and not completa and not hay_hueco):
                    continue
                nueva = Unidad(-1, n, 1)
                if len(j.tablero) < cap:
                    aporte = self._poder(env, jugador, j.tablero + [nueva]) - poder_actual
                else:
                    aporte = max(self._poder(env, jugador,
                                             j.tablero[:t] + j.tablero[t + 1:] + [nueva])
                                 for t in range(len(j.tablero))) - poder_actual
                variedad = (nueva.elemento in nombres_tab
                            and n not in nombres_tab[nueva.elemento])
                p = (aporte + self.perfil.peso_copias * copias[(n, 1)]
                     + self.perfil.peso_variedad * variedad + 0.1 * CATALOGO[n].poder(1))
                if p > mejor_p:
                    mejor, mejor_p = i, p
            if mejor is not None:
                return self._disparar("R4 comprar", Accion(TipoAccion.COMPRAR, mejor))

        # R5 renovar
        self.reglas_evaluadas += 1
        if (self._renovaciones < self.perfil.max_renovaciones
                and len(j.banca) < cfg.tamano_banca
                and j.oro - cfg.coste_renovar >= self.perfil.renovar_si_oro):
            self._renovaciones += 1
            return self._disparar("R5 renovar", Accion(TipoAccion.RENOVAR))

        # R6 terminar
        self.reglas_evaluadas += 1
        return self._disparar("R6 terminar", Accion(TipoAccion.TERMINAR))

    def lineas_estado(self) -> List[str]:
        return [self.nombre, f"regla: {self.ultima_regla}",
                f"evaluadas: {self.reglas_evaluadas}",
                f"disparadas: {self.reglas_disparadas}",
                f"reserva: {self.perfil.reserva} oro",
                f"renovaciones: {self._renovaciones}/{self.perfil.max_renovaciones}"]


class AgenteQLearning:
    """Q-learning tabular con estado simplificado.

    * Estado: (etapa de la partida, vidas propias, diferencia de vidas,
      nivel de oro, ¿mi tablero es más fuerte que el del rival?).
    * Acciones: el *perfil* de juego para la ronda (agresivo, equilibrado,
      ahorro, combos), que ejecuta ``AgenteHeuristico``.
    * Recompensa: +1/-1 por ronda ganada/perdida, +10/-10 al final.
    * Actualización: Q(s,a) <- Q(s,a) + alfa [r + gamma max Q(s',.) - Q(s,a)].

    Con ``aprendiendo=True`` explora (epsilon-greedy) y actualiza; con False
    solo explota la tabla aprendida.
    """

    nombre = "Q-learning"
    ACCIONES = list(PERFILES)

    def __init__(self, alfa: float = 0.1, gamma: float = 0.95,
                 epsilon: float = 0.1, aprendiendo: bool = False,
                 semilla: Optional[int] = None, archivo: Optional[str] = None) -> None:
        self.alfa, self.gamma, self.epsilon = alfa, gamma, epsilon
        self.aprendiendo = aprendiendo
        self.rng = crear_rng(semilla)
        self.Q: Dict[str, List[float]] = {}
        self.ejecutores = {p: AgenteHeuristico(p) for p in PERFILES}
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

    @staticmethod
    def estado_abstracto(env: MagicChessEnv, jugador: int) -> str:
        yo, otro = env.jugadores[jugador], env.jugadores[env.rival(jugador)]
        etapa = 0 if env.ronda <= 2 else (1 if env.ronda <= 4 else (2 if env.ronda <= 6 else 3))
        dif = max(-2, min(2, yo.vidas - otro.vidas))
        oro = 0 if yo.oro < 3 else (1 if yo.oro < 6 else 2)
        mio = env.poder_equipo(jugador).total
        suyo = env.poder_equipo(env.rival(jugador)).total
        fuerza = 1 if mio > suyo * 1.1 else (-1 if mio < suyo * 0.9 else 0)
        return f"{etapa}|{yo.vidas}|{dif}|{oro}|{fuerza}"

    def _q(self, s: str) -> List[float]:
        return self.Q.setdefault(s, [0.0] * len(self.ACCIONES))

    def _elegir(self, s: str) -> int:
        if self.aprendiendo and self.rng.random() < self.epsilon:
            return self.rng.randrange(len(self.ACCIONES))
        q = self._q(s)
        return self.rng.choice([i for i, v in enumerate(q) if v == max(q)])

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

    def actuar(self, env: MagicChessEnv, jugador: int) -> Accion:
        if env.ronda != self._ronda:
            s = self.estado_abstracto(env, jugador)
            self._actualizar(self._recompensa_ronda(env, jugador), s)
            self._ronda, self._s = env.ronda, s
            self._a = self._elegir(s)
        return self.ejecutores[self.ACCIONES[self._a]].actuar(env, jugador)

    def finalizar(self, env: MagicChessEnv, jugador: int) -> None:
        """Última actualización al terminar la partida."""
        r = self._recompensa_ronda(env, jugador)
        if env.ganador == jugador:
            r += MagicChessEnv.RECOMPENSA_PARTIDA
        elif env.ganador not in (None, EMPATE):
            r -= MagicChessEnv.RECOMPENSA_PARTIDA
        self._actualizar(r, None)

    def lineas_estado(self) -> List[str]:
        perfil = self.ACCIONES[self._a] if self._a is not None else "-"
        lineas = [self.nombre, f"s = {self._s or '-'}", f"perfil: {perfil}"]
        if self._s is not None:
            lineas += [f"Q[{a[:5]}] = {v:+.2f}" for a, v in zip(self.ACCIONES, self._q(self._s))]
        return lineas

    def guardar(self, archivo: str) -> None:
        with open(archivo, "w", encoding="utf-8") as f:
            json.dump({"acciones": self.ACCIONES, "Q": self.Q}, f, indent=1)

    def cargar(self, archivo: str) -> None:
        with open(archivo, encoding="utf-8") as f:
            datos = json.load(f)
        if datos.get("acciones") == self.ACCIONES:
            self.Q = datos["Q"]


# ===========================================================================
# 4. Experimentos
# ===========================================================================
def jugar_partida(env: MagicChessEnv, agentes: List[Any]) -> MagicChessEnv:
    """Juega una partida completa (cada jugador termina su turno y luego el
    otro)."""
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


def comparar_algoritmos(env: MagicChessEnv, jugador: int,
                        pesos: Optional[PesosEvaluacion] = None
                        ) -> List[Dict[str, Any]]:
    """Los tres algoritmos deciden el MISMO turno (cada uno sobre un clon).

    Se mide la utilidad V del estado en que cada uno termina el turno, si
    alcanzó el óptimo (el V del backtracking), tiempo, acciones y nodos /
    reglas. RENOVAR se trata como terminar en esta comparación (cambiaría
    la instancia al azar), para que los tres resuelvan el mismo problema."""
    agentes = {"Backtracking": AgenteBacktracking(pesos, max_renovaciones=0),
               "Heurística": AgenteHeuristico("equilibrado"),
               "Q-learning": FABRICAS["Q-learning"]()}
    filas = []
    referencia = None
    for nombre, agente in agentes.items():
        copia = env.clonar()
        agente.reiniciar()
        acciones: List[str] = []
        with Cronometro() as c:
            while True:
                a = agente.actuar(copia, jugador)
                if a.tipo in (TipoAccion.RENOVAR, TipoAccion.TERMINAR):
                    acciones.append("terminar")
                    break
                copia.step(jugador, a)
                acciones.append(str(a))
        v = valor_estado_real(copia, jugador, pesos)
        fila = {"algoritmo": nombre, "valor_V": round(v, 3),
                "poder_tablero": round(copia.poder_equipo(jugador).total, 3),
                "tiempo_s": c.segundos, "longitud_solucion": len(acciones),
                "plan": " > ".join(acciones)}
        if isinstance(agente, AgenteBacktracking):
            m = agente.ultimo_resultado.metricas
            fila.update(nodos_generados=m.nodos_generados,
                        nodos_expandidos=m.nodos_expandidos,
                        max_frontera=m.max_frontera)
            referencia = v
        if isinstance(agente, AgenteHeuristico):
            fila.update(reglas_evaluadas=agente.reglas_evaluadas,
                        reglas_disparadas=agente.reglas_disparadas)
        filas.append(fila)
    for f in filas:
        f["es_optima"] = f["valor_V"] >= round(referencia, 3) - 1e-6
        f["encontro_solucion"] = True
    return filas


def torneo(fabrica_a: Callable[[], Any], fabrica_b: Callable[[], Any],
           partidas: int = 50, dificultad: str = "normal",
           semilla: int = 0) -> Dict[str, Any]:
    """Partidas entre A y B alternando asiento. Métricas de A: % victorias,
    % empates, rondas promedio, vidas restantes, uso del oro y tiempo de
    decisión (si es backtracking)."""
    a, b = fabrica_a(), fabrica_b()
    gan = emp = 0
    rondas = vidas = uso_oro = 0.0
    tiempos: List[float] = []
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
        tiempos += [m.tiempo_s for m in getattr(a, "historial_metricas", [])]
    res = {"agente": getattr(a, "nombre", "A"), "rival": getattr(b, "nombre", "B"),
           "partidas": partidas, "victorias_%": 100 * gan / partidas,
           "empates_%": 100 * emp / partidas, "rondas_prom": rondas / partidas,
           "vidas_prom": vidas / partidas, "uso_oro_%": 100 * uso_oro / partidas}
    if tiempos:
        res["tiempo_decision_ms"] = 1000 * sum(tiempos) / len(tiempos)
    return res


def entrenar_qlearning(episodios: int = 3000, archivo: Optional[str] = None,
                       dificultad: str = "normal", semilla: int = 0,
                       epsilon_inicial: float = 0.3, epsilon_final: float = 0.02,
                       cada: int = 500) -> AgenteQLearning:
    """Entrena contra una mezcla de rivales (heurística con varios perfiles
    y backtracking) con epsilon decreciente, y guarda la tabla Q."""
    agente = AgenteQLearning(aprendiendo=True, semilla=semilla)
    rivales = [AgenteHeuristico("equilibrado"), AgenteHeuristico("agresivo"),
               AgenteHeuristico("ahorro"), AgenteBacktracking()]
    rng = crear_rng(semilla)
    victorias = 0
    for ep in range(episodios):
        agente.epsilon = (epsilon_inicial
                          + (epsilon_final - epsilon_inicial) * ep / max(1, episodios - 1))
        asiento = ep % 2
        rival = rng.choice(rivales)
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
    agente.guardar(archivo or RUTA_QTABLA)
    return agente


# ===========================================================================
# 5. Fábricas (las usa la GUI para el selector de algoritmo)
# ===========================================================================
RUTA_QTABLA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qtabla.json")

FABRICAS: Dict[str, Callable[[], Any]] = {
    "Heurística": lambda: AgenteHeuristico("equilibrado"),
    "Backtracking": lambda: AgenteBacktracking(),
    "Q-learning": lambda: AgenteQLearning(archivo=RUTA_QTABLA),
}


def _imprimir_tabla(filas: List[Dict[str, Any]], columnas: List[str]) -> None:
    def fmt(v):
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:.4f}" if abs(v) < 1 else f"{v:.2f}"
        return str(v)
    anchos = {c: max(len(c), *(len(fmt(f.get(c))) for f in filas)) for c in columnas}
    print("  ".join(c.ljust(anchos[c]) for c in columnas))
    for f in filas:
        print("  ".join(fmt(f.get(c)).ljust(anchos[c]) for c in columnas))


if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else "comparar"

    if modo == "comparar":
        # Instancia de ejemplo: ronda 5 (tablero de 4), 7 de oro.
        env = MagicChessEnv(ConfiguracionJuego(), semilla=3)
        env.fijar_ronda(5)
        env.fijar_oro(0, 7)
        for n, nv, z in [("Gota", 1, "tablero"), ("Chispa", 2, "tablero"),
                         ("Gota", 1, "banca"), ("Brote", 1, "banca")]:
            env.agregar_unidad(0, n, nv, z)
        for n, z in [("Brasa", "tablero"), ("Ent", "tablero"), ("Musgo", "tablero")]:
            env.agregar_unidad(1, n, 1, z)
        env.fijar_tienda(0, ["Burbuja", "Marea", "Espino"])
        filas = comparar_algoritmos(env, 0)
        _imprimir_tabla(filas, ["algoritmo", "valor_V", "poder_tablero", "es_optima",
                                "tiempo_s", "nodos_generados", "nodos_expandidos",
                                "max_frontera", "reglas_evaluadas", "longitud_solucion"])
        for f in filas:
            print(f"\n{f['algoritmo']}: {f['plan']}")

    elif modo == "torneo":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        filas = [torneo(FABRICAS[a], FABRICAS[b], partidas=n)
                 for a in FABRICAS for b in FABRICAS if a != b]
        _imprimir_tabla(filas, ["agente", "rival", "victorias_%", "empates_%",
                                "rondas_prom", "vidas_prom", "uso_oro_%",
                                "tiempo_decision_ms"])

    elif modo == "entrenar":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
        entrenar_qlearning(n)
        print(f"Tabla Q guardada en {RUTA_QTABLA}")
