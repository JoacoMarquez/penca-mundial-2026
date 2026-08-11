"""Asignación de N participaciones al portfolio, optimizando E[premio].

Dos etapas, igual que en la JMLM pero con el objetivo económico correcto:

  1. **Menú de candidatos por partido** — de los 36 marcadores solo unos pocos son
     jugables. Nos quedamos con los mejores por E[pts] y con los de mayor "hueco"
     (E[pts] por unidad de popularidad del pool). El segundo conjunto es el que mete
     al 0-0 y compañía: marcadores frecuentes en la liga que el pool subjuega, donde
     acertar no obliga a repartir el premio.

  2. **Ascenso por coordenadas sobre E[premio] simulado.** La participación 1 ancla en
     EV puro; el resto arranca desde ahí y se perturba partido a partido eligiendo el
     candidato que más sube E[premio] del portfolio COMPLETO. La diversificación
     emerge del objetivo (empatar contigo mismo divide el premio) en vez de imponerse
     por regla, que es la diferencia con las 5 objective functions fijas del Mundial.

Todas las evaluaciones comparten los mismos sorteos (common random numbers, semilla
fija en SimConfig): la diferencia entre dos portfolios es señal, no ruido de MC.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.clausura.economics import (
    N_SCORES,
    PrizeConfig,
    SeasonSimulator,
    SimConfig,
    SimResult,
    flatten_grid,
    index_score,
    score_index,
)
from src.clausura.pool import PoolConfig, pool_distribution
from src.clausura.scoring import expected_points_grid

log = logging.getLogger(__name__)


# Cómo se ordena la rama de "hueco" del menú de candidatos. Solo tiene efecto con
# K_HUECO > 0, que desde el 2026-08-08 ya no es el default (ver K_EV/K_HUECO). Se
# conserva porque el A/B del 8/8 midió `mispricing` peor que `legacy_hueco` por
# Δ E[premio] −$9.486 ± 2.154 (12 reps pareadas, negativo en 10/12, t≈−4.4): si
# algún día se reactiva la rama, se reactiva con esta métrica. Ver Candidato.hueco.
HUECO_METRIC = "legacy_hueco"

# Tamaño del menú de candidatos por partido: top-K_EV por E[pts] ∪ top-K_HUECO por
# hueco.
#
# EL TAMAÑO ÓPTIMO DEPENDE DE n_sims. No es una propiedad del menú, es la interacción
# entre cuántos candidatos hay y cuánto ruido tiene cada comparación. Cada candidato
# extra es una comparación más de dos estimaciones Monte Carlo, o sea un boleto más
# para que el ruido gane el argmax; pero también una chance más de encontrar el pick
# que de verdad conviene. Con pocos sorteos gana el primer efecto, con muchos el
# segundo. Medido el 2026-08-11, mismos brazos, misma verdad, solo cambian los sorteos:
#
#     Δ E[premio] de (5,0) vs (3,0)      a 2.400        a 9.600      a 19.200
#     chalk 1.0                          −$4.192       +$4.231       +$4.568
#     chalk 2.2                          −$1.138       +$9.433       +$9.481
#
# El signo se da vuelta. Por eso K_EV pasó de 5 a 3 el 2026-08-08 (+$9.737 ± 859,
# 16/16 reps, medido a 2.400) y vuelve a 5 hoy: aquella medición era correcta PARA SU
# RÉGIMEN y quedó vencida cuando subimos los sorteos a 9.600 y después a 19.200, sin
# que nada avisara. Producción hoy corre 19.200 (ver deploy/clausura-picks.service).
#
# >>> Si algún día se BAJAN los sorteos, hay que volver a medir esto. <<<
#
# K_HUECO sigue en 0: la rama de rareza mide casi no-op ((3,3) queda −$1.505 a +$1.712
# según el chalk, muy por debajo de (5,0)). Lo que sirve es más candidatos por E[pts],
# no por rareza — que es distinto de lo que suponíamos al escribir la rama.
#
# Reproducir: python scripts/backtest_chalk_menu.py --sims 19200 --reps 3 \
#                    --chalks 1.0,2.2 --menus 3-0,5-0
K_EV = 5
K_HUECO = 0


@dataclass(frozen=True)
class Candidato:
    pick: tuple[int, int]
    e_points: float
    pool_q: float
    p_scoreline: float

    @property
    def hueco(self) -> float:
        """E[pts] por unidad de popularidad del pool: valor NO DISPUTADO.

        DESACTIVADA desde el 2026-08-08 (K_HUECO=0). Lo que sigue explica por qué la
        métrica era la correcta PARA ORDENAR la rama; el barrido posterior mostró que
        la rama entera no paga, por ruido de Monte Carlo y no por mala métrica.

        Sí, ordena casi por rareza —de 1-0 a 1-4 el E[pts] cae a la mitad (2.19 →
        1.00) mientras el pool_q cae 50 veces (18.9% → 0.38%), así que el cociente es
        casi 1/pool_q— y por eso el menú incluye marcadores con P<0.7% (4-2, 1-4, 3-3
        en Liverpool–Albion) y deja afuera al 1-3 (2.28%). Parece un defecto y NO lo
        es: en esta penca el premio se REPARTE entre empatados, así que lo que vale no
        es que el pool se equivoque, es que nadie más lo tenga. Un 0-0 subjugado 2.2×
        lo juega igual el 4.4% del pool (30 rivales con quienes repartir); un 1-4, el
        0.38% (2.6 rivales). Dividir por pool_q compra exclusividad, y el E[pts] en el
        numerador impide que sea rareza pura.

        Se intentó reemplazarla por `mispricing` (P real / P del pool) el 2026-08-08 y
        el backtest la rechazó: Δ E[premio] **−$9.486 ± 2.154** en 12 reps pareadas
        (4 temporadas × 3 semillas, 600 sims), negativo en 10/12, t≈−4.4. Reproducir
        con `python -m src.clausura.backtest --experimento-menu --reps 3 --sims 600`.
        No volver a "arreglar" esto sin correr ese A/B."""
        return self.e_points / (self.pool_q + 1e-4)

    @property
    def mispricing(self) -> float:
        """Cuánto se equivoca el pool: P real / P del pool. RECHAZADA por backtest.

        Intuitivamente mejor que `hueco` —mide desajuste en vez de rareza, y numerador
        y denominador tienden a cero juntos— pero mide PEOR (ver `hueco`), porque
        ignora con cuántos habría que repartir el premio. Se conserva para el brazo B
        del A/B y como registro del resultado negativo."""
        return self.p_scoreline / (self.pool_q + 1e-4)


def build_candidates(
    grid: np.ndarray,
    pool_q: np.ndarray,
    preferencial: bool = False,
    k_ev: int | None = None,
    k_hueco: int | None = None,
    min_prob: float = 0.005,
    metrica: str | None = None,
) -> list[Candidato]:
    """Menú de marcadores jugables: top por E[pts] ∪ top por hueco de pool.

    `k_ev`/`k_hueco` en None toman los defaults de módulo K_EV/K_HUECO, que hoy son
    (3, 0): la rama de hueco está apagada porque el menú chico mide mejor (+$9.737 ±
    859, 16/16 reps — ver el comentario de K_EV). Se resuelven en tiempo de llamada,
    no en la firma, para que el A/B del backtest pueda barrer los tamaños.

    `metrica="mispricing"` es la alternativa rechazada por el A/B del 8/8 y existe
    para reproducirlo; solo tiene efecto con k_hueco > 0.
    """
    k_ev = K_EV if k_ev is None else k_ev
    k_hueco = K_HUECO if k_hueco is None else k_hueco
    metrica = metrica or HUECO_METRIC
    p = flatten_grid(grid)
    cands = [
        Candidato(
            pick=index_score(idx),
            e_points=expected_points_grid(index_score(idx), grid, preferencial),
            pool_q=float(pool_q[idx]),
            p_scoreline=float(p[idx]),
        )
        for idx in range(N_SCORES)
        if p[idx] >= min_prob
    ]
    if not cands:   # grilla degenerada: caemos al modal
        idx = int(np.argmax(p))
        return [Candidato(index_score(idx), expected_points_grid(index_score(idx), grid,
                                                                 preferencial),
                          float(pool_q[idx]), float(p[idx]))]

    by_ev = sorted(cands, key=lambda c: -c.e_points)[:k_ev]
    if k_hueco <= 0:
        by_hueco = []
    elif metrica == "legacy_hueco":
        by_hueco = sorted(cands, key=lambda c: -c.hueco)[:k_hueco]
    else:
        # Desempate por E[pts]: los marcadores "impopulares pero sin sesgo" quedan
        # todos en un mismo escalón de mispricing (~1.5 en el caso medido), y dentro
        # de ese escalón conviene el que más puntos rinde, no el más raro.
        by_hueco = sorted(cands, key=lambda c: (-c.mispricing, -c.e_points))[:k_hueco]

    out, seen = [], set()
    for c in by_ev + by_hueco:
        if c.pick not in seen:
            seen.add(c.pick)
            out.append(c)
    return out


@dataclass
class EspecialesInput:
    """Insumos para optimizar Campeón y Goleador dentro del portfolio."""
    local_de: np.ndarray               # (n_matches,) índice de equipo local
    visita_de: np.ndarray              # (n_matches,) índice de equipo visitante
    n_teams: int
    pool_q_campeon: np.ndarray         # (n_teams,) qué campeón pica el pool
    p_goleador: np.ndarray | None = None       # prior sobre opciones de goleador
    pool_q_goleador: np.ndarray | None = None
    frozen_campeon: np.ndarray | None = None   # (n_part,) equipo ya cargado, -1 = libre
    frozen_goleador: np.ndarray | None = None


@dataclass
class PortfolioClausura:
    picks: np.ndarray                  # (n_participaciones, n_partidos) índices de score
    candidatos: list[list[Candidato]]
    resultado: SimResult
    campeon: np.ndarray | None = None   # (n_participaciones,) índice de equipo
    goleador: np.ndarray | None = None  # (n_participaciones,) índice de opción
    p_campeon: np.ndarray | None = None  # (n_teams,) P(campeón) del modelo
    # Contexto para re-liquidar OTRAS matrices de picks con los mismos sorteos. Lo usa
    # el rerun de cierre para avisar por valor en vez de por diferencia de picks.
    evaluador: "EvaluadorPortfolio | None" = None

    def as_scores(self) -> list[list[tuple[int, int]]]:
        return [[index_score(int(i)) for i in fila] for fila in self.picks]

    def diversidad(self) -> float:
        """Fracción de partidos donde no todas las participaciones juegan lo mismo."""
        return float(np.mean([len(set(self.picks[:, m])) > 1 for m in range(self.picks.shape[1])]))


def build_portfolio(
    grids: list[np.ndarray],
    fecha_de_partido: list[int],
    preferencial: list[bool],
    n_participaciones: int = 5,
    pool_cfg: PoolConfig | None = None,
    prize: PrizeConfig | None = None,
    sim: SimConfig | None = None,
    # 3 → 6 el 2026-08-11. Es el MISMO lever que K_EV: cada pasada extra son más
    # comparaciones de estimaciones Monte Carlo (más chances de que el ruido gane el
    # argmax) pero también más chances de encontrar la mejora real. La auditoría del
    # 8/8 midió 3→6 como "85% overfitting" **explícitamente sin subir sorteos**, y con
    # 19.200 esa conclusión quedó vencida igual que la del menú.
    #
    # Dos mediciones independientes lo respaldan:
    #   * barrido directo de pasadas: +$778 a +$873 (control), ningún brazo negativo;
    #   * los reruns de cierre, que warm-startean desde la planilla de la mañana y por
    #     lo tanto le dan pasadas de más, miden +$1.071 de media sobre ella — 8 de 8
    #     positivos, con errores de 110-180 (scripts/backtest_umbral_aviso.py).
    #
    # La segunda es la que convence: llega al mismo número por un camino distinto, y
    # explica POR QUÉ el rerun venía encontrando plata gratis. Warm start con 3 pasadas
    # ≡ 6 pasadas en frío; la diferencia es que así la plata se captura sin pedirle al
    # usuario que recargue 12 planillas a mano.
    max_passes: int = 6,
    frozen_picks: np.ndarray | None = None,
    frozen_mask: np.ndarray | None = None,
    especiales: EspecialesInput | None = None,
    pool_qs: list[np.ndarray] | None = None,
    rivals=None,
    warm_start: np.ndarray | None = None,
) -> PortfolioClausura:
    """Construye el portfolio de N participaciones maximizando E[premio] simulado.

    `frozen_mask[m]=True` marca partidos cuyo pick YA fue cargado en la web (o ya se
    jugó): en esas columnas se usa `frozen_picks` tal cual y el optimizador no las toca.
    Es el mecanismo de re-optimización fecha a fecha: lo pasado queda fijo, lo futuro
    se replanifica con la información nueva.

    Con `especiales`, Campeón y Goleador entran al mismo ascenso por coordenadas como
    dos columnas más de cada participación (25 pts c/u sobre el total general).

    `pool_qs` permite pasar la distribución del pool por partido ya construida (por
    ejemplo la EMPÍRICA del snapshot post-inicio); sin ella se genera del prior.

    `rivals` (RivalModel de src.clausura.rivals) reemplaza el pool i.i.d. por el
    empírico por participación: picks conocidos, estilo γ, ausentismo y standings
    reales. Es el insumo correcto post-inicio del campeonato.

    `warm_start` (n_participaciones, n_matches) arranca el ascenso desde la planilla
    de la corrida anterior en vez del ancla de EV; -1 en una celda significa "no
    tengo dato" y esa columna cae al ancla. Vale por dos motivos, medidos el
    2026-08-08 con reps pareadas:

      * **+$3.622 ± 699 de E[premio]** (t=5,2). Tres pasadas de ascenso desde un
        punto ya bueno rinden más que tres desde el ancla.
      * **El churn baja de 49% a 22%.** Ese es el beneficio grande y es operativo: lo
        que hoy dispara una recarga manual de ~96 picks no es información nueva sino
        el salto entre óptimos locales equivalentes (el ascenso reasigna la mitad de
        las celdas aun con insumos idénticos, y eso NO se cura con más sorteos —
        medido a S=9600 sigue en 52%). Arrancar de la planilla de ayer hace que la
        de hoy se le parezca salvo donde los datos de verdad cambiaron.
    """
    pool_cfg = pool_cfg or PoolConfig()
    n_matches = len(grids)

    if frozen_mask is None:
        frozen_mask = np.zeros(n_matches, dtype=bool)
    if frozen_mask.any() and frozen_picks is None:
        raise ValueError("frozen_mask sin frozen_picks")

    if pool_qs is None:
        pool_qs = [pool_distribution(g, pool_cfg) for g in grids]
    elif len(pool_qs) != n_matches:
        raise ValueError(f"pool_qs tiene {len(pool_qs)} entradas, se esperaban {n_matches}")
    candidatos = [
        build_candidates(g, q, pref)
        for g, q, pref in zip(grids, pool_qs, preferencial)
    ]

    simulator = SeasonSimulator(grids, fecha_de_partido, preferencial, pool_qs, prize, sim,
                                rivals)

    # punto de partida: warm start donde haya, ancla de EV puro donde no
    if warm_start is not None and warm_start.shape != (n_participaciones, n_matches):
        raise ValueError(
            f"warm_start es {warm_start.shape}, se esperaba "
            f"({n_participaciones}, {n_matches})"
        )
    picks = np.zeros((n_participaciones, n_matches), dtype=np.int64)
    n_warm = 0
    for m in range(n_matches):
        if frozen_mask[m]:
            picks[:, m] = frozen_picks[:, m]
            continue
        # una columna se hereda entera o nada: mezclar warm y ancla dentro del mismo
        # partido inventaría un portfolio que nunca se evaluó
        if warm_start is not None and bool(np.all(warm_start[:, m] >= 0)):
            picks[:, m] = warm_start[:, m]
            n_warm += 1
            continue
        best = max(candidatos[m], key=lambda c: c.e_points)
        picks[:, m] = score_index(*best.pick)
    if warm_start is not None:
        log.info("warm start: %d/%d partidos heredados de la planilla anterior "
                 "(%d congelados, el resto arranca en el ancla EV)",
                 n_warm, n_matches, int(frozen_mask.sum()))
    simulator.load_picks(picks)

    # especiales: activar y anclar en el argmax de probabilidad
    p_champ = None
    if especiales is not None:
        simulator.enable_campeon(
            especiales.local_de, especiales.visita_de,
            especiales.n_teams, especiales.pool_q_campeon,
        )
        from src.clausura.especiales import p_campeon as _p_campeon
        p_champ = _p_campeon(simulator.champ_sim, especiales.n_teams)
        ancla_campeon = int(np.argmax(p_champ))
        for i in range(n_participaciones):
            fijo = especiales.frozen_campeon[i] if especiales.frozen_campeon is not None else -1
            simulator.set_campeon_pick(i, int(fijo) if fijo >= 0 else ancla_campeon)
        if especiales.p_goleador is not None:
            simulator.enable_goleador(especiales.p_goleador, especiales.pool_q_goleador)
            ancla_gol = int(np.argmax(especiales.p_goleador))
            for i in range(n_participaciones):
                fijo = (especiales.frozen_goleador[i]
                        if especiales.frozen_goleador is not None else -1)
                simulator.set_goleador_pick(i, int(fijo) if fijo >= 0 else ancla_gol)

    # ascenso por coordenadas: la participación 0 queda fija como ancla de EV
    actual = simulator.e_premio_total()
    log.info("ancla EV: E[premio]=%.0f", actual)

    for p in range(max_passes):
        mejoras = 0
        for i in range(1, n_participaciones):
            for m in range(n_matches):
                if frozen_mask[m]:
                    continue
                orig = int(simulator.picks[i, m])
                mejor_idx, mejor_val = orig, actual
                for c in candidatos[m]:
                    cand = score_index(*c.pick)
                    if cand == orig:
                        continue
                    simulator.set_pick(i, m, cand)
                    val = simulator.e_premio_total()
                    if val > mejor_val:
                        mejor_idx, mejor_val = cand, val
                simulator.set_pick(i, m, mejor_idx)
                if mejor_idx != orig:
                    mejoras += 1
                    actual = mejor_val

        # especiales como columnas extra (todas las participaciones, incluida la 0:
        # diversificar el campeón es barato y no compromete el ancla de marcadores)
        if especiales is not None:
            for i in range(n_participaciones):
                if (especiales.frozen_campeon is None
                        or especiales.frozen_campeon[i] < 0):
                    actual, cambio = _optimize_especial(
                        simulator, simulator.set_campeon_pick, simulator.campeon_picks,
                        i, especiales.n_teams, actual)
                    mejoras += cambio
                if (simulator.gol_sim is not None
                        and (especiales.frozen_goleador is None
                             or especiales.frozen_goleador[i] < 0)):
                    actual, cambio = _optimize_especial(
                        simulator, simulator.set_goleador_pick, simulator.goleador_picks,
                        i, len(especiales.p_goleador), actual)
                    mejoras += cambio

        log.info("pasada %d: %d cambios, E[premio]=%.0f", p + 1, mejoras, actual)
        if mejoras == 0:
            break

    # El E[premio] que se reporta se evalúa con sorteos FRESCOS (semilla distinta):
    # el valor in-sample del optimizador está sesgado hacia arriba por construcción
    # (el ascenso por coordenadas maximizó exactamente esos sorteos — winner's curse).
    resultado = _evaluate_fresh(
        simulator, grids, fecha_de_partido, preferencial, pool_qs, prize, especiales,
        rivals,
    )
    log.info("E[premio] out-of-sample: $%.0f (in-sample del optimizador: $%.0f)",
             resultado.e_premio_total, actual)

    return PortfolioClausura(
        picks=simulator.picks.copy(),
        candidatos=candidatos,
        resultado=resultado,
        campeon=simulator.campeon_picks.copy() if simulator.campeon_picks is not None else None,
        goleador=simulator.goleador_picks.copy() if simulator.goleador_picks is not None else None,
        p_campeon=p_champ,
        evaluador=EvaluadorPortfolio(
            grids, fecha_de_partido, preferencial, pool_qs, prize, simulator.cfg,
            especiales, rivals,
            campeon_picks=(simulator.campeon_picks.copy()
                           if simulator.campeon_picks is not None else None),
            goleador_picks=(simulator.goleador_picks.copy()
                            if simulator.goleador_picks is not None else None),
        ),
    )


# offset de la semilla de evaluación respecto de la de optimización (fix winner's curse)
EVAL_SEED_OFFSET = 900_001


def _evaluate_fresh(
    simulator: SeasonSimulator,
    grids: list[np.ndarray],
    fecha_de_partido: list[int],
    preferencial: list[bool],
    pool_qs: list[np.ndarray],
    prize: PrizeConfig | None,
    especiales: EspecialesInput | None,
    rivals,
) -> SimResult:
    """Re-liquida el portfolio final en un simulador con semilla independiente."""
    cfg = simulator.cfg
    eval_cfg = SimConfig(n_sims=cfg.n_sims, n_rivales=cfg.n_rivales,
                         seed=cfg.seed + EVAL_SEED_OFFSET)
    ev = SeasonSimulator(grids, fecha_de_partido, preferencial, pool_qs, prize,
                         eval_cfg, rivals)
    ev.load_picks(simulator.picks)
    if especiales is not None and simulator.champ_sim is not None:
        ev.enable_campeon(especiales.local_de, especiales.visita_de,
                          especiales.n_teams, especiales.pool_q_campeon)
        for i in range(simulator.campeon_picks.shape[0]):
            ev.set_campeon_pick(i, int(simulator.campeon_picks[i]))
        if simulator.gol_sim is not None:
            ev.enable_goleador(especiales.p_goleador, especiales.pool_q_goleador)
            for i in range(simulator.goleador_picks.shape[0]):
                ev.set_goleador_pick(i, int(simulator.goleador_picks[i]))
    return ev.result()


@dataclass
class ComparacionPortfolios:
    """Δ E[premio] entre dos planillas, con su error de Monte Carlo."""
    delta: float
    se: float
    valor_a: float
    valor_b: float
    n_seeds: int

    @property
    def significativa(self) -> bool:
        """Δ distinguible de cero con el ruido que tiene."""
        return self.se > 0 and abs(self.delta) > 2.0 * self.se

    def __str__(self) -> str:
        return (f"Δ E[premio] {self.delta:+,.0f} ± {self.se:,.0f} "
                f"({self.valor_a:,.0f} → {self.valor_b:,.0f}, {self.n_seeds} semillas)")


class EvaluadorPortfolio:
    """Compara dos matrices de picks bajo los MISMOS sorteos.

    Existe para que el rerun de cierre pueda avisar por VALOR en vez de por
    diferencia de picks. El 2026-08-08 quedó medido que el óptimo es plano: dos
    corridas con insumos idénticos reasignan 43 de 96 picks sin que nada haya
    cambiado, así que "los picks cambiaron" no dice nada sobre si conviene recargar.

    Dos detalles que hacen que el Δ sea usable:

      * **Sorteos comunes.** Las dos planillas se liquidan en la MISMA instancia del
        simulador, así que comparten resultados y picks rivales. La varianza del Δ
        queda muy por debajo de la de cada valor por separado — comparar dos números
        de corridas distintas (que es lo que hacíamos a mano) mezcla la diferencia
        real con el ruido de dos muestras independientes.
      * **Semillas independientes de la optimización.** Evaluar sobre los sorteos que
        el ascenso por coordenadas maximizó le daría ventaja espuria a la planilla
        nueva (winner's curse), que es exactamente el sesgo que haría avisar de más.
    """

    def __init__(self, grids, fecha_de_partido, preferencial, pool_qs, prize,
                 cfg: SimConfig, especiales=None, rivals=None,
                 campeon_picks=None, goleador_picks=None):
        self._args = (grids, fecha_de_partido, preferencial, pool_qs, prize)
        self._cfg = cfg
        self._especiales = especiales
        self._rivals = rivals
        self._campeon = campeon_picks
        self._goleador = goleador_picks

    def _simulador(self, seed: int) -> SeasonSimulator:
        """Simulador con el lado RIVAL listo. Nuestros picks los pone `_cargar`.

        `enable_campeon`/`enable_goleador` suman los 25 pts de los rivales a
        `rivals_total`, así que van UNA vez por simulador: llamarlos de nuevo se los
        contaría doble. Y no pueden setear nuestros especiales todavía, porque
        `set_campeon_pick` escribe sobre `mine_total`, que no existe hasta el
        `load_picks` de `_cargar`.
        """
        grids, fechas, pref, pool_qs, prize = self._args
        cfg = SimConfig(n_sims=self._cfg.n_sims, n_rivales=self._cfg.n_rivales, seed=seed)
        s = SeasonSimulator(grids, fechas, pref, pool_qs, prize, cfg, self._rivals)
        esp = self._especiales
        if esp is not None and self._campeon is not None:
            s.enable_campeon(esp.local_de, esp.visita_de, esp.n_teams, esp.pool_q_campeon)
            if self._goleador is not None and esp.p_goleador is not None:
                s.enable_goleador(esp.p_goleador, esp.pool_q_goleador)
        return s

    def _cargar(self, s: SeasonSimulator, picks: np.ndarray) -> float:
        """Carga una matriz de picks, RE-APLICA los especiales y liquida.

        Los especiales van después de CADA `load_picks` porque `load_picks` los
        resetea a None (economics.py). Este orden es el fix del 2026-08-08: antes
        `_simulador` intentaba setear el campeón ANTES de que existiera `mine_total`
        y tiraba AttributeError en el 100% de las corridas de producción —
        `rerun_cierre.valor_del_cambio` se lo comía en un except y avisaba igual, así
        que el gate por valor de los PR #147/#148 nunca llegó a correr. Y aun sin el
        crash, comparar sin re-aplicar habría medido las dos planillas SIN los 25+25
        puntos, que es justo lo que más define la tabla general.
        """
        s.load_picks(picks)
        if s.champ_sim is not None and self._campeon is not None:
            for i, op in enumerate(self._campeon):
                s.set_campeon_pick(i, int(op))
        if s.gol_sim is not None and self._goleador is not None:
            for i, op in enumerate(self._goleador):
                s.set_goleador_pick(i, int(op))
        return s.e_premio_total()

    def comparar(self, picks_a, picks_b, n_seeds: int = 5) -> ComparacionPortfolios:
        """Δ = valor(B) − valor(A), promediado sobre `n_seeds` evaluaciones pareadas."""
        deltas, va, vb = [], [], []
        for k in range(n_seeds):
            s = self._simulador(self._cfg.seed + EVAL_SEED_OFFSET + k)
            a = self._cargar(s, picks_a)
            b = self._cargar(s, picks_b)
            deltas.append(b - a)
            va.append(a)
            vb.append(b)
        se = (float(np.std(deltas, ddof=1) / np.sqrt(len(deltas)))
              if len(deltas) > 1 else 0.0)
        return ComparacionPortfolios(delta=float(np.mean(deltas)), se=se,
                                     valor_a=float(np.mean(va)),
                                     valor_b=float(np.mean(vb)), n_seeds=n_seeds)


def _optimize_especial(simulator, setter, current, i, n_opciones, actual) -> tuple[float, int]:
    """Prueba todas las opciones del especial para la participación i. (nuevo_valor, cambió)."""
    orig = int(current[i])
    mejor_op, mejor_val = orig, actual
    for op in range(n_opciones):
        if op == orig:
            continue
        setter(i, op)
        val = simulator.e_premio_total()
        if val > mejor_val:
            mejor_op, mejor_val = op, val
    setter(i, mejor_op)
    return mejor_val, int(mejor_op != orig)


# -------------------- baselines de comparación --------------------

def baseline_chalk(grids: list[np.ndarray], n_participaciones: int = 5) -> np.ndarray:
    """Todas las participaciones al marcador modal del mercado (chalk puro)."""
    picks = np.zeros((n_participaciones, len(grids)), dtype=np.int64)
    for m, g in enumerate(grids):
        picks[:, m] = int(np.argmax(flatten_grid(g)))
    return picks


def baseline_ev(
    grids: list[np.ndarray],
    preferencial: list[bool],
    n_participaciones: int = 5,
) -> np.ndarray:
    """Todas al argmax E[pts] — lo que haría quien optimiza puntaje esperado."""
    picks = np.zeros((n_participaciones, len(grids)), dtype=np.int64)
    for m, g in enumerate(grids):
        evs = [expected_points_grid(index_score(i), g, preferencial[m]) for i in range(N_SCORES)]
        picks[:, m] = int(np.argmax(evs))
    return picks


def baseline_random_diverse(
    grids: list[np.ndarray],
    preferencial: list[bool],
    n_participaciones: int = 5,
    seed: int = 7,
) -> np.ndarray:
    """Diversidad ingenua: cada participación toma un candidato al azar del top-EV.

    Sirve para separar "diversificar ayuda" de "diversificar BIEN ayuda".
    """
    rng = np.random.default_rng(seed)
    picks = np.zeros((n_participaciones, len(grids)), dtype=np.int64)
    for m, g in enumerate(grids):
        evs = np.array([expected_points_grid(index_score(i), g, preferencial[m])
                        for i in range(N_SCORES)])
        top = np.argsort(-evs)[:n_participaciones]
        picks[:, m] = rng.permutation(top)[:n_participaciones]
    return picks
