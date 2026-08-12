"""Objetivo económico de la Penca Supermatch: E[premio], no P(ganar).

Diferencia estructural con la penca JMLM del Mundial:

1. **El premio se REPARTE entre empatados** (Art. 7a y 8). No hay desempate por
   exactos ni por nada: si tres participaciones terminan con el mismo puntaje,
   cada una cobra $350.000/3. Entonces el objetivo correcto es

       E[premio] = Σ_escenarios P(escenario) · pozo · (nuestras en el máximo / total en el máximo)

   y no P(ser el máximo). Esto castiga converger con el pool: no alcanza con
   empatar arriba, empatar te divide el premio. Y empatar con vos mismo no es
   pérdida — si dos participaciones propias comparten el tope, cobramos las dos partes.

2. **Hay 15 premios por fecha** ($10.000 c/u = $150.000, casi la mitad del premio
   grande) que dependen SOLO de los 8 partidos de esa fecha (Art. 8: "no se acumulan
   los puntos precedentes"). Son 15 loterías cortas donde la varianza paga mucho más
   que en la general.

3. **Múltiples participaciones propias** (Art. 1), o sea portfolio real.

El simulador es Monte Carlo con **estado incremental**: sortea una vez los resultados
(common random numbers) y los picks de los rivales POR SIMULACIÓN (los futuros no
observados se re-sortean en cada sim: condicionar a una sola realización del pool
dejaba que el optimizador explotara huecos muestrales inexistentes en expectativa),
y después permite cambiar un pick propio y reliquidar en O(n_sims) en vez de
re-simular la temporada. Eso es lo que hace viable el ascenso por coordenadas de
src/clausura/strategy.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.clausura.scoring import supermatch_points

# Grilla de trabajo: 0..5 goles por lado cubre >99% de los resultados reales de la
# Primera uruguaya (598 partidos históricos).
MAX_GOALS = 5
SIDE = MAX_GOALS + 1
N_SCORES = SIDE * SIDE


def score_index(gL: int, gV: int) -> int:
    return gL * SIDE + gV


def index_score(idx: int) -> tuple[int, int]:
    return divmod(idx, SIDE)


def points_matrix(preferencial: bool = False) -> np.ndarray:
    """P[pick_idx, actual_idx] → puntos. Precalculada una vez, reusada en todo el MC."""
    m = np.zeros((N_SCORES, N_SCORES), dtype=np.int32)
    for pi in range(N_SCORES):
        pick = index_score(pi)
        for ai in range(N_SCORES):
            m[pi, ai] = supermatch_points(pick, index_score(ai), preferencial)
    return m


_POINTS_NORMAL = points_matrix(False)
_POINTS_PREF = points_matrix(True)


def flatten_grid(grid: np.ndarray) -> np.ndarray:
    """Grilla (n,n) del modelo → vector de probabilidades sobre los N_SCORES índices.

    La masa fuera de la grilla de trabajo (6+ goles) se redistribuye proporcionalmente.
    """
    n = grid.shape[0]
    flat = np.zeros(N_SCORES)
    lim = min(n, SIDE)
    flat[: lim * SIDE].reshape(lim, SIDE)[:, :lim] = grid[:lim, :lim]
    s = flat.sum()
    return flat / s if s > 0 else flat


@dataclass
class PrizeConfig:
    """Premios del reglamento (Art. 7-8). Montos de la penca paga 2026."""
    premio_penca: float = 350_000.0
    premio_fecha: float = 10_000.0
    costo_participacion: float = 400.0


@dataclass
class SimConfig:
    n_sims: int = 2_000
    n_rivales: int = 151         # totalElements del ranking público
    seed: int = 20260807


@dataclass
class SimResult:
    e_premio_total: float
    e_premio_penca: float
    e_premio_fechas: float
    p_gana_penca: float           # P(cobramos algo del premio grande)
    p_gana_alguna_fecha: float
    e_fechas_ganadas: float
    e_puntos_mejor: float
    e_puntos_pool_max: float
    costo: float
    roi: float = field(init=False)

    def __post_init__(self):
        self.roi = (self.e_premio_total - self.costo) / self.costo if self.costo else float("nan")


class SeasonSimulator:
    """Monte Carlo de la temporada con estado incremental.

    Sortea una sola vez los resultados de los partidos (common random numbers) y
    los picks de los rivales por simulación, y mantiene los acumulados propios
    para poder cambiar un pick y reliquidar barato.
    """

    def __init__(
        self,
        grids: list[np.ndarray],
        fecha_de_partido: list[int],
        preferencial: list[bool],
        pool_q: list[np.ndarray],
        prize: PrizeConfig | None = None,
        sim: SimConfig | None = None,
        rivals=None,
        compactar_fechas: bool = True,
    ):
        """`rivals` (opcional): un RivalModel de src.clausura.rivals — pool empírico
        por participación (picks conocidos, estilo γ, ausentismo, residuo vs tabla
        real). Sin él, el camino clásico: R rivales i.i.d. de la Q agregada."""
        self.prize = prize or PrizeConfig()
        self.cfg = sim or SimConfig()
        rng = np.random.default_rng(self.cfg.seed)

        self.n_matches = len(grids)
        self.preferencial = list(preferencial)
        self.pm = [_POINTS_PREF if p else _POINTS_NORMAL for p in preferencial]

        fechas = sorted(set(fecha_de_partido))
        self.fecha_idx = {f: i for i, f in enumerate(fechas)}
        self.match_fecha = [self.fecha_idx[f] for f in fecha_de_partido]
        self.n_fechas = len(fechas)

        self.rival_model = rivals
        S = self.cfg.n_sims
        R = self.n_rivales = rivals.n_rivales if rivals is not None else self.cfg.n_rivales

        # resultados sorteados: (n_matches, S)
        self.actual = np.stack([
            rng.choice(N_SCORES, size=S, p=flatten_grid(g)) for g in grids
        ])

        # picks de los rivales, POR SIMULACIÓN: (R, S) por partido → acumulados.
        #
        # El acumulado POR FECHA no se guarda entero: era una matriz
        # (n_fechas, R, S) —403 MB con S=9.600 y R=700, en un droplet de 1 GB— y de
        # ella solo se usa su (máximo, empatados) por simulación, que ocupa
        # (n_fechas, 2, S) = 2 MB. Ese array era el techo que impedía subir n_sims,
        # que es de donde sale la plata (a 2.400 sorteos el ascenso fitea ruido).
        #
        # Se acumula con UN buffer por fecha abierta, que se liquida a stats y se
        # libera apenas entra el último partido de esa fecha. El recorrido sigue
        # siendo m = 0..n_matches-1 en orden: agrupar por fecha cambiaría la
        # secuencia del rng y con ella los picks rivales, o sea que dejaría de ser
        # una optimización para pasar a mover los resultados. Con el fixture real
        # (fechas contiguas) hay un solo buffer vivo a la vez.
        self._cache_total: tuple[np.ndarray, np.ndarray] | None = None
        self._cache_fecha: list[tuple[np.ndarray, np.ndarray] | None] | None = None
        self.rivals_total = np.zeros((R, S), dtype=np.int32)
        self.rivals_fecha = None          # compactado; ver _stats_fecha

        # `compactar_fechas=False` conserva la matriz entera. No es para producción:
        # existe para inspeccionar los puntos de UN rival en UNA fecha (tests del
        # ausentismo, debugging). El camino compacto y el completo dan resultados
        # idénticos — hay un test que lo fija.
        ultimo_de_fecha = {}
        for m in range(self.n_matches):
            ultimo_de_fecha[self.match_fecha[m]] = m
        buffers: dict[int, np.ndarray] = {}
        self._cache_fecha = [None] * self.n_fechas
        if not compactar_fechas:
            self.rivals_fecha = np.zeros((self.n_fechas, R, S), dtype=np.int32)
            self._cache_fecha = None

        def _acumular(m: int, pts: np.ndarray, en_total: bool = True) -> None:
            fi = self.match_fecha[m]
            if en_total:
                self._rivals_total += pts
            if not compactar_fechas:
                self._rivals_fecha[fi] += pts
                return
            buf = buffers.get(fi)
            if buf is None:
                buf = buffers[fi] = np.zeros((R, S), dtype=np.int32)
            buf += pts
            if ultimo_de_fecha[fi] == m:       # no entran más partidos de esta fecha
                self._cache_fecha[fi] = self._stats(buf)
                del buffers[fi]

        if rivals is not None:
            for m in range(self.n_matches):
                if rivals.jugado_sin_observar(m):
                    # Jugado DESPUÉS del snapshot: los puntos reales de este partido
                    # ya viajan en el residuo (puntos vivos del ranking), así que al
                    # TOTAL no va nada acá. Pero el premio por FECHA computa solo sus
                    # partidos (Art. 8) y el residuo no llega ahí: sin esta imputación
                    # los R rivales sumaban 0 en el partido y el premio de $10k de la
                    # fecha se simulaba regalado. Se imputa como futuro (pick ∝ Q^γ,
                    # show ~ p_show) contra el resultado real, SOLO para la fecha.
                    rp, show = rivals.sample_picks_match(m, pool_q[m], rng, S,
                                                         forzar_futuro=True)
                    pts = self.pm[m][rp, self.actual[m][None, :]] * show
                    _acumular(m, pts, en_total=False)
                    continue
                rp, show = rivals.sample_picks_match(m, pool_q[m], rng, S)
                pts = self.pm[m][rp, self.actual[m][None, :]]
                pts = pts * show   # no cargó → 0 puntos
                _acumular(m, pts)
            # ancla a la tabla real: puntos del ranking − puntos implicados. Va SOLO
            # al total: los premios por fecha computan únicamente sus partidos (Art. 8).
            self.rivals_total = self._rivals_total + rivals.residuo.astype(np.int32)[:, None]
        else:
            for m in range(self.n_matches):
                # int16: índices de marcador, 110 MB → 27 por partido a S=19.200
                rp = rng.choice(N_SCORES, size=(R, S), p=pool_q[m]).astype(np.int16)
                pts = self.pm[m][rp, self.actual[m][None, :]]
                _acumular(m, pts)
            self.rivals_total = self._rivals_total

        # estado propio (se setea con load_picks)
        self.picks: np.ndarray | None = None
        self.mine_total: np.ndarray | None = None
        self.mine_fecha: np.ndarray | None = None

        # especiales (se activan con enable_campeon / enable_goleador)
        self._rng_especiales = np.random.default_rng(self.cfg.seed + 1)
        self.champ_sim: np.ndarray | None = None       # (S,) equipo campeón por sim
        self.gol_sim: np.ndarray | None = None         # (S,) goleador por sim
        self.campeon_picks: np.ndarray | None = None   # (n_mine,) equipo elegido
        self.goleador_picks: np.ndarray | None = None  # (n_mine,) opción elegida

    # ---------- lado rival: (máximo, empatados) cacheados ----------
    #
    # El lado rival NO cambia entre evaluaciones: se arma en el constructor (más los
    # especiales) y de ahí en más es constante, mientras el ascenso por coordenadas
    # cambia SOLO picks propios. Pero `_liquidar` recalculaba `rivals.max(axis=0)` y el
    # conteo de empatados en cada una de las ~25.000 evaluaciones por corrida, sobre
    # matrices (R,S) con R≈700 — o sea el 98% del trabajo era re-derivar lo mismo.
    # Medido 2026-08-08: cachear da 34-35x por evaluación (14,5 → 0,42 ms) con
    # resultados bit a bit idénticos, y con ese margen el ascenso puede correr a
    # n_sims 9.600 en vez de 2.400, que es donde deja de fitear ruido (+$10.206 ± 1.126).
    #
    # El caché es LAZY a propósito: `backtest.realized_prizes` y
    # `scripts/sweep_participaciones.build_sim` reescriben el lado rival DESPUÉS del
    # constructor, así que uno eager los corrompería en silencio. Las asignaciones
    # enteras (`sim.rivals_total = ...`, `sim.rivals_total += ...`) pasan por el setter
    # e invalidan solas. Lo que el setter NO puede ver es la mutación por índice
    # (`sim.rivals_fecha[fi] += pts`), así que al materializar el caché los arrays
    # quedan read-only: una mutación tardía tira ValueError en vez de devolver premios
    # calculados contra un máximo viejo.

    @property
    def rivals_total(self) -> np.ndarray:
        return self._rivals_total

    @rivals_total.setter
    def rivals_total(self, value: np.ndarray) -> None:
        self._rivals_total = value
        if value.base is None:
            value.setflags(write=True)
        self._cache_total = None

    @property
    def rivals_fecha(self) -> np.ndarray | None:
        """Acumulado rival por fecha, o None si está compactado (el caso normal).

        El constructor NO lo guarda: liquida cada fecha a (máximo, empatados) y tira
        la matriz. Escribirle una entera es un camino soportado —lo usa
        `backtest.realized_prizes`, que reemplaza el lado rival después del
        constructor— y a partir de ahí las stats se derivan de ella.
        """
        return self._rivals_fecha

    @rivals_fecha.setter
    def rivals_fecha(self, value: np.ndarray | None) -> None:
        self._rivals_fecha = value
        if value is not None and value.base is None:
            value.setflags(write=True)
        self._cache_fecha = None

    @staticmethod
    def _stats(rivals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(máximo del pool por sim, cuántos rivales lo empatan)."""
        top = rivals.max(axis=0)
        return top, (rivals == top[None, :]).sum(axis=0)

    def _stats_total(self) -> tuple[np.ndarray, np.ndarray]:
        if self._cache_total is None:
            self._rivals_total.setflags(write=False)
            self._cache_total = self._stats(self._rivals_total)
        return self._cache_total

    def _stats_fecha(self, fi: int) -> tuple[np.ndarray, np.ndarray]:
        if self._cache_fecha is None:
            if self._rivals_fecha is None:
                raise RuntimeError(
                    "no hay lado rival por fecha: se compactó en el constructor y "
                    "después alguien lo invalidó sin escribir uno nuevo"
                )
            self._rivals_fecha.setflags(write=False)
            self._cache_fecha = [None] * self.n_fechas
        if self._cache_fecha[fi] is None:
            if self._rivals_fecha is None:
                raise RuntimeError(f"fecha {fi} sin stats ni matriz rival")
            self._rivals_fecha.setflags(write=False)
            self._cache_fecha[fi] = self._stats(self._rivals_fecha[fi])
        return self._cache_fecha[fi]

    # ---------- especiales: Campeón y Goleador (25 pts c/u, solo total general) ----------

    def enable_campeon(
        self,
        local_de: np.ndarray,
        visita_de: np.ndarray,
        n_teams: int,
        pool_q_campeon: np.ndarray,
        puntos: int = 25,
    ) -> None:
        """Activa el especial Campeón: lo deriva de los MISMOS resultados sorteados.

        Eso preserva la correlación clave: la participación que pica campeón X cobra
        sus 25 puntos exactamente en los escenarios donde los partidos de X salieron
        bien — no en un sorteo independiente.

        Los picks de los rivales se sortean de pool_q_campeon y sus 25 puntos se
        suman a rivals_total (los especiales no cuentan para premios por fecha,
        Art. 8: la fecha computa "sí y solo sí" sus propios partidos).
        """
        from src.clausura.especiales import champions_from_results
        self._puntos_especial = puntos
        self.champ_sim = champions_from_results(
            self.actual, local_de, visita_de, n_teams, self._rng_especiales
        )
        rival_picks = self._rng_especiales.choice(
            n_teams, size=(self.n_rivales, self.cfg.n_sims), p=pool_q_campeon
        )
        # con modelo empírico, el campeón OBSERVADO de cada rival pisa al sorteado
        if self.rival_model is not None and self.rival_model.campeon_idx is not None:
            known = self.rival_model.campeon_idx
            rival_picks = np.where(known[:, None] >= 0, known[:, None], rival_picks)
        rival_picks = self._silenciar(rival_picks, "sin_campeon")
        self.rivals_total += puntos * (rival_picks == self.champ_sim[None, :])
        self.campeon_picks = None  # nuestras elecciones arrancan vacías (0 puntos)

    def enable_goleador(self, p_goleador: np.ndarray, pool_q_goleador: np.ndarray,
                        puntos: int = 25) -> None:
        """Activa el especial Goleador: categórica independiente de los partidos.

        Limitación honesta: no hay nivel jugador en las grillas, así que el goleador
        se sortea de su prior sin correlación con los marcadores. El campeón sí está
        correlacionado; el goleador queda como aproximación independiente.
        """
        self._puntos_especial = puntos
        n_op = len(p_goleador)
        self.gol_sim = self._rng_especiales.choice(n_op, size=self.cfg.n_sims, p=p_goleador)
        rival_picks = self._rng_especiales.choice(
            n_op, size=(self.n_rivales, self.cfg.n_sims), p=pool_q_goleador
        )
        # igual que el campeón: el goleador OBSERVADO pisa al sorteado
        if self.rival_model is not None and self.rival_model.goleador_idx is not None:
            known = self.rival_model.goleador_idx
            rival_picks = np.where(known[:, None] >= 0, known[:, None], rival_picks)
        rival_picks = self._silenciar(rival_picks, "sin_goleador")
        self.rivals_total += puntos * (rival_picks == self.gol_sim[None, :])
        self.goleador_picks = None

    def _silenciar(self, rival_picks: np.ndarray, campo: str) -> np.ndarray:
        """-1 a los rivales que no cargaron ESE especial: nunca matchean el resultado.

        Sin esto se les sortea un pick del pool y cobran como cualquiera. Con 217 de
        690 en blanco (post-lock del Clausura 2026) eso repartía ~7 puntos de
        esperanza fantasma a un tercio del pool, justo en la cola que define el
        premio: inflaba el umbral para ganar y nos hacía ver más difícil el torneo
        de lo que es."""
        marca = getattr(self.rival_model, campo, None) if self.rival_model else None
        if marca is None:
            return rival_picks
        return np.where(marca[:, None], -1, rival_picks)

    def set_campeon_pick(self, i: int, team: int) -> None:
        """Cambia el campeón de la participación i, con update incremental."""
        if self.champ_sim is None:
            raise RuntimeError("enable_campeon() primero")
        if self.campeon_picks is None:
            self.campeon_picks = np.full(self.mine_total.shape[0], -1, dtype=np.int64)
        old = int(self.campeon_picks[i])
        if old == team:
            return
        if old >= 0:
            self.mine_total[i] -= self._puntos_especial * (self.champ_sim == old)
        self.mine_total[i] += self._puntos_especial * (self.champ_sim == team)
        self.campeon_picks[i] = team

    def set_goleador_pick(self, i: int, opcion: int) -> None:
        """Cambia el goleador de la participación i, con update incremental."""
        if self.gol_sim is None:
            raise RuntimeError("enable_goleador() primero")
        if self.goleador_picks is None:
            self.goleador_picks = np.full(self.mine_total.shape[0], -1, dtype=np.int64)
        old = int(self.goleador_picks[i])
        if old == opcion:
            return
        if old >= 0:
            self.mine_total[i] -= self._puntos_especial * (self.gol_sim == old)
        self.mine_total[i] += self._puntos_especial * (self.gol_sim == opcion)
        self.goleador_picks[i] = opcion

    # ---------- estado propio ----------

    def match_points(self, m: int, pick_idx: int) -> np.ndarray:
        """Puntos que da ese pick en el partido m, por simulación. (S,)"""
        return self.pm[m][pick_idx, self.actual[m]]

    def load_picks(self, picks: np.ndarray) -> None:
        """Carga el portfolio completo (n_participaciones, n_matches) y acumula.

        Resetea también los picks de especiales: hay que re-setearlos después de
        cada load_picks (set_campeon_pick / set_goleador_pick).
        """
        if picks.shape[1] != self.n_matches:
            raise ValueError(f"picks tiene {picks.shape[1]} partidos, se esperaban {self.n_matches}")
        n_mine, S = picks.shape[0], self.cfg.n_sims
        self.picks = picks.copy()
        self.mine_total = np.zeros((n_mine, S), dtype=np.int32)
        self.mine_fecha = np.zeros((self.n_fechas, n_mine, S), dtype=np.int32)
        for m in range(self.n_matches):
            pts = self.pm[m][picks[:, m][:, None], self.actual[m][None, :]]
            self.mine_total += pts
            self.mine_fecha[self.match_fecha[m]] += pts
        self.campeon_picks = None
        self.goleador_picks = None

    def set_pick(self, i: int, m: int, pick_idx: int) -> None:
        """Cambia el pick de la participación i en el partido m, en O(n_sims)."""
        if self.picks is None:
            raise RuntimeError("load_picks() primero")
        old = int(self.picks[i, m])
        if old == pick_idx:
            return
        delta = self.match_points(m, pick_idx) - self.match_points(m, old)
        self.mine_total[i] += delta
        self.mine_fecha[self.match_fecha[m]][i] += delta
        self.picks[i, m] = pick_idx

    # ---------- liquidación ----------

    def _liquidar(self, mine: np.ndarray, rival_top: np.ndarray,
                  rival_empatados: np.ndarray, pozo: float) -> np.ndarray:
        """Premio cobrado en cada simulación, con reparto entre empatados (Art. 7a).

        `rival_top`/`rival_empatados` vienen cacheados de `_stats_*`. Equivalente
        exacto a contar los rivales en el máximo global: si el pool no llega al
        máximo, ningún rival lo empata y su aporte al reparto es 0.
        """
        top = np.maximum(mine.max(axis=0), rival_top)
        k = (mine == top[None, :]).sum(axis=0)
        j = np.where(rival_top == top, rival_empatados, 0)
        total = k + j
        return np.where(total > 0, pozo * k / np.maximum(total, 1), 0.0)

    def e_premio_total(self) -> float:
        """Objetivo escalar: E[premio del portfolio]. Es lo que optimiza la estrategia.

        Incluye los premios por fecha YA liquidados (las fechas pasadas entran como
        delta y aportan una constante), así que el número reportado en la planilla y
        el dashboard es mayor que la plata que queda por delante. Para el argmax da
        igual —una constante no mueve el óptimo— y para el gate del rerun también,
        porque compara dos planillas sobre los mismos sorteos y la constante se
        cancela. No confundirlo con "lo que falta ganar": ese número no es este.
        Y ver la nota de [[auditoria-clausura]]: los NIVELES no son creíbles, lo
        único que vale son los deltas pareados.
        """
        rt, rc = self._stats_total()
        premio = self._liquidar(self.mine_total, rt, rc, self.prize.premio_penca)
        for fi in range(self.n_fechas):
            ft, fc = self._stats_fecha(fi)
            premio = premio + self._liquidar(
                self.mine_fecha[fi], ft, fc, self.prize.premio_fecha
            )
        return float(premio.mean())

    def result(self) -> SimResult:
        """Reporte completo (más caro que e_premio_total, para el final)."""
        rt, rc = self._stats_total()
        premio_penca = self._liquidar(self.mine_total, rt, rc, self.prize.premio_penca)
        premio_fechas = np.zeros(self.cfg.n_sims)
        fechas_ganadas = np.zeros(self.cfg.n_sims)
        for fi in range(self.n_fechas):
            ft, fc = self._stats_fecha(fi)
            cobro = self._liquidar(
                self.mine_fecha[fi], ft, fc, self.prize.premio_fecha
            )
            premio_fechas += cobro
            fechas_ganadas += (cobro > 0).astype(float)

        n_mine = self.mine_total.shape[0]
        return SimResult(
            e_premio_total=float((premio_penca + premio_fechas).mean()),
            e_premio_penca=float(premio_penca.mean()),
            e_premio_fechas=float(premio_fechas.mean()),
            p_gana_penca=float((premio_penca > 0).mean()),
            p_gana_alguna_fecha=float((fechas_ganadas > 0).mean()),
            e_fechas_ganadas=float(fechas_ganadas.mean()),
            e_puntos_mejor=float(self.mine_total.max(axis=0).mean()),
            e_puntos_pool_max=float(rt.mean()),
            costo=n_mine * self.prize.costo_participacion,
        )


def simulate(
    grids: list[np.ndarray],
    fecha_de_partido: list[int],
    preferencial: list[bool],
    our_picks: np.ndarray,
    pool_q: list[np.ndarray],
    prize: PrizeConfig | None = None,
    sim: SimConfig | None = None,
    rivals=None,
) -> SimResult:
    """Wrapper de un solo tiro: construye el simulador, carga picks y liquida."""
    s = SeasonSimulator(grids, fecha_de_partido, preferencial, pool_q, prize, sim, rivals)
    s.load_picks(our_picks)
    return s.result()


def picks_to_index_matrix(picks: list[list[tuple[int, int]]]) -> np.ndarray:
    """[[(gL,gV) por partido] por participación] → matriz de índices."""
    return np.array(
        [[score_index(min(gL, MAX_GOALS), min(gV, MAX_GOALS)) for gL, gV in fila] for fila in picks],
        dtype=np.int64,
    )
