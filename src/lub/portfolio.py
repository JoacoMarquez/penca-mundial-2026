"""Portfolio de K participaciones: picks de la fecha actual + especiales → max E[premio].

Mismo enfoque que el Clausura (ascenso por coordenadas con sorteos comunes), en chico:

- Fecha actual: cada participación elige una de las 10 clases por partido.
- Fechas futuras: no se deciden hoy (se re-optimizan cuando llegan, con cuotas). Para
  valuar el premio de temporada se usa una POLÍTICA: cada participación elige
  argmax_c(E[pts_c] + ruido propio), que imita la diversificación que el optimizador
  va a hacer fecha a fecha.
- Especiales: campeón por participación (goleador cuando el API publique las opciones).

E[premio] = Σ_fechas E[premio_fecha] + E[premio_penca], con reparto en empates y
contra los rivales ya simulados en el Sorteo (riv_*_max / riv_*_cnt).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.lub.scoring import KERNEL, N_CLASES, expected_points
from src.lub.season import (PREMIO_FECHA, PREMIO_PENCA, PTS_ESPECIAL, PTS_EXACTO,
                            RHO_EXACTO, Previos, Sorteo, fechas_abiertas)


@dataclass
class Portfolio:
    picks_actual: np.ndarray        # (K, n_actual) clases de la fecha actual
    campeon: np.ndarray             # (K,) índice de equipo
    goleador: np.ndarray | None     # (K,) índice de candidato
    e_premio: float
    detalle: dict


class Evaluador:
    """Puntos por sorteo de K participaciones y liquidación incremental."""

    def __init__(self, so: Sorteo, K: int, slots_actual: list[int], seed: int = 7,
                 ruido_politica: float = 0.35, goleador_real: np.ndarray | None = None,
                 n_goleadores: int = 0, previos: Previos | None = None):
        """previos: temporada arrancada. Los partidos jugados no se valúan con la política
        (sus puntos reales entran como constantes: total vía tot_prop, fecha abierta vía
        parcial_prop) y las fechas ya cerradas no suman premio: E[premio] pasa a ser lo
        que TODAVÍA se puede ganar."""
        self.so, self.K = so, K
        S, J = so.clase.shape
        self.S = S
        self.actual = list(slots_actual)
        rng = np.random.default_rng(seed)
        F = len(so.fechas)
        self.fecha_actual = int(so.fecha_de_slot[self.actual[0]]) if self.actual else -1
        # exacto: sorteo fijo (CRN) por (slot, participación)
        self.u_exacto = rng.random((S, J, K), dtype=np.float32) < RHO_EXACTO
        # puntos de la política futura (todo slot que no es de la fecha actual)
        mult = np.where(so.pref, 2, 1).astype(np.int16)
        self.fecha_pts = np.zeros((S, F, K), np.int16)
        jugados = ({j for j, sl in enumerate(so.slots) if sl.resultado is not None}
                   if previos is not None else set())
        futuros = [j for j in range(J) if j not in set(self.actual) and j not in jugados]
        K_f = KERNEL.astype(np.float32)
        for j in futuros:
            ev = so.probs[:, j] @ K_f.T                                   # (S, 10) E[pts] por clase
            z = rng.normal(0, ruido_politica, (S, K, N_CLASES)).astype(np.float32)
            pk = (ev[:, None, :] + z).argmax(-1)                          # (S, K)
            pts = KERNEL[pk, so.clase[:, j][:, None]].astype(np.int16)
            pts += ((pk == so.clase[:, j][:, None]) & self.u_exacto[:, j]).astype(np.int16) * PTS_EXACTO
            pts *= so.jugado[:, j][:, None] * mult[j]
            self.fecha_pts[:, so.fecha_de_slot[j]] += pts
        self.total_offset = np.zeros(K, np.int16)
        self.fecha_mask = np.ones(F, bool)
        if previos is not None:
            fidx = {f: i for i, f in enumerate(so.fechas)}
            for f, pts in previos.parcial_prop.items():
                if f in fidx:
                    self.fecha_pts[:, fidx[f]] += np.asarray(pts, np.int16)[None, :]
            parciales = sum((np.asarray(v, np.int16) for f, v in previos.parcial_prop.items() if f in fidx),
                            np.zeros(K, np.int16))
            # el total real ya incluye los parciales, que también viven en fecha_pts
            self.total_offset = (np.asarray(previos.tot_prop, np.int16) - parciales).astype(np.int16)
            self.fecha_mask = fechas_abiertas(so)
        self.base_fecha_actual = self.fecha_pts[:, self.fecha_actual].copy() if self.actual else None
        self.mult = mult
        self.goleador_real = goleador_real
        self.n_goleadores = n_goleadores

    # ---- puntos de la fecha actual para una matriz de picks (K, n)
    def pts_actual(self, picks: np.ndarray) -> np.ndarray:
        so = self.so
        out = np.zeros((self.S, self.K), np.int16)
        for i, j in enumerate(self.actual):
            pk = picks[:, i][None, :]                                     # (1, K)
            c = so.clase[:, j][:, None]
            pts = KERNEL[pk, c].astype(np.int16)
            pts += ((pk == c) & self.u_exacto[:, j]).astype(np.int16) * PTS_EXACTO
            out += pts * so.jugado[:, j][:, None] * self.mult[j]
        return out

    def pts_slot(self, j_local: int, clase_pick: int, e: int) -> np.ndarray:
        j = self.actual[j_local]
        so = self.so
        c = so.clase[:, j]
        pts = KERNEL[clase_pick, c].astype(np.int16)
        pts += ((clase_pick == c) & self.u_exacto[:, j, e]).astype(np.int16) * PTS_EXACTO
        return pts * so.jugado[:, j] * self.mult[j]

    def especiales(self, campeon: np.ndarray, goleador: np.ndarray | None) -> np.ndarray:
        pts = (campeon[None, :] == self.so.campeon[:, None]).astype(np.int16) * PTS_ESPECIAL
        if goleador is not None and self.goleador_real is not None:
            pts += (goleador[None, :] == self.goleador_real[:, None]).astype(np.int16) * PTS_ESPECIAL
        return pts

    # ---- liquidación
    @staticmethod
    def _premio(ours: np.ndarray, rmax: np.ndarray, rcnt: np.ndarray, monto: float) -> np.ndarray:
        best = ours.max(-1)
        top = np.maximum(best, rmax)
        n_ours = (ours == top[..., None]).sum(-1)
        n_riv = np.where(rmax == top, rcnt, 0)
        return monto * n_ours / np.maximum(n_ours + n_riv, 1)

    def premios_fecha(self, fecha_pts: np.ndarray) -> np.ndarray:
        """(S, F) premio de fecha por sorteo; cero en las fechas ya cerradas."""
        so = self.so
        return self._premio(fecha_pts, so.riv_fecha_max, so.riv_fecha_cnt, PREMIO_FECHA) * self.fecha_mask

    def total(self, fecha_pts: np.ndarray, esp: np.ndarray) -> np.ndarray:
        """(S, K) puntos de temporada: lo real ya hecho + lo simulado + especiales."""
        return fecha_pts.sum(1) + esp + self.total_offset

    def valor(self, fecha_pts: np.ndarray, esp: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        """(E[premio], premio por sorteo (S,), premio por fecha medio (F,))."""
        so = self.so
        pf = self.premios_fecha(fecha_pts)
        total = self.total(fecha_pts, esp)
        pp = self._premio(total, so.riv_total_max, so.riv_total_cnt, PREMIO_PENCA)
        tot = pf.sum(1) + pp
        return float(tot.mean()), tot, pf.mean(0)


def evaluar(ev: Evaluador, picks: np.ndarray, campeon: np.ndarray,
            goleador: np.ndarray | None = None) -> dict:
    """Valor de un portfolio FIJO (sin optimizar) — para medir fuera de muestra.

    El ascenso elige sobre los mismos sorteos con los que se evalúa: su E[premio] está
    inflado por winner's curse (lección del Clausura). Re-evaluar sobre un Sorteo
    independiente da el número honesto."""
    fecha_pts = ev.fecha_pts.copy()
    if ev.actual:
        fecha_pts[:, ev.fecha_actual] = ev.base_fecha_actual + ev.pts_actual(picks)
    esp = ev.especiales(campeon, goleador)
    v, tot, _ = ev.valor(fecha_pts, esp)
    pp = ev._premio(ev.total(fecha_pts, esp), ev.so.riv_total_max, ev.so.riv_total_cnt, PREMIO_PENCA)
    pf = ev.premios_fecha(fecha_pts)   # (S, F)
    return {"e_premio": v, "se": float(tot.std() / np.sqrt(len(tot))), "e_penca": float(pp.mean()),
            "p_penca": float((pp > 0).mean()), "p_penca_entero": float((pp >= PREMIO_PENCA - 1e-6).mean()),
            "e_fechas": float(v - pp.mean()), "e_n_fechas": float((pf > 0).sum(1).mean()),
            "p_alguna_fecha": float(((pf > 0).sum(1) > 0).mean()), "_tot": tot}


def optimizar(ev: Evaluador, campeon_opts: list[int], campeon_init: np.ndarray | None = None,
              especiales_libres: bool = True, max_pasadas: int = 4, min_delta: float = 1.0,
              goleador_opts: list[int] | None = None,
              goleador_init: np.ndarray | None = None) -> Portfolio:
    """Ascenso por coordenadas: picks de la fecha actual y campeón (y goleador).

    campeon_init / goleador_init: los especiales ya cargados (−1 = ninguno); con
    especiales_libres=False entran a la valuación tal cual y no se mueven."""
    so, K = ev.so, ev.K
    n = len(ev.actual)
    # inicio: pick EV (según el modelo) en todas las participaciones
    ev_pick = [int(expected_points(so.probs[:, j].mean(0)).argmax()) for j in ev.actual]
    picks = np.tile(np.array(ev_pick, int), (K, 1))
    campeon = campeon_init.copy() if campeon_init is not None else np.full(K, campeon_opts[0])
    goleador = (goleador_init.copy() if goleador_init is not None
                else np.full(K, goleador_opts[0]) if goleador_opts else None)
    fecha_pts = ev.fecha_pts.copy()
    if n:
        fecha_pts[:, ev.fecha_actual] = ev.base_fecha_actual + ev.pts_actual(picks)
    esp = ev.especiales(campeon, goleador)
    mejor, tot, _ = ev.valor(fecha_pts, esp)
    inicial = mejor

    for _ in range(max_pasadas):
        cambio = False
        for e in range(K):
            for i in range(n):
                actual_pts = ev.pts_slot(i, picks[e, i], e)
                for c in range(N_CLASES):
                    if c == picks[e, i]:
                        continue
                    nuevo = ev.pts_slot(i, c, e)
                    fecha_pts[:, ev.fecha_actual, e] += nuevo - actual_pts
                    v, _, _ = ev.valor(fecha_pts, esp)
                    if v > mejor + min_delta:
                        mejor, picks[e, i], actual_pts, cambio = v, c, nuevo, True
                    else:
                        fecha_pts[:, ev.fecha_actual, e] -= nuevo - actual_pts
            if especiales_libres:
                for t in campeon_opts:
                    if t == campeon[e]:
                        continue
                    viejo = campeon[e]
                    campeon[e] = t
                    v, _, _ = ev.valor(fecha_pts, ev.especiales(campeon, goleador))
                    if v > mejor + min_delta:
                        mejor, cambio = v, True
                    else:
                        campeon[e] = viejo
                if goleador is not None and goleador_opts:
                    for g in goleador_opts:
                        if g == goleador[e]:
                            continue
                        viejo = goleador[e]
                        goleador[e] = g
                        v, _, _ = ev.valor(fecha_pts, ev.especiales(campeon, goleador))
                        if v > mejor + min_delta:
                            mejor, cambio = v, True
                        else:
                            goleador[e] = viejo
                esp = ev.especiales(campeon, goleador)
        if not cambio:
            break
    v, tot, por_fecha = ev.valor(fecha_pts, esp)
    pp = ev._premio(ev.total(fecha_pts, esp), so.riv_total_max, so.riv_total_cnt, PREMIO_PENCA)
    return Portfolio(picks_actual=picks, campeon=campeon, goleador=goleador, e_premio=v, detalle={
        "e_premio_inicial": inicial, "se": float(tot.std() / np.sqrt(len(tot))),
        "e_penca": float(pp.mean()), "p_penca": float((pp > 0).mean()),
        "e_fechas": float(v - pp.mean()), "e_fecha_actual": float(por_fecha[ev.fecha_actual]) if n else 0.0,
    })
