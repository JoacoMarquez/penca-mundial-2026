"""Tests del control frío del warm start (src.clausura.cold_check).

El warm start tiene un trinquete: un pick heredado solo se abandona si un candidato
del menú de HOY lo supera, y el gate del rerun compara warm-nueva vs warm-vigente,
o sea dos habitantes del mismo pozo. Sin una corrida fría, un óptimo local heredado
es invisible (auditoría 13/8).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from src.clausura.cold_check import (
    formatear_alerta,
    formatear_ok,
    guardar_historial,
    matriz_vigente,
    vale_avisar,
)


@dataclass
class _Comp:
    delta: float
    se: float
    valor_a: float = 100_000.0
    valor_b: float = 105_000.0
    n_seeds: int = 5


# -------------------- umbral --------------------

def test_avisa_solo_si_gana_de_verdad():
    """Doble condición, igual que el gate del rerun: significativo Y con plata."""
    assert vale_avisar(_Comp(delta=5_000, se=500))          # 10 SE y $5k
    assert not vale_avisar(_Comp(delta=5_000, se=4_000))    # plata pero es ruido
    assert not vale_avisar(_Comp(delta=800, se=10))         # nítido pero no paga la recarga
    assert not vale_avisar(_Comp(delta=-9_000, se=100))     # la cadena GANA: no se avisa


def test_umbrales_configurables():
    comp = _Comp(delta=1_500, se=100)
    assert not vale_avisar(comp)                             # default $2.000
    assert vale_avisar(comp, umbral_abs=1_000)


# -------------------- matriz vigente --------------------

def _contexto(picks_frios, eventos=None):
    class _Port:
        picks = picks_frios

    return {"portfolio": _Port(), "eventos": eventos or [{"evento_id": 1}, {"evento_id": 2}]}


def test_matriz_vigente_usa_el_warm_y_rellena_huecos_con_la_fria(monkeypatch):
    """Los -1 (partidos que ninguna planilla previa cubre) se toman de la fría: las
    dos matrices tienen que diferir SOLO donde hay decisión heredada que auditar."""
    from src.clausura import cold_check

    fria = np.array([[7, 8], [9, 10]], dtype=np.int64)
    warm = np.array([[3, -1], [-1, 4]], dtype=np.int64)
    monkeypatch.setattr("src.clausura.picks.load_warm_start",
                        lambda eventos, fecha, n: warm)

    out = matriz_vigente(_contexto(fria), target_fecha=3, n_participaciones=2)
    assert out.tolist() == [[3, 8], [9, 4]]
    assert cold_check is not None


def test_matriz_vigente_sin_cadena_previa_es_none(monkeypatch):
    """Arranque en frío legítimo: no hay nada que auditar todavía."""
    monkeypatch.setattr("src.clausura.picks.load_warm_start",
                        lambda eventos, fecha, n: None)
    fria = np.array([[1, 2]], dtype=np.int64)
    assert matriz_vigente(_contexto(fria), target_fecha=1, n_participaciones=1) is None


# -------------------- reporte e historial --------------------

def test_alerta_dice_el_delta_y_que_no_se_versiona():
    txt = formatear_alerta(_Comp(delta=6_200, se=900), target_fecha=5,
                           n_distintos=31, n_celdas=120)
    assert "6,200" in txt and "900" in txt
    assert "31/120" in txt
    assert "óptimo local" in txt
    assert "no versiona" in txt          # la fría es un control, no una planilla


def test_ok_no_grita():
    txt = formatear_ok(_Comp(delta=-300, se=800), target_fecha=5)
    assert "aguanta" in txt and "🧊" not in txt


def test_historial_acumula_y_se_poda(tmp_path, monkeypatch):
    """Sirve para ver si el trinquete aparece recién en fechas avanzadas, cuando la
    cadena es más profunda."""
    from datetime import datetime, timezone

    from src.clausura import cold_check

    monkeypatch.setattr(cold_check, "STATE_PATH", tmp_path / "cold_check.json")
    monkeypatch.setattr(cold_check, "MAX_HISTORIAL", 3)
    now = datetime(2026, 8, 20, 3, 0, tzinfo=timezone.utc)

    for i in range(5):
        guardar_historial(i, _Comp(delta=float(i), se=1.0), avisó=False, now=now)

    hist = json.loads((tmp_path / "cold_check.json").read_text(encoding="utf-8"))["corridas"]
    assert len(hist) == 3                       # podado
    assert [h["fecha"] for h in hist] == [2, 3, 4]
    assert hist[-1]["delta"] == 4.0 and hist[-1]["aviso"] is False


# -------------------- orquestación --------------------

def test_run_corre_en_frio_sin_guardar_y_compara(monkeypatch, tmp_path):
    """La corrida de control NO se versiona: guardarla la volvería el warm start de
    la próxima corrida, justo la cadena que viene a auditar."""
    from src.clausura import cold_check

    fria = np.array([[5, 6]], dtype=np.int64)
    warm = np.array([[1, 2]], dtype=np.int64)
    llamada = {}

    class _Port:
        picks = fria
        evaluador = None

    class _Ev:
        def comparar(self, a, b, n_seeds):
            llamada["comparados"] = (a.tolist(), b.tolist(), n_seeds)
            return _Comp(delta=9_000, se=500)

    def _picks_run(fecha, n_part, telegram, n_sims, contexto, usar_warm_start, guardar):
        llamada["flags"] = (usar_warm_start, guardar)
        contexto.update(portfolio=_Port(), evaluador=_Ev(),
                        eventos=[{"evento_id": 1}, {"evento_id": 2}], idx_of={})
        return None

    monkeypatch.setattr("src.clausura.picks.run", _picks_run)
    monkeypatch.setattr("src.clausura.picks.load_config", lambda: {})
    monkeypatch.setattr("src.clausura.picks.load_warm_start",
                        lambda eventos, fecha, n: warm)
    monkeypatch.setattr(cold_check, "STATE_PATH", tmp_path / "cold_check.json")

    msg = cold_check.run(fecha=4, n_participaciones=1, n_sims=800, dry_run=True)

    assert llamada["flags"] == (False, False)          # frío y sin versionar
    a, b, seeds = llamada["comparados"]
    assert a == [[1, 2]] and b == [[5, 6]]             # warm vs fría, en ese orden
    assert seeds == cold_check.EVAL_SEEDS
    assert msg is not None and "9,000" in msg


def test_run_calla_si_la_cadena_aguanta(monkeypatch, tmp_path):
    from src.clausura import cold_check

    class _Port:
        picks = np.array([[5]], dtype=np.int64)

    class _Ev:
        def comparar(self, a, b, n_seeds):
            return _Comp(delta=-450, se=700)

    monkeypatch.setattr("src.clausura.picks.run",
                        lambda *a, **k: k["contexto"].update(
                            portfolio=_Port(), evaluador=_Ev(),
                            eventos=[{"evento_id": 1}], idx_of={}))
    monkeypatch.setattr("src.clausura.picks.load_config", lambda: {})
    monkeypatch.setattr("src.clausura.picks.load_warm_start",
                        lambda eventos, fecha, n: np.array([[3]], dtype=np.int64))
    monkeypatch.setattr(cold_check, "STATE_PATH", tmp_path / "cold_check.json")

    assert cold_check.run(fecha=4, n_participaciones=1, n_sims=800, dry_run=True) is None
