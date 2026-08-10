"""El rerun avisa por VALOR, no por diferencia de picks.

Contexto (2026-08-08): el óptimo es plano — dos corridas con insumos idénticos
reasignan ~43 de 96 picks. Ese día el rerun pidió recargar 56 picks hacia una
planilla PEOR ($238k → $221k). El disparador correcto es el Δ E[premio] medido con
sorteos comunes, no la existencia de un diff.
"""

import numpy as np
import pytest

from src.clausura.economics import PrizeConfig, SimConfig, score_index
from src.clausura.rerun_cierre import (
    UMBRAL_ABS,
    picks_previos,
    valor_del_cambio,
    vale_avisar,
)
from src.clausura.strategy import ComparacionPortfolios, EvaluadorPortfolio
from src.model.poisson import score_grid


def comp(delta, se):
    return ComparacionPortfolios(delta=delta, se=se, valor_a=200_000.0,
                                 valor_b=200_000.0 + delta, n_seeds=5)


# -------------------- umbral --------------------

def test_no_avisa_por_churn():
    """Δ chico o negativo = el ruido del optimizador. Silencio."""
    assert not vale_avisar(comp(-16_489, 3_000))     # el caso real del 8/8
    assert not vale_avisar(comp(0, 3_000))
    assert not vale_avisar(comp(500, 100))           # positivo y significativo, pero migaja


def test_no_avisa_si_el_delta_no_supera_su_ruido():
    """Mejora grande pero indistinguible de cero: no se pide recargar 12 planillas."""
    assert not vale_avisar(comp(10_000, 8_000))      # 1.25 se
    assert vale_avisar(comp(10_000, 2_000))          # 5 se


def test_avisa_cuando_la_mejora_es_real():
    assert vale_avisar(comp(25_000, 4_000))
    assert comp(25_000, 4_000).significativa


def test_el_piso_absoluto_es_el_costo_de_recargar():
    assert not vale_avisar(comp(UMBRAL_ABS - 1, 1.0))
    assert vale_avisar(comp(UMBRAL_ABS + 1, 1.0))


# -------------------- comparación pareada --------------------

def _evaluador(n_matches=6, n_part=3):
    grids = [score_grid(1.3, 1.1, 0.0, max_goals=5) for _ in range(n_matches)]
    from src.clausura.pool import PoolConfig, pool_distribution
    pool_qs = [pool_distribution(g, PoolConfig()) for g in grids]
    fechas = [0] * n_matches
    pref = [False] * n_matches
    cfg = SimConfig(n_sims=200, n_rivales=40)
    return EvaluadorPortfolio(grids, fechas, pref, pool_qs, PrizeConfig(), cfg), n_matches, n_part


def test_comparar_una_planilla_contra_si_misma_da_cero_exacto():
    """Sorteos comunes: sin diferencia de picks no puede haber diferencia de valor.

    Es la propiedad que hace usable el Δ — si cada brazo se evaluara con sorteos
    propios, esto daría ruido en vez de cero.
    """
    ev, n_matches, n_part = _evaluador()
    picks = np.full((n_part, n_matches), score_index(1, 1), dtype=np.int64)
    c = ev.comparar(picks, picks, n_seeds=3)
    assert c.delta == 0.0
    assert c.se == 0.0
    assert not c.significativa


def test_comparar_detecta_una_planilla_peor():
    """Chalk razonable vs todas en 5-5: el Δ tiene que ser claramente negativo."""
    ev, n_matches, n_part = _evaluador()
    buena = np.full((n_part, n_matches), score_index(1, 1), dtype=np.int64)
    mala = np.full((n_part, n_matches), score_index(5, 5), dtype=np.int64)
    c = ev.comparar(buena, mala, n_seeds=3)
    assert c.delta < 0
    assert not vale_avisar(c)


# -------------------- reconstrucción de la matriz vieja --------------------

def test_picks_previos_solo_pisa_la_fecha_objetivo():
    class Port:
        picks = np.full((2, 4), score_index(1, 1), dtype=np.int64)

    contexto = {"portfolio": Port(), "idx_of": {100: 1, 101: 2}}
    prev = {"picks": [
        {"evento_id": 100, "scores": [[0, 0], [2, 1]]},
        {"evento_id": 101, "scores": [[3, 0], [0, 3]]},
        {"evento_id": 999, "scores": [[4, 4], [4, 4]]},   # evento ajeno: se ignora
    ]}
    m = picks_previos(prev, contexto, 2)
    assert m[0, 1] == score_index(0, 0) and m[1, 1] == score_index(2, 1)
    assert m[0, 2] == score_index(3, 0) and m[1, 2] == score_index(0, 3)
    # las columnas que la planilla vieja no menciona quedan como en la nueva
    assert m[0, 0] == score_index(1, 1) and m[0, 3] == score_index(1, 1)


def test_valor_del_cambio_sin_evaluador_no_explota():
    """Contexto incompleto → None, y el rerun avisa igual (no se queda mudo)."""
    assert valor_del_cambio({"picks": []}, {}, 12) is None


def test_build_portfolio_deja_el_evaluador_listo():
    """El seam que usa el rerun: el portfolio trae con qué re-liquidar otras picks."""
    from src.clausura.strategy import build_portfolio

    grids = [score_grid(1.3, 1.1, 0.0, max_goals=5) for _ in range(4)]
    port = build_portfolio(
        grids=grids, fecha_de_partido=[0] * 4, preferencial=[False] * 4,
        n_participaciones=3, sim=SimConfig(n_sims=120, n_rivales=30), max_passes=1,
    )
    assert port.evaluador is not None
    otra = np.full_like(port.picks, score_index(0, 0))
    c = port.evaluador.comparar(port.picks, otra, n_seeds=2)
    assert c.n_seeds == 2
    # el portfolio optimizado no puede ser peor que poner 0-0 en todo
    assert c.delta <= 0


# -------------------- el gate con especiales (regresión del 2026-08-08) --------------------

def _portfolio_con_especiales(n_part=3, n_matches=8):
    from src.clausura.strategy import EspecialesInput, build_portfolio

    grids = [score_grid(1.3, 1.1, 0.0, max_goals=5) for _ in range(n_matches)]
    esp = EspecialesInput(
        local_de=np.arange(n_matches) % 4,
        visita_de=(np.arange(n_matches) + 1) % 4,
        n_teams=4,
        pool_q_campeon=np.full(4, 0.25),
        p_goleador=np.array([0.4, 0.3, 0.2, 0.1]),
        pool_q_goleador=np.array([0.4, 0.3, 0.2, 0.1]),
    )
    return build_portfolio(
        grids=grids, fecha_de_partido=[1] * n_matches, preferencial=[False] * n_matches,
        n_participaciones=n_part, sim=SimConfig(n_sims=200, n_rivales=40),
        max_passes=1, especiales=esp,
    )


def test_evaluador_con_especiales_no_crashea():
    """El bug que dejó muerto el gate por valor durante días.

    `_simulador` seteaba el campeón ANTES de cualquier `load_picks`, y
    `set_campeon_pick` escribe sobre `mine_total`, que todavía era None →
    AttributeError en el 100% de las corridas (producción SIEMPRE pasa
    EspecialesInput). `rerun_cierre.valor_del_cambio` se lo comía en un except y
    avisaba por diferencia de picks, que es exactamente lo que los PR #147/#148
    querían dejar de hacer.
    """
    port = _portfolio_con_especiales()
    otra = port.picks.copy()
    otra[1, 0] = (otra[1, 0] + 1) % 36
    c = port.evaluador.comparar(port.picks, otra, n_seeds=2)   # antes: AttributeError
    assert c.n_seeds == 2
    assert np.isfinite(c.delta)


def test_evaluador_reaplica_los_especiales_despues_de_cada_carga():
    """La segunda capa del bug: sin esto se comparaba SIN los 25+25 puntos.

    `load_picks` resetea campeon_picks/goleador_picks a None, así que arreglar solo
    el orden habría dejado las dos planillas midiéndose sin los especiales — que en
    la tabla general pesan ~1,6 sigma del puntaje de temporada.
    """
    port = _portfolio_con_especiales()
    ev = port.evaluador
    s = ev._simulador(1234)

    ev._cargar(s, port.picks)
    assert s.campeon_picks is not None
    assert s.campeon_picks.tolist() == port.campeon.tolist()
    assert s.goleador_picks.tolist() == port.goleador.tolist()

    otra = port.picks.copy()
    otra[0, 0] = (otra[0, 0] + 1) % 36
    ev._cargar(s, otra)                                  # segunda carga: siguen ahí
    assert s.campeon_picks.tolist() == port.campeon.tolist()
    assert s.goleador_picks.tolist() == port.goleador.tolist()


def test_los_especiales_mueven_el_valor_que_reporta_el_evaluador():
    """Prueba de que los 25 pts realmente entran en la cuenta, no solo que existen."""
    port = _portfolio_con_especiales()
    ev = port.evaluador
    s = ev._simulador(4321)

    con = ev._cargar(s, port.picks)
    s.load_picks(port.picks)                             # misma planilla, SIN especiales
    sin = s.e_premio_total()
    assert con != sin, "los especiales no están afectando el E[premio] del evaluador"


def test_comparar_una_planilla_contra_si_misma_da_cero_con_especiales():
    """Sorteos comunes + especiales re-aplicados ⇒ Δ exactamente 0, no 'casi'."""
    port = _portfolio_con_especiales()
    c = port.evaluador.comparar(port.picks, port.picks.copy(), n_seeds=3)
    assert c.delta == 0.0
    assert c.se == 0.0


# -------------------- el veredicto llega al dashboard --------------------

def _planilla_min(tmp_path, generado="2026-08-10T19:40:59Z", scores=None):
    """Dos versiones de una fecha que difieren en un pick, como las escribe el rerun."""
    import json
    d = tmp_path / "fecha_01"
    d.mkdir(parents=True, exist_ok=True)
    base = {"generado_utc": generado, "n_sims": 9600, "picks": [{
        "evento_id": 1, "partido": "Deportivo Maldonado vs Racing",
        "cierre_pronostico_utc": "2099-01-01T00:00:00+00:00",
        "scores": scores or [[1, 0]]}]}
    return d, base


def test_veredicto_queda_escrito_en_la_planilla(tmp_path):
    """Si solo vive en el log, el panel muestra los picks sin decir que se descartaron."""
    import json
    from src.clausura.rerun_cierre import _guardar_veredicto

    p = tmp_path / "v26.json"
    p.write_text(json.dumps({"picks": []}), encoding="utf-8")
    _guardar_veredicto(p, comp(-456, 171), avisar=False, n_picks=7)

    v = json.loads(p.read_text(encoding="utf-8"))["veredicto_cambio"]
    assert v["avisar"] is False and v["medido"] is True
    assert v["n_picks"] == 7 and round(v["delta"]) == -456


def test_gate_roto_no_se_lee_como_cambio_sin_valor(tmp_path):
    """comp=None es 'no pude medir', que pide una acción DISTINTA a 'medí y da cero'."""
    import json
    from src.clausura.rerun_cierre import _guardar_veredicto

    p = tmp_path / "v26.json"
    p.write_text(json.dumps({"picks": []}), encoding="utf-8")
    _guardar_veredicto(p, None, avisar=True, n_picks=7)

    v = json.loads(p.read_text(encoding="utf-8"))["veredicto_cambio"]
    assert v["medido"] is False and v["avisar"] is True
    assert "delta" not in v


def test_guardar_veredicto_no_rompe_la_corrida(tmp_path):
    """Anotar el veredicto es cosmético: si falla, el rerun tiene que seguir."""
    from src.clausura.rerun_cierre import _guardar_veredicto
    _guardar_veredicto(tmp_path / "no-existe.json", comp(-456, 171), False, 7)


def test_el_panel_apaga_los_cambios_descartados():
    """El caso del 2026-08-10: 7 picks en amarillo con Δ -456 ± 171 detrás."""
    from jinja2 import Environment, FileSystemLoader

    def _data(c):
        return {"cambios": c, "fecha_n": 1}

    env = Environment(loader=FileSystemLoader("src/clausura/templates"))
    tpl = env.get_template("_cambios.html")
    cambios = {
        "n_versiones": 26, "fuentes": [], "especiales": [],
        "prev": {"generado_uy": "lun 10/08 08:18"}, "cur": {"generado_uy": "lun 10/08 16:40"},
        "cambios": [{"partido": "Deportivo Maldonado vs Racing", "numero": 899258856,
                     "antes": "1-0", "despues": "0-1", "cerrado": False}],
        "veredicto": {"avisar": False, "medido": True, "n_picks": 7,
                      "delta": -456.0, "se": 171.0, "n_seeds": 5, "significativa": True},
    }
    html = tpl.render(data=_data(cambios))

    assert "No toques nada" in html
    assert "-456" in html and "171" in html
    assert "text-amber-300" not in html and "border-amber" not in html

    cambios["veredicto"]["avisar"] = True
    html = tpl.render(data=_data(cambios))
    assert "conviene corregir" in html and "No toques nada" not in html

    cambios["veredicto"] = {"avisar": True, "medido": False, "n_picks": 7}
    html = tpl.render(data=_data(cambios))
    assert "no pudo medir" in html
