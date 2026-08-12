"""Tests del snapshot del pool: Q empírica, blending y persistencia (sin red)."""

import json

import numpy as np
import pytest

from src.clausura.economics import N_SCORES, score_index
from src.clausura.pool_snapshot import (
    RivalPicks,
    blended_q,
    empirical_campeon_counts,
    empirical_counts,
    empirical_goleador_counts,
    load_latest_snapshot,
    save_snapshot,
    snapshot_summary,
)


def _snapshot_dict():
    """Snapshot sintético: 3 rivales, 2 eventos, campeones cargados."""
    return {
        "generado_utc": "2026-08-08T12:00:00+00:00",
        "penca_id": 46,
        "n_participaciones": 3,
        "participaciones": [
            {"participacion_id": 1, "numero": 100, "puntos": 5, "exactos": 1,
             "picks": {"2080": [1, 0], "2085": [1, 1]}, "campeon": "Peñarol",
             "campeon_id": 80, "goleador": "X", "goleador_id": 9},
            {"participacion_id": 2, "numero": 101, "puntos": 3, "exactos": 0,
             "picks": {"2080": [1, 0]}, "campeon": "Nacional",
             "campeon_id": 79, "goleador": None, "goleador_id": None},
            {"participacion_id": 3, "numero": 102, "puntos": 0, "exactos": 0,
             "picks": {"2080": [2, 1]}, "campeon": "Peñarol",
             "campeon_id": 80, "goleador": None, "goleador_id": None},
        ],
    }


# -------------------- conteos empíricos --------------------

def test_empirical_counts_por_evento():
    counts = empirical_counts(_snapshot_dict())
    assert set(counts) == {2080, 2085}
    assert counts[2080][score_index(1, 0)] == 2
    assert counts[2080][score_index(2, 1)] == 1
    assert counts[2080].sum() == 3
    assert counts[2085][score_index(1, 1)] == 1


def test_empirical_counts_trunca_goleadas():
    snap = _snapshot_dict()
    snap["participaciones"][0]["picks"] = {"2080": [9, 0]}
    counts = empirical_counts(snap)
    assert counts[2080][score_index(5, 0)] == 1  # 6+ va al borde de la grilla


def test_empirical_counts_excluye_mis_numeros():
    """La Q empírica modela a los RIVALES: nuestras participaciones no cuentan."""
    counts = empirical_counts(_snapshot_dict(), mis_numeros={100})
    assert set(counts) == {2080}          # el 2085 solo lo tenía el 100 (nuestro)
    assert counts[2080].sum() == 2
    assert counts[2080][score_index(1, 0)] == 1


def test_empirical_campeon_counts():
    idx = {"Peñarol": 0, "Nacional": 1, "Cerro": 2}
    c = empirical_campeon_counts(_snapshot_dict(), idx, 3)
    assert c.tolist() == [2, 1, 0]


def test_empirical_campeon_counts_excluye_mis_numeros():
    idx = {"Peñarol": 0, "Nacional": 1, "Cerro": 2}
    c = empirical_campeon_counts(_snapshot_dict(), idx, 3, mis_numeros={102})
    assert c.tolist() == [1, 1, 0]


class _Opcion:
    """Duck-type de especiales.OpcionGoleador (id, nombre)."""
    def __init__(self, id, nombre):
        self.id, self.nombre = id, nombre


def test_empirical_goleador_counts_matchea_por_id():
    ops = [_Opcion(9, "X"), _Opcion(10, "Y")]
    c = empirical_goleador_counts(_snapshot_dict(), ops)
    assert c.tolist() == [1, 0]   # solo el rival 100 tiene goleador (id 9)


def test_empirical_goleador_counts_fallback_por_nombre_y_exclusion():
    snap = _snapshot_dict()
    # rival 101: id que no está en el menú pero nombre sí → cuenta por nombre
    snap["participaciones"][1]["goleador"] = "Y"
    snap["participaciones"][1]["goleador_id"] = 999
    ops = [_Opcion(9, "X"), _Opcion(10, "Y")]
    assert empirical_goleador_counts(snap, ops).tolist() == [1, 1]
    # excluyendo nuestra participación (100) desaparece su pick de X
    assert empirical_goleador_counts(snap, ops, mis_numeros={100}).tolist() == [0, 1]


# -------------------- blending Dirichlet --------------------

def test_blended_q_sin_observaciones_devuelve_prior():
    prior = np.full(N_SCORES, 1 / N_SCORES)
    assert np.array_equal(blended_q(prior, None), prior)
    assert np.array_equal(blended_q(prior, np.zeros(N_SCORES)), prior)


def test_blended_q_pocas_obs_cerca_del_prior_muchas_cerca_de_lo_observado():
    prior = np.full(N_SCORES, 1 / N_SCORES)
    obs = np.zeros(N_SCORES)
    idx = score_index(1, 0)

    obs[idx] = 2   # 2 observaciones contra strength=25
    q_pocas = blended_q(prior, obs, strength=25.0)

    obs_muchas = np.zeros(N_SCORES)
    obs_muchas[idx] = 500
    q_muchas = blended_q(prior, obs_muchas, strength=25.0)

    assert q_pocas[idx] < 0.15          # domina el prior
    assert q_muchas[idx] > 0.90         # domina lo observado
    assert q_pocas.sum() == pytest.approx(1.0)
    assert q_muchas.sum() == pytest.approx(1.0)


def test_blended_q_es_monotona_en_observaciones():
    prior = np.full(N_SCORES, 1 / N_SCORES)
    idx = score_index(0, 0)
    prev = prior[idx]
    for n in [1, 5, 20, 100]:
        obs = np.zeros(N_SCORES)
        obs[idx] = n
        q = blended_q(prior, obs)
        assert q[idx] > prev
        prev = q[idx]


# -------------------- persistencia --------------------

def test_save_y_load_snapshot(tmp_path, monkeypatch):
    import src.clausura.pool_snapshot as ps
    monkeypatch.setattr(ps, "SNAP_DIR", tmp_path)

    rivales = [
        RivalPicks(participacion_id=1, numero=100, puntos=0, exactos=0,
                   picks={2080: (1, 0)}, campeon="Peñarol", campeon_id=80),
    ]
    p1 = save_snapshot(rivales, penca_id=46)
    p2 = save_snapshot(rivales, penca_id=46)
    assert p1.name.startswith("v1_") and p2.name.startswith("v2_")

    snap = load_latest_snapshot()
    assert snap["penca_id"] == 46
    assert snap["participaciones"][0]["picks"] == {"2080": [1, 0]}


def test_load_snapshot_respeta_max_age(tmp_path, monkeypatch):
    import src.clausura.pool_snapshot as ps
    monkeypatch.setattr(ps, "SNAP_DIR", tmp_path)
    (tmp_path / "v1_20200101T000000Z.json").write_text(json.dumps({
        "generado_utc": "2020-01-01T00:00:00+00:00", "penca_id": 46,
        "n_participaciones": 0, "participaciones": [],
    }), encoding="utf-8")
    assert load_latest_snapshot(max_age_hours=48) is None
    assert load_latest_snapshot() is not None   # sin límite sí lo devuelve


def test_snapshot_summary():
    s = snapshot_summary(_snapshot_dict())
    assert "3 participaciones" in s
    assert "2 eventos" in s
    assert "3 con marcadores" in s
    assert "3 con campeón" in s


# -------------------- pacing, backoff y reuso de especiales --------------------

def _ranking_falso(n):
    from src.clausura.api import RankingRow
    return [RankingRow(participacion_id=100 + i, numero_participacion=900 + i,
                       puntos_totales=0, puntos_por_fecha=0, posicion_general=i + 1,
                       cant_resultados_exactos=0) for i in range(n)]


def _mock_api(monkeypatch, handler, n_rivales=2):
    """Parchea el ranking y enruta los GET del snapshot a `handler`."""
    import httpx
    import src.clausura.pool_snapshot as ps
    monkeypatch.setattr(ps, "_ranking_pacing", lambda *a, **k: _ranking_falso(n_rivales))
    monkeypatch.setattr(ps.time, "sleep", lambda s: None)   # sin esperas reales
    transport = httpx.MockTransport(handler)
    orig = httpx.Client

    def cliente(*a, **kw):
        kw["transport"] = transport
        return orig(*a, **kw)

    monkeypatch.setattr(ps.httpx, "Client", cliente)


def test_especiales_conocidos_ignora_filas_sin_campeon():
    from src.clausura.pool_snapshot import especiales_conocidos
    conocidos = especiales_conocidos(_snapshot_dict())
    assert set(conocidos) == {1, 2, 3}
    assert conocidos[1]["campeon"] == "Peñarol"
    assert conocidos[1]["goleador"] == "X"

    sin_campeon = _snapshot_dict()
    sin_campeon["participaciones"][0]["campeon"] = None
    assert 1 not in especiales_conocidos(sin_campeon)   # se vuelve a pedir


def test_especiales_conocidos_rechaza_snapshot_pre_lock():
    """Un snapshot anterior al lock NO sirve de base: hasta ese instante cualquiera
    (nosotros incluidos) podía cambiar campeón/goleador."""
    from datetime import datetime, timezone
    from src.clausura.pool_snapshot import especiales_conocidos

    lock = datetime(2026, 8, 7, 21, 45, tzinfo=timezone.utc)
    pre = _snapshot_dict()
    pre["generado_utc"] = "2026-08-07T14:13:30+00:00"
    assert especiales_conocidos(pre, lock_utc=lock) == {}

    post = _snapshot_dict()
    post["generado_utc"] = "2026-08-07T23:10:00+00:00"
    assert set(especiales_conocidos(post, lock_utc=lock)) == {1, 2, 3}


def test_lock_especiales_sale_del_primer_cierre_del_fixture():
    """Art. 4: los especiales se cierran 15' antes del primer partido = el cierre
    de pronóstico del primer evento del fixture."""
    from src.clausura.pool_snapshot import lock_especiales_utc
    lock = lock_especiales_utc()
    assert lock is not None and lock.isoformat() == "2026-08-07T21:45:00+00:00"


def test_fetch_snapshot_reusa_especiales_y_no_los_pide(monkeypatch):
    """Post-lock los especiales no cambian: reusarlos ahorra la MITAD de los
    requests. Los pids desconocidos SÍ se piden (participación comprada después)."""
    import httpx
    from src.clausura.pool_snapshot import fetch_snapshot

    pedidos = []

    def handler(request):
        pedidos.append(request.url.path)
        if "pronosticosEventos" in request.url.path:
            return httpx.Response(200, json={"data": [
                {"encuentroId": 2086, "golesEquipoLocal": 1, "golesEquipoVisitante": 0}]})
        return httpx.Response(200, json={
            "equipoCampeon": {"nombre": "Nacional", "id": 79},
            "opcionGoleador": {"goleador": "Arezo", "id": 700}})

    _mock_api(monkeypatch, handler, n_rivales=2)
    previos = {100: {"campeon": "Peñarol", "campeon_id": 80,
                     "goleador": "Gómez", "goleador_id": 622}}
    out = fetch_snapshot(46, pause_s=0, especiales_previos=previos)

    assert [p.campeon for p in out] == ["Peñarol", "Nacional"]   # 100 reusado, 101 pedido
    assert out[0].goleador == "Gómez" and out[0].goleador_id == 622
    assert out[0].picks == {2086: (1, 0)}                        # marcadores SIEMPRE se piden
    esp = [p for p in pedidos if "CampeonGoleador" in p]
    assert len(esp) == 1 and esp[0].endswith("/101/pronosticoCampeonGoleador")


def test_fetch_snapshot_reintenta_ante_429_en_vez_de_dejar_la_fila_vacia(monkeypatch):
    """El 7/8 un 429 tratado como fallo guardó 683 filas con 97 datos. Ahora se
    espera y se reintenta la misma request."""
    import httpx
    from src.clausura.pool_snapshot import fetch_snapshot

    estado = {"golpes": 0}

    def handler(request):
        if "pronosticosEventos" in request.url.path and estado["golpes"] < 2:
            estado["golpes"] += 1
            return httpx.Response(429, text="Too Many Requests")
        if "pronosticosEventos" in request.url.path:
            return httpx.Response(200, json={"data": [
                {"encuentroId": 2086, "golesEquipoLocal": 2, "golesEquipoVisitante": 1}]})
        return httpx.Response(200, json={"equipoCampeon": {"nombre": "Cerro", "id": 5},
                                         "opcionGoleador": {}})

    _mock_api(monkeypatch, handler, n_rivales=1)
    out = fetch_snapshot(46, pause_s=0)

    assert estado["golpes"] == 2                  # aguantó los dos 429…
    assert out[0].picks == {2086: (2, 1)}         # …y la fila quedó COMPLETA


def test_fetch_snapshot_429_persistente_aborta_en_vez_de_seguir(monkeypatch):
    """Bloqueo duro: abortar el escaneo ENTERO, no seguir con filas vacías.

    Seguir de largo tiene dos costos, los dos medidos el 7/8: se guarda un pool
    fantasma como si fuera válido, y cada rival paga sus reintentos (28 min
    trabado en el mismo) así que el escaneo no termina nunca."""
    import httpx
    import pytest as _pytest
    from src.clausura.pool_snapshot import RateLimited, fetch_snapshot

    _mock_api(monkeypatch, lambda r: httpx.Response(429, text="Too Many Requests"),
              n_rivales=40)
    with _pytest.raises(RateLimited):
        fetch_snapshot(46, pause_s=0)


def test_ranking_429_persistente_tambien_aborta(monkeypatch):
    """El 429 en el ranking mataba el escaneo con un HTTPStatusError crudo; ahora
    sale como RateLimited para que el CLI no guarde nada."""
    import httpx
    import pytest as _pytest
    import src.clausura.pool_snapshot as ps

    class ApiFalsa:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def ranking(self, _):
            req = httpx.Request("GET", "http://x/ranking")
            raise httpx.HTTPStatusError("429", request=req,
                                        response=httpx.Response(429, request=req))

    monkeypatch.setattr(ps, "PencaApiClient", ApiFalsa)
    monkeypatch.setattr(ps.time, "sleep", lambda s: None)
    with _pytest.raises(ps.RateLimited):
        ps._ranking_pacing(46)


# -------------------- integración con el optimizador --------------------

def test_build_portfolio_acepta_pool_qs_externo():
    from src.clausura.economics import SimConfig
    from src.clausura.strategy import build_portfolio
    from src.model.poisson import score_grid

    g = score_grid(1.4, 1.0, 0.0, max_goals=5)
    grids = [g] * 4
    q = np.full(N_SCORES, 1 / N_SCORES)
    port = build_portfolio(grids, [1, 1, 2, 2], [False] * 4, n_participaciones=2,
                           sim=SimConfig(n_sims=150, n_rivales=20, seed=3),
                           pool_qs=[q] * 4, max_passes=1)
    assert port.picks.shape == (2, 4)

    with pytest.raises(ValueError, match="pool_qs"):
        build_portfolio(grids, [1, 1, 2, 2], [False] * 4, n_participaciones=2,
                        sim=SimConfig(n_sims=150, n_rivales=20, seed=3),
                        pool_qs=[q] * 3)


# -------------------- exactos contados de los picks, no del contador del API --------------------

def _snap(participaciones):
    return {"n_participaciones": len(participaciones), "participaciones": participaciones}


def test_exact_rate_cuenta_los_picks_reales():
    """El numerador y el denominador salen del MISMO conjunto de pares rival-partido.

    Ese era el bug del canal viejo: el denominador contaba todos los partidos jugados
    de la temporada, incluidos los de la fecha en curso que el contador del API —que
    liquida al cierre— todavía no podía ver. Cuanto más avanzada la fecha, más abajo
    la tasa, y más disperso el pool modelado.
    """
    from src.clausura.pool_snapshot import exact_rate_desde_snapshot

    resultados = {1: (1, 0), 2: (2, 1)}
    snap = _snap([
        {"numero": 100, "picks": {"1": [1, 0], "2": [2, 1]}},   # 2 de 2
        {"numero": 101, "picks": {"1": [1, 0], "2": [0, 0]}},   # 1 de 2
        {"numero": 102, "picks": {"1": [3, 3], "2": [3, 3]}},   # 0 de 2
    ])
    assert exact_rate_desde_snapshot(snap, resultados) == 3 / 6


def test_exact_rate_ignora_partidos_sin_resultado():
    """Un pick de un partido que no se jugó no puede entrar en el denominador."""
    from src.clausura.pool_snapshot import exact_rate_desde_snapshot

    snap = _snap([{"numero": 100, "picks": {"1": [1, 0], "9": [2, 2]}}])
    assert exact_rate_desde_snapshot(snap, {1: (1, 0)}) == 1.0


def test_exact_rate_excluye_nuestras_participaciones():
    """El observable calibra a los RIVALES: nuestras 12 diversificadas lo sesgarían."""
    from src.clausura.pool_snapshot import exact_rate_desde_snapshot

    snap = _snap([
        {"numero": 899258848, "picks": {"1": [1, 0]}},   # nuestra, acierta
        {"numero": 500, "picks": {"1": [0, 0]}},         # rival, falla
    ])
    assert exact_rate_desde_snapshot(snap, {1: (1, 0)}, {899258848}) == 0.0


def test_exact_rate_sin_datos_devuelve_none():
    """Sin pares observables no se calibra: mejor el prior que un 0 que miente."""
    from src.clausura.pool_snapshot import exact_rate_desde_snapshot

    assert exact_rate_desde_snapshot(None, {1: (1, 0)}) is None
    assert exact_rate_desde_snapshot(_snap([]), {1: (1, 0)}) is None
    assert exact_rate_desde_snapshot(_snap([{"numero": 1, "picks": {}}]), {}) is None
