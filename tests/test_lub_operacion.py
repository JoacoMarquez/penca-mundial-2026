"""Operación diaria de la penca LUB: ventana del timer, goleador, especiales fijos,
formato de Telegram, recordatorios y modo carga. Todo sin red."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from src.lub import carga_alert as CA
from src.lub import dashboard as D
from src.lub import picks as P
from src.lub.data import Partido

T0 = datetime(2026, 10, 12, 15, 0, tzinfo=timezone.utc)


def _p(ev: int, fecha: str, cierre: datetime, jugado: bool = False, pref: bool = False) -> Partido:
    return Partido(campeonato_id=45, temporada=P.TEMPORADA, fecha_id=1, fecha_nombre=fecha,
                   fase="regular", evento_id=ev, local=f"L{ev}", visitante=f"V{ev}", local_id=1,
                   visitante_id=2, inicio_utc=(cierre + timedelta(minutes=15)).isoformat(),
                   cierre_utc=cierre.isoformat(), preferencial=pref, estado="",
                   pts_local=80 if jugado else None, pts_visitante=70 if jugado else None)


def test_ventana_toma_las_fechas_solapadas_del_dia():
    """La F3 de la 25/26 arrancó antes de que terminara la F2: un mismo día trae dos fechas."""
    ps = [_p(1, "Fecha 2", T0 + timedelta(hours=8)),
          _p(2, "Fecha 3", T0 + timedelta(hours=9)),
          _p(3, "Fecha 3", T0 + timedelta(hours=40)),       # otro día: no entra
          _p(4, "Fecha 4", T0 + timedelta(hours=60)),
          _p(5, "Fecha 2", T0 - timedelta(hours=1))]        # ya cerrado
    assert P.fechas_que_cierran(ps, T0, 24) == ["Fecha 2", "Fecha 3"]
    assert P.fechas_que_cierran(ps, T0 + timedelta(days=5), 24) == []
    # la fecha se optimiza ENTERA (lo abierto), no solo lo de hoy
    assert [p.evento_id for p in P.fecha_actual(ps, T0, "Fecha 3")] == [2, 3]


def test_goleador_prior_matchea_sin_tildes_y_normaliza(tmp_path):
    path = tmp_path / "prior.json"
    path.write_text(json.dumps({"santiago vescovi": 0.6, "JUAN PEREZ": 0.3, "Fuera Del Menu": 0.1}))
    nombres, p = P.cargar_goleador(["Santiago Véscovi", "Juan Pérez", "Otro"], path)
    assert nombres == ["Santiago Véscovi", "Juan Pérez", "Otro"]
    assert p.sum() == pytest.approx(1.0)
    assert p[0] > p[1] > p[2] > 0            # el de afuera del prior no queda en cero exacto


def test_goleador_sin_menu_o_sin_prior_no_asigna(tmp_path):
    path = tmp_path / "prior.json"
    assert P.cargar_goleador(["A"], path) is None            # falta el archivo
    path.write_text(json.dumps({"A": 1.0}))
    assert P.cargar_goleador(None, path) is None             # falta el menú (500)
    assert P.cargar_goleador(["B"], path) is None            # ningún nombre matchea


def test_especiales_fijos_desconocidos_nunca_aciertan():
    idx = P._especiales_idx(["Peñarol", None, "Equipo Fantasma"], ["Nacional", "Penarol"], 4)
    assert list(idx) == [1, -1, -1, -1]


def _planilla(**kw) -> dict:
    fila = {"evento_id": 2201, "fecha": "Fecha 1", "local": "Nacional", "visitante": "Aguada",
            "cierre_utc": "2026-10-12T22:45:00+00:00", "preferencial": True, "fuente": "ratings +6.6",
            "hoy": True, "p_local": 0.71, "probs_clase": [0.1] * 10, "picks": [[88, 65], [78, 75]]}
    oos = {"e_premio": 7301.0, "se": 587.0, "e_penca": 6250.0, "p_penca": 0.13, "e_fechas": 1051.0}
    pl = {"generado_utc": T0.isoformat(), "k": 2, "fecha": "Fecha 1", "partidos": [fila],
          "campeon": ["Peñarol", "Nacional"], "goleador": None, "especiales_libres": True,
          "oos": oos, "e_premio": 7400, "detalle": {}, "costo": 400.0, "numeros": [899301848, 899301849]}
    pl.update(kw)
    return pl


def test_telegram_lista_por_participacion_y_avisa_goleador_faltante():
    txt = P.formatear(_planilla())
    assert "<code> …848</code> 88-65" in txt and "<code> …849</code> 78-75" in txt
    assert "⭐ Nacional vs Aguada" in txt and "cierra 12/10 19:45" in txt
    assert "Goleador sin asignar" in txt


def test_telegram_sin_nada_para_hoy():
    pl = _planilla()
    pl["partidos"][0]["hoy"] = False
    assert "Nada para cargar hoy" in P.formatear(pl)


def test_telegram_escapa_html():
    pl = _planilla()
    pl["partidos"][0]["local"] = "A&B <x>"
    assert "A&amp;B &lt;x&gt;" in P.formatear(pl)


def test_recordatorio_agrupa_por_horario_de_cierre():
    cierre = T0 + timedelta(hours=5)
    evs = [{"evento_id": i, "local": f"L{i}", "visitante": f"V{i}", "preferencial": i == 2,
            "fecha": "Fecha 1", "cierre_pronostico_utc": cierre.isoformat()} for i in (1, 2, 3)]
    pend = CA.pendientes_de_alerta(evs, T0, set())
    assert len(pend) == 3
    msg = CA.formatear([(ev, c, t) for ev, c, t, _ in pend], 16)
    assert msg.startswith("⏰ <b>3 partido(s)") and "cierran a las 17:00 UY" in msg
    assert "L2 vs V2 ⭐x2" in msg and "las 16 participaciones" in msg


def test_modo_carga_lub_usa_su_prefijo_y_solo_lo_de_hoy(tmp_path, monkeypatch):
    pl = _planilla()
    otra = dict(pl["partidos"][0], evento_id=2202, hoy=False)
    pl["partidos"].append(otra)
    pl["_archivo"] = "v3_x.json"
    monkeypatch.setattr(D, "ultima_planilla", lambda: pl)
    d = D.load_lub_carga("tok", now=T0)
    assert d["ok"] and d["marca_prefijo"] == "lub:v1:" and d["verificar"] is False
    assert d["api_marcas_url"] == "/dash/tok/lub/api/carga-marcas"
    assert [r["evento_id"] for r in d["planilla"]["picks"]] == [2201]
    assert d["planilla"]["picks"][0]["scores_fmt"] == ["88-65", "78-75"]
    assert d["mis_numeros"] == [899301848, 899301849]
    assert d["planilla"]["especiales"]["por_participacion"][0] == {"campeon": "Peñarol", "goleador": None}


def test_modo_carga_lub_sin_planilla(monkeypatch):
    monkeypatch.setattr(D, "ultima_planilla", lambda: None)
    d = D.load_lub_carga("tok")
    assert d["ok"] is False and d["planilla"] is None


def test_clave_lub_valida_en_el_store():
    from src.clausura.carga_state import clave_valida
    assert clave_valida("lub:v1:0:3:2201") and clave_valida("lub:v1:esp:3")
    assert not clave_valida("lub:v1:0:filtro")
    assert not clave_valida("otra:v1:0:3:2201")


def test_optimizar_respeta_goleador_fijo():
    """Con la temporada arrancada los especiales no se mueven: goleador_init entra tal cual."""
    from src.lub.portfolio import optimizar

    class EvFalso:
        so = type("S", (), {"probs": np.zeros((1, 0, 10)), "riv_total_max": np.zeros(1),
                            "riv_total_cnt": np.zeros(1)})()
        K, actual, fecha_actual = 2, [], -1
        fecha_pts = np.zeros((1, 1, 2), np.int16)

        def especiales(self, c, g):
            return np.zeros((1, 2), np.int16)

        def valor(self, fp, esp):
            return 0.0, np.zeros(1), np.zeros(1)

        @staticmethod
        def _premio(ours, rmax, rcnt, monto):
            return np.zeros(1)

    port = optimizar(EvFalso(), [0, 1], campeon_init=np.array([1, -1]), especiales_libres=False,
                     goleador_opts=[0, 1], goleador_init=np.array([-1, 1]))
    assert list(port.campeon) == [1, -1] and list(port.goleador) == [-1, 1]
