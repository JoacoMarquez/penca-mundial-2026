"""Tests del vigía de pencas nuevas (src.clausura.pencas_watch)."""

from src.clausura.api import Penca, Premio
from src.clausura.pencas_watch import (formatear_falla, formatear_nueva,
                                       garantia_minima, nuevas)


def _penca(id_=49, precio=400.0, premios=None):
    premios = premios if premios is not None else (
        Premio("Ganador Penca", "PENCA", 350000.0, ""),
        Premio("Ganador Fecha", "FECHA", 10000.0, ""),
        Premio("Ganador Grupo Amigo", "GRUPOAMIGO", 3000.0, ""),
    )
    return Penca(id=id_, nombre="Torneo Apertura 2027", precio=precio,
                 campeonato_id=46, estado="ACTIVO", premios=premios)


def test_nuevas_filtra_las_vistas():
    vis = [_penca(46), _penca(48), _penca(49)]
    assert [p.id for p in nuevas(vis, {46, 47, 48})] == [49]
    assert nuevas(vis, {46, 48, 49}) == []


def test_garantia_suma_premios_por_fecha():
    assert garantia_minima(_penca(), 15) == 350000 + 15 * 10000 + 3000


def test_garantia_desconocida_sin_fixture():
    assert garantia_minima(_penca(), None) is None
    sin_fecha = _penca(premios=(Premio("Ganador Penca", "PENCA", 50000.0, ""),))
    assert garantia_minima(sin_fecha, None) == 50000


def test_mensaje_paga_con_break_even_y_payout():
    txt = formatear_nueva(_penca(), n_partic=200, n_fechas=15)
    assert "PAGA" in txt and "id 49" in txt and "Fechas publicadas: 15" in txt
    assert "$503,000" in txt
    assert "<b>1,258</b> participaciones" in txt       # 503000 / 400
    assert "payout 629%" in txt                         # 503000 / 80000


def test_mensaje_paga_sin_fixture_no_inventa_garantia():
    txt = formatear_nueva(_penca(), n_partic=None, n_fechas=None)
    assert "no calculable" in txt and "¿?" in txt


def test_mensaje_gratuita_muestra_premios_en_especie():
    gratis = _penca(precio=0.0, premios=(
        Premio("Ganador Penca", "PENCA", 0.0, "Play Station 5"),))
    txt = formatear_nueva(gratis, n_partic=1000, n_fechas=15)
    assert "GRATUITA" in txt and "Play Station 5" in txt
    assert "Precio" not in txt and "break" not in txt


def test_falla_escapa_html():
    txt = formatear_falla(4, "<html>502</html>")
    assert "4 corridas" in txt and "&lt;html&gt;" in txt


def test_run_avisa_una_sola_vez_tras_fallas_seguidas_y_registra_sin_avisar(
        tmp_path, monkeypatch):
    import src.clausura.pencas_watch as w

    class ApiCaida:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def pencas_visibles(self):
            raise RuntimeError("502")

    class ApiOk(ApiCaida):
        def pencas_visibles(self):
            return [_penca(46), _penca(48)]

    enviados = []
    monkeypatch.setattr(w, "STATE_PATH", tmp_path / "s.json")
    monkeypatch.setattr(w, "_enviar", enviados.append)

    monkeypatch.setattr(w, "PencaApiClient", ApiCaida)
    for _ in range(w.FALLOS_PARA_AVISAR + 2):
        w.run()
    assert len(enviados) == 1 and "sin poder leer" in enviados[0]

    # al volver el API: primera lectura exitosa registra sin avisar y resetea la falla
    monkeypatch.setattr(w, "PencaApiClient", ApiOk)
    assert w.run() == []
    st = w.load_state()
    assert st["vistos"] == [46, 48] and st["fallos"] == 0 and not st["falla_avisada"]
