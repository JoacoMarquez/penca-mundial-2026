"""Tests del parser del Intermedio 2026 (src.clausura.intermedio)."""

from src.clausura.intermedio import _clean_cell, parse_partidos

EQUIPOS = {"Peñarol", "Nacional", "Wanderers", "Liverpool", "Cerro",
           "Boston River", "Montevideo City Torque"}

WIKITEXT = """
{| class="wikitable"
! colspan="9" |Fecha 1
|-
! Serie !! Local !! Resultado !! Visitante
|- align="center"
|B
|Montevideo City Torque
|1:2
|bgcolor=#D0E7FF| '''Nacional
|Charrúa
|15 de mayo
|20:00
|rowspan=8|[[Archivo:Disney+_logo.svg|44x44px]]
|Gustavo Tejera
|- align="center"
|rowspan=3|A
|bgcolor=#D0E7FF| '''[[Club Atlético Peñarol|Peñarol]]
|2:1
|Liverpool
|Campeón del Siglo
|16 de mayo
|18:30
|[[Esteban Ostojich]]
|- align="center"
|B
|Albion FC Desconocido
|3:3
|Cerro
|Luis Franzini
|17:00
|Feres
|}
{| class="wikitable"
! colspan="9" |Final
|- align="center"
|
|Peñarol
|21:00
|bgcolor=#D0E7FF| '''Wanderers
|Centenario
|5 de agosto
|20:00
|Árbitro
|}
"""


def test_parsea_solo_filas_con_marcador_y_equipos_conocidos():
    ps = parse_partidos(WIKITEXT, EQUIPOS)
    # la fila de Albion (equipo desconocido) se descarta; la final sin marcador
    # real no matchea (21:00 tiene ':' pero es hora... ver test siguiente)
    partidos = {(p.local, p.visitante): (p.goles_local, p.goles_visitante) for p in ps}
    assert partidos[("Montevideo City Torque", "Nacional")] == (1, 2)
    assert partidos[("Peñarol", "Liverpool")] == (2, 1)
    assert ("Albion FC Desconocido", "Cerro") not in partidos


def test_hora_no_se_confunde_con_marcador():
    ps = parse_partidos(WIKITEXT, EQUIPOS)
    # la final del fixture tiene hora 21:00 y 20:00 pero sin resultado —
    # "Peñarol 21-0 Wanderers" sería un desastre silencioso en los ratings
    assert all({p.local, p.visitante} != {"Peñarol", "Wanderers"} for p in ps)


def test_clean_cell_saca_wikilinks_y_estilos():
    assert _clean_cell("bgcolor=#D0E7FF| '''[[Club Nacional|Nacional]]") == "Nacional"
    assert _clean_cell("[[Archivo:X.svg|44x44px]]") == ""
    assert _clean_cell("rowspan=3|A") == "A"


def test_fechas_se_asignan_del_contexto():
    ps = parse_partidos(WIKITEXT, EQUIPOS)
    torque = next(p for p in ps if p.local == "Montevideo City Torque")
    assert torque.inicio_utc.startswith("2026-05-15")
    assert torque.fecha_nombre == "Fecha 1"
    assert torque.campeonato == "Torneo Intermedio 2026"


def test_load_dataset_completo_avisa_ruidoso_si_falta_el_intermedio(tmp_path, monkeypatch, caplog):
    """El archivo está gitignored y el ExecStartPre ignora fallos: si falta, los
    ratings corren con datos hasta mayo y P(campeón) se invierte. El fallback
    tiene que avisar fuerte, no callar."""
    import src.clausura.intermedio as im
    monkeypatch.setattr(im, "OUT_PATH", tmp_path / "no_existe.json")
    monkeypatch.setattr(im, "EXTRA_PATH", tmp_path / "tampoco.yaml")
    monkeypatch.setattr(im, "load_dataset", lambda: [])
    with caplog.at_level("WARNING"):
        out = im.load_dataset_completo()
    assert out == []
    assert "AUSENTE" in caplog.text and "Intermedio" in caplog.text


def test_partidos_extra_entran_y_se_deduplican(tmp_path, monkeypatch):
    """La final del Intermedio (Wikipedia no la lista) entra desde el yaml; si la
    fuente automática la agrega después, el extra se descarta solo."""
    import src.clausura.intermedio as im

    yml = tmp_path / "partidos_extra.yaml"
    yml.write_text("""
partidos:
  - campeonato: "Torneo Intermedio 2026"
    fecha_nombre: "Final"
    local: "Peñarol"
    visitante: "Wanderers"
    goles_local: 5
    goles_visitante: 1
    inicio_utc: "2026-08-05T22:30:00+00:00"
""", encoding="utf-8")
    monkeypatch.setattr(im, "EXTRA_PATH", yml)

    extras = im.load_partidos_extra(set())
    assert len(extras) == 1
    p = extras[0]
    assert (p.local, p.goles_local, p.goles_visitante) == ("Peñarol", 5, 1)
    assert p.campeonato == "Torneo Intermedio 2026"

    # dedup: si ya está en la fuente automática, no entra dos veces
    assert im.load_partidos_extra({im._clave(p)}) == []


def test_partidos_extra_sin_archivo_devuelve_vacio(tmp_path, monkeypatch):
    import src.clausura.intermedio as im
    monkeypatch.setattr(im, "EXTRA_PATH", tmp_path / "no_existe.yaml")
    assert im.load_partidos_extra(set()) == []


def test_el_yaml_real_del_repo_carga_y_tiene_la_final():
    """El archivo commiteado tiene que parsear y contener la final 5-1."""
    from src.clausura.intermedio import load_partidos_extra
    extras = load_partidos_extra(set())
    finales = [p for p in extras if p.fecha_nombre == "Final"]
    assert len(finales) == 1
    assert (finales[0].goles_local, finales[0].goles_visitante) == (5, 1)
