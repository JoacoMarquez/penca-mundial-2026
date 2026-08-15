"""Marcas del modo carga compartidas entre dispositivos (src/clausura/carga_state.py)."""

import json

import pytest

from src.clausura.carga_state import (
    MAX_MARCAS, aplicar, clave_valida, fusionar, leer,
)

FILA = "carga:v2:2:899258848:2094"
ESP = "carga:v2:esp:899258848"


def test_claves_de_marca_si_preferencias_de_pantalla_no():
    assert clave_valida(FILA) and clave_valida(ESP)
    # El filtro y la marca de migración son del DISPOSITIVO: sincronizarlos haría
    # que esconder lo cargado en el celular te lo esconda también en la PC.
    assert not clave_valida("carga:v2:2:filtro")
    assert not clave_valida("carga:v2:2:migrado:v9_20260815T112325Z.json")
    # y nada que no sea del esquema
    assert not clave_valida("../../etc/passwd")
    assert not clave_valida("carga:v1:2:1:1")


def test_marcar_y_desmarcar(tmp_path):
    p = tmp_path / "marcas.json"
    assert aplicar(FILA, "1-0", path=p) == {FILA: "1-0"}
    assert aplicar(ESP, "Peñarol", path=p) == {FILA: "1-0", ESP: "Peñarol"}
    # se guarda EL VALOR, no un sí/no: re-marcar con otro marcador lo pisa
    assert aplicar(FILA, "2-1", path=p)[FILA] == "2-1"
    assert aplicar(FILA, None, path=p) == {ESP: "Peñarol"}
    assert leer(p) == {ESP: "Peñarol"}


def test_rechaza_clave_y_valor_invalidos(tmp_path):
    p = tmp_path / "marcas.json"
    with pytest.raises(ValueError):
        aplicar("carga:v2:2:filtro", "1", path=p)
    with pytest.raises(ValueError):
        aplicar(FILA, "x" * 100, path=p)
    assert not p.exists()          # nada se escribió


def test_fusionar_sube_lo_local_sin_pisar_el_servidor(tmp_path):
    """El primer dispositivo que abre trae su historial; el servidor no se pisa.

    Si una clave ya está arriba puede venir del OTRO dispositivo, que cargó
    después — adoptar la local ahí sería revivir una marca vieja.
    """
    p = tmp_path / "marcas.json"
    aplicar(FILA, "2-1", path=p)                       # el servidor ya sabe
    out = fusionar({FILA: "1-0", ESP: "Arezo", "carga:v2:2:filtro": "1"}, path=p)
    assert out[FILA] == "2-1"                          # gana el servidor
    assert out[ESP] == "Arezo"                         # la nueva sube
    assert "carga:v2:2:filtro" not in out              # la preferencia no viaja


def test_leer_ignora_basura_y_archivo_roto(tmp_path):
    p = tmp_path / "marcas.json"
    p.write_text('{"carga:v2:2:1:1": "1-0", "rota": "x"}', encoding="utf-8")
    assert leer(p) == {"carga:v2:2:1:1": "1-0"}
    p.write_text("{no soy json", encoding="utf-8")
    assert leer(p) == {}                               # arranca vacío, no explota


def test_tope_de_marcas(tmp_path):
    p = tmp_path / "marcas.json"
    p.write_text(json.dumps({f"carga:v2:1:{i}:1": "1-0" for i in range(MAX_MARCAS)}),
                 encoding="utf-8")
    with pytest.raises(ValueError):
        aplicar("carga:v2:9:999:9", "1-0", path=p)
    # pero desmarcar y re-marcar algo YA existente sigue andando
    assert aplicar("carga:v2:1:0:1", "2-0", path=p)["carga:v2:1:0:1"] == "2-0"


def test_escritura_atomica_no_deja_archivo_a_medias(tmp_path):
    """La PC y el celular pueden escribir casi a la vez: un archivo truncado
    borraría el progreso de la carga, que es justo lo que no se puede reconstruir
    (el gate del API no deja verificar hasta el cierre)."""
    p = tmp_path / "marcas.json"
    for i in range(30):
        aplicar(f"carga:v2:2:{i}:2094", "1-0", path=p)
        json.loads(p.read_text(encoding="utf-8"))      # siempre parseable
    assert len(leer(p)) == 30
    assert not list(tmp_path.glob("*.tmp"))            # sin temporales huérfanos
