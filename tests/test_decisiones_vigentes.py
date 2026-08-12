"""La alarma que nos faltó: avisa cuando una decisión medida queda obsoleta.

El 2026-08-11 encontramos que K_EV=3 se había elegido midiendo bien (+$9.737 ± 859,
16/16 reps) pero a 2.400 sorteos, y que al subir a 19.200 la respuesta correcta pasó a
ser 5 — un cambio de ~$8.500 que estuvo tres días perdido porque nada conecta una
decisión con los supuestos bajo los que se tomó.

Estos tests no re-miden nada (eso cuesta horas). Hacen dos cosas baratas:

  1. Verifican que `config/decisiones.yaml` diga lo que el código realmente hace, para
     que el registro no se convierta en otro comentario desactualizado.
  2. Comparan los supuestos de cada decisión contra los de producción y fallan si no
     coinciden, salvo que alguien haya escrito por qué está bien.

O sea: el día que alguien cambie `--sims` en el unit, la suite se pone roja y lista
exactamente qué decisiones hay que revisar. Que es lo que habría pasado el 10/8.
"""

import re
from pathlib import Path

import pytest
import yaml

RAIZ = Path(__file__).resolve().parents[1]
REGISTRO = RAIZ / "config" / "decisiones.yaml"
UNIT = RAIZ / "deploy" / "clausura-picks.service"


def cargar():
    return yaml.safe_load(REGISTRO.read_text(encoding="utf-8"))["decisiones"]


def produccion() -> dict:
    """Los supuestos con los que corre producción HOY, leídos del unit de systemd."""
    txt = UNIT.read_text(encoding="utf-8")
    sims = re.search(r"--sims (\d+)", txt)
    parts = re.search(r"--participaciones (\d+)", txt)
    assert sims and parts, "el unit de picks ya no pasa --sims/--participaciones"
    return {"sims": int(sims.group(1)), "participaciones": int(parts.group(1))}


def valor_real(donde: str):
    """Resuelve el valor que el código tiene HOY para esa decisión.

    Tres formas de apuntar, para no obligar a que toda decisión sea una constante:
      deploy/archivo --flag             → lo que corre en producción
      modulo.funcion(parametro=)        → el default de ese parámetro
      modulo.CONSTANTE / Clase.campo    → el atributo
    """
    import importlib
    import inspect

    if donde.startswith("deploy/"):
        return produccion()["sims"]

    if donde.endswith("=)"):
        ruta, param = donde[:-2].split("(")
        mod, fn = ruta.rsplit(".", 1)
        f = getattr(importlib.import_module(mod), fn)
        return inspect.signature(f).parameters[param].default

    mod, attr = donde.rsplit(".", 1)
    try:
        return getattr(importlib.import_module(mod), attr)
    except ModuleNotFoundError:
        # Clase.campo: se instancia para leer el default del dataclass
        mod, clase, attr = donde.rsplit(".", 2)
        return getattr(getattr(importlib.import_module(mod), clase)(), attr)


@pytest.mark.parametrize("d", cargar(), ids=lambda d: d["nombre"])
def test_el_registro_dice_lo_que_el_codigo_hace(d):
    """Si el registro miente, es peor que no tenerlo: da falsa tranquilidad."""
    real = valor_real(d["donde"])
    assert real == d["valor"], (
        f"«{d['nombre']}»: el registro dice {d['valor']!r} pero {d['donde']} "
        f"vale {real!r}. Actualizá config/decisiones.yaml (y si el cambio no se "
        f"midió, medilo antes)."
    )


def test_ninguna_decision_quedo_vencida_sin_declararlo():
    """LA alarma. Falla cuando producción se movió y una decisión no se revisó.

    Se apaga de dos formas legítimas: re-midiendo (y actualizando `medida_con`), o
    escribiendo `vencida:` con el motivo por el que se acepta el desfasaje. Lo que no
    se puede es ignorarla en silencio, que es exactamente lo que pasó con el menú.
    """
    prod = produccion()
    problemas = []
    for d in cargar():
        if d.get("inerte") or d.get("vencida"):
            continue
        for clave, valor_prod in prod.items():
            medido = d["medida_con"].get(clave)
            if medido is not None and medido != valor_prod:
                problemas.append(
                    f"  · «{d['nombre']}» ({d['donde']} = {d['valor']!r})\n"
                    f"      se midió con {clave}={medido}, producción corre {valor_prod}"
                )
    assert not problemas, (
        "Decisiones medidas en un régimen que ya no es el de producción:\n"
        + "\n".join(problemas)
        + "\n\n  Re-medí y actualizá `medida_con`, o declará `vencida:` con el motivo.\n"
          "  Precedente: K_EV se eligió a 2.400 sorteos y valía ~$8.500 corregirlo a 19.200."
    )


def test_las_vencidas_estan_a_la_vista(capsys):
    """Las declaradas vencidas no fallan, pero tienen que ser visibles y justificadas.

    Sin esto, `vencida:` sería un interruptor para silenciar la alarma para siempre.
    """
    vencidas = [d for d in cargar() if d.get("vencida")]
    for d in vencidas:
        assert len(str(d["vencida"]).strip()) > 40, (
            f"«{d['nombre']}» se declara vencida sin explicar por qué se acepta")
    with capsys.disabled():
        if vencidas:
            print(f"\n  ⚠️  {len(vencidas)} decisión(es) vencidas y aceptadas:")
            for d in vencidas:
                print(f"      · {d['nombre']} (medida a {d['medida_con'].get('sims')} sorteos)")


def test_toda_decision_declara_donde_vive_y_cuanto_midio():
    """Una entrada sin resultado medido es una opinión disfrazada de dato."""
    for d in cargar():
        for campo in ("nombre", "donde", "valor", "medida_con", "resultado", "fecha"):
            assert campo in d, f"«{d.get('nombre', '?')}» no declara {campo}"
        assert "sims" in d["medida_con"], f"«{d['nombre']}» no dice con cuántos sorteos"
