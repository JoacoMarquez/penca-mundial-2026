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

    Cuatro formas de apuntar, para no obligar a que toda decisión sea una constante:
      deploy/archivo --flag             → lo que corre en producción
      modulo.funcion(parametro=)        → el default de ese parámetro
      modulo.CONSTANTE / Clase.campo    → el atributo
      (nada — ...)                      → un RECHAZO: se midió y no se cambió nada,
                                          así que no hay valor que verificar
    """
    import importlib
    import inspect

    if donde.startswith("("):
        return None

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


# -------------------- potencia de los rechazos (auditoría 13/8) --------------------

# Efecto mínimo detectable al 80% de potencia, en múltiplos del error estándar.
#
# Sale de la regla del propio proyecto y no de un manual: se adopta cuando
# delta > 2·SE, así que para que un efecto REAL μ supere esa vara el 80% de las veces
# hace falta μ = 2·SE + 0,84·SE = 2,84·SE (0,84 es el cuantil 80 de la normal).
MDE_80_EN_SES = 2.84

# Piso de plata que ya decidimos que justifica actuar: es el mismo UMBRAL_ABS del gate
# del rerun. Un rechazo que no habría podido ver un efecto de este tamaño no dice nada
# útil sobre si conviene la perilla.
PISO_ACCIONABLE = 2_000.0


def mde_80(d: dict) -> float | None:
    """Efecto mínimo que esa medición habría detectado el 80% de las veces."""
    se = d.get("se")
    return None if se is None else MDE_80_EN_SES * float(se)


def es_nulo(d: dict) -> bool:
    """¿El resultado fue indistinguible de cero bajo la regla de 2·SE?

    Distingue el rechazo que MIDIÓ un efecto contrario y grande (decisivo: la
    potencia da igual) del que no encontró nada (donde la potencia lo es todo).
    """
    if d.get("delta") is None or d.get("se") is None:
        return False
    return abs(float(d["delta"])) < 2 * float(d["se"])


def test_toda_medicion_con_error_declara_su_error():
    """Si el resultado dice "±", el número va también en estructura.

    Sin esto la potencia no se puede calcular sin parsear prosa, y la prosa se
    escribe distinto cada vez. Es la única forma de que la alarma de potencia siga
    funcionando para las decisiones que todavía no se tomaron.
    """
    faltan = [d["nombre"] for d in cargar()
              if "±" in str(d["resultado"]) and not d.get("error_no_es_efecto") and
              (d.get("se") is None or d.get("delta") is None or d.get("unidad") is None)]
    assert not faltan, (
        "Estas entradas reportan un ± pero no declaran delta/se/unidad:\n  · "
        + "\n  · ".join(faltan)
        + "\n\n  Sin el error en estructura no se puede saber si un rechazo significa "
          "«no hay efecto» o «no lo habría visto».\n"
          "  Si ese ± NO es un efecto medido (por ejemplo, dispersión de Monte Carlo "
          "entre semillas),\n  declaralo con `error_no_es_efecto:` explicando por qué."
    )


def test_la_salida_de_potencia_viene_justificada():
    """`error_no_es_efecto` no puede ser un interruptor para saltear la alarma."""
    for d in cargar():
        if d.get("error_no_es_efecto"):
            assert len(str(d["error_no_es_efecto"]).strip()) > 40, (
                f"«{d['nombre']}» saltea el chequeo de potencia sin explicar por qué")
            assert d.get("se") is None, (
                f"«{d['nombre']}» declara `se` Y dice que su ± no es un efecto: "
                f"decidí cuál de las dos cosas es")


def test_las_unidades_son_conocidas():
    """`unidad` decide contra qué vara se juzga: el piso de $2.000 solo aplica a
    mediciones de E[premio]; nats y goles se juzgan con su propia escala."""
    validas = {"pesos", "nats", "goles"}
    raras = [(d["nombre"], d["unidad"]) for d in cargar()
             if d.get("unidad") and d["unidad"] not in validas]
    assert not raras, f"unidades desconocidas: {raras}"


def test_los_rechazos_subpotenciados_estan_a_la_vista(capsys):
    """No falla: informa. Un rechazo subpotenciado es un hecho histórico, no un bug —
    lo que hace falta es que no se lea como evidencia de que no hay nada.

    La auditoría del 13/8 lo pidió porque el registro escribía igual "medí y no hay
    efecto" que "medí y no pude ver nada", y en esta lista entran perillas que
    podrían valer plata y quedaron apagadas sin que nadie lo supiera.
    """
    subpotenciados = [
        d for d in cargar()
        if d.get("unidad") == "pesos" and es_nulo(d) and (mde_80(d) or 0) > PISO_ACCIONABLE
    ]
    with capsys.disabled():
        if subpotenciados:
            print(f"\n  🔍 {len(subpotenciados)} rechazo(s) SUBPOTENCIADO(s) "
                  f"— nulos, pero ciegos a efectos accionables:")
            for d in sorted(subpotenciados, key=lambda x: -(mde_80(x) or 0)):
                print(f"      · {d['nombre']}: midió {d['delta']:+,.0f} ± {d['se']:,.0f} "
                      f"⇒ indistinguible de un efecto real de hasta "
                      f"${mde_80(d):,.0f}")
            print(f"      (vara: el piso de ${PISO_ACCIONABLE:,.0f} del gate del rerun)")


def test_el_mde_se_calcula_de_la_regla_del_proyecto():
    """Fija la aritmética para que nadie la cambie sin querer."""
    assert mde_80({"se": 1000}) == pytest.approx(2840.0)
    assert mde_80({}) is None
    # nulo = indistinguible de cero con la vara de 2·SE que usa el resto del sistema
    assert es_nulo({"delta": -626, "se": 1359})
    assert not es_nulo({"delta": -9486, "se": 2154})     # rechazo decisivo, no ciego
    assert not es_nulo({"delta": 8531, "se": 10})        # adopción nítida


def test_los_rechazos_conocidos_quedan_bien_clasificados():
    """Ancla el resultado del análisis del 13/8 sobre las entradas reales.

    Si alguien re-mide una de estas con más réplicas y la saca de la lista, este test
    se lo hace notar (y ahí se actualiza, que es justo lo que se quiere que pase).
    """
    por_nombre = {d["nombre"]: d for d in cargar()}

    # ciegos: no vieron nada, pero tampoco podrían haber visto $2.000
    for nombre in ("ausentismo por fecha en el simulador",
                   "cobertura de desenlace en el menú"):
        d = por_nombre[nombre]
        assert es_nulo(d) and mde_80(d) > PISO_ACCIONABLE, (
            f"«{nombre}» dejó de ser un rechazo subpotenciado — actualizá el registro")

    # con potencia suficiente: el rechazo del goleador SÍ descarta un efecto accionable
    d = por_nombre["goleador dentro del Monte Carlo"]
    assert es_nulo(d) and mde_80(d) < PISO_ACCIONABLE

    # decisivo: efecto grande y contrario, la potencia es irrelevante
    assert not es_nulo(por_nombre["métrica de orden de la rama de hueco"])
