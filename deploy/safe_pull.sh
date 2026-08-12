#!/usr/bin/env bash
# Safe pull: hace git pull, valida que todos los módulos importen, y SOLO si pasa,
# acepta el cambio. Si falla, hace rollback al commit anterior.
#
# Uso: ssh root@VPS /opt/penca/deploy/safe_pull.sh
#       o en cron: cada 5 min ANTES del scheduler.

set -e
cd /opt/penca

# Re-instala los units de systemd que hayan cambiado. SIN esto, un cambio en un
# .service o .timer que llega por pull NO se aplica nunca y nadie se entera: los
# units viven COPIADOS en /etc/systemd/system (así los instala setup_clausura.sh),
# no como symlink al repo, así que `git pull` + `daemon-reload` es un no-op para
# ellos. Pasó el 2026-08-08 con `--sims 2400 → 3200`: el pull entró, daemon-reload
# corrió, y systemd siguió ejecutando el valor viejo. Se detectó de casualidad
# comparando con `systemctl cat`.
#
# Compara CONTENIDO en vez de mirar el diff del pull, así también corrige la deriva
# que ya exista (un unit editado a mano en el VPS, que es cómo se pierden los fixes
# cuando el droplet se recrea). El repo es la fuente de verdad, pero se avisa qué se
# pisó para que un cambio local no desaparezca en silencio.
#
# Solo toca units YA instalados: dar de alta uno nuevo es trabajo de
# setup_clausura.sh, que además lo habilita.
sync_units() {
    local cambiados=() faltantes=() u dst tipo
    for src in /opt/penca/deploy/*.service /opt/penca/deploy/*.timer; do
        u=$(basename "$src")
        dst="/etc/systemd/system/$u"
        if [ ! -f "$dst" ]; then
            # Unit del repo que el VPS no tiene: instalarlo requiere decidir si va
            # habilitado (setup_clausura.sh), pero callarlo es peor — gate-watch y
            # heartbeat nacieron después del setup inicial y nadie se enteraba.
            case "$u" in clausura-*) faltantes+=("$u");; esac
            continue
        fi
        if ! cmp -s "$src" "$dst"; then
            cp "$src" "$dst"
            cambiados+=("$u")
        fi
    done

    if [ ${#faltantes[@]} -gt 0 ]; then
        echo "⚠️  Units del repo NO instalados en el VPS: ${faltantes[*]}" >&2
        echo "    Si tienen que correr, instalalos con: bash deploy/setup_clausura.sh" >&2
    fi

    if [ ${#cambiados[@]} -eq 0 ]; then
        return 0
    fi

    echo "⚙️  Units actualizados desde el repo: ${cambiados[*]}"
    systemctl daemon-reload

    # Los .service disparados por timer toman el archivo nuevo en su próximo disparo.
    # Los de larga vida (dashboard) NO: siguen corriendo con la definición vieja hasta
    # que se reinicien. Se filtra por Type para no reiniciar un oneshot que justo esté
    # a mitad de corrida.
    for u in "${cambiados[@]}"; do
        case "$u" in
            *.service) ;;
            *) continue ;;
        esac
        tipo=$(systemctl show -p Type --value "$u" 2>/dev/null || echo "")
        case "$tipo" in
            simple|exec|notify|notify-reload)
                if systemctl is-active --quiet "$u"; then
                    echo "   ↻ reiniciando $u (Type=$tipo, estaba activo)"
                    systemctl restart "$u"
                fi
                ;;
        esac
    done
}

# Archivos TRACKEADOS que el VPS REGENERA en cada corrida. Su copia local siempre
# difiere del repo, y eso hace abortar el pull con "Please commit your changes or
# stash them before you merge" en cuanto un commit los toca.
#
# Pasó el 2026-08-10 con config/clausura2026.yaml y dejó al VPS DOS COMMITS ATRÁS sin
# que nadie lo notara: el postmortem corrió con código viejo y volvió a fallar por el
# mismo bug que se acababa de arreglar. El deploy fallaba en silencio, que es
# exactamente lo que safe_pull existe para evitar.
#
# Descartarlos es seguro y hay que verificarlo antes de agregar uno a esta lista:
#   * config/clausura2026.yaml — lo reescribe `src.clausura.sync`, que corre como
#     ExecStartPre (sin "-", o sea obligatorio) de clausura-picks y de
#     clausura-rerun-cierre. Su diff local es solo `generado_utc` y `estado`, y
#     NADIE lee `estado` del config (verificado por grep 2026-08-10).
REGENERADOS=(config/clausura2026.yaml)

descartar_regenerados() {
    local f sucios=()
    for f in "${REGENERADOS[@]}"; do
        if ! git diff --quiet -- "$f" 2>/dev/null; then
            git checkout -- "$f" && sucios+=("$f")
        fi
    done
    if [ ${#sucios[@]} -gt 0 ]; then
        echo "↺ descartados cambios locales de archivos regenerados: ${sucios[*]}"
        echo "  (los reescribe src.clausura.sync antes de la próxima corrida)"
    fi
}

PREVIOUS=$(git rev-parse HEAD)
git fetch -q origin main
INCOMING=$(git rev-parse origin/main)

if [ "$PREVIOUS" = "$INCOMING" ]; then
    # Nada nuevo que validar, pero los units SÍ se reconcilian: la deriva puede venir
    # de un pull anterior a este cambio o de una edición a mano en el VPS, y en los dos
    # casos sigue ahí mientras nadie la corrija.
    sync_units
    exit 0
fi

echo "Pulling $PREVIOUS → $INCOMING"
descartar_regenerados
if ! git pull -q; then
    # Que NO quede en silencio: el modo de falla del 10/8 fue justamente este —
    # abortar y seguir corriendo con código viejo como si nada hubiera pasado.
    echo "❌ git pull FALLÓ — el VPS queda en $PREVIOUS, con código viejo" >&2
    git status --short | head -10 >&2
    set -a; source /etc/penca/env 2>/dev/null; set +a
    .venv/bin/python -c "
from src.notifier.telegram import TelegramNotifier, TelegramConfig
try:
    TelegramNotifier(TelegramConfig.from_env()).send_error(
        'deploy bloqueado',
        'git pull falló en el VPS: sigue en $PREVIOUS. Revisá git status en /opt/penca.')
except Exception:
    pass
" 2>/dev/null || true
    exit 1
fi

# Validar que los módulos críticos importan. El stack VIVO es src.clausura (el del
# Mundial está apagado desde el 29/7): un ImportError en picks/rerun/drift pasaba
# esta validación y se descubría al día siguiente vía OnFailure, con la ventana de
# la fecha ya corriendo.
if .venv/bin/python -c "
import sys
try:
    from src.clausura import (api, carga_alert, dashboard_loader, drift_audit,
                              gate_watch, heartbeat, picks, pool, pool_snapshot,
                              postmortem, rerun_cierre, rivals, scoring, strategy,
                              sync, verificar_carga, webapp)
    from src.notifier import telegram
    print('OK')
except Exception as e:
    print(f'IMPORT ERROR: {type(e).__name__}: {e}', file=sys.stderr)
    sys.exit(1)
" 2>&1; then
    echo "✅ Validation passed at $INCOMING"
    sync_units
else
    echo "❌ Validation FAILED — rolling back to $PREVIOUS" >&2
    git reset --hard "$PREVIOUS" -q
    # Notificar via Telegram (sourcing env)
    set -a
    source /etc/penca/env 2>/dev/null
    set +a
    .venv/bin/python -c "
from src.notifier.telegram import TelegramNotifier, TelegramConfig
try:
    n = TelegramNotifier(TelegramConfig.from_env())
    n.send_error('safe_pull rollback', 'Commit $INCOMING falló validación, rollback a $PREVIOUS')
except Exception:
    pass
" 2>/dev/null || true
    exit 1
fi
