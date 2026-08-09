# Deploy del Clausura en el VPS

Stack mínimo para la penca Supermatch del Clausura 2026. **Nada del Mundial ni del
valuebet se habilita** — este deploy es solo dashboard + planilla automática.

Qué corre en el droplet:

| Unit | Qué hace |
|---|---|
| `clausura-dashboard.service` | uvicorn `src.clausura.webapp` en `:8000` — el dashboard con token en la URL, accesible desde el celular |
| `clausura-picks.timer` | jue-dom 12:00 UTC (09:00 UY): re-sync del fixture + pipeline de picks (`--fecha auto`) + **planilla por Telegram** |
| `clausura-carga-alert.timer` | cada hora 11-23 UTC: aviso por Telegram si faltan cargar picks a 6h/2h del cierre |
| `clausura-drift-audit.timer` | 13:20 / 18:20 / 23:50 UTC: compara lo cargado en la web vs la planilla guardada y avisa discrepancias (pre-inicio sale en silencio) |
| `clausura-postmortem.timer` | 03:20 UTC diario: si una fecha quedó completa, snapshot fresco del pool + postmortem por Telegram (puntos reales vs esperados, pool, exactos) |
| `clausura-rerun-cierre.timer` | ~2h antes del primer cierre del día: re-corre el pipeline (odds+pool frescos) y avisa SOLO si algún pick abierto cambió vs la planilla de la mañana |
| `clausura-goleador-watch.timer` | cada hora: cuando el admin publique los menús de Campeón/Goleador (hoy 500), avisa qué cargar por participación; con goleador regenera la planilla para asignarlo |
| `penca-failure-notify@.service` | template `OnFailure=`: si un service del Clausura falla, aviso por Telegram con el nombre del unit |

La carga de picks en supermatch.com.uy sigue siendo **manual** (decisión 2026-08-04).

## Levantar de cero (~15 min)

```bash
# 1. Crear droplet (DO_API_TOKEN en .env de la raíz del repo local)
python3 deploy/provision_droplet.py --name penca-clausura-2026
# → imprime la IP. export IP=<IP_NUEVA>
```

```bash
# 2. Setup base (igual que el Mundial: clona /opt/penca, venv, swap)
ssh root@$IP "curl -fsSL https://raw.githubusercontent.com/JoacoMarquez/penca-mundial-2026/main/deploy/setup_droplet.sh | bash"
```

```bash
# 3. Secrets mínimos (desde el backup del Mundial sirven los mismos)
BK=/Users/joaquinmarquez/Documents/Personal/Automatizaciones/penca-backups/vps-20260729
scp $BK/etc-penca/env root@$IP:/etc/penca/env
ssh root@$IP "chmod 600 /etc/penca/env"
# Claves que usa este stack: DASHBOARD_TOKEN, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID.
# (el resto de las claves del env viejo no molestan; DRY_RUN da igual acá)
```

```bash
# 4. Stack del Clausura
ssh root@$IP "bash /opt/penca/deploy/setup_clausura.sh"
# → imprime la URL del dashboard con tu token
```

En el celular: guardar `http://<IP>:8000/dash/<DASHBOARD_TOKEN>/` como acceso directo.

Páginas:

| Ruta | Qué muestra |
|---|---|
| `/dash/<TOKEN>/` | planilla de la fecha + ranking en vivo (top-15 + las nuestras) |
| `/dash/<TOKEN>/carga/` | modo carga: una tarjeta por participación, marcas por valor y verificación contra la web |
| `/dash/<TOKEN>/pool/` | el pool: líder y empatados en la cima, puesto real de cada participación nuestra, distribución de puntos, premio de la fecha y trayectoria |

### Modo carga: dos defensas contra el error humano

1. **Marcas por valor.** Tocás cada fila al copiarla en la web y se guarda *el
   marcador que cargaste* (localStorage, clave `carga:v2:<fecha>:<part>:<evento>`).
   Si una corrida posterior mueve ese pick, la fila avisa `cargaste 2-1 → corregí a
   1-1` y el cambio queda listado arriba. Antes la marca era un sí/no con la versión
   de planilla en la clave: planilla nueva ⇒ progreso borrado y cambio invisible.
2. **Verificación real** (botón *🔎 Verificar contra la web*, endpoint
   `/dash/<TOKEN>/api/verificar`). Lee los pronósticos efectivamente cargados
   (`pronosticosEventos` + `pronosticoCampeonGoleador`, públicos post-inicio) y los
   compara con la planilla: ~24 requests con el mismo pacing que el escaneo del pool,
   cacheadas 60 s. Sincroniza las marcas con lo que dice la web. Si el API no expone
   los pronósticos de esa fecha, lo dice en vez de reportar "sin cargar".
   También como CLI: `python3 -m src.clausura.verificar_carga --fecha N`.

El pool sale de **una** request al penca-api (`ranking?size=1000` trae las ~700 filas
enteras), cacheada 120 s y compartida con el home. El escaneo caro —2 requests por
participación— es otra cosa: sirve para ver los PICKS ajenos (`pool_snapshot`), no
la tabla. La trayectoria se alimenta sola de esas lecturas en
`data/pool_history/clausura/ranking.jsonl` (no versionado).

## Nota de seguridad

Igual que en el Mundial: HTTP plano con el token como secreto en la URL. Aceptable
para un dashboard read-only de una penca; si algún día muestra algo sensible,
ponerle Caddy con TLS adelante es media hora.

## Apagar al final del torneo

```bash
ssh root@$IP "systemctl disable --now clausura-dashboard.service clausura-picks.timer"
bash deploy/backup_vps.sh      # si querés conservar data/predictions/clausura/
bash deploy/destroy_droplet.sh
```
