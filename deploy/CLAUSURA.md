# Deploy del Clausura en el VPS

Stack mínimo para la penca Supermatch del Clausura 2026. **Nada del Mundial ni del
valuebet se habilita** — este deploy es solo dashboard + planilla automática.

Qué corre en el droplet:

| Unit | Qué hace |
|---|---|
| `clausura-dashboard.service` | uvicorn `src.clausura.webapp` en `:8000` — el dashboard con token en la URL, accesible desde el celular |
| `clausura-picks.timer` | jue-dom 12:00 UTC (09:00 UY): re-sync del fixture + pipeline de picks (`--fecha auto`) + **planilla por Telegram** |

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
