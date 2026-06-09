# Runbook operacional — Penca Mundial 2026

Guía rápida para operar el agente durante el Mundial. Asume que el sistema ya
está desplegado en el VPS (ver `deploy/README.md` para provisioning inicial).

---

## 0. URLs y accesos

- **Dashboard:** `https://<VPS_IP>/dash/<DASHBOARD_TOKEN>/`
- **Telegram bot:** notificaciones llegan al chat configurado en `TELEGRAM_CHAT_ID`.
- **VPS:** `ssh root@<VPS_IP>` (clave SSH).
- **GitHub:** push a `main` → `git pull` en el VPS (manual o vía Action).

Secrets viven en `/etc/penca/env` en el VPS. **Nunca commitearlos.**

---

## 1. Estado del sistema (chequeo de 30 segundos)

```bash
ssh root@<VPS_IP> 'cd /opt/penca && \
  systemctl status penca-scheduler.timer penca-heartbeat.timer --no-pager | head -30 && \
  echo "---" && \
  journalctl -u penca-scheduler.service -n 20 --no-pager'
```

Señales verdes:
- Ambos timers `active (waiting)`.
- `next trigger` en menos de 5 min (scheduler).
- Último log del scheduler termina en `idle (no match in window)` o `pipeline DONE`.

Señales rojas:
- `failed` en cualquier service.
- `python: command not found` o `ModuleNotFoundError` → venv roto.
- Heartbeat más viejo de 26h → cron no corrió.

---

## 2. Telegram heartbeats

Llega 1 por día (~09:00 UY). Si no llegó por >26h:

```bash
ssh root@<VPS_IP> 'systemctl list-timers penca-heartbeat.timer'
ssh root@<VPS_IP> 'journalctl -u penca-heartbeat.service -n 50 --no-pager'
```

El heartbeat que recibís va pineado en el chat. Si ves `DRY_RUN activo` en el
footer, el sistema NO publica predicciones — verificar `/etc/penca/env`.

---

## 3. Comandos comunes

### Ver picks de un partido específico
```bash
ssh root@<VPS_IP> 'ls /var/lib/penca/predictions/<MATCH_ID>/'
ssh root@<VPS_IP> 'cat /var/lib/penca/predictions/<MATCH_ID>/v3_*.json | jq .portfolio.picks'
```

### Forzar una pasada manualmente (sin esperar al timer)
```bash
ssh root@<VPS_IP> 'cd /opt/penca && /opt/penca/.venv/bin/python -m src.agent.pipeline <MATCH_ID> T_24h'
```

### Dry-run de la pipeline en local (no publica)
```bash
python -m scripts.dry_run --next                    # próximo partido
python -m scripts.dry_run --match-id 1234 --phase T_3h
python -m scripts.dry_run --next --no-telegram      # también muteado
```

### Deploy nueva versión
```bash
git push origin main
ssh root@<VPS_IP> 'cd /opt/penca && ./deploy/safe_pull.sh && systemctl daemon-reload && systemctl restart penca-scheduler.timer'
```
`safe_pull.sh` corre `python -m compileall` antes del restart — si hay syntax error aborta.

### Detener todo (kill switch)
```bash
ssh root@<VPS_IP> 'systemctl stop penca-scheduler.timer penca-heartbeat.timer'
```

### Activar/desactivar DRY_RUN
```bash
ssh root@<VPS_IP> 'sed -i "s/^DRY_RUN=.*/DRY_RUN=true/" /etc/penca/env && systemctl restart penca-scheduler.service'
```

---

## 4. Troubleshooting por síntoma

| Síntoma | Causa probable | Fix |
|---|---|---|
| Telegram no llega | `TELEGRAM_BOT_TOKEN`/`CHAT_ID` malos | revisar `/etc/penca/env`, mandar `/start` al bot |
| Pipeline crashea con `ValueError teams.yaml` | fixture sin alias en `teams.yaml` | agregar alias en aliases ES→EN |
| `429 Too Many Requests` (Anthropic) | rate limit | esperar; reducir frecuencia de pasadas |
| Pinnacle 503/blocked | guest API caída | el sistema cae a sólo capa 2 — aceptar y monitorear |
| Demasiadas pencas con el mismo marcador | exposición mal calibrada | revisar `greedy_assignment` / `PENCA_MAX_CANDIDATES`; correr `python -m pytest tests/test_assignment.py` |
| Dashboard 500 | cache stale o data_loader bug | `systemctl restart penca-dashboard.service` |
| `odds_anomaly` detectada | Pinnacle se movió >5pp entre pasadas | leer el alert en Telegram; investigar manualmente (lesión? suspensión?) — el sistema ya re-corrió |
| Heartbeat muestra `predicciones=0` | scheduler nunca disparó | revisar `kickoff_utc` en fixtures.yaml; chequear timezone |
| `DRY_RUN` activo cuando debería estar off | env mal cargado | grep `DRY_RUN` en `/etc/penca/env`, restart service |

---

## 5. Flujo por partido (lo esperado)

1. **T-24h:** llega Telegram con el menú de objetivos + la **exposición de las N pencas por marcador** + dossier. *Tu opción:* revisar el dashboard; si algo te chirría, ssh al VPS y editás manualmente el último JSON antes del T-30min.
2. **T-3h:** si hubo cambio de alineación o movimiento de odds, llega notif con diff. Si no, silencio.
3. **T-30min:** lock-in. Llega notif "PUBLISHED" con las picks finales que el publisher mandó a la API.
4. **Post-match:** postmortem automático al detectar resultado final. Aparece en `/dash/<token>/history`.

---

## 6. Costos y límites

- **VPS:** ~$6/mes fijo, sin sorpresas (Droplet Basic).
- **Anthropic:** acumulador en `data/usage_log.jsonl`. Heartbeat muestra MTD y 24h.
  Si te acercás al budget, bajar `LLM_MODEL` a Haiku para partidos no-clave.
- **Telegram / ESPN / Pinnacle / Betfair / Google News:** gratis.

Budget esperado total Mundial: **US$15-25** (Anthropic).

---

## 7. Pre-Mundial checklist (correr 48h antes del 2026-06-11)

- [ ] `python -m scripts.dry_run --next` corre clean local.
- [ ] Dashboard accesible desde celular (`https://<IP>/dash/<token>/`).
- [ ] Heartbeat de hoy llegó OK.
- [ ] `DRY_RUN=false` en `/etc/penca/env`.
- [ ] PENCA_API specs confirmadas y `PENCA_API_KEY` cargada.
- [ ] `systemctl list-timers` muestra próximo trigger de scheduler en <5 min.
- [ ] `journalctl -u penca-scheduler -n 100 --no-pager` sin errores.
- [ ] Backup del repo a una segunda ubicación (GitHub ya cuenta).
- [ ] Postmortem listo (carpeta `data/predictions/` con permisos OK).

---

## 8. Contactos / handoff

- Admin de la penca: **TBD** (specs API).
- Repo: `github.com/JoacoMarquez/penca-mundial-2026`.
- VPS provider: DigitalOcean (login con cuenta personal).
