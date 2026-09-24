# Experimento del goleador en el Monte Carlo

**Fecha del diseño:** 2026-09-08  
**Objetivo primario:** determinar si incluir los 25 puntos del goleador al optimizar el portfolio aumenta la probabilidad de que al menos una de nuestras 12 participaciones cobre el premio general del Clausura 2026.

## Alcance

El trabajo agrega un experimento reproducible y de solo lectura. No cambia `GOLEADOR_EN_MC`, no genera una nueva versión de planilla, no modifica producción y no envía Telegram. El resultado será evidencia para una decisión posterior.

El experimento comparará:

- **Control:** el comportamiento vigente, con campeón dentro del Monte Carlo y goleador fuera.
- **Tratamiento:** el mismo pipeline, pero incorporando el goleador propio y rival al Monte Carlo durante la optimización.

Ambos brazos usarán el mismo estado del torneo, la misma cantidad de participaciones, los mismos sorteos y las mismas semillas. Los partidos jugados o cerrados conservarán los picks realmente cargados.

## Datos

La corrida leerá el config sincronizado, resultados finalizados, ranking vivo, último snapshot completo del pool y última planilla versionada. Excluirá nuestras 12 participaciones del conjunto de rivales.

El menú y las elecciones de goleador se reconstruirán desde el snapshot cuando el endpoint de opciones no esté disponible, siguiendo el camino que ya existe en `src/clausura/picks.py`. El prior central se obtendrá con `goleador_prior_desde_pool`, aplicando el encogimiento vigente para evitar tratar la popularidad del pool como probabilidad deportiva pura.

El script deberá registrar en su salida los archivos y timestamps de entrada, el número de rivales, partidos congelados, picks observados y opciones de goleador. Si falta una entrada necesaria, abortará con un error claro en vez de sustituirla silenciosamente por un default.

## Escenarios de sensibilidad

El escenario central usará el prior actualizado que produce el sistema. Además, se evaluarán tres inclinaciones controladas hacia Matías Arezo, Maximiliano Gómez y Abel Hernández.

Cada inclinación multiplicará por 1,5 la probabilidad del jugador elegido y renormalizará el vector completo. Este factor es deliberadamente moderado: prueba si el signo depende de un favorito concreto sin convertir el escenario en una certeza artificial. Los cuatro escenarios compartirán las mismas semillas dentro de cada comparación.

## Construcción y evaluación

Cada brazo construirá un portfolio de 12 filas con 19.200 simulaciones y los parámetros vigentes de producción. La comparación final se hará con semillas frescas, distintas de las usadas para optimizar, y números aleatorios comunes entre control y tratamiento.

La métrica primaria será:

`P(al menos una de nuestras 12 participaciones cobra una parte del premio general)`

El estimador de cada escenario informará el delta tratamiento menos control y su error estándar entre semillas de evaluación. También informará:

- premio general esperado;
- premio total esperado, incluidos los premios por fecha;
- puntos esperados de la mejor fila;
- cantidad y detalle de celdas que cambian;
- participación propia favorecida en cada combinación de campeón y goleador.

La probabilidad de cobrar se usa porque el reglamento reparte el premio entre empatados. El informe no la presentará como una probabilidad real calibrada: es una comparación interna bajo el modelo.

## Regla de decisión

El experimento recomendará habilitar el goleador en producción solamente si se cumplen las condiciones siguientes:

1. En el escenario central, el delta de la métrica primaria es positivo y supera dos errores estándar.
2. El delta no es negativo y supera dos errores estándar en ninguno de los tres escenarios inclinados.
3. El premio general esperado no cae más de $2.000 en el escenario central.
4. La conclusión se mantiene al repetir la evaluación con al menos cinco semillas frescas.

Si la señal es positiva pero no supera la incertidumbre, el resultado será **inconcluso**. Si alguna sensibilidad produce un deterioro estadísticamente claro, el resultado será **rechazado por fragilidad**. Ninguno de estos resultados cambia producción automáticamente.

## Componentes

Se creará `scripts/backtest_goleador_actual.py` como harness del experimento. Reutilizará las funciones puras y modelos existentes de `src/clausura/` y mantendrá la lógica experimental fuera del pipeline productivo.

Se agregarán pruebas en `tests/test_backtest_goleador_actual.py` para:

- la inclinación y renormalización del prior;
- la clasificación de resultados como adoptar, inconcluso o rechazado;
- el cálculo pareado del delta de probabilidad y su error estándar;
- el rechazo temprano cuando faltan snapshot, planilla u opciones de goleador.

La salida completa se persistirá en `data/experimentos/goleador_actual_20260908.json`, con parámetros, entradas, resultados por semilla y resumen. El archivo será versionable porque documenta la evidencia de una decisión de estrategia.

## Ejecución prevista

La validación rápida correrá con pocos sorteos para verificar el cableado. La medición decisiva correrá a 19.200 simulaciones, 12 participaciones y al menos cinco semillas de evaluación por escenario. Por consumo de memoria y por necesitar el snapshot vivo, se ejecutará en el VPS una vez que el código y las pruebas locales estén listos.

Si el tratamiento satisface la regla de decisión, el paso posterior será actualizar `config/decisiones.yaml`, habilitar la perilla de producción de forma explícita y correr la suite completa antes del deploy. Ese cambio queda fuera del alcance de este experimento.
