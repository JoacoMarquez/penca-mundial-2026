"""Fuentes de cuotas del sistema valuebet.

Interfaz común (duck typing / Protocol):
    fetch_quotes(cfg: VBConfig) -> list[OddsQuote]

Roles:
    supermatch.py  — soft book objetivo (las cuotas que apostamos)
    pinnacle_vb.py — sharp de referencia (precio justo). Fork parametrizado del
                     scraper penca; NO importa src.scrapers.pinnacle.
    oddsapi_vb.py  — fallback de soft book + fuente barata de closing lines.
"""

from __future__ import annotations

from typing import Protocol

from src.valuebet.config import VBConfig
from src.valuebet.types import OddsQuote


class Book(Protocol):
    def fetch_quotes(self, cfg: VBConfig) -> list[OddsQuote]: ...
