import logging

import src.notifier.telegram  # noqa: F401  (instala el filtro)


def test_el_log_de_httpx_no_muestra_el_token(caplog):
    url = "https://api.telegram.org/bot123456:AAHsecreto-_x/sendMessage"
    with caplog.at_level(logging.INFO, logger="httpx"):
        logging.getLogger("httpx").info('HTTP Request: %s %s "%s"', "POST", url, "HTTP/1.1 200 OK")
    assert "AAHsecreto" not in caplog.text
    assert "/bot<token>/sendMessage" in caplog.text


def test_otros_requests_quedan_intactos(caplog):
    url = "https://penca.supermatch.com.uy/penca-api/v1/front/pencas/46/ranking"
    with caplog.at_level(logging.INFO, logger="httpx"):
        logging.getLogger("httpx").info("HTTP Request: GET %s", url)
    assert url in caplog.text
