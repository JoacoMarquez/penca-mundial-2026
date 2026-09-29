"""Mensajes largos: Telegram rechaza > 4096 caracteres y el postmortem de la F8 se
perdió entero por eso (28/9). send() los parte por secciones."""

from src.notifier.telegram import MAX_CHARS, TelegramConfig, TelegramNotifier, _partir


def test_corto_no_se_parte():
    assert _partir("hola\n\nchau") == ["hola\n\nchau"]


def test_parte_por_parrafos_sin_perder_texto():
    parrafos = [f"<b>Sección {i}</b>\n" + "x" * 900 for i in range(10)]
    text = "\n\n".join(parrafos)
    trozos = _partir(text)
    assert len(trozos) > 1
    assert all(len(t) <= MAX_CHARS for t in trozos)
    assert "\n\n".join(trozos) == text
    # ningún párrafo quedó cortado al medio
    assert all(t.count("<b>") == t.count("</b>") for t in trozos)


def test_parrafo_gigante_se_parte_por_lineas():
    lineas = [f"línea {i} " + "y" * 90 for i in range(100)]
    text = "\n".join(lineas)
    trozos = _partir(text, limite=1000)
    assert all(len(t) <= 1000 for t in trozos)
    assert "\n".join(trozos) == text


def test_linea_gigante_se_corta_a_lo_bruto():
    trozos = _partir("z" * 2500, limite=1000)
    assert [len(t) for t in trozos] == [1000, 1000, 500]


def test_send_manda_varios_y_devuelve_el_primer_id(monkeypatch):
    enviados = []

    class Resp:
        status_code = 200

        def __init__(self, n):
            self.n = n

        def json(self):
            return {"result": {"message_id": self.n}}

    notif = TelegramNotifier(TelegramConfig(bot_token="t", chat_id="c"))

    def post(url, json):
        enviados.append(json["text"])
        return Resp(100 + len(enviados))

    monkeypatch.setattr(notif._client, "post", post)
    text = "\n\n".join("w" * 1500 for _ in range(6))
    assert notif.send(text) == 101
    assert len(enviados) > 1
    assert all(len(t) <= MAX_CHARS for t in enviados)
