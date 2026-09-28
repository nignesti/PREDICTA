import pandas as pd
import pytest

import predicta.nba.quote.infortuni_nba as inf


def _voce(nome, stato, data):
    return {"athlete": {"displayName": nome}, "status": stato, "date": data, "shortComment": f"{nome} {stato}"}


DATI = {"injuries": [
    {"displayName": "Boston Celtics", "injuries": [
        _voce("Vecchia Notizia", "Out", "2026-11-01T10:00Z"),
        _voce("Nuova Notizia", "Out", "2026-11-01T20:00Z"),
        _voce("Nel Margine", "Day-To-Day", "2026-11-01T17:55Z"),
        _voce("Dopo Inizio", "Out", "2026-11-02T01:00Z"),
    ]},
    {"displayName": "LA Clippers", "injuries": [_voce("Altra Partita", "Out", "2026-11-01T20:00Z")]},
]}

PARTITE = pd.DataFrame({"Casa": ["BOS"], "Trasferta": ["NYK"],
                        "inizio": [pd.Timestamp("2026-11-02T00:00Z")]})


def test_tabella_usa_le_abbreviazioni_dello_storico():
    t = inf.infortuni_in_tabella(DATI)
    assert set(t["squadra"]) == {"BOS", "LAC"}
    assert str(t["aggiornato"].dt.tz) == "UTC"


def test_solo_notizie_dopo_le_quote_e_prima_dell_inizio():
    t = inf.infortuni_in_tabella(DATI)
    # quote scaricate alle 18:00 UTC (19:00 a Roma), margine 10 minuti
    n = inf.notizie_dopo_quote(t, PARTITE, pd.Timestamp("2026-11-01T19:00", tz="Europe/Rome"))
    assert list(n["giocatore"]) == ["Nuova Notizia", "Nel Margine"]
    assert (n["Casa"] == "BOS").all()


def test_formato_inatteso_solleva_errore_dedicato():
    with pytest.raises(inf.ErroreInfortuni):
        inf.infortuni_in_tabella({"injuries": [{"displayName": "X", "injuries": [{"status": "Out"}]}]})
