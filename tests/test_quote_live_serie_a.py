import pandas as pd
import pytest

import predicta.serie_a.quote.quote_live_serie_a as ql


def _book(chiave, h2h=None, totale=None):
    mercati = []
    if h2h:
        mercati.append({"key": "h2h", "outcomes": [
            {"name": "Inter Milan", "price": h2h[0]},
            {"name": "Draw", "price": h2h[1]},
            {"name": "Napoli", "price": h2h[2]},
        ]})
    if totale:
        linea, q_over, q_under = totale
        mercati.append({"key": "totals", "outcomes": [
            {"name": "Over", "price": q_over, "point": linea},
            {"name": "Under", "price": q_under, "point": linea},
        ]})
    return {"key": chiave, "title": chiave.title(), "markets": mercati}


EVENTI = [{
    "id": "e1", "sport_key": "soccer_italy_serie_a", "commence_time": "2026-10-21T18:00:00Z",
    "home_team": "Inter Milan", "away_team": "Napoli",
    "bookmakers": [
        _book("pinnacle", h2h=(1.95, 3.60, 3.80), totale=(2.5, 1.95, 1.91)),
        _book("unibet_eu", h2h=(1.90, 3.50, 4.00), totale=(2.5, 2.10, 1.75)),
        _book("betsson", h2h=(1.85, 3.40, 4.20)),
    ],
}]


def test_quote_in_tabella_mappa_squadre_ed_esiti():
    t = ql.quote_in_tabella(EVENTI)
    assert set(t["Casa"]) == {"Inter"} and set(t["Trasferta"]) == {"Napoli"}
    assert set(t[t["mercato"] == "h2h"]["esito"]) == {"casa", "pareggio", "trasferta"}
    assert len(t) == 3 * 3 + 2 * 2


def test_probabilita_eque_usano_il_book_sharp_senza_margine():
    eque = ql.probabilita_eque(ql.quote_in_tabella(EVENTI))
    ml = eque[eque["mercato"] == "h2h"]
    assert ml["fonte"].iloc[0] == "pinnacle"
    p_casa, p_pareggio, p_trasferta = ml["probabilita"].iloc[0]
    assert p_casa + p_pareggio + p_trasferta == pytest.approx(1.0)
    assert 1 / 1.95 > p_casa > 1 / 3.80


def test_trova_valore_confronta_solo_la_stessa_linea_ed_esclude_lo_sharp():
    valore = ql.trova_valore(ql.quote_in_tabella(EVENTI))
    assert "Pinnacle" not in set(valore["book"])
    # Betsson trasferta 4.20 paga piu' della quota equa di Pinnacle: valore positivo
    ml = valore[(valore["mercato"] == "h2h") & (valore["book"] == "Betsson")]
    assert list(ml["esito"]) == ["trasferta"] and ml["EV"].iloc[0] > 0
    # Unibet ha un'unica linea di totale (2.5) uguale a Pinnacle: puo' comparire
    assert (valore["Kelly"] > 0).all()


def test_consenso_quando_manca_lo_sharp():
    senza_sharp = EVENTI[0]["bookmakers"][1:] + [
        _book("tipico_de", h2h=(1.88, 3.55, 4.10)),
        _book("sport888", h2h=(1.92, 3.45, 4.05)),
    ]
    eque = ql.probabilita_eque(ql.quote_in_tabella([dict(EVENTI[0], bookmakers=senza_sharp)]))
    assert set(eque["fonte"]) == {"mediana book"}
    ml = eque[eque["mercato"] == "h2h"].iloc[0]
    assert ml["n_book"] == 4
    p_casa, p_pareggio, p_trasferta = ml["probabilita"]
    assert p_casa + p_pareggio + p_trasferta == pytest.approx(1.0)
    # il totale lo quota solo Unibet, sotto la soglia minima di book
    assert "totals" not in set(eque["mercato"])


def test_consenso_non_calcolato_con_pochi_book():
    eventi = [dict(EVENTI[0], bookmakers=EVENTI[0]["bookmakers"][1:2])]
    assert ql.probabilita_eque(ql.quote_in_tabella(eventi)).empty


def test_quote_sharp_per_schedina_nel_formato_della_pagina():
    df = ql.quote_sharp_per_schedina(ql.quote_in_tabella(EVENTI))
    riga = df.iloc[0]
    assert (riga["Casa"], riga["Trasferta"]) == ("Inter", "Napoli")
    assert riga["Quota 1"] == 1.95 and riga["Quota X"] == 3.60 and riga["Quota 2"] == 3.80
    assert riga["Quota Over 2.5"] == 1.95 and riga["Quota Under 2.5"] == 1.91


def test_raccolta_salta_odds_senza_partite_vicine(monkeypatch):
    import predicta.serie_a.quote.raccogli_quote_serie_a as raccogli_quote_serie_a

    chiamate = []
    monkeypatch.setenv("ODDS_API_KEY", "x")
    monkeypatch.setattr(ql, "partite_in_programma", lambda chiave, fino_a, da=None: 0)
    monkeypatch.setattr(ql, "scarica_quote", lambda *a, **k: chiamate.append(k))
    assert raccogli_quote_serie_a.main() == 0 and chiamate == []


def test_raccolta_prepartita_usa_finestra_stretta(monkeypatch):
    import predicta.serie_a.quote.raccogli_quote_serie_a as raccogli_quote_serie_a

    finestre, chiamate = [], []
    monkeypatch.setenv("ODDS_API_KEY", "x")
    monkeypatch.setenv("QUOTE_DA_MINUTI", "25")
    monkeypatch.setenv("QUOTE_ORIZZONTE_ORE", "0.75")
    monkeypatch.setattr(ql, "partite_in_programma", lambda chiave, fino_a, da=None: finestre.append((da, fino_a)) or 1)
    monkeypatch.setattr(ql, "scarica_quote", lambda *a, **k: chiamate.append(k) or ([], {"ultima": 1, "rimanenti": 1}))
    assert raccogli_quote_serie_a.main() == 0
    da, fino_a = finestre[0]
    assert fino_a - da == pd.Timedelta(minutes=20)
    assert chiamate[0]["da"] == da and chiamate[0]["fino_a"] == fino_a
