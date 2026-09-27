import pandas as pd
import pytest

import quote_live_nba as ql


def _book(chiave, h2h=None, spread=None, totale=None):
    mercati = []
    if h2h:
        mercati.append({"key": "h2h", "outcomes": [
            {"name": "Boston Celtics", "price": h2h[0]}, {"name": "New York Knicks", "price": h2h[1]}]})
    if spread:
        linea, q_casa, q_trasf = spread
        mercati.append({"key": "spreads", "outcomes": [
            {"name": "Boston Celtics", "price": q_casa, "point": linea},
            {"name": "New York Knicks", "price": q_trasf, "point": -linea}]})
    if totale:
        linea, q_over, q_under = totale
        mercati.append({"key": "totals", "outcomes": [
            {"name": "Over", "price": q_over, "point": linea},
            {"name": "Under", "price": q_under, "point": linea}]})
    return {"key": chiave, "title": chiave.title(), "markets": mercati}


EVENTI = [{
    "id": "e1", "sport_key": "basketball_nba", "commence_time": "2026-10-21T23:30:00Z",
    "home_team": "Boston Celtics", "away_team": "New York Knicks",
    "bookmakers": [
        _book("pinnacle", h2h=(1.60, 2.45), spread=(-4.5, 1.93, 1.93), totale=(224.5, 1.95, 1.91)),
        _book("unibet_eu", h2h=(1.57, 2.60), spread=(-4.5, 1.90, 1.90), totale=(226.5, 2.10, 1.75)),
        _book("betsson", h2h=(1.55, 2.40), spread=(-5.0, 2.05, 1.80)),
    ],
}]


def test_quote_in_tabella_mappa_squadre_ed_esiti():
    t = ql.quote_in_tabella(EVENTI)
    assert set(t["Casa"]) == {"BOS"} and set(t["Trasferta"]) == {"NYK"}
    spread_trasf = t[(t["book"] == "pinnacle") & (t["mercato"] == "spreads") & (t["esito"] == "trasferta")]
    assert spread_trasf["linea"].iloc[0] == 4.5
    assert len(t) == 2 * 3 + 2 * 3 + 2 * 2


def test_probabilita_eque_usano_il_book_sharp_senza_margine():
    eque = ql.probabilita_eque(ql.quote_in_tabella(EVENTI))
    spread = eque[eque["mercato"] == "spreads"]
    assert len(spread) == 1 and spread["linea_rif"].iloc[0] == -4.5
    assert spread["p1"].iloc[0] == pytest.approx(0.5)
    ml = eque[eque["mercato"] == "h2h"]
    assert ml["fonte"].iloc[0] == "pinnacle"
    assert 1 / 1.60 > ml["p1"].iloc[0] > 0.5


def test_trova_valore_confronta_solo_la_stessa_linea_ed_esclude_lo_sharp():
    valore = ql.trova_valore(ql.quote_in_tabella(EVENTI))
    assert "Pinnacle" not in set(valore["book"])
    # Unibet trasferta 2.60 contro una quota equa di ~2.40: valore positivo
    ml = valore[(valore["mercato"] == "h2h") & (valore["book"] == "Unibet_Eu")]
    assert list(ml["esito"]) == ["trasferta"] and ml["EV"].iloc[0] > 0
    # Betsson spread -5.0 e Unibet totale 226.5 hanno linee diverse da Pinnacle: esclusi
    assert not ((valore["mercato"] == "spreads") & (valore["book"] == "Betsson")).any()
    assert not (valore["mercato"] == "totals").any()
    assert (valore["Kelly"] > 0).all()


def test_consenso_quando_manca_lo_sharp():
    senza_sharp = EVENTI[0]["bookmakers"][1:] + [
        _book("tipico_de", h2h=(1.58, 2.50), spread=(-4.5, 1.88, 1.92)),
        _book("sport888", spread=(-4.5, 1.90, 1.90)),
    ]
    eque = ql.probabilita_eque(ql.quote_in_tabella([dict(EVENTI[0], bookmakers=senza_sharp)]))
    assert set(eque["fonte"]) == {"mediana book"}
    ml = eque[eque["mercato"] == "h2h"].iloc[0]
    assert ml["n_book"] == 3
    # spread: -4.5 quotato da 3 book batte -5.0 (Betsson da solo)
    sp = eque[eque["mercato"] == "spreads"].iloc[0]
    assert sp["linea_rif"] == -4.5 and sp["n_book"] == 3
    # totale: solo Unibet lo quota, sotto la soglia minima di book
    assert "totals" not in set(eque["mercato"])


def test_consenso_non_calcolato_con_pochi_book():
    eventi = [dict(EVENTI[0], bookmakers=EVENTI[0]["bookmakers"][1:])]
    assert ql.probabilita_eque(ql.quote_in_tabella(eventi)).empty


def test_quote_sharp_per_schedina_nel_formato_della_pagina():
    df = ql.quote_sharp_per_schedina(ql.quote_in_tabella(EVENTI))
    riga = df.iloc[0]
    assert (riga["Casa"], riga["Trasferta"]) == ("BOS", "NYK")
    assert riga["Quota vittoria Casa"] == 1.60 and riga["Linea spread (Casa)"] == -4.5
    assert riga["Linea totale"] == 224.5 and riga["Quota Under"] == 1.91


def test_squadre_coprono_lo_storico():
    import unisci_quote_nba
    assert set(unisci_quote_nba.MAPPA_SQUADRE.values()) == set(ql.SQUADRE.values())


def test_raccolta_salta_odds_senza_partite_vicine(monkeypatch):
    import raccogli_quote_nba

    chiamate = []
    monkeypatch.setenv("ODDS_API_KEY", "x")
    monkeypatch.setattr(ql, "partite_in_programma", lambda chiave, fino_a: 0)
    monkeypatch.setattr(ql, "scarica_quote", lambda *a, **k: chiamate.append(k))
    assert raccogli_quote_nba.main() == 0 and chiamate == []
