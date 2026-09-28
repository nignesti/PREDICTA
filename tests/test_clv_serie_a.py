import pandas as pd
import pytest

import predicta.serie_a.quote.clv_serie_a as clv_serie_a
import predicta.serie_a.quote.quote_live_serie_a as ql
from test_quote_live_serie_a import EVENTI, _book


def _storico(tmp_path):
    apertura = EVENTI  # Betsson paga la trasferta 4.20 contro una quota equa Pinnacle piu' bassa
    chiusura = [dict(EVENTI[0], bookmakers=[
        _book("pinnacle", h2h=(1.90, 3.55, 4.10), totale=(2.5, 1.91, 1.95)),
    ])]
    for eventi, istante in ((apertura, "2026-10-21T10:00Z"), (chiusura, "2026-10-21T17:00Z")):
        ql.salva_snapshot_quote(ql.quote_in_tabella(eventi), istante, cartella=tmp_path)
    return ql.carica_snapshot(tmp_path)


def test_snapshot_salvati_e_ricaricati(tmp_path):
    snapshot = _storico(tmp_path)
    assert snapshot["istante"].nunique() == 2
    assert str(snapshot["inizio"].dt.tz) == "UTC"


def test_chiusura_e_l_ultimo_snapshot_prima_dell_inizio(tmp_path):
    c = clv_serie_a.chiusure(_storico(tmp_path))
    assert (c["istante_chiusura"] == pd.Timestamp("2026-10-21T17:00Z")).all()
    assert set(c["mercato"]) == {"h2h", "totals"}


def test_snapshot_prepartita_solo_1x2_non_toglie_la_chiusura_del_totale(tmp_path):
    _storico(tmp_path)
    prepartita = [dict(EVENTI[0], bookmakers=[_book("pinnacle", h2h=(1.85, 3.60, 4.30))])]
    ql.salva_snapshot_quote(ql.quote_in_tabella(prepartita), "2026-10-21T17:30Z", cartella=tmp_path)
    c = clv_serie_a.chiusure(ql.carica_snapshot(tmp_path)).set_index("mercato")["istante_chiusura"]
    assert c["h2h"] == pd.Timestamp("2026-10-21T17:30Z")
    assert c["totals"] == pd.Timestamp("2026-10-21T17:00Z")


def test_clv_calcolato_sulla_probabilita_di_chiusura(tmp_path):
    clv, linea_mossa = clv_serie_a.calcola_clv(_storico(tmp_path), soglia_ev=0.0)
    ml = clv[clv["mercato"] == "h2h"]
    assert "trasferta" in set(ml["esito"])
    riga = ml[ml["esito"] == "trasferta"].iloc[0]
    assert riga["CLV"] == pytest.approx(riga["quota"] * riga["p_chiusura"] - 1)
    assert linea_mossa == 0  # l'1X2 non ha linea, il totale ha la stessa linea (2.5) in apertura e chiusura


def test_riepilogo_con_intervallo_di_confidenza(tmp_path):
    clv, _ = clv_serie_a.calcola_clv(_storico(tmp_path), soglia_ev=0.0)
    r = clv_serie_a.riepilogo_clv(clv)
    assert r["n"] == len(clv) and r["ic_95"][0] <= r["clv_medio"] <= r["ic_95"][1]


def test_nessuno_snapshot(tmp_path):
    assert ql.carica_snapshot(tmp_path).empty
