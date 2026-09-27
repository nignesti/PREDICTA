import pandas as pd
import pytest

import predicta.nba.quote.clv_nba as clv_nba
import predicta.nba.quote.quote_live_nba as ql
from test_quote_live_nba import EVENTI, _book


def _snapshot(eventi, istante):
    return ql.quote_in_tabella(eventi).assign(istante=pd.Timestamp(istante))


def _storico(tmp_path):
    apertura = EVENTI  # Unibet paga NYK 2.60 contro una quota equa Pinnacle ~2.55
    chiusura = [dict(EVENTI[0], bookmakers=[
        _book("pinnacle", h2h=(1.66, 2.35), spread=(-5.5, 1.93, 1.93), totale=(224.5, 1.91, 1.95)),
    ])]
    for eventi, istante in ((apertura, "2026-10-21T15:00Z"), (chiusura, "2026-10-21T23:00Z")):
        ql.salva_snapshot_quote(ql.quote_in_tabella(eventi), istante, cartella=tmp_path)
    return ql.carica_snapshot(tmp_path)


def test_snapshot_salvati_e_ricaricati(tmp_path):
    snapshot = _storico(tmp_path)
    assert snapshot["istante"].nunique() == 2
    assert str(snapshot["inizio"].dt.tz) == "UTC"


def test_chiusura_e_l_ultimo_snapshot_prima_dell_inizio(tmp_path):
    c = clv_nba.chiusure(_storico(tmp_path))
    assert (c["istante_chiusura"] == pd.Timestamp("2026-10-21T23:00Z")).all()
    assert c.loc[c["mercato"] == "spreads", "linea_chiusura"].iloc[0] == -5.5


def test_clv_calcolato_sulla_probabilita_di_chiusura(tmp_path):
    clv, linea_mossa = clv_nba.calcola_clv(_storico(tmp_path), soglia_ev=0.0)
    ml = clv[clv["mercato"] == "h2h"]
    assert list(ml["esito"]) == ["trasferta"]
    # NYK si e' accorciata in chiusura (2.45 -> 2.35 da Pinnacle): la quota presa a 2.60 batte la chiusura
    assert ml["CLV"].iloc[0] == pytest.approx(2.60 * ml["p_chiusura"].iloc[0] - 1)
    assert ml["CLV"].iloc[0] > 0
    # lo spread segnalato a -4.5 non si confronta con la chiusura a -5.5
    assert (clv["mercato"] != "spreads").all()


def test_riepilogo_con_intervallo_di_confidenza(tmp_path):
    clv, _ = clv_nba.calcola_clv(_storico(tmp_path), soglia_ev=0.0)
    r = clv_nba.riepilogo_clv(clv)
    assert r["n"] == len(clv) and r["ic_95"][0] <= r["clv_medio"] <= r["ic_95"][1]


def test_nessuno_snapshot(tmp_path):
    assert ql.carica_snapshot(tmp_path).empty
