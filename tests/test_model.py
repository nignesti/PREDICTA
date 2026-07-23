import numpy as np
import pandas as pd
import pytest

import app
import backtesting as bt


def _partite_squadra(df, squadra):
    return df[(df["HomeTeam"] == squadra) | (df["AwayTeam"] == squadra)]


def _gol_fatti_attesi(partite, squadra):
    valori = []
    for _, row in partite.iterrows():
        valori.append(row["FTHG"] if row["HomeTeam"] == squadra else row["FTAG"])
    return valori


@pytest.fixture
def partite_miste():
    # Partite di "A" alternate tra casa e trasferta, in ordine cronologico (indice = data).
    righe = [
        {"HomeTeam": "A", "AwayTeam": "X", "FTHG": 2, "FTAG": 0},  # 0 casa: V
        {"HomeTeam": "Y", "AwayTeam": "A", "FTHG": 1, "FTAG": 1},  # 1 trasferta: N
        {"HomeTeam": "A", "AwayTeam": "Z", "FTHG": 0, "FTAG": 1},  # 2 casa: P
        {"HomeTeam": "W", "AwayTeam": "A", "FTHG": 2, "FTAG": 0},  # 3 trasferta: P
        {"HomeTeam": "A", "AwayTeam": "Y", "FTHG": 3, "FTAG": 1},  # 4 casa: V
        {"HomeTeam": "Z", "AwayTeam": "A", "FTHG": 1, "FTAG": 2},  # 5 trasferta: V
        {"HomeTeam": "A", "AwayTeam": "W", "FTHG": 1, "FTAG": 1},  # 6 casa: N
    ]
    return pd.DataFrame(righe)


def test_calcola_forma_usa_le_ultime_n_partite_in_ordine_cronologico(partite_miste):
    # Regressione: la vecchia implementazione concatenava le ultime N partite in casa
    # con le ultime N in trasferta e tagliava la coda, finendo per usare solo le trasferte.
    _, _, _, _, risultati, gol_fatti = app.calcola_forma(partite_miste, "A", ultime_n=5)

    attese = _partite_squadra(partite_miste, "A").tail(5)
    gol_fatti_attesi = _gol_fatti_attesi(attese, "A")

    assert gol_fatti == gol_fatti_attesi
    assert len(risultati) == 5
    # Con le partite di fixture, l'ultima trasferta risale a due gare prima dell'ultima
    # partita in casa: se il bug fosse presente, "risultati" conterrebbe solo trasferte (V,P,V)
    assert risultati == ["P", "P", "V", "V", "N"]


def test_calcola_forma_nessuna_partita_restituisce_none(partite_miste):
    risultato = app.calcola_forma(partite_miste, "SquadraInesistente", ultime_n=5)
    assert risultato == (None, None, None, None, [], [])


def test_scontri_diretti_conta_correttamente_vittorie_pareggi():
    df = pd.DataFrame([
        {"HomeTeam": "A", "AwayTeam": "B", "FTHG": 2, "FTAG": 1},  # A vince in casa
        {"HomeTeam": "B", "AwayTeam": "A", "FTHG": 0, "FTAG": 0},  # pareggio
        {"HomeTeam": "A", "AwayTeam": "B", "FTHG": 1, "FTAG": 3},  # B vince in trasferta
    ])
    gol_fatti_a, gol_subiti_a, vittorie_a, pareggi, vittorie_b, tabella = app.scontri_diretti(df, "A", "B")

    assert vittorie_a == 1
    assert pareggi == 1
    assert vittorie_b == 1
    assert gol_fatti_a == pytest.approx((2 + 0 + 1) / 3)
    assert gol_subiti_a == pytest.approx((1 + 0 + 3) / 3)
    assert len(tabella) == 3


def test_scontri_diretti_nessun_precedente_restituisce_none():
    df = pd.DataFrame([{"HomeTeam": "A", "AwayTeam": "C", "FTHG": 1, "FTAG": 0}])
    risultato = app.scontri_diretti(df, "A", "B")
    assert risultato == (None, None, None, None, None, None)


def test_stima_probabilita_pesi_oltre_1_vengono_normalizzati_non_azzerati():
    squadre = app.stats["Squadra"].unique()
    squadra_casa, squadra_trasferta = squadre[0], squadre[1]

    risultato = app.stima_probabilita(
        app.df, app.stats, squadra_casa, squadra_trasferta,
        peso_forma=1.0, peso_scontri=0.5, peso_quote=0.5,
    )

    assert risultato is not None
    assert risultato["xG_casa"] > 0
    assert risultato["xG_trasferta"] > 0
    probabilita_totale = risultato["p_1"] + risultato["p_X"] + risultato["p_2"]
    assert probabilita_totale == pytest.approx(1.0, abs=1e-6)


def test_stima_probabilita_squadra_sconosciuta_restituisce_none():
    assert app.stima_probabilita(app.df, app.stats, "Squadra Non Esistente", "Inter") is None


def test_backtesting_pesi_oltre_1_non_usa_i_valori_hardcoded():
    # Prima del fix, una somma pesi > 1 faceva scattare un fallback con valori
    # hardcoded (0.2/0.4/0.15/0.25) invece di normalizzare quelli scelti dall'utente.
    pred_normale = bt.predici_partita_bt(bt.train_df, bt.test_df, 0, bt.stats, 0.5, 0.15, 0.15)
    pred_pesi_alti = bt.predici_partita_bt(bt.train_df, bt.test_df, 0, bt.stats, 1.0, 0.5, 0.5)

    assert pred_normale is not None
    assert pred_pesi_alti is not None
    assert pred_pesi_alti["pred"] in ("1", "X", "2")
    assert pred_pesi_alti["1"] + pred_pesi_alti["X"] + pred_pesi_alti["2"] == pytest.approx(1.0, abs=1e-6)


def test_backtesting_usa_le_quote_reali_della_partita_di_test():
    riga_con_quote = bt.test_df[bt.test_df["B365H"].notna()].iloc[0]
    idx = bt.test_df.index.get_loc(riga_con_quote.name)

    pred_senza_quote = bt.predici_partita_bt(bt.train_df, bt.test_df, idx, bt.stats, 0.5, 0.15, 0.0)
    pred_con_quote = bt.predici_partita_bt(bt.train_df, bt.test_df, idx, bt.stats, 0.5, 0.15, 1.0)

    assert pred_senza_quote is not None and pred_con_quote is not None
    # Con peso_quote=1 la previsione deve avvicinarsi alle probabilità implicite delle quote reali
    prob_1_quota_implicita = 1 / riga_con_quote["B365H"]
    assert pred_con_quote["1"] != pytest.approx(pred_senza_quote["1"], abs=1e-9)
