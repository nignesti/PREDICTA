import numpy as np
import pandas as pd
import pytest

import app
import backtesting as bt
import modello


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


def test_stima_probabilita_peso_quote_alto_non_schiaccia_xg():
    # Regressione: quando peso_quote e' alto (es. 0.85), storico+forma+scontri
    # restavano pesati per la loro quota originale (es. 0.15) invece di essere
    # rinormalizzati a sommare 1 tra loro, producendo xG assurdamente bassi
    # (es. 0.24 invece di ~1.6) perche' "quote" non entra nel calcolo dell'xG.
    squadre = app.stats["Squadra"].unique()
    squadra_casa, squadra_trasferta = squadre[0], squadre[1]

    risultato = app.stima_probabilita(
        app.df, app.stats, squadra_casa, squadra_trasferta,
        peso_forma=0.0, peso_scontri=0.15, peso_quote=0.85,
    )

    assert risultato is not None
    # Un xG sotto 0.5 per una squadra di Serie A e' irrealistico: e' la firma del bug.
    assert risultato["xG_casa"] > 0.5
    assert risultato["xG_trasferta"] > 0.5


def test_backtesting_pesi_oltre_1_non_usa_i_valori_hardcoded():
    # Prima del fix, una somma pesi > 1 faceva scattare un fallback con valori
    # hardcoded (0.2/0.4/0.15/0.25) invece di normalizzare quelli scelti dall'utente.
    pred_normale = bt.predici_partita_bt(bt.train_df, bt.test_df, 0, 0.5, 0.15, 0.15, 730, -0.10)
    pred_pesi_alti = bt.predici_partita_bt(bt.train_df, bt.test_df, 0, 1.0, 0.5, 0.5, 730, -0.10)

    assert pred_normale is not None
    assert pred_pesi_alti is not None
    assert pred_pesi_alti["pred"] in ("1", "X", "2")
    assert pred_pesi_alti["1"] + pred_pesi_alti["X"] + pred_pesi_alti["2"] == pytest.approx(1.0, abs=1e-6)


def test_backtesting_usa_le_quote_reali_della_partita_di_test():
    colonna_quota = "OddsAvgH" if "OddsAvgH" in bt.test_df.columns else "B365H"
    riga_con_quote = bt.test_df[bt.test_df[colonna_quota].notna()].iloc[0]
    idx = bt.test_df.index.get_loc(riga_con_quote.name)

    pred_senza_quote = bt.predici_partita_bt(bt.train_df, bt.test_df, idx, 0.5, 0.15, 0.0, 730, -0.10)
    pred_con_quote = bt.predici_partita_bt(bt.train_df, bt.test_df, idx, 0.5, 0.15, 1.0, 730, -0.10)

    assert pred_senza_quote is not None and pred_con_quote is not None
    # Con peso_quote=1 la previsione deve spostarsi rispetto a quella senza quote
    assert pred_con_quote["1"] != pytest.approx(pred_senza_quote["1"], abs=1e-9)


def test_backtesting_peso_quote_alto_non_schiaccia_xg():
    # Stessa regressione di test_stima_probabilita_peso_quote_alto_non_schiaccia_xg
    # ma nel motore di backtesting: le due implementazioni erano gia' divergenti in
    # passato (bug sugli scontri diretti), quindi si testano entrambe separatamente.
    pred = bt.predici_partita_bt(bt.train_df, bt.test_df, 0, 0.0, 0.15, 0.85, 730, -0.10)
    assert pred is not None
    # Se il bug fosse presente, p_X (pareggio) sarebbe innaturalmente alto perche'
    # l'xG di base collasserebbe verso 0 nella componente "senza quote".
    assert pred["X"] < 0.45


# ------------------------------------------------------------
# modello.py: Dixon-Coles, decadimento temporale, RPS
# ------------------------------------------------------------

def test_tau_dixon_coles_valori_neutri_fuori_dai_bassi_punteggi():
    assert modello.tau_dixon_coles(2, 2, 1.5, 1.2, -0.1) == 1.0
    assert modello.tau_dixon_coles(3, 0, 1.5, 1.2, -0.1) == 1.0


def test_tau_dixon_coles_rho_negativo_aumenta_probabilita_pareggio():
    matrice_neutra = modello.distribuzione_punteggi(1.5, 1.1, rho=0.0)
    matrice_corretta = modello.distribuzione_punteggi(1.5, 1.1, rho=-0.1)

    esiti_neutri = modello.esiti_da_matrice(matrice_neutra)
    esiti_corretti = modello.esiti_da_matrice(matrice_corretta)

    # Un rho negativo tipico di Dixon-Coles alza la probabilita' di pareggio
    # (corregge la sottostima dei pareggi di un Poisson indipendente puro).
    assert esiti_corretti["p_X"] > esiti_neutri["p_X"]


def test_probabilita_shin_senza_overround_coincide_col_proporzionale():
    # Quote "eque" (nessun margine, somma di 1/quota == 1): Shin e proporzionale
    # devono coincidere, non c'e' margine da ripartire in modo diverso.
    quote = [2.0, 4.0, 4.0]  # 1/2 + 1/4 + 1/4 = 1.0
    proporzionale = [1 / q / sum(1 / q for q in quote) for q in quote]
    shin = modello.probabilita_shin(quote)
    assert shin == pytest.approx(proporzionale, abs=1e-6)


def test_probabilita_shin_somma_a_uno_con_overround():
    quote = [1.8, 3.6, 4.5]  # overround tipico da quote reali di calcio
    shin = modello.probabilita_shin(quote)
    assert sum(shin) == pytest.approx(1.0, abs=1e-9)
    assert all(p > 0 for p in shin)


def test_probabilita_shin_mantiene_ordine_di_favorito_e_differisce_dal_proporzionale():
    quote = [1.8, 3.6, 4.5]
    pi = [1 / q for q in quote]
    proporzionale = [p / sum(pi) for p in pi]
    shin = modello.probabilita_shin(quote)

    # L'ordine (favorito > pareggio > sfavorito, dato l'ordine delle quote) va
    # preservato: Shin corregge QUANTO margine togliere da ciascun esito, non
    # l'ordinamento delle probabilita'.
    assert shin[0] > shin[1] > shin[2]
    # Con overround, la correzione di Shin non e' una normalizzazione uniforme:
    # deve produrre probabilita' diverse dal metodo proporzionale.
    assert shin != pytest.approx(proporzionale, abs=1e-6)


def test_distribuzione_punteggi_somma_a_uno():
    for xg_casa, xg_trasf, rho in [(0.8, 0.8, -0.1), (2.5, 0.3, -0.2), (1.2, 1.2, 0.0)]:
        matrice = modello.distribuzione_punteggi(xg_casa, xg_trasf, rho=rho)
        assert matrice.sum() == pytest.approx(1.0, abs=1e-9)
        assert (matrice >= 0).all()


def test_rps_previsione_perfetta_e_peggiore():
    assert modello.rps({"1": 1.0, "X": 0.0, "2": 0.0}, "1") == pytest.approx(0.0)
    assert modello.rps({"1": 1.0, "X": 0.0, "2": 0.0}, "2") == pytest.approx(1.0)


def test_peso_esponenziale_dimezza_dopo_emivita():
    pesi = modello.peso_esponenziale(np.array([0, 365, 730]), half_life_giorni=365)
    assert pesi[0] == pytest.approx(1.0)
    assert pesi[1] == pytest.approx(0.5)
    assert pesi[2] == pytest.approx(0.25)


def test_peso_esponenziale_nessun_decadimento_se_half_life_none():
    pesi = modello.peso_esponenziale(np.array([0, 10000]), half_life_giorni=None)
    assert (pesi == 1.0).all()


def test_stats_pesate_squadre_emivita_corta_ignora_partite_vecchie():
    df = pd.DataFrame([
        {"HomeTeam": "A", "AwayTeam": "B", "FTHG": 0, "FTAG": 0, "Date": pd.Timestamp("2000-01-01")},
        {"HomeTeam": "A", "AwayTeam": "B", "FTHG": 5, "FTAG": 0, "Date": pd.Timestamp("2024-01-01")},
    ])
    stats = modello.stats_pesate_squadre(df, data_riferimento=pd.Timestamp("2024-01-01"), half_life_giorni=1)
    riga_a = stats[stats["Squadra"] == "A"].iloc[0]
    # Con un'emivita di 1 giorno, la partita del 2000 pesa ~0: la media deve
    # essere quella della partita recente (5 gol), non una via di mezzo.
    assert riga_a["gol_fatti_casa_storico"] == pytest.approx(5.0, abs=0.01)


# ------------------------------------------------------------
# Integrita' del dataset (regressione: stagione 2009/10 duplicata, 2010/11 mancante)
# ------------------------------------------------------------

def test_dataset_nessuna_partita_duplicata():
    duplicati = app.df.duplicated(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]).sum()
    assert duplicati == 0


def test_dataset_ordinato_cronologicamente():
    assert app.df["Date"].is_monotonic_increasing


def test_dataset_nessuna_stagione_mancante_tra_min_e_max():
    stagioni = sorted(int(s) for s in app.df["Stagione"].unique())
    attese = list(range(stagioni[0], stagioni[-1] + 1))
    assert stagioni == attese


# ------------------------------------------------------------
# Elo (clubelo.com): lookup point-in-time e regressione di calibrazione
# ------------------------------------------------------------

@pytest.fixture
def storico_elo():
    # Due periodi contigui per "A": rating basso fino al 10/01, poi alto dall'11/01
    # (come i dati reali di ClubElo, dove il rating si aggiorna il giorno dopo la partita).
    righe = [
        {"Squadra": "A", "Elo": 1500.0, "From": pd.Timestamp("2024-01-01"), "To": pd.Timestamp("2024-01-10")},
        {"Squadra": "A", "Elo": 1550.0, "From": pd.Timestamp("2024-01-11"), "To": pd.Timestamp("2024-01-20")},
        {"Squadra": "B", "Elo": 1400.0, "From": pd.Timestamp("2024-01-01"), "To": pd.Timestamp("2024-01-20")},
    ]
    return modello.prepara_elo(pd.DataFrame(righe))


def test_elo_asof_batch_usa_il_rating_pre_partita(storico_elo):
    # Una partita giocata il 10/01 deve prendere il rating del periodo che finisce
    # quel giorno (1500), non quello nuovo che parte l'11/01 (1550): il rating
    # ClubElo per un giorno riflette sempre la situazione PRIMA della partita.
    risultato = modello.elo_asof_batch(storico_elo, ["A", "A"], [pd.Timestamp("2024-01-10"), pd.Timestamp("2024-01-11")])
    assert risultato[0] == pytest.approx(1500.0)
    assert risultato[1] == pytest.approx(1550.0)


def test_elo_asof_batch_squadra_sconosciuta_da_nan(storico_elo):
    risultato = modello.elo_asof_batch(storico_elo, ["SquadraInesistente"], [pd.Timestamp("2024-01-10")])
    assert np.isnan(risultato[0])


def test_elo_asof_batch_data_fuori_da_ogni_periodo_da_nan(storico_elo):
    # Richiesta precedente all'inizio dello storico Elo della squadra: nessun
    # periodo la copre, quindi deve restituire NaN e non un valore-spazzatura.
    risultato = modello.elo_asof_batch(storico_elo, ["A"], [pd.Timestamp("2023-01-01")])
    assert np.isnan(risultato[0])


def test_calibra_regressione_elo_squadra_piu_forte_segna_di_piu():
    # Dataset sintetico dove "A" (Elo alto) segna sistematicamente piu' di "B" (Elo
    # basso): la regressione calibrata deve imparare xG piu' alto per chi ha Elo
    # maggiore, non un valore piatto uguale per tutti.
    partite = []
    for i in range(30):
        partite.append({"HomeTeam": "A", "AwayTeam": "B", "FTHG": 3, "FTAG": 0,
                        "Date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i)})
        partite.append({"HomeTeam": "B", "AwayTeam": "A", "FTHG": 0, "FTAG": 3,
                        "Date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i)})
    df = pd.DataFrame(partite)

    elo = modello.prepara_elo(pd.DataFrame([
        {"Squadra": "A", "Elo": 1700.0, "From": pd.Timestamp("2019-01-01"), "To": pd.Timestamp("2021-01-01")},
        {"Squadra": "B", "Elo": 1300.0, "From": pd.Timestamp("2019-01-01"), "To": pd.Timestamp("2021-01-01")},
    ]))

    modello_casa, modello_trasferta = modello.calibra_regressione_elo(df, elo)
    xg_a_casa, xg_b_trasferta = modello.xg_da_elo_calibrato(1700.0, 1300.0, modello_casa, modello_trasferta)
    xg_b_casa, xg_a_trasferta = modello.xg_da_elo_calibrato(1300.0, 1700.0, modello_casa, modello_trasferta)

    assert xg_a_casa > xg_b_casa  # "A" (Elo alto) in casa deve avere xG maggiore di "B" (Elo basso) in casa
    assert xg_b_trasferta < xg_a_trasferta  # "B" in trasferta contro Elo alto deve avere xG minore


def test_xg_da_elo_calibrato_rating_mancante_usa_elo_diff_zero():
    # Training con copertura Elo parziale (caso realistico: una squadra senza
    # storico Elo, es. neopromossa non ancora tracciata) mescolata a partite con
    # Elo valido, cosi' la regressione ha comunque campioni su cui calibrarsi.
    df = pd.DataFrame([
        {"HomeTeam": "A", "AwayTeam": "C", "FTHG": 2, "FTAG": 1, "Date": pd.Timestamp("2020-01-01")},
        {"HomeTeam": "C", "AwayTeam": "A", "FTHG": 1, "FTAG": 2, "Date": pd.Timestamp("2020-01-08")},
        {"HomeTeam": "A", "AwayTeam": "B", "FTHG": 1, "FTAG": 1, "Date": pd.Timestamp("2020-01-15")},
        {"HomeTeam": "B", "AwayTeam": "A", "FTHG": 1, "FTAG": 1, "Date": pd.Timestamp("2020-01-22")},
    ])
    elo = modello.prepara_elo(pd.DataFrame([
        {"Squadra": "A", "Elo": 1500.0, "From": pd.Timestamp("2019-01-01"), "To": pd.Timestamp("2021-01-01")},
        {"Squadra": "C", "Elo": 1450.0, "From": pd.Timestamp("2019-01-01"), "To": pd.Timestamp("2021-01-01")},
        # "B" non ha storico Elo: le partite A-B/B-A avranno elo_diff NaN e verranno escluse dal fit.
    ]))
    modello_casa, modello_trasferta = modello.calibra_regressione_elo(df, elo)
    # elo_trasferta NaN (squadra "B" non nello storico Elo): non deve esplodere,
    # deve ricadere su elo_diff=0.
    xg_casa, xg_trasferta = modello.xg_da_elo_calibrato(1500.0, np.nan, modello_casa, modello_trasferta)
    assert np.isfinite(xg_casa) and np.isfinite(xg_trasferta)
