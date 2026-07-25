"""
Test del modulo multi-lega (Fase 0, priorita' 1).

Il rischio principale nell'allargare il test set a 5 campionati e' il data
leakage temporale: e' l'errore che la ROADMAP elenca per primo fra quelli
"facili da reintrodurre", e su un modulo nuovo che itera su indici di
DataFrame e' particolarmente insidioso. Il test centrale qui verifica
direttamente che le componenti di una partita non cambino se si cancella
tutto cio' che viene dopo di essa.
"""
import numpy as np
import pandas as pd
import pytest

import multilega as ml


def _lega_sintetica(n_partite=600, n_squadre=10, seed=0):
    """Campionato finto con date crescenti e squadre che si affrontano a
    rotazione, abbastanza lungo da superare la soglia di storico minimo."""
    rng = np.random.default_rng(seed)
    squadre = [f"S{i}" for i in range(n_squadre)]
    righe = []
    data = pd.Timestamp("2015-08-15")
    for i in range(n_partite):
        casa, trasferta = squadre[i % n_squadre], squadre[(i + 1 + i // n_squadre) % n_squadre]
        if casa == trasferta:
            trasferta = squadre[(i + 2) % n_squadre]
        data = data + pd.Timedelta(days=3)
        righe.append({
            "HomeTeam": casa, "AwayTeam": trasferta,
            "FTHG": int(rng.integers(0, 4)), "FTAG": int(rng.integers(0, 4)),
            "Date": data,
            "OddsAvgH": 2.10, "OddsAvgD": 3.40, "OddsAvgA": 3.60,
            "OddsAvgCH": 2.05, "OddsAvgCD": 3.45, "OddsAvgCA": 3.70,
        })
    df = pd.DataFrame(righe)
    df["Stagione"] = ml.stagione_da_data(df["Date"])
    return df


def test_stagione_da_data_usa_la_convenzione_agosto_maggio():
    date = pd.Series(pd.to_datetime([
        "2023-08-20",  # inizio stagione 2023/24 -> 2023
        "2023-12-30",  # ancora 2023/24        -> 2023
        "2024-05-25",  # fine stagione 2023/24 -> 2023
        "2024-08-18",  # inizio 2024/25        -> 2024
    ]))
    assert list(ml.stagione_da_data(date)) == [2023, 2023, 2023, 2024]


def test_componenti_non_usano_dati_futuri():
    """Il test decisivo contro il data leakage: le componenti di una partita
    devono essere identiche che il DataFrame contenga o meno le partite
    successive. Se una qualsiasi delle medie guardasse avanti, i due calcoli
    divergerebbero — soprattutto qui, dove le partite future hanno punteggi
    volutamente assurdi."""
    df = _lega_sintetica()
    stagione_bersaglio = int(df["Stagione"].iloc[400])
    indice_bersaglio = 400

    # Versione "futuro alterato": stessi dati fino alla partita bersaglio,
    # poi risultati completamente diversi.
    df_alterato = df.copy()
    futuro = df_alterato.index > indice_bersaglio
    df_alterato.loc[futuro, "FTHG"] = 9
    df_alterato.loc[futuro, "FTAG"] = 0

    comp_a = ml.componenti_lega(df, [stagione_bersaglio], "TEST")
    comp_b = ml.componenti_lega(df_alterato, [stagione_bersaglio], "TEST")

    assert len(comp_a) == len(comp_b) > 0
    campi_xg = ["xG_casa_storico", "xG_trasf_storico", "xG_casa_forma",
                "xG_trasf_forma", "xG_casa_scontri", "xG_trasf_scontri"]
    # Solo le partite fino al bersaglio sono confrontabili: dopo, il "passato"
    # delle due versioni diverge legittimamente.
    for a, b in zip(comp_a, comp_b):
        if a["data"] > df["Date"].iloc[indice_bersaglio]:
            break
        for campo in campi_xg:
            assert a[campo] == pytest.approx(b[campo]), f"{campo} contaminato da dati futuri"


def test_componenti_solo_delle_stagioni_richieste():
    df = _lega_sintetica()
    stagioni = sorted(df["Stagione"].unique())
    bersaglio = int(stagioni[-1])
    comp = ml.componenti_lega(df, [bersaglio], "TEST")
    assert comp
    assert {c["stagione"] for c in comp} == {str(bersaglio)}


def test_probabilita_modello_sommano_a_uno():
    df = _lega_sintetica()
    bersaglio = int(sorted(df["Stagione"].unique())[-1])
    comp = ml.componenti_lega(df, [bersaglio], "TEST")
    for p in ml.probabilita_modello(comp):
        assert sum(p) == pytest.approx(1.0, abs=1e-9)
        assert all(x >= 0 for x in p)


def test_peso_quote_uno_restituisce_le_probabilita_di_mercato():
    # Con peso_forma=0 e peso_quote=1 il blend deve coincidere esattamente con
    # il mercato: e' il baseline usato da valida_multilega.py, quindi se
    # divergesse il confronto "modello vs mercato" sarebbe falsato.
    df = _lega_sintetica()
    bersaglio = int(sorted(df["Stagione"].unique())[-1])
    comp = ml.componenti_lega(df, [bersaglio], "TEST")

    solo_quote = ml.probabilita_modello(comp, peso_forma=0.0, peso_scontri=0.0, peso_quote=1.0)
    mercato = ml.probabilita_mercato(comp)

    for a, b in zip(solo_quote, mercato):
        assert a == pytest.approx(b, abs=1e-9)


def test_quote_di_chiusura_preferite_all_apertura():
    riga = pd.Series({"OddsAvgH": 2.10, "OddsAvgD": 3.40, "OddsAvgA": 3.60,
                      "OddsAvgCH": 1.50, "OddsAvgCD": 4.00, "OddsAvgCA": 7.00})
    da_chiusura = ml._probabilita_quote(riga)
    from modello import probabilita_shin
    assert da_chiusura == pytest.approx(probabilita_shin([1.50, 4.00, 7.00]))


def test_fallback_su_apertura_se_manca_la_chiusura():
    riga = pd.Series({"OddsAvgH": 2.10, "OddsAvgD": 3.40, "OddsAvgA": 3.60,
                      "OddsAvgCH": np.nan, "OddsAvgCD": np.nan, "OddsAvgCA": np.nan})
    from modello import probabilita_shin
    assert ml._probabilita_quote(riga) == pytest.approx(probabilita_shin([2.10, 3.40, 3.60]))


def test_nessuna_quota_disponibile_restituisce_none():
    riga = pd.Series({"OddsAvgH": np.nan, "OddsAvgD": np.nan, "OddsAvgA": np.nan,
                      "OddsAvgCH": np.nan, "OddsAvgCD": np.nan, "OddsAvgCA": np.nan})
    assert ml._probabilita_quote(riga) is None


def test_serie_a_caricata_con_le_stesse_colonne_delle_altre_leghe():
    # Le 5 leghe devono essere trattate in modo identico: se la Serie A
    # arrivasse con colonne diverse, il confronto fra campionati sarebbe
    # apples-to-oranges.
    serie_a = ml.carica_lega("I1")
    attese = {"HomeTeam", "AwayTeam", "FTHG", "FTAG", "Date", "Stagione",
              "OddsAvgH", "OddsAvgD", "OddsAvgA", "OddsAvgCH", "OddsAvgCD", "OddsAvgCA"}
    assert attese <= set(serie_a.columns)
    assert serie_a["Date"].is_monotonic_increasing
