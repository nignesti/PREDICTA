import numpy as np
import pandas as pd
import pytest

import modello_nba as mn


def _partite(righe):
    df = pd.DataFrame(righe, columns=["Date", "HomeTeam", "AwayTeam", "PTS_Home", "PTS_Away", "Stagione"])
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@pytest.mark.parametrize("data, attesa", [
    ("2009-10-27", 2010), ("2026-06-13", 2026), ("2026-09-27", 2027), ("2027-01-15", 2027),
])
def test_stagione_di(data, attesa):
    assert mn.stagione_di(data) == attesa


def test_stagioni_precedenti_esclude_la_stagione_e_rispetta_la_finestra():
    df = pd.DataFrame({"Stagione": range(2015, 2027)})
    assert sorted(mn.stagioni_precedenti(df, 2026, n=4)["Stagione"]) == [2022, 2023, 2024, 2025]


def test_regredisci_verso_media_coincide_con_la_regressione_di_calcola_elo_storico():
    df = _partite([
        ("2025-01-01", "A", "B", 120, 100, 2025),
        ("2025-01-03", "B", "A", 110, 100, 2025),
        ("2025-11-01", "A", "B", 100, 100 + 1, 2026),
    ])
    _, rating_fine = mn.calcola_elo_storico(df.iloc[:2])
    df_elo, _ = mn.calcola_elo_storico(df)
    regrediti = mn.regredisci_verso_media(rating_fine)
    assert df_elo["EloHomePre"].iloc[2] == pytest.approx(regrediti["A"])
    assert df_elo["EloAwayPre"].iloc[2] == pytest.approx(regrediti["B"])
    assert sum(regrediti.values()) == pytest.approx(2 * mn.RATING_INIZIALE)


def test_totale_atteso_storico_non_guarda_la_partita_stessa():
    righe = [(f"2025-01-{g:02d}", "A", "B", 100 + g, 100, 2025) for g in range(1, 8)]
    df = _partite(righe)
    atteso = mn.totale_atteso_storico(df, finestra=3, minimo_partite=2)
    assert atteso.iloc[:2].isna().all()
    # partita 4 (indice 3): media dei totali delle partite 1-3 = 201, 202, 203
    assert atteso.iloc[3] == pytest.approx(202.0)


def test_sigma_residui_ignora_il_livello_medio():
    previsto = np.array([200.0, 230.0, 200.0, 230.0])
    reale = previsto + np.array([-5.0, 5.0, -5.0, 5.0])
    assert mn.sigma_residui(previsto, reale) == pytest.approx(5.0)
