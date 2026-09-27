"""
Valida il modello Elo NBA (modello_nba.py) contro due baseline, con lo stesso
protocollo usato per il calcio (protocollo.py): Brier con bootstrap appaiato
come criterio primario, accuratezza con McNemar come descrittiva.

1. "Vince sempre la squadra di casa": baseline piu' semplice che usi comunque
   un'informazione reale (in NBA la squadra di casa vince ~57-60% delle
   partite).
2. Mercato (moneyline): ora disponibile grazie a nba_2010-2026.csv, quindi si
   puo' rifare per l'NBA la stessa domanda che il calcio ha gia' risposto
   (readme.md): il modello aggiunge qualcosa al mercato o e' solo rumore?

Split walk-forward: calibra la pendenza Elo->probabilita' sulle stagioni piu'
vecchie, valuta sulle ultime STAGIONI_TEST. Il rating stesso non ha bisogno di
split (e' gia' calcolato partita per partita senza guardare il futuro), solo
la calibrazione della regressione logistica lo richiede.
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import norm

import modello_nba
import protocollo
import unisci_quote_nba

STAGIONI_TEST = 7  # come protocollo.py per il calcio: piu' stagioni, piu' potenza statistica
COPERTURA_MINIMA_MERCATO = 0.9


def carica_storico():
    """Preferisce nba_storico.csv (box score reali via nba_api, se scaricato
    con scarica_nba.py + unisci_dati_nba.py) unito alle quote; se non esiste
    ancora usa il file derivato dalle sole quote (autosufficiente per
    l'Elo: ha gia' Stagione/PTS/Winner, vedi unisci_quote_nba.carica_quote)."""
    quote = unisci_quote_nba.carica_quote(unisci_quote_nba.FILE_GREZZO)
    if os.path.exists("nba_storico.csv"):
        df = unisci_quote_nba.unisci_con_storico(quote)
    else:
        df = quote
    return df.sort_values("Date", kind="stable").reset_index(drop=True)


def probabilita_mercato(moneyline_home, moneyline_away):
    """Converte moneyline americano in probabilita' implicita casa, rimuovendo
    il vig con normalizzazione proporzionale. Con due soli esiti (niente
    pareggio in NBA) la correzione di Shin usata per il calcio non aggiunge
    nulla: il favorite-longshot bias che corregge si manifesta a tre esiti."""
    def implicita(ml):
        ml = np.asarray(ml, dtype=float)
        negativo = ml < 0
        risultato = np.empty_like(ml)
        risultato[negativo] = -ml[negativo] / (-ml[negativo] + 100)
        risultato[~negativo] = 100.0 / (ml[~negativo] + 100)
        return risultato

    p_casa, p_trasferta = implicita(moneyline_home), implicita(moneyline_away)
    return p_casa / (p_casa + p_trasferta)


def spread_casa_segnato(df):
    """Spread nel file e' sempre un numero positivo (il margine del
    favorito), con 'whos_favored' a dire chi e' favorito: lo riconverte nella
    convenzione standard (negativo se la casa e' favorita, es. -4.5)."""
    segno = np.where(df["whos_favored"] == "home", -1.0, 1.0)
    return segno * df["spread"].to_numpy(dtype=float)


def calibra_mercato_da_spread(df):
    """Regressione lineare logit(probabilita' di mercato reale) ~ spread
    firmato, calibrata su TUTTE le partite dove il moneyline reale e'
    disponibile (2010 - 16/1/2023, ~17.100 partite): serve a ricostruire una
    proxy di probabilita' di mercato per le partite successive, dove il feed
    moneyline si e' fermato ma lo spread resta coperto al 100%.

    Niente split train/test qui: le partite su cui verra' applicata (dal
    17/1/2023 in poi) non hanno mai moneyline, quindi non c'e' overlap con
    i dati di calibrazione e nessun rischio di leakage."""
    from sklearn.linear_model import LinearRegression

    con_moneyline = df["moneyline_home"].notna() & df["moneyline_away"].notna() & df["spread"].notna()
    sotto = df[con_moneyline]
    p_reale = probabilita_mercato(sotto["moneyline_home"], sotto["moneyline_away"])
    spread_segnato = spread_casa_segnato(sotto)

    logit_p = np.log(p_reale / (1 - p_reale))
    X = spread_segnato.reshape(-1, 1)
    modello = LinearRegression().fit(X, logit_p)
    print(f"Mercato-da-spread calibrato su {len(sotto):,} partite (R^2 = {modello.score(X, logit_p):.3f})")
    return modello


def probabilita_mercato_da_spread(modello, spread_segnato):
    X = np.asarray(spread_segnato, dtype=float).reshape(-1, 1)
    logit_p = modello.predict(X)
    return 1.0 / (1.0 + np.exp(-logit_p))


def stagioni_test_con_spread(df_elo, n=STAGIONI_TEST, copertura_minima=COPERTURA_MINIMA_MERCATO):
    """Le ultime n stagioni con copertura spread sufficiente (in pratica
    tutte tranne l'inizio dello storico, dove pure lo spread ha buchi
    sporadici): a differenza del moneyline, lo spread resta coperto al 100%
    anche dopo l'interruzione del 17/1/2023, quindi la proxy di mercato
    permette di valutare su un campione molto piu' ampio delle sole
    stagioni con moneyline reale."""
    copertura = df_elo.groupby("Stagione")["spread"].apply(lambda s: s.notna().mean())
    idonee = sorted(s for s in copertura.index if copertura[s] >= copertura_minima)
    scartate = sorted(set(copertura.index) - set(idonee))
    if scartate:
        print(f"Stagioni escluse per copertura spread < {copertura_minima:.0%}: {scartate}")
    return set(idonee[-n:])


def probabilita_mercato_completo(df, modello_spread):
    """Probabilita' di mercato per ogni riga: moneyline reale quando c'e',
    altrimenti la proxy ricostruita dallo spread (calibra_mercato_da_spread).
    Copre l'intero storico, non solo le partite fino al 17/1/2023."""
    con_moneyline = df["moneyline_home"].notna().to_numpy() & df["moneyline_away"].notna().to_numpy()
    prob = np.empty(len(df))
    prob[con_moneyline] = probabilita_mercato(
        df.loc[con_moneyline, "moneyline_home"], df.loc[con_moneyline, "moneyline_away"]
    )
    prob[~con_moneyline] = probabilita_mercato_da_spread(
        modello_spread, spread_casa_segnato(df[~con_moneyline])
    )
    return prob, con_moneyline


def probabilita_elo_walk_forward(df_elo, stagioni_test):
    """Probabilita' Elo calibrata stagione per stagione sulle
    STAGIONI_CALIBRAZIONE precedenti (finestra mobile), come fa l'app in
    produzione: una calibrazione unica sulle stagioni pre-2020 sovrastima il
    vantaggio campo, sceso da ~59% a ~55% di vittorie in casa."""
    prob = pd.Series(np.nan, index=df_elo.index)
    for stagione in sorted(stagioni_test):
        train = modello_nba.stagioni_precedenti(df_elo, stagione)
        riga = df_elo["Stagione"] == stagione
        modello = modello_nba.calibra_probabilita(train["EloDiff"], train["Winner"] == "H")
        prob[riga] = modello_nba.probabilita_calibrata(modello, df_elo.loc[riga, "EloDiff"])
    return prob


def confronto_spread(df_elo, stagioni_test):
    """Il modello Elo->margine (calibra_margine) contro la linea spread del
    mercato, che per costruzione da' ~50% a entrambi i lati. Esclude i push
    (margine esattamente uguale alla linea): nessun vincitore."""
    prob, esiti = [], []
    for stagione in sorted(stagioni_test):
        train = modello_nba.stagioni_precedenti(df_elo, stagione).dropna(subset=["spread"])
        test = df_elo[df_elo["Stagione"] == stagione].dropna(subset=["spread"])
        modello_margine, sigma = modello_nba.calibra_margine(
            train["EloDiff"], train["PTS_Home"] - train["PTS_Away"])
        atteso = modello_nba.margine_atteso(modello_margine, test["EloDiff"])
        copertura = (test["PTS_Home"] - test["PTS_Away"]).to_numpy() + spread_casa_segnato(test)
        valide = copertura != 0
        prob.extend(norm.cdf((atteso + spread_casa_segnato(test)) / sigma)[valide])
        esiti.extend(copertura[valide] > 0)
    return protocollo.confronta_binario(
        "Modello spread (Elo)", prob, "Linea spread (50%)", np.full(len(esiti), 0.5), esiti)


def confronto_totale(df_elo, stagioni_test):
    """Lo stimatore del totale usato dall'app (media ultime 10 partite delle due
    squadre) contro la linea Over/Under del mercato (~50% a entrambi i lati)."""
    totale_atteso = modello_nba.totale_atteso_storico(df_elo)
    totale = df_elo["PTS_Home"] + df_elo["PTS_Away"]
    prob, esiti = [], []
    for stagione in sorted(stagioni_test):
        train = modello_nba.stagioni_precedenti(df_elo, stagione).index
        sigma = modello_nba.sigma_residui(totale_atteso[train], totale[train])
        test = df_elo[(df_elo["Stagione"] == stagione) & df_elo["total"].notna() & totale_atteso.notna()]
        valide = (totale[test.index] != test["total"]).to_numpy()
        p = norm.cdf((totale_atteso[test.index] - test["total"]) / sigma)
        prob.extend(np.asarray(p)[valide])
        esiti.extend((totale[test.index] > test["total"]).to_numpy()[valide])
    return protocollo.confronta_binario(
        "Modello totale (media 10)", prob, "Linea totale (50%)", np.full(len(esiti), 0.5), esiti)


def main():
    df = carica_storico()
    df_elo, rating_finali = modello_nba.calcola_elo_storico(df)
    modello_spread = calibra_mercato_da_spread(df_elo)

    stagioni_ordinate = sorted(df_elo["Stagione"].unique())
    stagioni_test = stagioni_test_con_spread(df_elo)
    train = df_elo[~df_elo["Stagione"].isin(stagioni_test)]
    test = df_elo[df_elo["Stagione"].isin(stagioni_test)]
    print(f"Train: {len(train):,} partite ({len(stagioni_ordinate) - len(stagioni_test)} stagioni)")
    print(f"Test:  {len(test):,} partite ({len(stagioni_test)} stagioni: {sorted(stagioni_test)})")
    print(f"Calibrazione Elo walk-forward sulle {modello_nba.STAGIONI_CALIBRAZIONE} stagioni precedenti")

    prob_elo = probabilita_elo_walk_forward(df_elo, stagioni_test)[test.index].to_numpy()

    tasso_casa_train = (train["Winner"] == "H").mean()
    prob_baseline = np.full(len(test), tasso_casa_train)

    esiti = (test["Winner"] == "H").to_numpy()
    esiti_da_confrontare = [
        protocollo.confronta_binario(
            "Elo calibrato", prob_elo, "Solo vantaggio campo", prob_baseline, esiti
        )
    ]

    prob_mercato, con_moneyline_reale = probabilita_mercato_completo(test, modello_spread)
    esiti_da_confrontare.append(
        protocollo.confronta_binario(
            "Elo calibrato", prob_elo, "Mercato (moneyline + proxy da spread)", prob_mercato, esiti
        )
    )
    print(f"({con_moneyline_reale.sum():,}/{len(test):,} partite di test con moneyline reale, "
          f"il resto usa la proxy da spread)")

    esiti_da_confrontare.append(confronto_spread(df_elo, stagioni_test))
    esiti_da_confrontare.append(confronto_totale(df_elo, stagioni_test))

    print()
    protocollo.riepiloga(esiti_da_confrontare)


if __name__ == "__main__":
    main()
