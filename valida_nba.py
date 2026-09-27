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

    modello_calibrato = modello_nba.calibra_probabilita(
        train["EloDiff"], train["Winner"] == "H"
    )
    prob_elo = modello_nba.probabilita_calibrata(modello_calibrato, test["EloDiff"])

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

    print()
    protocollo.riepiloga(esiti_da_confrontare)


if __name__ == "__main__":
    main()
