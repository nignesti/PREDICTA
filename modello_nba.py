"""
Modello predittivo per l'NBA: rating Elo con moltiplicatore sul margine di
vittoria (metodologia FiveThirtyEight), calibrato su dati reali invece che su
costanti indovinate — stesso principio di calibra_regressione_elo() in
modello.py per il calcio.

Perche' non Dixon-Coles: quel modello stima gol attesi per convertirli in un
mercato a tre esiti (1X2, con pareggio). L'NBA non ha pareggi: il mercato e'
vittoria/sconfitta (moneyline), quindi qui basta e conviene un rating a somma
zero con probabilita' derivata direttamente dalla differenza di rating.

Fasi:
1. calcola_elo_storico: cammina le partite in ordine cronologico, aggiorna i
   rating dopo ognuna. Nessun lookahead per costruzione (il rating pre-partita
   usato per la previsione riflette solo partite precedenti).
2. calibra_probabilita: invece di fissare la pendenza a 1/400 (convenzione
   Elo scacchistica, arbitraria per l'NBA), la stima con una regressione
   logistica sulla differenza di rating pre-partita -> vittoria in casa.
"""
import numpy as np
import pandas as pd

RATING_INIZIALE = 1500.0
VANTAGGIO_CASA = 100.0  # punti Elo aggiunti alla squadra di casa nel calcolo della probabilita'
K_BASE = 20.0
PESO_REGRESSIONE_STAGIONALE = 1.0 / 3.0  # quota di regressione verso la media a inizio stagione (538)


def probabilita_vittoria_casa(elo_casa, elo_trasferta, vantaggio_casa=VANTAGGIO_CASA):
    """Probabilita' Elo standard (formula logistica in base 10, scala 400)."""
    return 1.0 / (1.0 + 10 ** (-(elo_casa + vantaggio_casa - elo_trasferta) / 400.0))


def moltiplicatore_mov(margine, elo_diff_vincitore):
    """Moltiplicatore sul margine di vittoria (formula FiveThirtyEight): le
    vittorie nette pesano di piu' sull'aggiornamento del rating, ma l'effetto
    si attenua se il favorito vince come previsto (elo_diff_vincitore alto)."""
    return ((abs(margine) + 3) ** 0.8) / (7.5 + 0.006 * elo_diff_vincitore)


def calcola_elo_storico(df, rating_iniziale=RATING_INIZIALE, vantaggio_casa=VANTAGGIO_CASA,
                         k_base=K_BASE, peso_regressione=PESO_REGRESSIONE_STAGIONALE):
    """Richiede df ordinato per Date con colonne HomeTeam, AwayTeam, PTS_Home,
    PTS_Away, Stagione. Restituisce (df_con_rating_pre_partita, rating_finali).

    A ogni cambio di Stagione per una squadra, il suo rating viene prima
    regredito verso la media (le rose cambiano in offseason: un rating di fine
    stagione riflette una squadra che in parte non esiste piu')."""
    rating = {}
    stagione_corrente = {}
    elo_casa_pre, elo_trasferta_pre, prob_casa_pre = [], [], []

    for riga in df.itertuples(index=False):
        casa, trasferta, stagione = riga.HomeTeam, riga.AwayTeam, riga.Stagione

        for squadra in (casa, trasferta):
            if squadra not in rating:
                rating[squadra] = rating_iniziale
                stagione_corrente[squadra] = stagione
            elif stagione_corrente[squadra] != stagione:
                rating[squadra] = rating[squadra] * (1 - peso_regressione) + rating_iniziale * peso_regressione
                stagione_corrente[squadra] = stagione

        r_casa, r_trasferta = rating[casa], rating[trasferta]
        p_casa = probabilita_vittoria_casa(r_casa, r_trasferta, vantaggio_casa)
        elo_casa_pre.append(r_casa)
        elo_trasferta_pre.append(r_trasferta)
        prob_casa_pre.append(p_casa)

        margine = riga.PTS_Home - riga.PTS_Away
        vince_casa = margine > 0
        elo_diff_vincitore = (r_casa + vantaggio_casa - r_trasferta) if vince_casa else (r_trasferta - vantaggio_casa - r_casa)
        mov = moltiplicatore_mov(margine, max(elo_diff_vincitore, 0.0))

        risultato_casa = 1.0 if vince_casa else 0.0
        delta = k_base * mov * (risultato_casa - p_casa)
        rating[casa] = r_casa + delta
        rating[trasferta] = r_trasferta - delta

    df_out = df.copy()
    df_out["EloHomePre"] = elo_casa_pre
    df_out["EloAwayPre"] = elo_trasferta_pre
    df_out["EloDiff"] = np.array(elo_casa_pre) + vantaggio_casa - np.array(elo_trasferta_pre)
    df_out["ProbHomeElo"] = prob_casa_pre
    return df_out, rating


def calibra_probabilita(elo_diff, vittoria_casa):
    """Regressione logistica di EloDiff -> vittoria in casa: sostituisce la
    pendenza fissa 1/400 con una stimata sui dati reali, stesso principio di
    calibra_regressione_elo() per il calcio. Va chiamata solo sul training set
    per evitare data leakage."""
    from sklearn.linear_model import LogisticRegression

    X = np.asarray(elo_diff, dtype=float).reshape(-1, 1)
    y = np.asarray(vittoria_casa).astype(int)
    modello = LogisticRegression()
    modello.fit(X, y)
    return modello


def probabilita_calibrata(modello, elo_diff):
    X = np.asarray(elo_diff, dtype=float).reshape(-1, 1)
    return modello.predict_proba(X)[:, 1]
