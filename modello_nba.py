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
STAGIONI_CALIBRAZIONE = 4  # finestra mobile per le calibrazioni Elo->probabilita' e Elo->margine:
                           # il vantaggio campo NBA e' sceso (~59% di vittorie in casa fino al
                           # 2019, ~55% dal 2020), quindi una calibrazione su tutto lo storico
                           # sovrastima la squadra di casa. Con 4 stagioni il Brier sul test
                           # 2020-2026 passa da 0.2199 (statica) a 0.2186 (valida_nba.py).
MESE_INIZIO_STAGIONE = 8  # la stagione NBA chiude a giugno e riparte a ottobre: da agosto in
                          # poi una partita appartiene gia' alla stagione successiva


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


def stagione_di(data):
    """Stagione NBA (anno in cui finisce, stessa convenzione della colonna
    Stagione dello storico: 2009-10-27 -> 2010) di una data."""
    data = pd.Timestamp(data)
    return data.year + 1 if data.month >= MESE_INIZIO_STAGIONE else data.year


def stagioni_precedenti(df, stagione, n=STAGIONI_CALIBRAZIONE):
    """Righe delle n stagioni immediatamente precedenti a 'stagione' (esclusa):
    il training set di una calibrazione walk-forward a finestra mobile."""
    return df[(df["Stagione"] < stagione) & (df["Stagione"] >= stagione - n)]


def regredisci_verso_media(rating, peso=PESO_REGRESSIONE_STAGIONALE, rating_iniziale=RATING_INIZIALE):
    """Stessa regressione d'inizio stagione di calcola_elo_storico, ma applicata
    a tutti i rating finali: serve a prevedere partite di una stagione che nello
    storico non e' ancora iniziata (calcola_elo_storico la applica solo quando
    incontra la prima partita della stagione nuova)."""
    return {squadra: r * (1 - peso) + rating_iniziale * peso for squadra, r in rating.items()}


def calibra_margine(elo_diff, margine):
    """Regressione lineare EloDiff -> margine di vittoria in casa (punti), piu'
    la deviazione standard dei residui: serve al mercato spread, che chiede una
    probabilita' di "copertura" di una linea (es. Casa -4.5), non solo chi
    vince. Trattare il margine come Normale(margine_atteso, sigma) attorno alla
    proiezione e' l'approccio standard nei modelli Elo->spread per l'NBA."""
    from sklearn.linear_model import LinearRegression

    X = np.asarray(elo_diff, dtype=float).reshape(-1, 1)
    y = np.asarray(margine, dtype=float)
    modello = LinearRegression().fit(X, y)
    residui = y - modello.predict(X)
    sigma = float(residui.std())
    return modello, sigma


def margine_atteso(modello_margine, elo_diff):
    X = np.asarray(elo_diff, dtype=float).reshape(-1, 1)
    return modello_margine.predict(X)


def probabilita_copre_spread(margine_atteso, sigma_margine, linea_spread_casa):
    """Probabilita' che la squadra di casa copra 'linea_spread_casa' (negativa
    se favorita, es. -4.5): vince la copertura se margine_reale + linea > 0.
    Con margine ~ Normale(margine_atteso, sigma_margine), la probabilita' e' il
    CDF normale valutato in quel punto, riscalato dalla deviazione standard."""
    from scipy.stats import norm

    if sigma_margine <= 0:
        return 1.0 if margine_atteso + linea_spread_casa > 0 else 0.0
    return float(norm.cdf((margine_atteso + linea_spread_casa) / sigma_margine))


def probabilita_over_totale(totale_atteso, sigma_totale, linea_totale):
    """Probabilita' che il totale punti superi 'linea_totale', stessa logica
    normale di probabilita_copre_spread ma centrata sul totale invece che sul
    margine."""
    from scipy.stats import norm

    if sigma_totale <= 0:
        return 1.0 if totale_atteso > linea_totale else 0.0
    return float(norm.cdf((totale_atteso - linea_totale) / sigma_totale))


def totale_atteso_storico(df, finestra=10, minimo_partite=5):
    """Per ogni partita, la media dei punti totali (Casa+Trasferta) delle
    ultime 'finestra' partite di ciascuna delle due squadre, mediata fra le
    due: lo stesso stimatore di pronostico_nba._media_punti_totali, ma
    vettoriale e senza lookahead (shift(1)), per misurarne l'errore storico.
    NaN finche' una squadra non ha almeno 'minimo_partite' partite."""
    totale = df["PTS_Home"] + df["PTS_Away"]
    lungo = pd.concat([
        pd.DataFrame({"riga": df.index, "Date": df["Date"], "squadra": df["HomeTeam"], "totale": totale}),
        pd.DataFrame({"riga": df.index, "Date": df["Date"], "squadra": df["AwayTeam"], "totale": totale}),
    ]).sort_values(["squadra", "Date"], kind="stable")
    lungo["media"] = lungo.groupby("squadra")["totale"].transform(
        lambda s: s.shift(1).rolling(finestra, min_periods=minimo_partite).mean())
    per_partita = lungo.groupby("riga")["media"]
    media = per_partita.mean().where(per_partita.count() == 2)
    return media.reindex(df.index)


def sigma_residui(previsto, reale):
    """Deviazione standard dello scarto reale - previsto: e' l'incertezza
    giusta da usare nella Normale di probabilita_over_totale. La deviazione
    standard dei totali grezzi su tutto lo storico la gonfierebbe (23.5 punti
    contro ~19), perche' la media lega e' salita da ~192 (2012) a ~230 (2026)."""
    scarto = np.asarray(reale, dtype=float) - np.asarray(previsto, dtype=float)
    return float(np.nanstd(scarto))


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
