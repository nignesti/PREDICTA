"""
Funzioni di modellazione condivise tra app.py e pages/backtesting.py.

Contiene la parte di Dixon & Coles (1997) che avevamo in sospeso:
- correzione tau per i punteggi bassi (il modello Poisson puro sottostima i pareggi)
- decadimento temporale esponenziale delle statistiche storiche (le partite recenti
  contano di più di quelle di 30 anni fa)
- calcolo esatto della distribuzione dei punteggi (sostituisce la simulazione Monte
  Carlo: stessa idea ma senza rumore campionario e permette di applicare tau)
- Ranked Probability Score, la metrica standard in letteratura per valutare
  previsioni 1X2 probabilistiche (l'accuratezza da sola non basta: premia
  previsioni "decise" anche se mal calibrate)
"""

import numpy as np
from scipy.stats import poisson

MAX_GOL = 10


def peso_esponenziale(giorni_trascorsi, half_life_giorni):
    """Peso di una partita in funzione di quanto tempo fa è stata giocata.

    half_life_giorni: dopo quanti giorni il peso di una partita si dimezza.
    Se half_life_giorni è None o <= 0, restituisce peso 1 per tutti (nessun decadimento).
    """
    if half_life_giorni is None or half_life_giorni <= 0:
        return np.ones_like(giorni_trascorsi, dtype=float)
    return 0.5 ** (np.clip(giorni_trascorsi, a_min=0, a_max=None) / half_life_giorni)


def media_pesata_per_squadra(df, colonna_gruppo, colonna_valore, colonna_data, data_riferimento, half_life_giorni):
    """Equivalente di df.groupby(colonna_gruppo)[colonna_valore].mean(), ma con
    decadimento temporale esponenziale rispetto a data_riferimento invece di una
    media semplice su tutta la storia."""
    giorni = (data_riferimento - df[colonna_data]).dt.days.to_numpy()
    pesi = peso_esponenziale(giorni, half_life_giorni)
    tmp = df[[colonna_gruppo]].copy()
    tmp["_peso"] = pesi
    tmp["_valore_pesato"] = df[colonna_valore].to_numpy() * pesi
    agg = tmp.groupby(colonna_gruppo).agg(_somma_pesata=("_valore_pesato", "sum"), _somma_pesi=("_peso", "sum"))
    return (agg["_somma_pesata"] / agg["_somma_pesi"]).rename(colonna_valore)


def stats_pesate_squadre(df, data_riferimento, half_life_giorni):
    """Tabella attacco/difesa casa e trasferta per ogni squadra, pesata con
    decadimento temporale esponenziale rispetto a data_riferimento (2 groupby
    invece di una chiamata per colonna, per tenere il backtesting veloce anche
    dovendola ricalcolare a ogni partita)."""
    giorni = (data_riferimento - df["Date"]).dt.days.to_numpy()
    pesi = peso_esponenziale(giorni, half_life_giorni)

    tmp = df[["HomeTeam", "AwayTeam"]].copy()
    tmp["_peso"] = pesi
    tmp["_fatti_casa"] = df["FTHG"].to_numpy() * pesi
    tmp["_subiti_casa"] = df["FTAG"].to_numpy() * pesi
    tmp["_fatti_trasferta"] = df["FTAG"].to_numpy() * pesi
    tmp["_subiti_trasferta"] = df["FTHG"].to_numpy() * pesi

    casa = tmp.groupby("HomeTeam").agg(
        _peso_casa=("_peso", "sum"), _fatti_casa=("_fatti_casa", "sum"), _subiti_casa=("_subiti_casa", "sum"))
    trasferta = tmp.groupby("AwayTeam").agg(
        _peso_trasferta=("_peso", "sum"), _fatti_trasferta=("_fatti_trasferta", "sum"), _subiti_trasferta=("_subiti_trasferta", "sum"))

    stats = casa.join(trasferta, how="outer")
    stats["gol_fatti_casa_storico"] = stats["_fatti_casa"] / stats["_peso_casa"]
    stats["gol_subiti_casa_storico"] = stats["_subiti_casa"] / stats["_peso_casa"]
    stats["gol_fatti_trasferta_storico"] = stats["_fatti_trasferta"] / stats["_peso_trasferta"]
    stats["gol_subiti_trasferta_storico"] = stats["_subiti_trasferta"] / stats["_peso_trasferta"]

    stats = stats[["gol_fatti_casa_storico", "gol_subiti_casa_storico",
                   "gol_fatti_trasferta_storico", "gol_subiti_trasferta_storico"]].fillna(0)
    stats.index.name = "Squadra"
    return stats.reset_index()


def tau_dixon_coles(x, y, lam, mu, rho):
    """Fattore di correzione di Dixon & Coles (1997) per i 4 risultati a basso
    punteggio, dove il Poisson indipendente sottostima sistematicamente i pareggi."""
    if x == 0 and y == 0:
        return 1 - lam * mu * rho
    if x == 0 and y == 1:
        return 1 + lam * rho
    if x == 1 and y == 0:
        return 1 + mu * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


def distribuzione_punteggi(xg_casa, xg_trasferta, rho, max_gol=MAX_GOL):
    """Matrice P(gol_casa=i, gol_trasferta=j) esatta (Poisson indipendenti +
    correzione tau di Dixon-Coles), al posto della simulazione Monte Carlo:
    stesso modello concettuale ma deterministico e senza rumore campionario."""
    gol = np.arange(0, max_gol + 1)
    p_casa = poisson.pmf(gol, xg_casa)
    p_trasferta = poisson.pmf(gol, xg_trasferta)
    matrice = np.outer(p_casa, p_trasferta)

    matrice[0, 0] *= tau_dixon_coles(0, 0, xg_casa, xg_trasferta, rho)
    matrice[0, 1] *= tau_dixon_coles(0, 1, xg_casa, xg_trasferta, rho)
    matrice[1, 0] *= tau_dixon_coles(1, 0, xg_casa, xg_trasferta, rho)
    matrice[1, 1] *= tau_dixon_coles(1, 1, xg_casa, xg_trasferta, rho)

    matrice = matrice / matrice.sum()
    return matrice


def esiti_da_matrice(matrice, n_top=5):
    """Estrae probabilità 1X2, over/under e i risultati esatti più probabili
    dalla matrice congiunta dei punteggi."""
    n = matrice.shape[0]
    gol_casa, gol_trasferta = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")

    p_1 = matrice[gol_casa > gol_trasferta].sum()
    p_X = matrice[gol_casa == gol_trasferta].sum()
    p_2 = matrice[gol_casa < gol_trasferta].sum()
    over_25 = matrice[(gol_casa + gol_trasferta) > 2.5].sum()
    over_15 = matrice[(gol_casa + gol_trasferta) > 1.5].sum()

    indici_ordinati = np.dstack(np.unravel_index(np.argsort(-matrice, axis=None), matrice.shape))[0]
    top_risultati = [(f"{i}-{j}", matrice[i, j]) for i, j in indici_ordinati[:n_top]]

    return {
        "p_1": p_1, "p_X": p_X, "p_2": p_2,
        "over_25": over_25, "under_25": 1 - over_25, "over_15": over_15,
        "top_risultati": top_risultati,
    }


def rps(probabilita, esito_reale, ordine=("1", "X", "2")):
    """Ranked Probability Score per una singola previsione 1X2 (Constantinou &
    Fenton). 0 = previsione perfetta, valori più bassi sono migliori. A differenza
    dell'accuratezza pura, penalizza le probabilità mal calibrate e non solo la
    classe con probabilità massima."""
    p = [probabilita[o] for o in ordine]
    e = [1.0 if o == esito_reale else 0.0 for o in ordine]
    cum_p = cum_e = 0.0
    somma = 0.0
    for i in range(len(ordine) - 1):
        cum_p += p[i]
        cum_e += e[i]
        somma += (cum_p - cum_e) ** 2
    return somma / (len(ordine) - 1)
