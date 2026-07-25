"""
Fase 3, punto 1 della ROADMAP: modello Bayesiano gerarchico (Baio &
Blangiardo, 2010) con partial pooling tra squadre, per gestire meglio
neopromosse e campioni piccoli rispetto alle medie storiche pesate nel tempo
gia' in produzione (che trattano ogni squadra in isolamento, senza "prendere
in prestito forza" dalle altre quando i dati sono pochi).

NOTA METODOLOGICA IMPORTANTE: l'implementazione "da manuale" userebbe MCMC
(PyMC/Stan) per campionare la distribuzione a posteriori completa. In questo
ambiente `pip install pymc` fallisce: la dipendenza `llvmlite` (necessaria a
`numba`, il motore JIT di PyMC) non ha ancora una wheel precompilata per
Python 3.14 e non compila da sorgente qui (stesso genere di incompatibilita'
di sistema gia' incontrata con XGBoost/libomp in Fase 2). Invece di forzare
una compilazione locale di LLVM (fuori scopo, fragile), si usa una stima
puntuale equivalente: massima verosimiglianza PENALIZZATA (Poisson, log-link,
vantaggio-casa comune), con una penalita' L2 su attacco/difesa per squadra.
Matematicamente questa penalita' e' il logaritmo di un prior Normale centrato
sulla media di lega: minimizzare la verosimiglianza penalizzata equivale a
trovare la moda a posteriori (MAP) dello stesso modello gerarchico che si
userebbe con MCMC, con lo stesso identico effetto di "partial pooling" (le
squadre con pochi dati vengono tirate verso la media di lega invece di avere
stime instabili), ma senza l'incertezza a posteriori completa (solo la stima
puntuale, non l'intervallo di credibilita').

Il modello viene ri-adattato periodicamente (non ad ogni partita, per
tenerlo veloce: circa ogni 10 partite elaborate, un'approssimazione di
"ogni giornata") usando solo le partite precedenti (walk-forward, nessun
dato futuro). Le stime di attacco/difesa sostituiscono la componente
"storico" E "forma" (xG_casa_storico/xG_trasf_storico E xG_casa_forma/
xG_trasf_forma) nel blend gia' validato: ATTENZIONE, non e' una scelta
stilistica ma una necessita' matematica scoperta durante lo sviluppo — ai
pesi ottimali gia' validati (forma=0.10, quote=0.90, che sommano esattamente
a 1) il peso della sola componente "storico" nel blend risulta 0 per
costruzione (vedi `valuta_componente` in pages/backtesting.py: peso_storico =
1 - pf - ps - pe - pq), quindi sostituire SOLO xG_casa_storico non avrebbe
alcun effetto misurabile sulla previsione finale, a prescindere dal suo
valore. Scontri diretti/quote (Shin + chiusura) restano invariati.
"""
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "pages")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score

import backtesting as bt
import modello
from modello import rps, distribuzione_punteggi, esiti_da_matrice

RIADATTA_OGNI_N_PARTITE = 10  # approssima "una volta a giornata" senza ricostruire il calendario esatto
LAMBDA_REG = 5.0  # forza dello shrinkage verso la media di lega (iperparametro, non ottimizzato a fondo)
PESO_FORMA, PESO_SCONTRI, PESO_QUOTE, RHO = 0.10, 0.0, 0.90, -0.10


def adatta_modello_gerarchico(df_train, lambda_reg=LAMBDA_REG):
    """Stima puntuale (MAP) di attacco/difesa per squadra e vantaggio-casa
    comune, vedi docstring del modulo per l'equivalenza con un modello
    gerarchico Bayesiano. df_train: partite con HomeTeam/AwayTeam/FTHG/FTAG."""
    squadre = sorted(set(df_train["HomeTeam"]) | set(df_train["AwayTeam"]))
    idx = {s: i for i, s in enumerate(squadre)}
    n = len(squadre)

    home_idx = df_train["HomeTeam"].map(idx).to_numpy()
    away_idx = df_train["AwayTeam"].map(idx).to_numpy()
    fthg = df_train["FTHG"].to_numpy(dtype=float)
    ftag = df_train["FTAG"].to_numpy(dtype=float)

    def neg_log_posteriori(theta):
        attacco, difesa, vantaggio_casa = theta[:n], theta[n:2 * n], theta[2 * n]
        log_lambda_home = attacco[home_idx] + difesa[away_idx] + vantaggio_casa
        log_lambda_away = attacco[away_idx] + difesa[home_idx]
        lambda_home = np.exp(log_lambda_home)
        lambda_away = np.exp(log_lambda_away)
        log_verosimiglianza = np.sum(fthg * log_lambda_home - lambda_home) + np.sum(ftag * log_lambda_away - lambda_away)
        penalita = lambda_reg * (np.sum(attacco ** 2) + np.sum(difesa ** 2))
        return -log_verosimiglianza + penalita

    risultato = minimize(neg_log_posteriori, np.zeros(2 * n + 1), method="L-BFGS-B")
    attacco = dict(zip(squadre, risultato.x[:n]))
    difesa = dict(zip(squadre, risultato.x[n:2 * n]))
    return attacco, difesa, risultato.x[2 * n]


def xg_gerarchico(attacco, difesa, vantaggio_casa, casa, trasferta):
    if casa not in attacco or trasferta not in attacco:
        return None
    xg_casa = np.exp(attacco[casa] + difesa[trasferta] + vantaggio_casa)
    xg_trasf = np.exp(attacco[trasferta] + difesa[casa])
    return xg_casa, xg_trasf


def valuta_stagione(stagione_test):
    bt.stagioni_test = [stagione_test]
    bt.train_df = bt.df[~bt.df["Stagione"].astype(str).isin(bt.stagioni_test)].copy()
    bt.test_df = bt.df[bt.df["Stagione"].astype(str).isin(bt.stagioni_test)].copy()
    bt.media_gol_casa = bt.train_df["FTHG"].mean()
    bt.media_gol_trasferta = bt.train_df["FTAG"].mean()
    bt.media_gol_generale = (bt.media_gol_casa + bt.media_gol_trasferta) / 2
    bt.modello_elo_casa, bt.modello_elo_trasferta = (None, None)  # Elo non usato in questo confronto (gia' negativo)

    predizioni_base, predizioni_ger, reali = [], [], []
    prob_base, prob_ger = [], []

    attacco = difesa = vantaggio_casa = None
    for i in range(len(bt.test_df)):
        riga = bt.test_df.iloc[i]
        comp = bt.precompute_componente(i, 730, 3, metodo_quote="shin", fonte_quote="chiusura")
        if comp is None:
            continue

        idx_globale = len(bt.train_df) + i
        if attacco is None or idx_globale % RIADATTA_OGNI_N_PARTITE == 0:
            df_fino_a_ora = pd.concat([bt.train_df, bt.test_df.iloc[:i]])
            attacco, difesa, vantaggio_casa = adatta_modello_gerarchico(df_fino_a_ora)

        xg = xg_gerarchico(attacco, difesa, vantaggio_casa, riga["HomeTeam"], riga["AwayTeam"])

        r_base = bt.valuta_componente(comp, PESO_FORMA, PESO_SCONTRI, PESO_QUOTE, RHO)
        predizioni_base.append(r_base["pred"]); prob_base.append([r_base["1"], r_base["2"], r_base["X"]])

        if xg is not None:
            comp_ger = dict(comp)
            comp_ger["xG_casa_storico"] = comp_ger["xG_casa_forma"] = xg[0]
            comp_ger["xG_trasf_storico"] = comp_ger["xG_trasf_forma"] = xg[1]
            r_ger = bt.valuta_componente(comp_ger, PESO_FORMA, PESO_SCONTRI, PESO_QUOTE, RHO)
        else:
            r_ger = r_base
        predizioni_ger.append(r_ger["pred"]); prob_ger.append([r_ger["1"], r_ger["2"], r_ger["X"]])
        reali.append(comp["esito"])

    def metriche(predizioni, probabilita):
        acc = accuracy_score(reali, predizioni)
        rps_medio = np.mean([rps({"1": p[0], "X": p[2], "2": p[1]}, r) for p, r in zip(probabilita, reali)])
        return acc, rps_medio

    return metriche(predizioni_base, prob_base), metriche(predizioni_ger, prob_ger)


if __name__ == "__main__":
    print("Confronto: storico gol reali (attuale, media pesata nel tempo) vs modello Bayesiano gerarchico (MAP)")
    print(f"Ri-adattamento ogni {RIADATTA_OGNI_N_PARTITE} partite, lambda_reg={LAMBDA_REG}\n", flush=True)

    riepilogo_base, riepilogo_ger = [], []
    for stagione in ["2025", "2024", "2023"]:
        print(f"=== Stagione di test: {stagione} ===", flush=True)
        (acc_b, rps_b), (acc_g, rps_g) = valuta_stagione(stagione)
        riepilogo_base.append((acc_b, rps_b)); riepilogo_ger.append((acc_g, rps_g))
        print(f"  Storico gol reali:  acc={acc_b:.2%} rps={rps_b:.4f}")
        print(f"  Bayesiano gerarchico: acc={acc_g:.2%} rps={rps_g:.4f}\n", flush=True)

    accs_b = [v[0] for v in riepilogo_base]; rpss_b = [v[1] for v in riepilogo_base]
    accs_g = [v[0] for v in riepilogo_ger]; rpss_g = [v[1] for v in riepilogo_ger]
    print("=== RIEPILOGO (confronta con il modello statistico: 54.87% acc, 0.1889 rps medi su 3 stagioni) ===")
    print(f"  Storico gol reali:    acc={np.mean(accs_b):.2%} rps={np.mean(rpss_b):.4f}")
    print(f"  Bayesiano gerarchico: acc={np.mean(accs_g):.2%} rps={np.mean(rpss_g):.4f}")
