"""
Valida le MEDIE DI LEGA PESATE NEL TEMPO come normalizzatore dell'xG, al posto
della media semplice su tutto il training set usata finora.

Motivazione. Le formule dell'xG rapportano la forza di una squadra alla media di
campionato:

    xG_casa = (attacco_casa / media_gol_casa) * (difesa_trasferta / media_gol_trasferta) * media_gol_casa

Le statistiche di squadra (attacco_casa, difesa_trasferta) decadono con emivita
730 giorni, quindi descrivono di fatto gli ultimi 2-4 anni. La media di lega
invece era calcolata come media semplice su ~30 stagioni. Si stanno confrontando
due epoche diverse, e su questo dataset la differenza non e' trascurabile:

    tutto lo storico:  1.5135 gol casa / 1.1433 gol trasferta
    dal 2021 in poi:   1.3932 gol casa / 1.2121 gol trasferta

Semplificando algebricamente, media_gol_casa sparisce dalla formula dell'xG di
casa e resta solo il divisore media_gol_trasferta (piu' basso del 6% del dovuto,
quindi xG di casa gonfiato del 6%); simmetricamente l'xG di trasferta e'
sgonfiato dell'8%. Il rapporto fra i due e' distorto di circa il 15% a favore
della casa su OGNI partita. E' la stessa modalita' di errore gia' corretta una
volta in Fase 1 ("il modello collassa su vince sempre la casa").

Protocollo identico a valida_shin.py / valida_quote_chiusura.py: 3 stagioni di
test indipendenti (2023, 2024, 2025) walk-forward, pesi fissi alla configurazione
gia' validata, Shin + quote di chiusura come default correnti. Nessun dato nuovo
e nessun addestramento: cambia solo il normalizzatore.
"""
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "pages")

import numpy as np
from sklearn.metrics import accuracy_score

import backtesting as bt
from modello import rps

PESO_FORMA, PESO_SCONTRI, PESO_QUOTE, RHO = 0.10, 0.0, 0.90, -0.10
EMIVITA_GIORNI, N_PARTITE_FORMA = 730, 3
STAGIONI = ["2025", "2024", "2023"]


def valuta_stagione(stagione_test, medie_lega):
    bt.stagioni_test = [stagione_test]
    bt.train_df = bt.df[~bt.df["Stagione"].astype(str).isin(bt.stagioni_test)].copy()
    bt.test_df = bt.df[bt.df["Stagione"].astype(str).isin(bt.stagioni_test)].copy()
    bt.media_gol_casa = bt.train_df["FTHG"].mean()
    bt.media_gol_trasferta = bt.train_df["FTAG"].mean()
    bt.media_gol_generale = (bt.media_gol_casa + bt.media_gol_trasferta) / 2
    bt.modello_elo_casa, bt.modello_elo_trasferta = (
        bt.calibra_regressione_elo(bt.train_df, bt.elo_df) if bt.ELO_DISPONIBILE else (None, None))

    componenti = bt.precompute_tutte(EMIVITA_GIORNI, N_PARTITE_FORMA, metodo_quote="shin",
                                     fonte_quote="chiusura", medie_lega=medie_lega)
    predizioni, reali, _, probabilita = bt.valuta_tutte(componenti, PESO_FORMA, PESO_SCONTRI, PESO_QUOTE, RHO)
    acc = accuracy_score(reali, predizioni)
    rps_medio = np.mean([rps({"1": p[0], "X": p[2], "2": p[1]}, r) for p, r in zip(probabilita, reali)])
    return acc, rps_medio


def valuta_stagione_solo_modello(stagione_test, medie_lega):
    """Stesso calcolo ma con peso_quote=0: isola l'effetto sul modello statistico
    puro. Con peso_quote=0.90 il blend e' dominato dal mercato e mascherebbe quasi
    del tutto una modifica che agisce solo sull'xG."""
    bt.stagioni_test = [stagione_test]
    bt.train_df = bt.df[~bt.df["Stagione"].astype(str).isin(bt.stagioni_test)].copy()
    bt.test_df = bt.df[bt.df["Stagione"].astype(str).isin(bt.stagioni_test)].copy()
    bt.media_gol_casa = bt.train_df["FTHG"].mean()
    bt.media_gol_trasferta = bt.train_df["FTAG"].mean()
    bt.media_gol_generale = (bt.media_gol_casa + bt.media_gol_trasferta) / 2

    componenti = bt.precompute_tutte(EMIVITA_GIORNI, N_PARTITE_FORMA, metodo_quote="shin",
                                     fonte_quote="chiusura", medie_lega=medie_lega)
    predizioni, reali, _, probabilita = bt.valuta_tutte(componenti, PESO_FORMA, PESO_SCONTRI, 0.0, RHO)
    acc = accuracy_score(reali, predizioni)
    rps_medio = np.mean([rps({"1": p[0], "X": p[2], "2": p[1]}, r) for p, r in zip(probabilita, reali)])
    quota_pred_1 = predizioni.count("1") / len(predizioni)
    return acc, rps_medio, quota_pred_1


if __name__ == "__main__":
    print("Confronto normalizzatore xG: media di lega STORICA vs PESATA nel tempo")
    print(f"Pesi fissi: forma={PESO_FORMA}, scontri={PESO_SCONTRI}, quote={PESO_QUOTE}, "
          f"rho={RHO}, Shin + quote di chiusura\n", flush=True)

    print("### A) Configurazione di produzione (peso_quote=0.90)\n", flush=True)
    riepilogo = {}
    for medie in ["storiche", "pesate"]:
        print(f"=== Medie di lega: {medie} ===", flush=True)
        accs, rpss = [], []
        for stagione in STAGIONI:
            acc, rps_medio = valuta_stagione(stagione, medie)
            accs.append(acc); rpss.append(rps_medio)
            print(f"  {stagione}: acc={acc:.2%} rps={rps_medio:.4f}", flush=True)
        riepilogo[medie] = (np.mean(accs), np.mean(rpss))
        print(f"  MEDIA: acc={riepilogo[medie][0]:.2%} rps={riepilogo[medie][1]:.4f}\n", flush=True)

    print("### B) Modello statistico puro (peso_quote=0), per isolare l'effetto sull'xG\n", flush=True)
    riepilogo_puro = {}
    for medie in ["storiche", "pesate"]:
        print(f"=== Medie di lega: {medie} ===", flush=True)
        accs, rpss, quote1 = [], [], []
        for stagione in STAGIONI:
            acc, rps_medio, q1 = valuta_stagione_solo_modello(stagione, medie)
            accs.append(acc); rpss.append(rps_medio); quote1.append(q1)
            print(f"  {stagione}: acc={acc:.2%} rps={rps_medio:.4f} (previsioni '1': {q1:.1%})", flush=True)
        riepilogo_puro[medie] = (np.mean(accs), np.mean(rpss), np.mean(quote1))
        print(f"  MEDIA: acc={riepilogo_puro[medie][0]:.2%} rps={riepilogo_puro[medie][1]:.4f} "
              f"(previsioni '1': {riepilogo_puro[medie][2]:.1%})\n", flush=True)

    print("=== RIEPILOGO FINALE ===")
    print("Produzione (quote=0.90):")
    for medie, (acc, r) in riepilogo.items():
        print(f"  {medie:10s}: acc={acc:.2%} rps={r:.4f}")
    print("Modello statistico puro (quote=0):")
    for medie, (acc, r, q1) in riepilogo_puro.items():
        print(f"  {medie:10s}: acc={acc:.2%} rps={r:.4f} previsioni '1'={q1:.1%}")
