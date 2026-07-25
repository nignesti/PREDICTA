"""
Valida l'ipotesi di Fase 2 punto 4 della ROADMAP: sostituire i gol grezzi
(FTHG/FTAG) con l'xG reale di Understat.com nel calcolo delle componenti
storico/forma del modello statistico, tenendo tutto il resto (scontri
diretti, quote di chiusura + Shin, pesi) identico.

Dataset: archive/game_stats.csv (esportazione Kaggle di Understat.com,
scaricata manualmente dall'utente il 2026-07-25 perche' lo scraping diretto
di understat.com/fbref.com e' bloccato in questo ambiente - vedi ROADMAP.md).
Non e' un match-level dataset (non ha un id di partita condiviso tra le due
squadre): un record e' una riga per squadra per partita, senza il nome
dell'avversario. Le partite vengono quindi ricostruite abbinando ogni riga
alla lista ufficiale delle partite in serie_a.csv per (squadra, data) - non
serve un id di match perche' sappiamo gia' chi ha giocato contro chi.

Copertura Serie A nel dataset: stagioni 2014-2023 complete (380 partite
ciascuna), stagione 2024 parziale (59 partite, l'esportazione si ferma a fine
settembre 2024), stagione 2025 assente (non ancora giocata al momento
dell'esportazione, avvenuta a settembre 2024). Le 3 stagioni di test usate
nel resto della Fase 2 (2023/2024/2025) non sono quindi utilizzabili qui:
si testa invece sulle ultime 3 stagioni con copertura piena (2021/2022/2023),
ricalcolando ANCHE il modello statistico attuale (senza Understat) sulla
STESSA finestra per un confronto onesto — il numero di riferimento 54.87%/
0.1889 e' calcolato su 2023/2024/2025 e non è comparabile direttamente.
"""
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "pages")

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

import backtesting as bt
import modello
from modello import rps

MAPPATURA_NOMI_UNDERSTAT = {"AC Milan": "Milan", "SPAL 2013": "Spal", "Parma Calcio 1913": "Parma"}
HALF_LIFE, N_FORMA = 730, 3
PESO_FORMA, PESO_SCONTRI, PESO_QUOTE, RHO = 0.10, 0.0, 0.90, -0.10
STAGIONI_TEST = ["2023", "2022", "2021"]  # le piu' recenti con copertura Understat piena


def carica_xg_understat(percorso="archive/game_stats.csv"):
    df = pd.read_csv(percorso, parse_dates=["date"])
    sa = df[df["league"] == "Serie A"].copy()
    sa["club_name"] = sa["club_name"].replace(MAPPATURA_NOMI_UNDERSTAT)
    sa["Date"] = sa["date"].dt.normalize()
    casa = sa[sa["home_away"] == "h"].set_index(["club_name", "Date"])["xG"]
    trasf = sa[sa["home_away"] == "a"].set_index(["club_name", "Date"])["xG"]

    base = bt.df[["Date", "HomeTeam", "AwayTeam", "Stagione"]].copy()
    base["xG_casa"] = base.set_index(["HomeTeam", "Date"]).index.map(casa)
    base["xG_trasf"] = base.set_index(["AwayTeam", "Date"]).index.map(trasf)
    coperte = base.dropna(subset=["xG_casa", "xG_trasf"]).sort_values("Date").reset_index(drop=True)
    # Rinominate FTHG/FTAG per riusare modello.stats_pesate_squadre/bt.calcola_forma_bt
    # cosi' come sono: la matematica (media pesata nel tempo per squadra) e'
    # identica, cambia solo se il "gol" e' un gol vero o un xG.
    return coperte.rename(columns={"xG_casa": "FTHG", "xG_trasf": "FTAG"})


def xg_storico_e_forma_understat(df_xg, squadra_casa, squadra_trasf, data_partita):
    idx_xg = int(df_xg["Date"].searchsorted(data_partita, side="left"))
    if idx_xg == 0:
        return None
    df_prima = df_xg.iloc[:idx_xg]
    media_casa = df_prima["FTHG"].mean()
    media_trasf = df_prima["FTAG"].mean()

    storico = modello.stats_pesate_squadre(df_prima, data_riferimento=data_partita, half_life_giorni=HALF_LIFE)
    riga_c = storico[storico["Squadra"] == squadra_casa]
    riga_t = storico[storico["Squadra"] == squadra_trasf]
    if riga_c.empty or riga_t.empty:
        return None

    xg_casa_storico = (riga_c["gol_fatti_casa_storico"].values[0] / media_casa) * (riga_t["gol_subiti_trasferta_storico"].values[0] / media_trasf) * media_casa
    xg_trasf_storico = (riga_t["gol_fatti_trasferta_storico"].values[0] / media_trasf) * (riga_c["gol_subiti_casa_storico"].values[0] / media_casa) * media_trasf

    fatti_c, subiti_c, fatti_c_home, _ = bt.calcola_forma_bt(df_xg, squadra_casa, idx_xg, N_FORMA)
    fatti_t, subiti_t, _, fatti_t_away = bt.calcola_forma_bt(df_xg, squadra_trasf, idx_xg, N_FORMA)
    xg_casa_forma = (fatti_c_home / media_casa) * (max(subiti_t, 0.3) / media_trasf) * media_casa if fatti_c_home > 0 else xg_casa_storico
    xg_trasf_forma = (fatti_t_away / media_trasf) * (max(subiti_c, 0.3) / media_casa) * media_trasf if fatti_t_away > 0 else xg_trasf_storico

    return xg_casa_storico, xg_trasf_storico, xg_casa_forma, xg_trasf_forma


def valuta_stagione(stagione_test, df_xg):
    bt.stagioni_test = [stagione_test]
    bt.train_df = bt.df[~bt.df["Stagione"].astype(str).isin(bt.stagioni_test)].copy()
    bt.test_df = bt.df[bt.df["Stagione"].astype(str).isin(bt.stagioni_test)].copy()
    bt.media_gol_casa = bt.train_df["FTHG"].mean()
    bt.media_gol_trasferta = bt.train_df["FTAG"].mean()
    bt.media_gol_generale = (bt.media_gol_casa + bt.media_gol_trasferta) / 2
    bt.modello_elo_casa, bt.modello_elo_trasferta = (None, None)  # Elo non usato in questo confronto

    componenti_base, componenti_understat = [], []
    for i in range(len(bt.test_df)):
        riga = bt.test_df.iloc[i]
        comp = bt.precompute_componente(i, HALF_LIFE, N_FORMA, metodo_quote="shin", fonte_quote="chiusura")
        if comp is None:
            continue
        extra = xg_storico_e_forma_understat(df_xg, riga["HomeTeam"], riga["AwayTeam"], riga["Date"])
        if extra is None:
            continue
        xg_casa_storico, xg_trasf_storico, xg_casa_forma, xg_trasf_forma = extra
        comp_u = dict(comp)
        comp_u.update(xG_casa_storico=xg_casa_storico, xG_trasf_storico=xg_trasf_storico,
                      xG_casa_forma=xg_casa_forma, xG_trasf_forma=xg_trasf_forma)
        componenti_base.append(comp)
        componenti_understat.append(comp_u)
    return componenti_base, componenti_understat


def metriche(componenti):
    predizioni, reali, probabilita = [], [], []
    for comp in componenti:
        r = bt.valuta_componente(comp, PESO_FORMA, PESO_SCONTRI, PESO_QUOTE, RHO)
        predizioni.append(r["pred"]); reali.append(comp["esito"])
        probabilita.append([r["1"], r["2"], r["X"]])
    acc = accuracy_score(reali, predizioni)
    rps_medio = np.mean([rps({"1": p[0], "X": p[2], "2": p[1]}, r) for p, r in zip(probabilita, reali)])
    return acc, rps_medio


if __name__ == "__main__":
    print("Confronto: storico/forma da gol reali (attuale) vs da xG Understat (Fase 2, punto 4)")
    print(f"Stagioni di test: {STAGIONI_TEST} (uniche con copertura Understat piena; NON le stesse")
    print("2023/2024/2025 degli altri risultati di Fase 2 - vedi docstring per il perche'.\n", flush=True)

    df_xg = carica_xg_understat()

    risultati_base, risultati_understat = {}, {}
    for stagione in STAGIONI_TEST:
        print(f"=== Stagione di test: {stagione} ===", flush=True)
        comp_base, comp_understat = valuta_stagione(stagione, df_xg)
        print(f"  partite valutabili (con copertura Understat): {len(comp_understat)}/{len(bt.test_df)}", flush=True)
        acc_b, rps_b = metriche(comp_base)
        acc_u, rps_u = metriche(comp_understat)
        risultati_base[stagione] = (acc_b, rps_b)
        risultati_understat[stagione] = (acc_u, rps_u)
        print(f"  Gol reali (baseline): acc={acc_b:.2%} rps={rps_b:.4f}")
        print(f"  xG Understat:         acc={acc_u:.2%} rps={rps_u:.4f}\n", flush=True)

    print("=== RIEPILOGO ===")
    accs_b = [v[0] for v in risultati_base.values()]; rpss_b = [v[1] for v in risultati_base.values()]
    accs_u = [v[0] for v in risultati_understat.values()]; rpss_u = [v[1] for v in risultati_understat.values()]
    print(f"  Gol reali (baseline, ricalcolato su {STAGIONI_TEST}): acc={np.mean(accs_b):.2%} rps={np.mean(rpss_b):.4f}")
    print(f"  xG Understat:                                          acc={np.mean(accs_u):.2%} rps={np.mean(rpss_u):.4f}")
