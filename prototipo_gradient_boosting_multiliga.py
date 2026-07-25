"""
Fase 3, punto 3 della ROADMAP: estensione multi-campionato per aumentare il
volume dati della componente gradient boosting, che finora e' rimasta sotto
il modello statistico (54.17%/0.1896 contro 54.87%/0.1889 su Serie A, vedi
Fase 2 punto 3), probabile causa la scarsita' di dati (~380 partite/stagione
di una sola lega). Ipotesi: usare le partite di altre leghe europee (Premier
League, Liga, Bundesliga, Ligue 1, scaricate con scarica_altre_leghe.py) come
righe di training AGGIUNTIVE aumenta il volume abbastanza da chiudere il
divario. Il TEST resta sempre e solo Serie A (2023/2024/2025, walk-forward):
le altre leghe entrano SOLO nel pool di training.

Ogni lega ha le proprie medie storiche di gol (attacco/difesa) calcolate solo
sui propri dati: non ha senso confrontare i gol grezzi tra Bundesliga e Serie
A, ma le feature finali (rapporti normalizzati sulla media di lega,
probabilita' implicite delle quote) sono su una scala comparabile tra
campionati, motivo per cui il classificatore puo' comunque imparare pattern
trasferibili (es. "quanto le quote predicono bene" e' universale).

Semplificazione rispetto a bt.precompute_componente: la media gol per la
normalizzazione delle altre leghe e' calcolata UNA SOLA VOLTA su tutto lo
storico disponibile della lega (non walk-forward stagione per stagione come
in produzione), perche' queste partite servono solo da training ausiliario,
mai da previsione reale. Le statistiche squadra per squadra (attacco/difesa,
forma, scontri diretti) restano invece rigorosamente walk-forward, nessun
dato futuro rispetto alla partita usata come esempio di training.
"""
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "pages")

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss

import backtesting as bt
import modello
from modello import rps, probabilita_shin, stats_pesate_squadre, distribuzione_punteggi, esiti_da_matrice
import prototipo_gradient_boosting as pgb

CARTELLA_ALTRE_LEGHE = "altre_leghe"
LEGHE = ["E0", "SP1", "D1", "F1"]
HALF_LIFE, N_FORMA = 730, 3

CASCATA_QUOTA_APERTURA = {"H": ["AvgH", "BbAvH", "B365H"], "D": ["AvgD", "BbAvD", "B365D"], "A": ["AvgA", "BbAvA", "B365A"]}
CASCATA_QUOTA_CHIUSURA = {"H": ["AvgCH", "B365CH"], "D": ["AvgCD", "B365CD"], "A": ["AvgCA", "B365CA"]}


def carica_lega(codice_lega):
    """Carica e concatena tutte le stagioni scaricate di una lega, con la
    stessa cascata di priorita' per le quote di consenso (apertura/chiusura)
    gia' usata in unisci_dati.py, duplicata qui per non toccare lo script di
    produzione con una dipendenza da una cartella dati puramente sperimentale."""
    cartella = os.path.join(CARTELLA_ALTRE_LEGHE, codice_lega)
    tutti = []
    for file in sorted(os.listdir(cartella)):
        if not file.endswith(".txt"):
            continue
        percorso = os.path.join(cartella, file)
        try:
            df = pd.read_csv(percorso, encoding="latin1", on_bad_lines="skip")
        except Exception:
            df = pd.read_csv(percorso, encoding="cp1252", on_bad_lines="skip", engine="python")
        if "HomeTeam" not in df.columns or "AwayTeam" not in df.columns or "FTHG" not in df.columns:
            continue
        for esito, candidate in CASCATA_QUOTA_APERTURA.items():
            colonna = next((c for c in candidate if c in df.columns), None)
            df[f"OddsAvg{esito}"] = pd.to_numeric(df[colonna], errors="coerce") if colonna else np.nan
        for esito, candidate in CASCATA_QUOTA_CHIUSURA.items():
            colonna = next((c for c in candidate if c in df.columns), None)
            df[f"OddsAvgC{esito}"] = pd.to_numeric(df[colonna], errors="coerce") if colonna else np.nan
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
        df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
        df = df.dropna(subset=["FTHG", "FTAG", "Date"])
        tutti.append(df)
    lega = pd.concat(tutti, ignore_index=True).sort_values("Date", kind="stable").reset_index(drop=True)
    return lega


MAX_STORICO_PARTITE = 1600  # ~4 stagioni: con half_life=730gg i contributi piu' vecchi sono comunque trascurabili


def calcola_componenti_lega(df_lega, min_partite_storico=380):
    """Componenti per-partita di un'intera lega (walk-forward: solo partite
    precedenti la data di riferimento), a partire dalla min_partite_storico-esima
    partita (serve un po' di storico prima che le medie squadra abbiano senso;
    default 380 = una stagione, come minimo prima di iniziare).

    La finestra di storico passata a stats_pesate_squadre/calcola_forma_bt e'
    limitata a MAX_STORICO_PARTITE (non l'intero storico crescente): senza
    questo limite il costo cresce quadraticamente con il numero di partite
    della lega (ogni chiamata riscansiona da capo tutto lo storico fino a
    quel punto), diventando impraticabile su 30+ stagioni per lega. Con
    half_life=730 giorni i contributi oltre ~4 stagioni fa sono comunque
    trascurabili nella media pesata, quindi il troncamento non cambia il
    risultato in modo apprezzabile."""
    media_gol_casa = df_lega["FTHG"].mean()
    media_gol_trasferta = df_lega["FTAG"].mean()
    media_gol_generale = (media_gol_casa + media_gol_trasferta) / 2

    righe = []
    for i in range(min_partite_storico, len(df_lega)):
        riga = df_lega.iloc[i]
        casa, trasferta, data = riga["HomeTeam"], riga["AwayTeam"], riga["Date"]
        inizio_finestra = max(0, i - MAX_STORICO_PARTITE)
        df_prima = df_lega.iloc[inizio_finestra:i]
        idx_locale = len(df_prima)  # "questa partita" e' subito dopo la finestra troncata

        stats = stats_pesate_squadre(df_prima, data_riferimento=data, half_life_giorni=HALF_LIFE)
        c = stats[stats["Squadra"] == casa]
        t = stats[stats["Squadra"] == trasferta]
        if c.empty or t.empty:
            continue

        xG_casa_storico = (c["gol_fatti_casa_storico"].values[0] / media_gol_casa) * (t["gol_subiti_trasferta_storico"].values[0] / media_gol_trasferta) * media_gol_casa
        xG_trasf_storico = (t["gol_fatti_trasferta_storico"].values[0] / media_gol_trasferta) * (c["gol_subiti_casa_storico"].values[0] / media_gol_casa) * media_gol_trasferta

        fatti_c, subiti_c, fatti_c_home, _ = bt.calcola_forma_bt(df_prima, casa, idx_locale, N_FORMA)
        fatti_t, subiti_t, _, fatti_t_away = bt.calcola_forma_bt(df_prima, trasferta, idx_locale, N_FORMA)
        xG_casa_forma = (fatti_c_home / media_gol_casa) * (max(subiti_t, 0.3) / media_gol_trasferta) * media_gol_casa if fatti_c_home > 0 else xG_casa_storico
        xG_trasf_forma = (fatti_t_away / media_gol_trasferta) * (max(subiti_c, 0.3) / media_gol_casa) * media_gol_trasferta if fatti_t_away > 0 else xG_trasf_storico

        gol_fatti_scontri, gol_subiti_scontri = bt.scontri_diretti_bt(df_prima, casa, trasferta, ultimi_n=10)
        if gol_fatti_scontri is not None:
            xG_casa_scontri = (gol_fatti_scontri / media_gol_generale) * media_gol_casa
            xG_trasf_scontri = (gol_subiti_scontri / media_gol_generale) * media_gol_trasferta
        else:
            xG_casa_scontri, xG_trasf_scontri = xG_casa_storico, xG_trasf_storico

        prob_1_quote = prob_X_quote = prob_2_quote = np.nan
        if pd.notna(riga.get("OddsAvgCH")) and pd.notna(riga.get("OddsAvgCD")) and pd.notna(riga.get("OddsAvgCA")):
            prob_1_quote, prob_X_quote, prob_2_quote = probabilita_shin([riga["OddsAvgCH"], riga["OddsAvgCD"], riga["OddsAvgCA"]])
        elif pd.notna(riga.get("OddsAvgH")) and pd.notna(riga.get("OddsAvgD")) and pd.notna(riga.get("OddsAvgA")):
            prob_1_quote, prob_X_quote, prob_2_quote = probabilita_shin([riga["OddsAvgH"], riga["OddsAvgD"], riga["OddsAvgA"]])

        esito = "1" if riga["FTHG"] > riga["FTAG"] else ("X" if riga["FTHG"] == riga["FTAG"] else "2")

        righe.append({
            "xG_casa_storico": xG_casa_storico, "xG_trasf_storico": xG_trasf_storico,
            "xG_casa_forma": xG_casa_forma, "xG_trasf_forma": xG_trasf_forma,
            "xG_casa_scontri": xG_casa_scontri, "xG_trasf_scontri": xG_trasf_scontri,
            "elo_casa": np.nan, "elo_trasferta": np.nan, "elo_diff": np.nan,  # Elo non usato qui (gia' negativo, vedi Fase 2)
            "prob_1_quote": prob_1_quote, "prob_X_quote": prob_X_quote, "prob_2_quote": prob_2_quote,
            "esito": esito,
        })
    return pd.DataFrame(righe)


MAX_PARTITE_PER_LEGA = 6000  # ~15-16 stagioni: profondita' paragonabile a N_STAGIONI_TRAINING di Serie A,
# tiene il tempo di calcolo gestibile senza scaricare/processare 30+ stagioni per lega


def costruisci_pool_altre_leghe():
    pool = []
    for codice in LEGHE:
        print(f"  Calcolo componenti {codice}...", flush=True)
        df_lega = carica_lega(codice)
        if len(df_lega) > MAX_PARTITE_PER_LEGA:
            df_lega = df_lega.iloc[-MAX_PARTITE_PER_LEGA:].reset_index(drop=True)
        comp = calcola_componenti_lega(df_lega)
        print(f"    {len(df_lega)} partite grezze -> {len(comp)} componenti valide", flush=True)
        pool.append(comp)
    return pd.concat(pool, ignore_index=True)


def valuta_stagione_multiliga(stagione_test, pool_altre_leghe, n_stagioni_training=pgb.N_STAGIONI_TRAINING):
    colonne = pgb.COLONNE_FEATURE
    comp_train_serie_a = pd.concat([
        pgb.componenti_in_dataframe(pgb.calcola_componenti_per_stagione(s))
        for s in sorted(bt.df["Stagione"].astype(str).unique())[
            max(0, sorted(bt.df["Stagione"].astype(str).unique()).index(stagione_test) - n_stagioni_training):
            sorted(bt.df["Stagione"].astype(str).unique()).index(stagione_test)
        ]
    ], ignore_index=True)
    df_test = pgb.componenti_in_dataframe(pgb.calcola_componenti_per_stagione(stagione_test))

    df_train_esteso = pd.concat([comp_train_serie_a, pool_altre_leghe], ignore_index=True)

    risultati = {}
    for nome, df_train in [("Solo Serie A", comp_train_serie_a), ("Serie A + altre leghe", df_train_esteso)]:
        modello_gb = HistGradientBoostingClassifier(
            max_iter=100, max_depth=2, learning_rate=0.03, l2_regularization=5.0,
            early_stopping=True, validation_fraction=0.2, n_iter_no_change=10, random_state=0,
        )
        modello_gb.fit(df_train[colonne], df_train["esito"])
        classi = list(modello_gb.classes_)
        probabilita_test = modello_gb.predict_proba(df_test[colonne])
        predizioni = modello_gb.predict(df_test[colonne])
        acc = accuracy_score(df_test["esito"], predizioni)
        ll = log_loss(df_test["esito"], probabilita_test, labels=classi)
        idx_1, idx_2, idx_X = classi.index("1"), classi.index("2"), classi.index("X")
        rps_medio = np.mean([
            rps({"1": p[idx_1], "2": p[idx_2], "X": p[idx_X]}, r)
            for p, r in zip(probabilita_test, df_test["esito"])
        ])
        risultati[nome] = (acc, rps_medio, ll, len(df_train))
    return risultati


if __name__ == "__main__":
    print("Calcolo del pool di training aggiuntivo dalle altre leghe (una tantum)...", flush=True)
    pool_altre_leghe = costruisci_pool_altre_leghe()
    print(f"Pool totale altre leghe: {len(pool_altre_leghe)} partite\n", flush=True)

    riepilogo = {"Solo Serie A": [], "Serie A + altre leghe": []}
    for stagione in ["2025", "2024", "2023"]:
        print(f"=== Stagione di test: {stagione} ===", flush=True)
        risultati = valuta_stagione_multiliga(stagione, pool_altre_leghe)
        for nome, (acc, rps_medio, ll, n_train) in risultati.items():
            print(f"  {nome}: n_train={n_train} acc={acc:.2%} rps={rps_medio:.4f} logloss={ll:.4f}", flush=True)
            riepilogo[nome].append((acc, rps_medio))
        print()

    print("=== RIEPILOGO (confronta con il modello statistico: 54.87% acc, 0.1889 rps medi su 3 stagioni) ===")
    for nome, valori in riepilogo.items():
        accs = [v[0] for v in valori]
        rpss = [v[1] for v in valori]
        print(f"  {nome}: acc medio={np.mean(accs):.2%} rps medio={np.mean(rpss):.4f}")
