"""
Prototipo di modello a gradient boosting (Fase 2, punto 3 della ROADMAP), che usa
le stesse componenti già validate del modello statistico (xG storico/forma/
scontri diretti, rating Elo, probabilità implicite delle quote) come feature per
un classificatore HistGradientBoosting, invece di combinarle con una media
pesata scelta a mano.

Usiamo HistGradientBoostingClassifier di scikit-learn (già una dipendenza del
progetto) al posto di XGBoost: XGBoost richiede la libreria di sistema libomp,
non disponibile in questo ambiente, mentre HistGradientBoosting è concettualmente
lo stesso tipo di modello (alberi con istogrammi) e gestisce nativamente i NaN
(comodo per le partite senza quote o senza scontri diretti pregressi).

Le feature per OGNI partita (training e test) sono calcolate con la stessa
logica walk-forward point-in-time già validata in pages/backtesting.py: nessuna
approssimazione, nessun dato futuro rispetto alla partita da prevedere.
"""
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "pages")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss

import backtesting as bt
from modello import rps

N_STAGIONI_TRAINING = 10  # con 5 c'era overfitting severo; con 10 il divario train/test si riduce parecchio
N_PARTITE_TIRI = 5  # finestra per le medie recenti di tiri in porta / corner
N_GIORNI_CONGESTIONE = 14  # finestra per "quante partite nelle ultime N giornate"
POSIZIONE_SALVEZZA = 17  # in Serie A (20 squadre) le ultime 3 retrocedono
POSIZIONE_EUROPA = 6  # approssima Champions+Europa League+Conference nella maggior parte delle stagioni

COLONNE_FEATURE = ["xG_casa_storico", "xG_trasf_storico", "xG_casa_forma", "xG_trasf_forma",
                   "xG_casa_scontri", "xG_trasf_scontri", "elo_casa", "elo_trasferta", "elo_diff",
                   "prob_1_quote", "prob_X_quote", "prob_2_quote"]

COLONNE_TIRI = [
    "tiri_porta_fatti_casa", "tiri_porta_subiti_casa", "corner_fatti_casa", "corner_subiti_casa",
    "tiri_porta_fatti_trasf", "tiri_porta_subiti_trasf", "corner_fatti_trasf", "corner_subiti_trasf",
]

COLONNE_RIPOSO = [
    "giorni_riposo_casa", "giorni_riposo_trasf", "partite_congestione_casa", "partite_congestione_trasf",
    "trasferta_precedente_casa", "trasferta_precedente_trasf",
]

COLONNE_MOTIVAZIONE = [
    "distanza_salvezza_casa", "distanza_salvezza_trasf", "distanza_europa_casa", "distanza_europa_trasf",
    "giornata_casa", "giornata_trasf",
]

COLONNE_FEATURE_TIRI = COLONNE_FEATURE + COLONNE_TIRI


def calcola_media_recente(df, squadra, prima_di_idx, col_casa, col_trasferta, n=N_PARTITE_TIRI):
    """Media recente (ultime n partite, casa+trasferta) di una statistica di
    partita (tiri in porta, corner...) non ancora usata dal modello di
    produzione. col_casa è la colonna quando la squadra gioca in casa,
    col_trasferta quando gioca fuori (es. per "fatti": HST/AST; per "subiti":
    AST/HST, invertite). Stessa logica walk-forward di calcola_forma_bt."""
    df_prima = df.iloc[:prima_di_idx]
    casa = df_prima[df_prima["HomeTeam"] == squadra].tail(n)
    trasferta = df_prima[df_prima["AwayTeam"] == squadra].tail(n)
    valori = list(casa[col_casa].dropna()) + list(trasferta[col_trasferta].dropna())
    return np.mean(valori) if valori else np.nan


def calcola_giorni_riposo(df, squadra, prima_di_idx, data_partita):
    """Giorni trascorsi dall'ultima partita giocata dalla squadra (casa o
    trasferta) prima di questa, calcolo walk-forward point-in-time (nessun dato
    futuro). NaN se e' la prima partita della squadra nel dataset fino a qui."""
    df_prima = df.iloc[:prima_di_idx]
    partite_squadra = df_prima[(df_prima["HomeTeam"] == squadra) | (df_prima["AwayTeam"] == squadra)]
    if partite_squadra.empty:
        return np.nan
    return (data_partita - partite_squadra["Date"].max()).days


def conta_partite_congestione(df, squadra, prima_di_idx, data_partita, giorni=N_GIORNI_CONGESTIONE):
    """Numero di partite giocate dalla squadra negli ultimi 'giorni' prima
    della data della partita da prevedere (congestione di calendario: coppe,
    recuperi...)."""
    df_prima = df.iloc[:prima_di_idx]
    partite_squadra = df_prima[(df_prima["HomeTeam"] == squadra) | (df_prima["AwayTeam"] == squadra)]
    soglia = data_partita - pd.Timedelta(days=giorni)
    return int(((partite_squadra["Date"] > soglia) & (partite_squadra["Date"] < data_partita)).sum())


def ultima_partita_fu_trasferta(df, squadra, prima_di_idx):
    """1.0 se l'ultima partita giocata dalla squadra prima di questa era in
    trasferta, 0.0 se in casa, NaN se non ci sono partite precedenti (serve a
    intercettare "trasferta dopo trasferta", segnalata come possibile fattore
    di affaticamento da una revisione esterna)."""
    df_prima = df.iloc[:prima_di_idx]
    partite_squadra = df_prima[(df_prima["HomeTeam"] == squadra) | (df_prima["AwayTeam"] == squadra)]
    if partite_squadra.empty:
        return np.nan
    ultima = partite_squadra.sort_values("Date").iloc[-1]
    return 1.0 if ultima["AwayTeam"] == squadra else 0.0


def costruisci_classifiche_progressive(df_stagione):
    """df_stagione: le partite di UNA sola stagione, ordinate per data (come
    e' gia' bt.test_df quando si processa una singola stagione). Restituisce
    una lista della stessa lunghezza dove l'elemento i e' lo stato della
    classifica (punti, partite giocate per squadra) calcolato SOLO sulle
    partite con indice < i: walk-forward, nessun dato futuro rispetto alla
    partita i-esima. Le squadre non ancora scese in campo non compaiono nei
    dict (trattate a 0 punti/0 giocate da chi le consuma)."""
    punti, giocate = {}, {}
    stati = []
    for _, row in df_stagione.iterrows():
        stati.append((dict(punti), dict(giocate)))
        h, a = row["HomeTeam"], row["AwayTeam"]
        giocate[h] = giocate.get(h, 0) + 1
        giocate[a] = giocate.get(a, 0) + 1
        if row["FTHG"] > row["FTAG"]:
            punti[h] = punti.get(h, 0) + 3
        elif row["FTHG"] < row["FTAG"]:
            punti[a] = punti.get(a, 0) + 3
        else:
            punti[h] = punti.get(h, 0) + 1
            punti[a] = punti.get(a, 0) + 1
    return stati


def distanza_da_soglie(stato, squadre_tutte, squadra):
    """Punti sopra/sotto le soglie di salvezza ed Europa (positivo = sopra la
    soglia, es. al sicuro dalla retrocessione o in corsa per l'Europa) e
    partite giocate dalla squadra. Richiede l'elenco completo delle squadre
    della stagione per considerare a 0 punti quelle non ancora in classifica
    (es. a inizio stagione)."""
    punti, giocate = stato
    if len(squadre_tutte) < POSIZIONE_SALVEZZA:
        return np.nan, np.nan, giocate.get(squadra, 0)
    punti_completi = {s: punti.get(s, 0) for s in squadre_tutte}
    ordinati = sorted(punti_completi.values(), reverse=True)
    soglia_salvezza = ordinati[POSIZIONE_SALVEZZA - 1]
    soglia_europa = ordinati[POSIZIONE_EUROPA - 1]
    p = punti_completi[squadra]
    return p - soglia_salvezza, p - soglia_europa, giocate.get(squadra, 0)


def componenti_in_dataframe(componenti):
    righe = []
    for c in componenti:
        righe.append({
            "xG_casa_storico": c["xG_casa_storico"], "xG_trasf_storico": c["xG_trasf_storico"],
            "xG_casa_forma": c["xG_casa_forma"], "xG_trasf_forma": c["xG_trasf_forma"],
            "xG_casa_scontri": c["xG_casa_scontri"] if c["scontri_validi"] else np.nan,
            "xG_trasf_scontri": c["xG_trasf_scontri"] if c["scontri_validi"] else np.nan,
            "elo_casa": c["elo_casa"], "elo_trasferta": c["elo_trasferta"],
            "elo_diff": (c["elo_casa"] - c["elo_trasferta"]) if (pd.notna(c["elo_casa"]) and pd.notna(c["elo_trasferta"])) else np.nan,
            "prob_1_quote": c["prob_1_quote"] if c["quote_presenti"] else np.nan,
            "prob_X_quote": c["prob_X_quote"] if c["quote_presenti"] else np.nan,
            "prob_2_quote": c["prob_2_quote"] if c["quote_presenti"] else np.nan,
            "tiri_porta_fatti_casa": c.get("tiri_porta_fatti_casa", np.nan),
            "tiri_porta_subiti_casa": c.get("tiri_porta_subiti_casa", np.nan),
            "corner_fatti_casa": c.get("corner_fatti_casa", np.nan),
            "corner_subiti_casa": c.get("corner_subiti_casa", np.nan),
            "tiri_porta_fatti_trasf": c.get("tiri_porta_fatti_trasf", np.nan),
            "tiri_porta_subiti_trasf": c.get("tiri_porta_subiti_trasf", np.nan),
            "corner_fatti_trasf": c.get("corner_fatti_trasf", np.nan),
            "corner_subiti_trasf": c.get("corner_subiti_trasf", np.nan),
            "giorni_riposo_casa": c.get("giorni_riposo_casa", np.nan),
            "giorni_riposo_trasf": c.get("giorni_riposo_trasf", np.nan),
            "partite_congestione_casa": c.get("partite_congestione_casa", np.nan),
            "partite_congestione_trasf": c.get("partite_congestione_trasf", np.nan),
            "trasferta_precedente_casa": c.get("trasferta_precedente_casa", np.nan),
            "trasferta_precedente_trasf": c.get("trasferta_precedente_trasf", np.nan),
            "distanza_salvezza_casa": c.get("distanza_salvezza_casa", np.nan),
            "distanza_salvezza_trasf": c.get("distanza_salvezza_trasf", np.nan),
            "distanza_europa_casa": c.get("distanza_europa_casa", np.nan),
            "distanza_europa_trasf": c.get("distanza_europa_trasf", np.nan),
            "giornata_casa": c.get("giornata_casa", np.nan),
            "giornata_trasf": c.get("giornata_trasf", np.nan),
            "esito": c["esito"],
        })
    return pd.DataFrame(righe)


def calcola_componenti_per_stagione(stagione_test, half_life=730, n_forma=3, con_tiri=False, con_riposo=False,
                                    con_motivazione=False):
    """Rialloca train/test/statistiche del modulo backtesting per una singola
    stagione (walk-forward corretto: train = tutto ciò che precede) e restituisce
    le componenti per-partita di quella stagione. Se con_tiri=True, aggiunge le
    medie recenti di tiri in porta/corner; se con_riposo=True, aggiunge giorni
    di riposo/congestione di calendario; se con_motivazione=True, aggiunge la
    distanza dalle soglie di salvezza/Europa in classifica (nessuna delle tre
    e' nel modello di produzione).

    Non usa bt.precompute_tutte così com'è: quella funzione scarta le partite
    senza storico valido, perdendo l'allineamento con l'indice originale di
    test_df di cui questa funzione ha bisogno per calcolare le feature extra
    sullo stesso sottoinsieme di partite."""
    bt.stagioni_test = [stagione_test]
    bt.train_df = bt.df[~bt.df["Stagione"].astype(str).isin(bt.stagioni_test)].copy()
    bt.test_df = bt.df[bt.df["Stagione"].astype(str).isin(bt.stagioni_test)].copy()
    bt.media_gol_casa = bt.train_df["FTHG"].mean()
    bt.media_gol_trasferta = bt.train_df["FTAG"].mean()
    bt.media_gol_generale = (bt.media_gol_casa + bt.media_gol_trasferta) / 2
    bt.modello_elo_casa, bt.modello_elo_trasferta = (
        bt.calibra_regressione_elo(bt.train_df, bt.elo_df) if bt.ELO_DISPONIBILE else (None, None)
    )

    train_df, test_df = bt.train_df, bt.test_df
    if bt.ELO_DISPONIBILE:
        elo_casa_arr = bt.elo_asof_batch(bt.elo_df, test_df["HomeTeam"], test_df["Date"])
        elo_trasf_arr = bt.elo_asof_batch(bt.elo_df, test_df["AwayTeam"], test_df["Date"])
    else:
        elo_casa_arr = np.full(len(test_df), np.nan)
        elo_trasf_arr = np.full(len(test_df), np.nan)

    if con_motivazione:
        classifiche = costruisci_classifiche_progressive(test_df)
        squadre_stagione = sorted(set(test_df["HomeTeam"]) | set(test_df["AwayTeam"]))

    componenti = []
    for i in range(len(test_df)):
        comp = bt.precompute_componente(i, half_life, n_forma, elo_casa=elo_casa_arr[i], elo_trasferta=elo_trasf_arr[i])
        if comp is None:
            continue
        riga = test_df.iloc[i]
        casa, trasferta = riga["HomeTeam"], riga["AwayTeam"]
        idx_globale = len(train_df) + i
        df_fino_a_ora = pd.concat([train_df, test_df.iloc[:i + 1]])
        if con_tiri:
            comp.update(
                tiri_porta_fatti_casa=calcola_media_recente(df_fino_a_ora, casa, idx_globale, "HST", "AST"),
                tiri_porta_subiti_casa=calcola_media_recente(df_fino_a_ora, casa, idx_globale, "AST", "HST"),
                corner_fatti_casa=calcola_media_recente(df_fino_a_ora, casa, idx_globale, "HC", "AC"),
                corner_subiti_casa=calcola_media_recente(df_fino_a_ora, casa, idx_globale, "AC", "HC"),
                tiri_porta_fatti_trasf=calcola_media_recente(df_fino_a_ora, trasferta, idx_globale, "AST", "HST"),
                tiri_porta_subiti_trasf=calcola_media_recente(df_fino_a_ora, trasferta, idx_globale, "HST", "AST"),
                corner_fatti_trasf=calcola_media_recente(df_fino_a_ora, trasferta, idx_globale, "AC", "HC"),
                corner_subiti_trasf=calcola_media_recente(df_fino_a_ora, trasferta, idx_globale, "HC", "AC"),
            )
        if con_riposo:
            data_partita = riga["Date"]
            comp.update(
                giorni_riposo_casa=calcola_giorni_riposo(df_fino_a_ora, casa, idx_globale, data_partita),
                giorni_riposo_trasf=calcola_giorni_riposo(df_fino_a_ora, trasferta, idx_globale, data_partita),
                partite_congestione_casa=conta_partite_congestione(df_fino_a_ora, casa, idx_globale, data_partita),
                partite_congestione_trasf=conta_partite_congestione(df_fino_a_ora, trasferta, idx_globale, data_partita),
                trasferta_precedente_casa=ultima_partita_fu_trasferta(df_fino_a_ora, casa, idx_globale),
                trasferta_precedente_trasf=ultima_partita_fu_trasferta(df_fino_a_ora, trasferta, idx_globale),
            )
        if con_motivazione:
            dist_salv_c, dist_eur_c, giornata_c = distanza_da_soglie(classifiche[i], squadre_stagione, casa)
            dist_salv_t, dist_eur_t, giornata_t = distanza_da_soglie(classifiche[i], squadre_stagione, trasferta)
            comp.update(
                distanza_salvezza_casa=dist_salv_c, distanza_salvezza_trasf=dist_salv_t,
                distanza_europa_casa=dist_eur_c, distanza_europa_trasf=dist_eur_t,
                giornata_casa=giornata_c, giornata_trasf=giornata_t,
            )
        componenti.append(comp)
    return componenti


def valuta_stagione(stagione_test, n_stagioni_training=N_STAGIONI_TRAINING, con_tiri=False, con_riposo=False,
                    con_motivazione=False):
    colonne = (COLONNE_FEATURE + (COLONNE_TIRI if con_tiri else [])
               + (COLONNE_RIPOSO if con_riposo else []) + (COLONNE_MOTIVAZIONE if con_motivazione else []))
    stagioni_disponibili = sorted(bt.df["Stagione"].astype(str).unique())
    idx_test = stagioni_disponibili.index(stagione_test)
    stagioni_training = stagioni_disponibili[max(0, idx_test - n_stagioni_training):idx_test]

    df_train = pd.concat([componenti_in_dataframe(calcola_componenti_per_stagione(
        s, con_tiri=con_tiri, con_riposo=con_riposo, con_motivazione=con_motivazione)) for s in stagioni_training], ignore_index=True)
    df_test = componenti_in_dataframe(calcola_componenti_per_stagione(
        stagione_test, con_tiri=con_tiri, con_riposo=con_riposo, con_motivazione=con_motivazione))

    modello = HistGradientBoostingClassifier(
        max_iter=100, max_depth=2, learning_rate=0.03, l2_regularization=5.0,
        early_stopping=True, validation_fraction=0.2, n_iter_no_change=10, random_state=0,
    )
    modello.fit(df_train[colonne], df_train["esito"])

    classi = list(modello.classes_)
    probabilita_test = modello.predict_proba(df_test[colonne])
    predizioni = modello.predict(df_test[colonne])

    acc_train = accuracy_score(df_train["esito"], modello.predict(df_train[colonne]))
    acc = accuracy_score(df_test["esito"], predizioni)
    ll = log_loss(df_test["esito"], probabilita_test, labels=classi)
    idx_1, idx_2, idx_X = classi.index("1"), classi.index("2"), classi.index("X")
    rps_medio = np.mean([
        rps({"1": p[idx_1], "2": p[idx_2], "X": p[idx_X]}, r)
        for p, r in zip(probabilita_test, df_test["esito"])
    ])
    return acc, rps_medio, ll, acc_train, len(df_train), len(df_test)


if __name__ == "__main__":
    con_tiri = "--con-tiri" in sys.argv
    con_riposo = "--con-riposo" in sys.argv
    con_motivazione = "--con-motivazione" in sys.argv
    print(f"Feature tiri/corner: {'SI' if con_tiri else 'NO'} | Feature riposo/congestione: {'SI' if con_riposo else 'NO'} "
          f"| Feature motivazione (salvezza/Europa): {'SI' if con_motivazione else 'NO'}\n", flush=True)
    risultati = {}
    for stagione in ["2025", "2024", "2023"]:
        print(f"=== Stagione di test: {stagione} ===", flush=True)
        acc, rps_medio, ll, acc_train, n_train, n_test = valuta_stagione(
            stagione, con_tiri=con_tiri, con_riposo=con_riposo, con_motivazione=con_motivazione)
        risultati[stagione] = (acc, rps_medio, ll)
        print(f"  n_train={n_train} n_test={n_test} acc_train={acc_train:.1%}")
        print(f"  HistGradientBoosting: acc={acc:.1%} rps={rps_medio:.4f} logloss={ll:.4f}", flush=True)
        print()

    print("=== RIEPILOGO (confronta con il modello statistico: 54.87% acc, 0.1889 rps medi su 3 stagioni) ===")
    accs = [v[0] for v in risultati.values()]
    rpss = [v[1] for v in risultati.values()]
    for s, (acc, rps_medio, ll) in risultati.items():
        print(f"  {s}: acc={acc:.1%} rps={rps_medio:.4f} logloss={ll:.4f}")
    print(f"  MEDIA: acc={np.mean(accs):.2%} rps={np.mean(rpss):.4f}")
