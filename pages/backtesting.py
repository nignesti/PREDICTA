import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, log_loss

from modello import (stats_pesate_squadre, distribuzione_punteggi, esiti_da_matrice, rps,
                     prepara_elo, elo_asof_batch, calibra_regressione_elo, xg_da_elo_calibrato)

st.set_page_config(page_title="PredictA — Backtesting", page_icon=":material/bar_chart:", layout="wide")

st.title("Backtesting — Validazione del modello", text_alignment="center")
st.markdown("Simulazione walk-forward su stagioni consecutive per validare accuratezza e calibrazione del modello predittivo.", text_alignment="center")

# ------------------------------------------------------------
# CARICA DATI
# ------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("serie_a.csv")
    df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
    df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["FTHG", "FTAG"])
    return df

@st.cache_data
def load_elo():
    elo = pd.read_csv("elo_storico.csv", parse_dates=["From", "To"])
    return prepara_elo(elo)

df = load_data()
elo_df = load_elo()

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
st.sidebar.markdown("### :material/tune: Impostazioni backtesting")

stage = st.sidebar.container(border=True)
with stage:
    st.markdown("**Walk-forward**")
    stagioni_disponibili = sorted(df["Stagione"].astype(str).unique())
    n_stagioni_test = st.slider(
        "Stagioni di test", 1, min(5, len(stagioni_disponibili) - 1), 1,
        help="Le ultime N stagioni vengono usate come test set (walk-forward), tutte le precedenti come training."
    )
    stagioni_test = stagioni_disponibili[-n_stagioni_test:]

with st.sidebar.container(border=True):
    st.markdown("**Pesi del modello**")
    peso_forma_bt = st.slider("Forma recente", 0.0, 1.0, 0.10, 0.05,
                        help="Validato su 3 stagioni indipendenti: un peso piccolo aiuta, oltre 0.20-0.25 peggiora.")
    peso_scontri_bt = st.slider("Scontri diretti", 0.0, 0.5, 0.0, 0.05,
                        help="Su 3 stagioni di backtest non aggiunge valore misurabile una volta pesate bene le quote.")
    n_partite_forma = st.slider("Partite per forma", 3, 10, 3,
                        help="Poco sensibile in questo range; 3 e' risultato leggermente migliore nella grid search.")
    peso_elo_bt = st.slider("Elo (ClubElo.com)", 0.0, 1.0, 0.0, 0.05,
                        help="Rating Elo storico da clubelo.com, non ancora validato: default a 0 finché non testato.")
    peso_quote_bt = st.slider("Quote bookmaker", 0.0, 1.0, 0.90, 0.05)

with st.sidebar.container(border=True):
    st.markdown("**Iperparametri Dixon-Coles**")
    emivita_giorni_bt = st.slider(
        "Emivita storica (giorni)", 90, 3650, 730, 90,
        help="Dopo quanti giorni una partita pesa la metà nelle medie storiche. Sostituisce la media semplice su 33 stagioni."
    )
    rho_bt = st.slider(
        "Rho (correzione bassi punteggi)", -0.30, 0.10, -0.10, 0.02,
        help="Corregge la sottostima dei pareggi tipica di un Poisson indipendente. Valori tipici in letteratura: -0.05/-0.20."
    )

train_df = df[~df["Stagione"].astype(str).isin(stagioni_test)].copy()
test_df = df[df["Stagione"].astype(str).isin(stagioni_test)].copy()

st.sidebar.caption(f":material/database: Training: **{len(train_df):,}** partite")
st.sidebar.caption(f":material/science: Test: **{len(test_df):,}** partite ({', '.join(stagioni_test)})")

# ------------------------------------------------------------
# STATISTICHE SU TRAINING
# ------------------------------------------------------------
media_gol_casa = train_df["FTHG"].mean()
media_gol_trasferta = train_df["FTAG"].mean()
media_gol_generale = (media_gol_casa + media_gol_trasferta) / 2
vantaggio_casa = media_gol_casa / media_gol_trasferta

# Regressione di Poisson che calibra su dati reali di training come la differenza
# di rating Elo si traduce in gol attesi (al posto di una costante indovinata):
# va rifatta solo quando cambia il training set (stagioni di test), non a ogni
# combinazione di pesi provata.
modello_elo_casa, modello_elo_trasferta = calibra_regressione_elo(train_df, elo_df)

# ------------------------------------------------------------
# FUNZIONI
# ------------------------------------------------------------
def calcola_forma_bt(df, squadra, prima_di_idx, n=5):
    df_prima = df.iloc[:prima_di_idx]
    casa = df_prima[df_prima["HomeTeam"] == squadra].tail(n)
    trasferta = df_prima[df_prima["AwayTeam"] == squadra].tail(n)

    gol_fatti, gol_subiti = [], []
    for _, row in casa.iterrows():
        gol_fatti.append(row["FTHG"]); gol_subiti.append(row["FTAG"])
    for _, row in trasferta.iterrows():
        gol_fatti.append(row["FTAG"]); gol_subiti.append(row["FTHG"])

    if len(gol_fatti) == 0:
        return 0, 0, 0, 0
    return (np.mean(gol_fatti), np.mean(gol_subiti),
            np.mean([row["FTHG"] for _, row in casa.iterrows()]) if len(casa) > 0 else 0,
            np.mean([row["FTAG"] for _, row in trasferta.iterrows()]) if len(trasferta) > 0 else 0)

def scontri_diretti_bt(df_prima, squadra1, squadra2, ultimi_n=10):
    scontri = df_prima[((df_prima["HomeTeam"] == squadra1) & (df_prima["AwayTeam"] == squadra2)) |
                        ((df_prima["HomeTeam"] == squadra2) & (df_prima["AwayTeam"] == squadra1))].tail(ultimi_n)
    if len(scontri) == 0:
        return None, None
    gol_fatti_s1, gol_subiti_s1 = [], []
    for _, row in scontri.iterrows():
        if row["HomeTeam"] == squadra1:
            gol_fatti_s1.append(row["FTHG"]); gol_subiti_s1.append(row["FTAG"])
        else:
            gol_fatti_s1.append(row["FTAG"]); gol_subiti_s1.append(row["FTHG"])
    return np.mean(gol_fatti_s1), np.mean(gol_subiti_s1)

def precompute_componente(idx_test, half_life_giorni, n_partite_forma, elo_casa=None, elo_trasferta=None):
    """Calcola le componenti costose (storico pesato nel tempo, forma, scontri
    diretti, quote) per una partita di test. Dipendono solo da half_life_giorni e
    n_partite_forma, MAI dai pesi forma/scontri/quote: si possono quindi calcolare
    una volta sola e riusare per confrontare più configurazioni di pesi. I rating
    Elo (se passati) sono già stati calcolati in batch da precompute_tutte."""
    riga = test_df.iloc[idx_test]
    casa = riga["HomeTeam"]
    trasferta = riga["AwayTeam"]

    idx_globale = len(train_df) + idx_test
    df_fino_a_ora = pd.concat([train_df, test_df.iloc[:idx_test + 1]])
    df_prima = df_fino_a_ora.iloc[:idx_globale]

    stats = stats_pesate_squadre(df_prima, data_riferimento=riga["Date"], half_life_giorni=half_life_giorni)
    c = stats[stats["Squadra"] == casa]
    t = stats[stats["Squadra"] == trasferta]
    if c.empty or t.empty:
        return None

    xG_casa_storico = (c["gol_fatti_casa_storico"].values[0] / media_gol_casa) * (t["gol_subiti_trasferta_storico"].values[0] / media_gol_trasferta) * media_gol_casa
    xG_trasf_storico = (t["gol_fatti_trasferta_storico"].values[0] / media_gol_trasferta) * (c["gol_subiti_casa_storico"].values[0] / media_gol_casa) * media_gol_trasferta

    fatti_c, subiti_c, fatti_c_home, _ = calcola_forma_bt(df_fino_a_ora, casa, idx_globale, n_partite_forma)
    fatti_t, subiti_t, _, fatti_t_away = calcola_forma_bt(df_fino_a_ora, trasferta, idx_globale, n_partite_forma)
    xG_casa_forma = (fatti_c_home / media_gol_casa) * (max(subiti_t, 0.3) / media_gol_trasferta) * media_gol_casa if fatti_c_home > 0 else xG_casa_storico
    xG_trasf_forma = (fatti_t_away / media_gol_trasferta) * (max(subiti_c, 0.3) / media_gol_casa) * media_gol_trasferta if fatti_t_away > 0 else xG_trasf_storico

    gol_fatti_scontri, gol_subiti_scontri = scontri_diretti_bt(df_prima, casa, trasferta, ultimi_n=10)
    if gol_fatti_scontri is not None:
        xG_casa_scontri = (gol_fatti_scontri / media_gol_generale) * media_gol_casa
        xG_trasf_scontri = (gol_subiti_scontri / media_gol_generale) * media_gol_trasferta
        scontri_validi = True
    else:
        xG_casa_scontri, xG_trasf_scontri = xG_casa_storico, xG_trasf_storico
        scontri_validi = False

    quote_presenti = False
    prob_1_quote, prob_X_quote, prob_2_quote = 0, 0, 0
    colonne_quota = ("OddsAvgH", "OddsAvgD", "OddsAvgA") if "OddsAvgH" in riga.index else ("B365H", "B365D", "B365A")
    if all(col in riga.index for col in colonne_quota):
        qh, qd, qa = riga[colonne_quota[0]], riga[colonne_quota[1]], riga[colonne_quota[2]]
        if pd.notna(qh) and pd.notna(qd) and pd.notna(qa):
            prob_1_quote, prob_X_quote, prob_2_quote = 1 / qh, 1 / qd, 1 / qa
            somma = prob_1_quote + prob_X_quote + prob_2_quote
            prob_1_quote /= somma; prob_X_quote /= somma; prob_2_quote /= somma
            quote_presenti = True

    esito = "1" if riga["FTHG"] > riga["FTAG"] else ("X" if riga["FTHG"] == riga["FTAG"] else "2")

    return dict(xG_casa_storico=xG_casa_storico, xG_trasf_storico=xG_trasf_storico,
                xG_casa_forma=xG_casa_forma, xG_trasf_forma=xG_trasf_forma,
                xG_casa_scontri=xG_casa_scontri, xG_trasf_scontri=xG_trasf_scontri,
                scontri_validi=scontri_validi, quote_presenti=quote_presenti,
                prob_1_quote=prob_1_quote, prob_X_quote=prob_X_quote, prob_2_quote=prob_2_quote,
                elo_casa=elo_casa, elo_trasferta=elo_trasferta,
                esito=esito, stagione=str(riga["Stagione"]))

def valuta_componente(comp, peso_forma, peso_scontri, peso_quote, rho, peso_elo=0.0):
    """Combina le componenti precalcolate con i pesi scelti: veloce (nessun accesso
    al DataFrame), si può richiamare molte volte sulle stesse componenti."""
    peso_scontri_eff = peso_scontri if comp["scontri_validi"] else 0
    elo_valido = comp.get("elo_casa") is not None and pd.notna(comp.get("elo_casa")) and pd.notna(comp.get("elo_trasferta"))
    peso_elo_eff = peso_elo if elo_valido else 0
    peso_totale = peso_forma + peso_scontri_eff + peso_elo_eff + peso_quote
    if peso_totale > 1:
        pf, ps, pe, pq = peso_forma / peso_totale, peso_scontri_eff / peso_totale, peso_elo_eff / peso_totale, peso_quote / peso_totale
        peso_storico = 0
    else:
        pf, ps, pe, pq = peso_forma, peso_scontri_eff, peso_elo_eff, peso_quote
        peso_storico = 1 - pf - ps - pe - pq

    if not comp["quote_presenti"]:
        peso_storico += pq
        pq = 0

    # Storico+forma+scontri+elo vanno rinormalizzati a sommare 1 tra loro: "quote" non
    # entra nell'xG (entra dopo, sulle probabilità finali), altrimenti con peso_quote
    # alto l'xG si schiaccia verso 0 invece di usare per intero il peso rimanente.
    peso_xg_totale = peso_storico + pf + ps + pe
    if peso_xg_totale > 0:
        xG_casa_elo, xG_trasf_elo = xg_da_elo_calibrato(
            comp.get("elo_casa"), comp.get("elo_trasferta"), modello_elo_casa, modello_elo_trasferta
        ) if pe > 0 else (comp["xG_casa_storico"], comp["xG_trasf_storico"])
        xG_casa = (peso_storico * comp["xG_casa_storico"] + pf * comp["xG_casa_forma"] + ps * comp["xG_casa_scontri"] + pe * xG_casa_elo) / peso_xg_totale
        xG_trasferta = (peso_storico * comp["xG_trasf_storico"] + pf * comp["xG_trasf_forma"] + ps * comp["xG_trasf_scontri"] + pe * xG_trasf_elo) / peso_xg_totale
    else:
        xG_casa, xG_trasferta = comp["xG_casa_storico"], comp["xG_trasf_storico"]

    matrice = distribuzione_punteggi(max(0.05, xG_casa), max(0.05, xG_trasferta), rho=rho)
    esiti = esiti_da_matrice(matrice)
    p_1_base, p_X_base, p_2_base = esiti["p_1"], esiti["p_X"], esiti["p_2"]

    if comp["quote_presenti"]:
        p_1 = (1 - pq) * p_1_base + pq * comp["prob_1_quote"]
        p_X = (1 - pq) * p_X_base + pq * comp["prob_X_quote"]
        p_2 = (1 - pq) * p_2_base + pq * comp["prob_2_quote"]
    else:
        p_1, p_X, p_2 = p_1_base, p_X_base, p_2_base

    return {"1": p_1, "X": p_X, "2": p_2, "pred": "1" if p_1 > p_X and p_1 > p_2 else ("X" if p_X > p_2 else "2")}

def predici_partita_bt(train_df_arg, test_df_arg, idx_test, peso_forma, peso_scontri, peso_quote,
                       half_life_giorni, rho, n_partite_forma_val=5, peso_elo=0.0):
    """Comodo per una singola previsione (test automatici, uso puntuale): calcola
    componenti + valutazione in un solo passo. Il backtest sull'intero test set usa
    invece precompute_componente/valuta_componente separati, per non ripetere il
    calcolo costoso a ogni configurazione di pesi provata."""
    riga = test_df.iloc[idx_test]
    elo_c, elo_t = elo_asof_batch(elo_df, [riga["HomeTeam"], riga["AwayTeam"]], [riga["Date"], riga["Date"]])
    comp = precompute_componente(idx_test, half_life_giorni, n_partite_forma_val, elo_casa=elo_c, elo_trasferta=elo_t)
    if comp is None:
        return None
    return valuta_componente(comp, peso_forma, peso_scontri, peso_quote, rho, peso_elo)

def precompute_tutte(half_life_giorni, n_partite_forma_val, mostra_progress=False):
    # Lookup Elo calcolato in batch per tutte le partite di test in una volta sola
    # (molto più veloce che una query per partita dentro il ciclo).
    elo_casa_arr = elo_asof_batch(elo_df, test_df["HomeTeam"], test_df["Date"])
    elo_trasf_arr = elo_asof_batch(elo_df, test_df["AwayTeam"], test_df["Date"])

    componenti = []
    progress_bar = st.progress(0) if mostra_progress else None
    for i in range(len(test_df)):
        comp = precompute_componente(i, half_life_giorni, n_partite_forma_val,
                                     elo_casa=elo_casa_arr[i], elo_trasferta=elo_trasf_arr[i])
        if comp is not None:
            componenti.append(comp)
        if mostra_progress:
            progress_bar.progress((i + 1) / len(test_df))
    if mostra_progress:
        progress_bar.empty()
    return componenti

def valuta_tutte(componenti, peso_forma, peso_scontri, peso_quote, rho, peso_elo=0.0):
    predizioni, reali, stagioni_pred, probabilita = [], [], [], []
    for comp in componenti:
        r = valuta_componente(comp, peso_forma, peso_scontri, peso_quote, rho, peso_elo)
        predizioni.append(r["pred"])
        reali.append(comp["esito"])
        stagioni_pred.append(comp["stagione"])
        # Ordine (1, 2, X) e non (1, X, 2): sklearn.log_loss richiede che le colonne
        # di probabilita' siano nell'ordine lessicografico delle label, altrimenti
        # calcola il log-loss sulle colonne sbagliate senza errore (bug silenzioso).
        probabilita.append([r["1"], r["2"], r["X"]])
    return predizioni, reali, stagioni_pred, probabilita

# ------------------------------------------------------------
# BOTTONE BACKTESTING
# ------------------------------------------------------------
if st.sidebar.button(":material/play_arrow: Esegui backtesting", width="stretch", type="primary"):
    with st.spinner(":material/hourglass_top: Simulando le previsioni... Potrebbe richiedere qualche minuto."):
        componenti = precompute_tutte(emivita_giorni_bt, n_partite_forma, mostra_progress=True)
        predizioni, reali, stagioni_pred, probabilita = valuta_tutte(
            componenti, peso_forma_bt, peso_scontri_bt, peso_quote_bt, rho_bt, peso_elo_bt)

        # Metriche
        acc = accuracy_score(reali, predizioni)
        cm = confusion_matrix(reali, predizioni, labels=["1", "X", "2"])
        precision, recall, f1, support = precision_recall_fscore_support(reali, predizioni, labels=["1", "X", "2"])
        benchmark = reali.count("1") / len(reali)

        # Metriche probabilistiche
        rps_medio = np.mean([rps({"1": p[0], "X": p[2], "2": p[1]}, r) for p, r in zip(probabilita, reali)])
        logloss = log_loss(reali, probabilita, labels=["1", "2", "X"])

        # ------------------------------------------------------------
        # VISUALIZZA RISULTATI
        # ------------------------------------------------------------
        st.space("medium")
        st.markdown("### Risultati backtesting")

        col_m1, col_m2, col_m3, col_m4 = st.columns(4, gap="medium")

        with col_m1:
            with st.container(border=True):
                st.markdown(f"## {acc:.1%}")
                st.caption("Accuratezza 1X2")
                st.badge("✅ Sopra benchmark" if acc > benchmark else "❌ Sotto benchmark", color="green" if acc > benchmark else "red")

        with col_m2:
            corrette = sum(1 for p, r in zip(predizioni, reali) if p == r)
            with st.container(border=True):
                st.markdown(f"## {corrette}/{len(reali)}")
                st.caption("Partite indovinate")
                st.badge(f"Benchmark: {benchmark:.0%}", color="gray")

        with col_m3:
            with st.container(border=True):
                st.markdown(f"## {rps_medio:.3f}")
                st.caption("RPS medio")
                st.badge("0 = perfetto", color="blue")

        with col_m4:
            with st.container(border=True):
                st.markdown(f"## {logloss:.3f}")
                st.caption("Log-loss")
                st.badge("Più basso = meglio", color="orange")

        st.caption(f"RPS e log-loss valutano quanto sono ben calibrate le probabilità, non solo se la previsione più probabile è quella giusta. Benchmark \"predici sempre 1\": {benchmark:.0%}.")

        st.space("medium")

        # --- Confusion matrix + per-class metrics row ---
        col_cm1, col_cm2 = st.columns([3, 2], gap="medium")

        with col_cm1:
            with st.container(border=True):
                st.markdown("**Matrice di confusione**")
                fig_cm = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=["1 (Casa)", "X (Pareggio)", "2 (Trasferta)"],
                    y=["1 (Casa)", "X (Pareggio)", "2 (Trasferta)"],
                    text=cm,
                    texttemplate="%{text}",
                    textfont={"size": 16},
                    colorscale="Greens"
                ))
                fig_cm.update_layout(
                    height=380, xaxis_title="Predetto", yaxis_title="Reale",
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#E6EDF3')
                )
                st.plotly_chart(fig_cm, width='stretch')

        with col_cm2:
            with st.container(border=True):
                st.markdown("**Metriche per classe**")
                metrics_df = pd.DataFrame({
                    "Classe": ["1 (Casa)", "X (Pareggio)", "2 (Trasferta)"],
                    "Precision": [f"{p:.1%}" for p in precision],
                    "Recall": [f"{r:.1%}" for r in recall],
                    "F1": [f"{f:.1%}" for f in f1],
                    "N": support
                })
                st.dataframe(metrics_df, hide_index=True, width="stretch")

            with st.container(border=True):
                st.markdown("**Precision / Recall / F1**")
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Bar(name="Precision", x=["1", "X", "2"], y=precision,
                                        marker_color="#00E676"))
                fig_pr.add_trace(go.Bar(name="Recall", x=["1", "X", "2"], y=recall,
                                        marker_color="#448AFF"))
                fig_pr.add_trace(go.Bar(name="F1", x=["1", "X", "2"], y=f1,
                                        marker_color="#FF9100"))
                fig_pr.update_layout(
                    height=280, barmode='group',
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#E6EDF3'), margin=dict(l=10, r=10, t=10, b=10)
                )
                st.plotly_chart(fig_pr, width='stretch')

        # Accuratezza per stagione (solo se multistagione)
        if n_stagioni_test > 1:
            st.space("medium")
            with st.container(border=True):
                st.markdown("**Accuratezza per stagione**")
                df_ris = pd.DataFrame({"Stagione": stagioni_pred, "corretta": [p == r for p, r in zip(predizioni, reali)]})
                breakdown = df_ris.groupby("Stagione")["corretta"].mean().reset_index()
                breakdown.columns = ["Stagione", "Accuratezza"]
                breakdown["Accuratezza"] = breakdown["Accuratezza"].apply(lambda x: f"{x:.1%}")
                st.dataframe(breakdown, hide_index=True, width="stretch")

# ------------------------------------------------------------
# CONFRONTO TRA CONFIGURAZIONI
# ------------------------------------------------------------
if st.sidebar.button(":material/compare_arrows: Confronta configurazioni", width="stretch"):
    configurazioni = [
        ("Solo storico", 0.0, 0.0, 0.0, 0.0),
        ("Storico + Forma", 0.5, 0.0, 0.0, 0.0),
        ("Primo default (storico+forma+scontri+quote)", 0.5, 0.15, 0.15, 0.0),
        ("Solo quote bookmaker", 0.0, 0.0, 1.0, 0.0),
        ("Secondo default (forma=0,scontri=0.15,quote=0.85)", 0.0, 0.15, 0.85, 0.0),
        ("Ottimale validato su 3 stagioni (senza Elo)", 0.10, 0.0, 0.90, 0.0),
        ("Solo Elo (ClubElo.com)", 0.0, 0.0, 0.0, 1.0),
        ("Ottimale + Elo (pesi correnti della sidebar)", peso_forma_bt, peso_scontri_bt, peso_quote_bt, peso_elo_bt),
    ]
    with st.spinner(":material/hourglass_top: Calcolo componenti una sola volta per tutte le configurazioni..."):
        componenti = precompute_tutte(emivita_giorni_bt, n_partite_forma, mostra_progress=True)

    risultati_confronto = []
    for nome, pf, ps, pq, pe in configurazioni:
        predizioni_c, reali_c, _, probabilita_c = valuta_tutte(componenti, pf, ps, pq, rho_bt, pe)
        rps_c = np.mean([rps({"1": p[0], "X": p[2], "2": p[1]}, r) for p, r in zip(probabilita_c, reali_c)])
        risultati_confronto.append({
            "Configurazione": nome,
            "Accuratezza": accuracy_score(reali_c, predizioni_c),
            "RPS medio": rps_c,
        })

    st.space("medium")
    st.markdown("### Confronto configurazioni del modello")

    df_confronto = pd.DataFrame(risultati_confronto)
    fig_confronto = go.Figure(go.Bar(
        x=df_confronto["Configurazione"], y=df_confronto["Accuratezza"],
        text=[f"{a:.1%}" for a in df_confronto["Accuratezza"]], textposition="outside",
        marker_color=["#448AFF", "#00E676", "#B388FF", "#FF9100", "#8B949E", "#00BCD4", "#FF1744", "#FFD600"]
    ))
    fig_confronto.update_layout(
        height=400, yaxis_tickformat=".0%", yaxis_title="Accuratezza 1X2",
        yaxis_range=[0, 1], paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E6EDF3')
    )
    st.plotly_chart(fig_confronto, width='stretch')
    df_confronto_display = df_confronto.copy()
    df_confronto_display["Accuratezza"] = df_confronto_display["Accuratezza"].apply(lambda x: f"{x:.1%}")
    df_confronto_display["RPS medio"] = df_confronto_display["RPS medio"].apply(lambda x: f"{x:.3f}")
    st.dataframe(df_confronto_display, hide_index=True, width="stretch")

st.space("large")
st.caption(":material/bar_chart: Backtesting walk-forward: le stagioni di test sono sempre le più recenti, quelle precedenti sono usate per l'addestramento.", text_alignment="center")
