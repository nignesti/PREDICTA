import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

st.set_page_config(page_title="Backtesting", page_icon="📊", layout="wide")

# ------------------------------------------------------------
# CSS CORRETTO (testo bianco su sfondo colorato)
# ------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1.5rem;
        color: #1a1a1a;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        color: white;
        margin: 0.3rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card.green { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .metric-card.blue { background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); }
    .metric-card.orange { background: linear-gradient(135deg, #f12711 0%, #f5af19 100%); }
    .metric-card.purple { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: white;
    }
    .metric-label {
        font-size: 0.9rem;
        color: rgba(255,255,255,0.9);
        margin-top: 0.3rem;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 0.3rem;
        color: #1a1a1a;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">📊 Backtesting - Validazione del Modello</p>', unsafe_allow_html=True)

# ------------------------------------------------------------
# CARICA DATI
# ------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("serie_a.csv")
    df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
    df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
    df = df.dropna(subset=["FTHG", "FTAG"])
    return df

df = load_data()

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
st.sidebar.markdown("### ⚙️ Impostazioni Backtesting")

stagioni_disponibili = sorted(df["Stagione"].astype(str).unique())
n_stagioni_test = st.sidebar.slider(
    "Stagioni di test", 1, min(5, len(stagioni_disponibili) - 1), 1,
    help="Le ultime N stagioni vengono usate come test set (walk-forward), tutte le precedenti come training."
)
stagioni_test = stagioni_disponibili[-n_stagioni_test:]

peso_forma_bt = st.sidebar.slider("Peso forma", 0.0, 1.0, 0.0, 0.05,
                    help="Da grid search: la forma sulle ultime partite è troppo rumorosa e non aggiunge valore.")
peso_scontri_bt = st.sidebar.slider("Peso scontri", 0.0, 0.5, 0.15, 0.05)
n_partite_forma = st.sidebar.slider("Partite per forma", 3, 10, 5,
                    help="Ininfluente quando 'Peso forma' è 0.")
peso_quote_bt = st.sidebar.slider("Peso quote", 0.0, 1.0, 0.85, 0.05)

train_df = df[~df["Stagione"].astype(str).isin(stagioni_test)].copy()
test_df = df[df["Stagione"].astype(str).isin(stagioni_test)].copy()

st.sidebar.caption(f"Training: {len(train_df):,} partite")
st.sidebar.caption(f"Test: {len(test_df):,} partite ({', '.join(stagioni_test)})")

# ------------------------------------------------------------
# STATISTICHE SU TRAINING
# ------------------------------------------------------------
media_gol_casa = train_df["FTHG"].mean()
media_gol_trasferta = train_df["FTAG"].mean()
media_gol_generale = (media_gol_casa + media_gol_trasferta) / 2
vantaggio_casa = media_gol_casa / media_gol_trasferta

home_stats = train_df.groupby("HomeTeam").agg(
    gol_fatti_casa=("FTHG", "mean"), gol_subiti_casa=("FTAG", "mean")).reset_index()
away_stats = train_df.groupby("AwayTeam").agg(
    gol_fatti_trasferta=("FTAG", "mean"), gol_subiti_trasferta=("FTHG", "mean")).reset_index()
stats = pd.merge(home_stats, away_stats, left_on="HomeTeam", right_on="AwayTeam", how="outer")
stats = stats.rename(columns={"HomeTeam": "Squadra"}).fillna(0)

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

def predici_partita_bt(train_df, test_df, idx_test, stats, peso_forma, peso_scontri, peso_quote=0.15):
    riga = test_df.iloc[idx_test]
    casa = riga["HomeTeam"]
    trasferta = riga["AwayTeam"]

    c = stats[stats["Squadra"] == casa]
    t = stats[stats["Squadra"] == trasferta]
    if c.empty or t.empty:
        return None

    # --- STORICO (forza attacco/difesa relativa alla media di lega, come in app.py) ---
    xG_casa_storico = (c["gol_fatti_casa"].values[0] / media_gol_casa) * (t["gol_subiti_trasferta"].values[0] / media_gol_trasferta) * media_gol_casa
    xG_trasf_storico = (t["gol_fatti_trasferta"].values[0] / media_gol_trasferta) * (c["gol_subiti_casa"].values[0] / media_gol_casa) * media_gol_trasferta

    idx_globale = len(train_df) + idx_test
    df_fino_a_ora = pd.concat([train_df, test_df.iloc[:idx_test+1]])
    fatti_c, subiti_c, fatti_c_home, _ = calcola_forma_bt(df_fino_a_ora, casa, idx_globale, n_partite_forma)
    fatti_t, subiti_t, _, fatti_t_away = calcola_forma_bt(df_fino_a_ora, trasferta, idx_globale, n_partite_forma)

    if fatti_c_home > 0:
        xG_casa_forma = (fatti_c_home / media_gol_casa) * (max(subiti_t, 0.3) / media_gol_trasferta) * media_gol_casa
    else:
        xG_casa_forma = xG_casa_storico
    if fatti_t_away > 0:
        xG_trasf_forma = (fatti_t_away / media_gol_trasferta) * (max(subiti_c, 0.3) / media_gol_casa) * media_gol_trasferta
    else:
        xG_trasf_forma = xG_trasf_storico

    # --- SCONTRI DIRETTI (solo partite precedenti a quella corrente, niente lookahead) ---
    df_prima = df_fino_a_ora.iloc[:idx_globale]
    gol_fatti_scontri, gol_subiti_scontri = scontri_diretti_bt(df_prima, casa, trasferta, ultimi_n=10)
    if gol_fatti_scontri is not None:
        xG_casa_scontri = (gol_fatti_scontri / media_gol_generale) * media_gol_casa
        xG_trasf_scontri = (gol_subiti_scontri / media_gol_generale) * media_gol_trasferta
    else:
        xG_casa_scontri, xG_trasf_scontri = xG_casa_storico, xG_trasf_storico
        peso_scontri = 0

    # --- QUOTE BOOKMAKER: usa le quote reali pre-partita di questo fixture, se presenti ---
    quote_presenti = False
    prob_1_quote, prob_X_quote, prob_2_quote = 0, 0, 0
    if all(col in riga.index for col in ["B365H", "B365D", "B365A"]):
        b365h, b365d, b365a = riga["B365H"], riga["B365D"], riga["B365A"]
        if pd.notna(b365h) and pd.notna(b365d) and pd.notna(b365a):
            prob_1_quote, prob_X_quote, prob_2_quote = 1 / b365h, 1 / b365d, 1 / b365a
            somma = prob_1_quote + prob_X_quote + prob_2_quote
            prob_1_quote /= somma; prob_X_quote /= somma; prob_2_quote /= somma
            quote_presenti = True

    # --- COMBINAZIONE PESI (storico + forma + scontri per gli xG, quote in blend finale, come in app.py) ---
    peso_totale = peso_forma + peso_scontri + peso_quote
    if peso_totale > 1:
        peso_forma /= peso_totale
        peso_scontri /= peso_totale
        peso_quote /= peso_totale
        peso_storico = 0
    else:
        peso_storico = 1 - peso_forma - peso_scontri - peso_quote

    if not quote_presenti:
        peso_storico += peso_quote
        peso_quote = 0

    xG_casa = (peso_storico * xG_casa_storico +
               peso_forma * xG_casa_forma +
               peso_scontri * xG_casa_scontri)
    xG_trasferta = (peso_storico * xG_trasf_storico +
                    peso_forma * xG_trasf_forma +
                    peso_scontri * xG_trasf_scontri)

    # Il vantaggio campo è già incorporato sopra tramite media_gol_casa/media_gol_trasferta:
    # non va riapplicato, altrimenti si torna a "vince sempre la casa" a prescindere dalle squadre.

    n_sim = 5000
    gol_casa_sim = np.random.poisson(max(0.1, xG_casa), n_sim)
    gol_trasferta_sim = np.random.poisson(max(0.1, xG_trasferta), n_sim)

    p_1_base = np.sum(gol_casa_sim > gol_trasferta_sim) / n_sim
    p_X_base = np.sum(gol_casa_sim == gol_trasferta_sim) / n_sim
    p_2_base = np.sum(gol_casa_sim < gol_trasferta_sim) / n_sim

    if quote_presenti:
        p_1 = (1 - peso_quote) * p_1_base + peso_quote * prob_1_quote
        p_X = (1 - peso_quote) * p_X_base + peso_quote * prob_X_quote
        p_2 = (1 - peso_quote) * p_2_base + peso_quote * prob_2_quote
    else:
        p_1, p_X, p_2 = p_1_base, p_X_base, p_2_base

    return {"1": p_1, "X": p_X, "2": p_2, "pred": "1" if p_1 > p_X and p_1 > p_2 else ("X" if p_X > p_2 else "2")}

def esegui_backtest(peso_forma, peso_scontri, peso_quote, mostra_progress=False):
    predizioni, reali, stagioni_pred = [], [], []
    progress_bar = st.progress(0) if mostra_progress else None
    for i in range(len(test_df)):
        pred = predici_partita_bt(train_df, test_df, i, stats, peso_forma, peso_scontri, peso_quote)
        if pred is not None:
            riga = test_df.iloc[i]
            if riga["FTHG"] > riga["FTAG"]:
                reale = "1"
            elif riga["FTHG"] == riga["FTAG"]:
                reale = "X"
            else:
                reale = "2"
            predizioni.append(pred["pred"])
            reali.append(reale)
            stagioni_pred.append(str(riga["Stagione"]))
        if mostra_progress:
            progress_bar.progress((i + 1) / len(test_df))
    if mostra_progress:
        progress_bar.empty()
    return predizioni, reali, stagioni_pred

# ------------------------------------------------------------
# BOTTONE BACKTESTING
# ------------------------------------------------------------
if st.sidebar.button("🚀 Esegui Backtesting", width='stretch', type="primary"):
    with st.spinner("Simulando le previsioni... Questo potrebbe richiedere qualche minuto."):
        predizioni, reali, stagioni_pred = esegui_backtest(peso_forma_bt, peso_scontri_bt, peso_quote_bt, mostra_progress=True)

        # Metriche
        acc = accuracy_score(reali, predizioni)
        cm = confusion_matrix(reali, predizioni, labels=["1", "X", "2"])
        precision, recall, f1, support = precision_recall_fscore_support(reali, predizioni, labels=["1", "X", "2"])
        benchmark = reali.count("1") / len(reali)  # accuratezza di "predici sempre 1", calcolata sul test set reale

        # ------------------------------------------------------------
        # VISUALIZZA RISULTATI
        # ------------------------------------------------------------
        st.markdown("---")
        st.markdown("### 📊 Risultati Backtesting")

        # Box metriche con colori corretti
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.markdown(f"""<div class="metric-card green"><div class="metric-value">{acc:.1%}</div><div class="metric-label">Accuratezza 1X2</div></div>""", unsafe_allow_html=True)
        with col_m2:
            corrette = sum(1 for p, r in zip(predizioni, reali) if p == r)
            st.markdown(f"""<div class="metric-card blue"><div class="metric-value">{corrette}/{len(reali)}</div><div class="metric-label">Partite indovinate</div></div>""", unsafe_allow_html=True)
        with col_m3:
            f1_avg = np.mean(f1)
            st.markdown(f"""<div class="metric-card purple"><div class="metric-value">{f1_avg:.1%}</div><div class="metric-label">F1 Score medio</div></div>""", unsafe_allow_html=True)
        with col_m4:
            st.markdown(f"""<div class="metric-card orange"><div class="metric-value">{'✅' if acc > benchmark else '❌'}</div><div class="metric-label">vs Benchmark "sempre 1" ({benchmark:.0%})</div></div>""", unsafe_allow_html=True)

        # Matrice di confusione
        st.markdown("---")
        st.markdown("### 🔍 Matrice di Confusione")

        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=["1 (Casa)", "X (Pareggio)", "2 (Trasferta)"],
            y=["1 (Casa)", "X (Pareggio)", "2 (Trasferta)"],
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16},
            colorscale="Blues"
        ))
        fig_cm.update_layout(height=400, xaxis_title="Predetto", yaxis_title="Reale")
        st.plotly_chart(fig_cm, width='stretch')

        # Metriche per classe
        st.markdown("---")
        st.markdown("### 📋 Metriche per Classe")

        metrics_df = pd.DataFrame({
            "Classe": ["1 (Casa)", "X (Pareggio)", "2 (Trasferta)"],
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Supporto": support
        })
        st.dataframe(metrics_df, hide_index=True, width='stretch')

        # Grafico precision/recall
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Bar(name="Precision", x=["1", "X", "2"], y=precision, marker_color="#2a5298"))
        fig_pr.add_trace(go.Bar(name="Recall", x=["1", "X", "2"], y=recall, marker_color="#38ef7d"))
        fig_pr.add_trace(go.Bar(name="F1-Score", x=["1", "X", "2"], y=f1, marker_color="#f5af19"))
        fig_pr.update_layout(height=400, barmode='group')
        st.plotly_chart(fig_pr, width='stretch')

        # Accuratezza per stagione (solo se il test set copre più di una stagione)
        if n_stagioni_test > 1:
            st.markdown("---")
            st.markdown("### 📅 Accuratezza per Stagione")
            df_ris = pd.DataFrame({"Stagione": stagioni_pred, "corretta": [p == r for p, r in zip(predizioni, reali)]})
            breakdown = df_ris.groupby("Stagione")["corretta"].mean().reset_index()
            breakdown.columns = ["Stagione", "Accuratezza"]
            st.dataframe(breakdown.style.format({"Accuratezza": "{:.1%}"}), hide_index=True, width='stretch')

# ------------------------------------------------------------
# CONFRONTO TRA CONFIGURAZIONI FISSE DEL MODELLO
# ------------------------------------------------------------
st.sidebar.markdown("---")
if st.sidebar.button("🔬 Confronta configurazioni", width='stretch'):
    configurazioni = [
        ("Solo storico", 0.0, 0.0, 0.0),
        ("Storico + Forma", 0.5, 0.0, 0.0),
        ("Vecchio default (storico+forma+scontri+quote)", 0.5, 0.15, 0.15),
        ("Solo quote bookmaker", 0.0, 0.0, 1.0),
        ("Ottimale da grid search (nuovo default)", 0.0, 0.15, 0.85),
    ]
    risultati_confronto = []
    with st.spinner("Eseguo il backtest per le configurazioni di riferimento..."):
        for nome, pf, ps, pq in configurazioni:
            predizioni_c, reali_c, _ = esegui_backtest(pf, ps, pq)
            risultati_confronto.append({"Configurazione": nome, "Accuratezza": accuracy_score(reali_c, predizioni_c)})

    st.markdown("---")
    st.markdown("### 🔬 Confronto tra Configurazioni del Modello")
    df_confronto = pd.DataFrame(risultati_confronto)
    fig_confronto = go.Figure(go.Bar(
        x=df_confronto["Configurazione"], y=df_confronto["Accuratezza"],
        text=[f"{a:.1%}" for a in df_confronto["Accuratezza"]], textposition="outside",
        marker_color=["#3498db", "#2ecc71", "#9b59b6", "#e67e22"]
    ))
    fig_confronto.update_layout(height=400, yaxis_tickformat=".0%", yaxis_title="Accuratezza 1X2", yaxis_range=[0, 1])
    st.plotly_chart(fig_confronto, width='stretch')
    st.dataframe(df_confronto.style.format({"Accuratezza": "{:.1%}"}), hide_index=True, width='stretch')

st.markdown("---")
st.caption("📊 Backtesting walk-forward: le stagioni di test sono sempre le più recenti, quelle precedenti sono usate per l'addestramento.")
