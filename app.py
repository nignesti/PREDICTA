import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ------------------------------------------------------------
# CONFIGURAZIONE PAGINA
# ------------------------------------------------------------
st.set_page_config(
    page_title="Pronostici Serie A",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; text-align: center; margin-bottom: 2rem; }
    .probability-card { border-radius: 15px; padding: 1.5rem; color: white; text-align: center; margin: 0.5rem; }
    .probability-card.home { background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); }
    .probability-card.draw { background: linear-gradient(135deg, #8e9eab 0%, #eef2f3 100%); color: #333; }
    .probability-card.away { background: linear-gradient(135deg, #c31432 0%, #240b36 100%); }
    .big-number { font-size: 2rem; font-weight: 800; }
    .section-title { font-size: 1.3rem; font-weight: 600; margin-top: 1.5rem; margin-bottom: 0.8rem; border-bottom: 2px solid #e0e0e0; padding-bottom: 0.3rem; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# DATI
# ------------------------------------------------------------
DATA_FILE = "serie_a.csv"

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
    df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
    df = df.dropna(subset=["FTHG", "FTAG"])
    return df

df = load_data(DATA_FILE)
media_gol_casa = df["FTHG"].mean()
media_gol_trasferta = df["FTAG"].mean()
vantaggio_casa = media_gol_casa / media_gol_trasferta

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
with st.sidebar:
    st.image("serie_a_logo.svg", width=120)
    st.markdown("### ⚙️ Impostazioni Modello")

    st.markdown("---")
    st.markdown("#### 🏟️ Statistiche Campionato")
    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("Gol Casa", f"{media_gol_casa:.2f}")
    col_s2.metric("Gol Trasferta", f"{media_gol_trasferta:.2f}")
    col_s3.metric("Vantaggio", f"+{((vantaggio_casa-1)*100):.0f}%")

    st.markdown("---")
    st.markdown("#### 🎚️ Pesi del Modello")
    peso_forma = st.slider("Forma recente", 0.0, 1.0, 0.50, 0.05)
    peso_scontri = st.slider("Scontri diretti", 0.0, 0.5, 0.15, 0.05)
    peso_quote = st.slider("Quote bookmaker", 0.0, 0.4, 0.15, 0.05,
                       help="Peso delle quote storiche Bet365/Pinnacle")
    peso_storico = 1 - peso_forma - peso_scontri - peso_quote

    if peso_storico < 0:
        st.error("⚠️ Somma pesi > 100%")
    else:
        st.caption(f"Peso storico: **{peso_storico:.0%}**")
        fig_pesi = go.Figure(go.Bar(
            x=[peso_storico, peso_forma, peso_scontri],
            y=["Pesi"], orientation='h',
            marker_color=['#3498db', '#2ecc71', '#e74c3c'],
            text=[f"Storico {peso_storico:.0%}", f"Forma {peso_forma:.0%}", f"Scontri {peso_scontri:.0%}"],
            textposition='inside', insidetextanchor='middle', textfont=dict(size=10)
        ))
        fig_pesi.update_layout(height=80, margin=dict(l=0, r=0, t=0, b=0), barmode='stack', showlegend=False,
                               xaxis=dict(range=[0, 1], showticklabels=False), yaxis=dict(showticklabels=False))
        st.plotly_chart(fig_pesi, width='stretch')

    st.markdown("---")
    st.caption(f"Partite: **{len(df):,}** | Squadre: **{df['HomeTeam'].nunique()}**")

# ------------------------------------------------------------
# FUNZIONI DI CALCOLO (condivise)
# ------------------------------------------------------------
def calcola_forma(df, squadra, ultime_n=5):
    casa = df[df["HomeTeam"] == squadra].tail(ultime_n)
    trasferta = df[df["AwayTeam"] == squadra].tail(ultime_n)
    if len(casa) == 0 and len(trasferta) == 0:
        return None, None, None, None, [], []

    # Unisce casa e trasferta in ordine cronologico prima di tagliare le ultime N,
    # altrimenti (essendo trasferta aggiunta per seconda) la coda prenderebbe solo partite in trasferta.
    partite = pd.concat([casa, trasferta]).sort_index().tail(ultime_n)

    gol_fatti, gol_subiti, risultati = [], [], []
    for _, row in partite.iterrows():
        if row["HomeTeam"] == squadra:
            gol_fatti.append(row["FTHG"]); gol_subiti.append(row["FTAG"])
            risultati.append("V" if row["FTHG"] > row["FTAG"] else ("N" if row["FTHG"] == row["FTAG"] else "P"))
        else:
            gol_fatti.append(row["FTAG"]); gol_subiti.append(row["FTHG"])
            risultati.append("V" if row["FTAG"] > row["FTHG"] else ("N" if row["FTAG"] == row["FTHG"] else "P"))

    media_fatti = np.mean(gol_fatti) if gol_fatti else 0
    media_subiti = np.mean(gol_subiti) if gol_subiti else 0
    media_fatti_casa = casa["FTHG"].mean() if len(casa) > 0 else 0
    media_fatti_trasferta = trasferta["FTAG"].mean() if len(trasferta) > 0 else 0
    return media_fatti, media_subiti, media_fatti_casa, media_fatti_trasferta, risultati, gol_fatti

def scontri_diretti(df, squadra1, squadra2, ultimi_n=10):
    scontri = df[((df["HomeTeam"] == squadra1) & (df["AwayTeam"] == squadra2)) |
                 ((df["HomeTeam"] == squadra2) & (df["AwayTeam"] == squadra1))].tail(ultimi_n)
    if len(scontri) == 0:
        return None, None, None, None, None, None
    gol_fatti_s1, gol_subiti_s1 = [], []
    vittorie_s1, pareggi, vittorie_s2 = 0, 0, 0
    for _, row in scontri.iterrows():
        if row["HomeTeam"] == squadra1:
            gol_fatti_s1.append(row["FTHG"]); gol_subiti_s1.append(row["FTAG"])
            if row["FTHG"] > row["FTAG"]: vittorie_s1 += 1
            elif row["FTHG"] == row["FTAG"]: pareggi += 1
            else: vittorie_s2 += 1
        else:
            gol_fatti_s1.append(row["FTAG"]); gol_subiti_s1.append(row["FTHG"])
            if row["FTAG"] > row["FTHG"]: vittorie_s1 += 1
            elif row["FTAG"] == row["FTHG"]: pareggi += 1
            else: vittorie_s2 += 1
    return np.mean(gol_fatti_s1), np.mean(gol_subiti_s1), vittorie_s1, pareggi, vittorie_s2, scontri

home_stats = df.groupby("HomeTeam").agg(
    gol_fatti_casa_storico=("FTHG", "mean"), gol_subiti_casa_storico=("FTAG", "mean")).reset_index()
away_stats = df.groupby("AwayTeam").agg(
    gol_fatti_trasferta_storico=("FTAG", "mean"), gol_subiti_trasferta_storico=("FTHG", "mean")).reset_index()
stats = pd.merge(home_stats, away_stats, left_on="HomeTeam", right_on="AwayTeam", how="outer")
stats = stats.rename(columns={"HomeTeam": "Squadra"}).fillna(0)

def stima_probabilita(df, stats, squadra_casa, squadra_trasferta,
                      peso_forma=0.5, peso_scontri=0.15, peso_quote=0.20):
    """
    Combina: storico + forma + scontri diretti + quote dei bookmaker
    """
    casa = stats[stats["Squadra"] == squadra_casa]
    trasferta = stats[stats["Squadra"] == squadra_trasferta]
    if casa.empty or trasferta.empty:
        return None

    # --- STORICO ---
    # Forza attacco/difesa relativa alla media di campionato (stile Poisson classico),
    # non una media semplice: mediare ripetutamente con media_gol_casa/trasferta annulla
    # quasi ogni differenza tra squadre e fa collassare il modello su "vince sempre la casa".
    attacco_casa_storico = casa["gol_fatti_casa_storico"].values[0]
    difesa_casa_storico = casa["gol_subiti_casa_storico"].values[0]
    attacco_trasf_storico = trasferta["gol_fatti_trasferta_storico"].values[0]
    difesa_trasf_storico = trasferta["gol_subiti_trasferta_storico"].values[0]

    xG_casa_storico = (attacco_casa_storico / media_gol_casa) * (difesa_trasf_storico / media_gol_trasferta) * media_gol_casa
    xG_trasf_storico = (attacco_trasf_storico / media_gol_trasferta) * (difesa_casa_storico / media_gol_casa) * media_gol_trasferta

    # --- FORMA ---
    fatti_casa_forma, subiti_casa_forma, fatti_casa_home, _, _, _ = calcola_forma(df, squadra_casa)
    fatti_trasf_forma, subiti_trasf_forma, _, fatti_trasf_away, _, _ = calcola_forma(df, squadra_trasferta)

    if fatti_casa_forma is None:
        fatti_casa_forma, subiti_casa_forma, fatti_casa_home = attacco_casa_storico, difesa_casa_storico, attacco_casa_storico
    if fatti_trasf_forma is None:
        fatti_trasf_forma, subiti_trasf_forma, fatti_trasf_away = attacco_trasf_storico, difesa_trasf_storico, attacco_trasf_storico

    if fatti_casa_home > 0:
        xG_casa_forma = (fatti_casa_home / media_gol_casa) * (max(subiti_trasf_forma, 0.3) / media_gol_trasferta) * media_gol_casa
    else:
        xG_casa_forma = xG_casa_storico
    if fatti_trasf_away > 0:
        xG_trasf_forma = (fatti_trasf_away / media_gol_trasferta) * (max(subiti_casa_forma, 0.3) / media_gol_casa) * media_gol_trasferta
    else:
        xG_trasf_forma = xG_trasf_storico

    # --- SCONTRI DIRETTI ---
    media_gol_generale = (media_gol_casa + media_gol_trasferta) / 2
    scontri = scontri_diretti(df, squadra_casa, squadra_trasferta, ultimi_n=10)
    if scontri[0] is not None:
        gol_fatti_scontri, gol_subiti_scontri, _, _, _, _ = scontri
        xG_casa_scontri = (gol_fatti_scontri / media_gol_generale) * media_gol_casa
        xG_trasf_scontri = (gol_subiti_scontri / media_gol_generale) * media_gol_trasferta
    else:
        xG_casa_scontri, xG_trasf_scontri = xG_casa_storico, xG_trasf_storico
        peso_scontri = 0

    # --- QUOTE BOOKMAKER (NOVITÀ) ---
    # Prendiamo le quote medie degli ultimi scontri diretti (se disponibili)
    quote_presenti = False
    prob_1_quote, prob_X_quote, prob_2_quote = 0, 0, 0

    if "B365H" in df.columns and scontri[0] is not None:
        _, _, _, _, _, tabella_scontri = scontri
        if "B365H" in tabella_scontri.columns:
            quote_valide = tabella_scontri.dropna(subset=["B365H", "B365D", "B365A"])
            if len(quote_valide) > 0:
                # Converti quote in probabilità implicite e fai la media
                prob_1_quote = (1 / quote_valide["B365H"]).mean()
                prob_X_quote = (1 / quote_valide["B365D"]).mean()
                prob_2_quote = (1 / quote_valide["B365A"]).mean()
                # Normalizza (rimuovi il margine del bookmaker)
                somma = prob_1_quote + prob_X_quote + prob_2_quote
                prob_1_quote /= somma
                prob_X_quote /= somma
                prob_2_quote /= somma
                quote_presenti = True

    # --- COMBINAZIONE ---
    # I pesi ora sono: storico + forma + scontri + quote = 1
    peso_totale = peso_forma + peso_scontri + peso_quote
    if peso_totale > 1:
        # Normalizza se supera 1
        peso_forma /= peso_totale
        peso_scontri /= peso_totale
        peso_quote /= peso_totale
        peso_storico_final = 0
    else:
        peso_storico_final = 1 - peso_forma - peso_scontri - peso_quote

    if not quote_presenti:
        # Se non ci sono quote, ridistribuisci il peso
        peso_storico_final += peso_quote
        peso_quote = 0

    # Combinazione pesata per i gol attesi
    xG_casa = (peso_storico_final * xG_casa_storico +
               peso_forma * xG_casa_forma +
               peso_scontri * xG_casa_scontri)
    xG_trasferta = (peso_storico_final * xG_trasf_storico +
                    peso_forma * xG_trasf_forma +
                    peso_scontri * xG_trasf_scontri)

    # Il vantaggio campo è già incorporato sopra (ogni componente è scalata su
    # media_gol_casa o media_gol_trasferta), quindi qui non va riapplicato: raddoppiarlo
    # schiacciava il modello su "vince sempre la casa" a prescindere dalle squadre.

    # --- SIMULAZIONE MONTECARLO ---
    n_sim = 10000
    gol_casa_sim = np.random.poisson(max(0.1, xG_casa), n_sim)
    gol_trasferta_sim = np.random.poisson(max(0.1, xG_trasferta), n_sim)

    p_1_base = np.sum(gol_casa_sim > gol_trasferta_sim) / n_sim
    p_X_base = np.sum(gol_casa_sim == gol_trasferta_sim) / n_sim
    p_2_base = np.sum(gol_casa_sim < gol_trasferta_sim) / n_sim

    # Se abbiamo le quote, facciamo un blend finale
    if quote_presenti:
        p_1 = (1 - peso_quote) * p_1_base + peso_quote * prob_1_quote
        p_X = (1 - peso_quote) * p_X_base + peso_quote * prob_X_quote
        p_2 = (1 - peso_quote) * p_2_base + peso_quote * prob_2_quote
    else:
        p_1, p_X, p_2 = p_1_base, p_X_base, p_2_base

    # Risultati esatti
    risultati_esatti = {}
    for gc, gt in zip(gol_casa_sim, gol_trasferta_sim):
        chiave = f"{gc}-{gt}"
        risultati_esatti[chiave] = risultati_esatti.get(chiave, 0) + 1
    top_risultati = sorted(risultati_esatti.items(), key=lambda x: x[1], reverse=True)[:5]
    top_risultati = [(r, c / n_sim) for r, c in top_risultati]

    over_25 = np.sum((gol_casa_sim + gol_trasferta_sim) > 2.5) / n_sim
    over_15 = np.sum((gol_casa_sim + gol_trasferta_sim) > 1.5) / n_sim
    gol_totali_attesi = xG_casa + xG_trasferta

    return {
        "xG_casa": xG_casa,
        "xG_trasferta": xG_trasferta,
        "p_1": p_1,
        "p_X": p_X,
        "p_2": p_2,
        "p_1_base": p_1_base,  # senza quote
        "p_X_base": p_X_base,
        "p_2_base": p_2_base,
        "quote_presenti": quote_presenti,
        "prob_1_quote": prob_1_quote if quote_presenti else None,
        "prob_X_quote": prob_X_quote if quote_presenti else None,
        "prob_2_quote": prob_2_quote if quote_presenti else None,
        "top_risultati": top_risultati,
        "over_25": over_25,
        "under_25": 1 - over_25,
        "over_15": over_15,
        "gol_totali_attesi": gol_totali_attesi,
        "scontri": scontri
    }

# ------------------------------------------------------------
# INTERFACCIA PRINCIPALE
# ------------------------------------------------------------
st.markdown('<p class="main-header">🔮 Pronostici Serie A</p>', unsafe_allow_html=True)
st.markdown("---")

lista_squadre = sorted(stats["Squadra"].unique().tolist())
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    squadra_casa = st.selectbox("🏠 Squadra in casa", lista_squadre, key="home")
with col2:
    squadra_trasferta = st.selectbox("🚌 Squadra in trasferta", lista_squadre, key="away")
with col3:
    st.write(""); st.write("")
    calcola = st.button("📊 CALCOLA", width='stretch', type="primary")

if calcola:
    if squadra_casa == squadra_trasferta:
        st.warning("⚠️ Scegli due squadre diverse!")
    elif peso_storico < 0:
        st.error("⚠️ Somma pesi > 1.")
    else:
        risultato = stima_probabilita(df, stats, squadra_casa, squadra_trasferta,
                              peso_forma, peso_scontri, peso_quote)
        if risultato is None:
            st.error("Dati insufficienti.")
        else:
            st.markdown("---")
            col_c1, col_c2, col_c3 = st.columns(3)
            with col_c1:
                st.markdown(f"""<div class="probability-card home"><div>Vittoria {squadra_casa}</div><div class="big-number">{risultato['p_1']:.1%}</div><div>1</div></div>""", unsafe_allow_html=True)
            with col_c2:
                st.markdown(f"""<div class="probability-card draw"><div>Pareggio</div><div class="big-number">{risultato['p_X']:.1%}</div><div>X</div></div>""", unsafe_allow_html=True)
            with col_c3:
                st.markdown(f"""<div class="probability-card away"><div>Vittoria {squadra_trasferta}</div><div class="big-number">{risultato['p_2']:.1%}</div><div>2</div></div>""", unsafe_allow_html=True)

            col_g1, col_g2 = st.columns(2)
            with col_g1:
                st.markdown('<p class="section-title">⚽ Gol Attesi</p>', unsafe_allow_html=True)
                met1, met2 = st.columns(2)
                met1.metric(squadra_casa, f"{risultato['xG_casa']:.2f}")
                met2.metric(squadra_trasferta, f"{risultato['xG_trasferta']:.2f}")
            with col_g2:
                st.markdown('<p class="section-title">📊 Over / Under</p>', unsafe_allow_html=True)
                st.metric("Gol totali attesi", f"{risultato['gol_totali_attesi']:.2f}")
                st.markdown(f"Over 1.5: **{risultato['over_15']:.1%}** | Over 2.5: **{risultato['over_25']:.1%}** | Under 2.5: **{risultato['under_25']:.1%}**")

            # 👇 NUOVO BLOCCO QUOTE
            if risultato['quote_presenti']:
                st.info("📊 Quote bookmaker disponibili per questo match! Il modello le sta usando.")
                with st.expander("🔍 Vedi effetto quote"):
                    col_q1, col_q2, col_q3 = st.columns(3)
                    col_q1.metric("1 (senza quote)", f"{risultato['p_1_base']:.1%}")
                    col_q2.metric("X (senza quote)", f"{risultato['p_X_base']:.1%}")
                    col_q3.metric("2 (senza quote)", f"{risultato['p_2_base']:.1%}")
            # 👆 FINE NUOVO BLOCCO

            st.markdown("---")
            col_re1, col_re2 = st.columns(2)
            with col_re1:
                st.markdown('<p class="section-title">🎯 Risultati Esatti più Probabili</p>', unsafe_allow_html=True)
                for re, prob in risultato['top_risultati']:
                    st.markdown(f"**{re}** — {prob:.1%}")
            with col_re2:
                st.markdown('<p class="section-title">📈 Forma Recente</p>', unsafe_allow_html=True)
                fatti_c, subiti_c, _, _, risultati_c, _ = calcola_forma(df, squadra_casa)
                fatti_t, subiti_t, _, _, risultati_t, _ = calcola_forma(df, squadra_trasferta)
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    st.write(f"**{squadra_casa}**")
                    if fatti_c is not None:
                        pallini = "".join(["🟢" if r=="V" else "🟡" if r=="N" else "🔴" for r in risultati_c])
                        st.write(f"Forma: {pallini}")
                with col_f2:
                    st.write(f"**{squadra_trasferta}**")
                    if fatti_t is not None:
                        pallini = "".join(["🟢" if r=="V" else "🟡" if r=="N" else "🔴" for r in risultati_t])
                        st.write(f"Forma: {pallini}")

            st.markdown("---")
            st.markdown('<p class="section-title">⚔️ Ultimi Scontri Diretti</p>', unsafe_allow_html=True)
            scontri = risultato['scontri']
            if scontri is not None and scontri[0] is not None:
                _, _, v1, pareggi_s, v2, tabella_scontri = scontri
                st.write(f"{squadra_casa} {v1} - {pareggi_s} - {v2} {squadra_trasferta}")
                st.dataframe(tabella_scontri[["HomeTeam", "AwayTeam", "FTHG", "FTAG"]], hide_index=True)
            else:
                st.info("Nessuno scontro diretto.")

st.markdown("---")
st.caption("⚽ Modello predittivo a scopo dimostrativo.")