"""
Pagina del pronostico su una singola partita.

Solo interfaccia: il modello vive in pronostico.py, cosi' che i test possano
importarlo senza far girare Streamlit. Fino alla costruzione della pagina
Schedina questa era la homepage (app.py).
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from pronostico import (df, stats, media_gol_casa, media_gol_trasferta, vantaggio_casa,
                        calcola_forma, scontri_diretti, stima_probabilita)

st.set_page_config(
    page_title="PredictA — Pronostico partita",
    page_icon=":material/sports_soccer:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
with st.sidebar:
    st.logo("serie_a_logo.svg", size="large")
    st.markdown("### :material/tune: Impostazioni modello")

    with st.container(border=True):
        st.markdown("**Statistiche campionato**")
        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.metric("Gol casa", f"{media_gol_casa:.2f}")
        col_s2.metric("Gol trasferta", f"{media_gol_trasferta:.2f}")
        col_s3.metric("Vantaggio casa", f"+{((vantaggio_casa-1)*100):.0f}%")

    with st.container(border=True):
        st.markdown("**Pesi del modello**")
        peso_forma = st.slider("Forma recente", 0.0, 1.0, 0.0, 0.05,
                           help="Default 0: misurato su 12.421 partite di 5 campionati, ogni peso sopra 0.02 peggiora "
                                "la calibrazione in modo statisticamente significativo, e il danno cresce in modo "
                                "monotono. Resta regolabile per esplorare, ma l'ottimo misurato e' zero.")
        peso_scontri = st.slider("Scontri diretti", 0.0, 0.5, 0.0, 0.05,
                           help="Su 3 stagioni di backtest non aggiunge valore misurabile una volta pesate bene le quote: default a 0.")
        peso_quote = st.slider("Quote bookmaker", 0.0, 1.0, 1.0, 0.05,
                           help="Peso della quota di consenso (chiusura quando disponibile, convertita con Shin). "
                                "Default 1.0: su 12.421 partite di 5 campionati il mercato da solo batte qualunque "
                                "miscela con il modello statistico.")
        peso_storico = 1 - peso_forma - peso_scontri - peso_quote

        if peso_storico < 0:
            st.error(":material/error: La somma dei pesi supera il 100%")
        else:
            st.caption(f"Peso storico: **{peso_storico:.0%}**")
            fig_pesi = go.Figure(go.Bar(
                x=[peso_storico, peso_forma, peso_scontri],
                y=["Pesi"], orientation='h',
                marker_color=['#448AFF', '#00E676', '#FF1744'],
                text=[f"Storico {peso_storico:.0%}", f"Forma {peso_forma:.0%}", f"Scontri {peso_scontri:.0%}"],
                textposition='inside', insidetextanchor='middle', textfont=dict(size=10, color='white')
            ))
            fig_pesi.update_layout(height=80, margin=dict(l=0, r=0, t=0, b=0), barmode='stack', showlegend=False,
                                   xaxis=dict(range=[0, 1], showticklabels=False), yaxis=dict(showticklabels=False),
                                   paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_pesi, width='stretch')

    st.caption(f":material/bar_chart: Partite: **{len(df):,}** | Squadre: **{df['HomeTeam'].nunique()}**")

# ------------------------------------------------------------
# INTERFACCIA PRINCIPALE
# ------------------------------------------------------------
st.title("PredictA — Pronostici Serie A", text_alignment="center")
st.markdown("Modello predittivo basato su Dixon-Coles, forma recente e quote dei bookmaker", text_alignment="center")

st.space("medium")

# Team selector row
lista_squadre = sorted(stats["Squadra"].unique().tolist())
col1, col2, col3 = st.columns([2, 2, 1], vertical_alignment="bottom")
with col1:
    squadra_casa = st.selectbox(":material/home: Squadra in casa", lista_squadre, key="home")
with col2:
    squadra_trasferta = st.selectbox(":material/directions_bus: Squadra in trasferta", lista_squadre, key="away")
with col3:
    calcola = st.button(":material/calculate: Calcola", width="stretch", type="primary")

if calcola:
    if squadra_casa == squadra_trasferta:
        st.warning(":material/warning: Scegli due squadre diverse!")
    elif peso_storico < 0:
        st.error(":material/error: La somma dei pesi supera il 100%")
    else:
        risultato = stima_probabilita(df, stats, squadra_casa, squadra_trasferta,
                              peso_forma, peso_scontri, peso_quote)
        if risultato is None:
            st.error(":material/error: Dati insufficienti per il calcolo.")
        else:
            st.space("medium")

            # --- Probability cards ---
            col_c1, col_c2, col_c3 = st.columns(3, gap="medium")

            with col_c1:
                with st.container(border=True):
                    st.markdown(f"**:material/home: {squadra_casa}**")
                    st.markdown(f"## {risultato['p_1']:.1%}")
                    st.badge("1", color="green")

            with col_c2:
                with st.container(border=True):
                    st.markdown("**:material/handshake: Pareggio**")
                    st.markdown(f"## {risultato['p_X']:.1%}")
                    st.badge("X", color="gray")

            with col_c3:
                with st.container(border=True):
                    st.markdown(f"**:material/directions_bus: {squadra_trasferta}**")
                    st.markdown(f"## {risultato['p_2']:.1%}")
                    st.badge("2", color="red")

            st.space("medium")

            # --- xG + Over/Under row ---
            col_g1, col_g2 = st.columns(2, gap="medium")

            with col_g1:
                with st.container(border=True):
                    st.markdown("**Gol attesi (xG)**")
                    met1, met2 = st.columns(2)
                    met1.metric(
                        squadra_casa,
                        f"{risultato['xG_casa']:.2f}",
                        help="Gol attesi per la squadra di casa"
                    )
                    met2.metric(
                        squadra_trasferta,
                        f"{risultato['xG_trasferta']:.2f}",
                        help="Gol attesi per la squadra in trasferta"
                    )

            with col_g2:
                with st.container(border=True):
                    st.markdown("**Over / Under**")
                    if risultato['quote_presenti'] and peso_quote >= 1.0:
                        st.caption(":material/info: Con le quote al 100% il pronostico 1X2 viene interamente dal "
                                   "mercato, mentre gol attesi, risultati esatti e Over/Under restano calcolati "
                                   "dal modello statistico: le due parti possono non concordare.")
                    st.metric("Gol totali attesi", f"{risultato['gol_totali_attesi']:.2f}")
                    st.markdown(
                        f":green-badge[Over 1.5 {risultato['over_15']:.0%}] "
                        f":orange-badge[Over 2.5 {risultato['over_25']:.0%}] "
                        f":blue-badge[Under 2.5 {risultato['under_25']:.0%}]"
                    )

            # --- Quote effect expander ---
            if risultato['quote_presenti']:
                st.info(":material/analytics: Quote bookmaker disponibili per questo match — il modello le sta usando.", icon=":material/analytics:")
                with st.expander(":material/compare_arrows: Vedi effetto delle quote"):
                    col_q1, col_q2, col_q3 = st.columns(3)
                    col_q1.metric(f"1 ({squadra_casa}) senza quote", f"{risultato['p_1_base']:.1%}")
                    col_q2.metric("X senza quote", f"{risultato['p_X_base']:.1%}")
                    col_q3.metric(f"2 ({squadra_trasferta}) senza quote", f"{risultato['p_2_base']:.1%}")

            st.space("medium")

            # --- Exact scores + Form row ---
            col_re1, col_re2 = st.columns(2, gap="medium")

            with col_re1:
                with st.container(border=True):
                    st.markdown("**Risultati esatti più probabili**")
                    for re, prob in risultato['top_risultati']:
                        st.markdown(f":material/sports_score: **{re}** — {prob:.1%}")

            with col_re2:
                with st.container(border=True):
                    st.markdown("**Forma recente**")
                    fatti_c, subiti_c, _, _, risultati_c, _ = calcola_forma(df, squadra_casa)
                    fatti_t, subiti_t, _, _, risultati_t, _ = calcola_forma(df, squadra_trasferta)
                    col_f1, col_f2 = st.columns(2)
                    with col_f1:
                        st.markdown(f"**:material/home: {squadra_casa}**")
                        if fatti_c is not None:
                            pallini = "".join([":green[●]" if r=="V" else ":orange[●]" if r=="N" else ":red[●]" for r in risultati_c])
                            st.markdown(f"Forma: {pallini}")
                    with col_f2:
                        st.markdown(f"**:material/directions_bus: {squadra_trasferta}**")
                        if fatti_t is not None:
                            pallini = "".join([":green[●]" if r=="V" else ":orange[●]" if r=="N" else ":red[●]" for r in risultati_t])
                            st.markdown(f"Forma: {pallini}")

            st.space("medium")

            # --- Head-to-head ---
            with st.container(border=True):
                st.markdown("**Ultimi scontri diretti**")
                scontri = risultato['scontri']
                if scontri is not None and scontri[0] is not None:
                    _, _, v1, pareggi_s, v2, tabella_scontri = scontri
                    st.markdown(f"{squadra_casa} **{v1}** — {pareggi_s} — **{v2}** {squadra_trasferta}")
                    st.dataframe(
                        tabella_scontri[["HomeTeam", "AwayTeam", "FTHG", "FTAG"]]
                        .rename(columns={"HomeTeam": "Casa", "AwayTeam": "Trasferta", "FTHG": "Gol C", "FTAG": "Gol T"}),
                        hide_index=True, width="stretch"
                    )
                else:
                    st.caption(":material/info: Nessuno scontro diretto disponibile.")

st.space("large")
st.caption(":material/sports_soccer: PredictA — Modello predittivo a scopo dimostrativo. I dati provengono da fonti pubbliche.", text_alignment="center")