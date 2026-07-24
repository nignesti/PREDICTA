import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from modello import stats_pesate_squadre, distribuzione_punteggi, esiti_da_matrice, probabilita_shin

# Iperparametri Dixon-Coles validati via backtest (vedi pages/backtesting.py):
# EMIVITA_GIORNI: dopo quanti giorni una partita storica pesa la metà nelle medie
#   squadra (decadimento esponenziale, invece di media semplice su 33 stagioni).
# RHO_DIXON_COLES: correzione per i punteggi bassi (0-0, 1-0, 0-1, 1-1), dove un
#   Poisson indipendente sottostima sistematicamente i pareggi.
EMIVITA_GIORNI = 730
RHO_DIXON_COLES = -0.10

# ------------------------------------------------------------
# CONFIGURAZIONE PAGINA
# ------------------------------------------------------------
st.set_page_config(
    page_title="PredictA — Pronostici Serie A",
    page_icon=":material/sports_soccer:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# DATI
# ------------------------------------------------------------
DATA_FILE = "serie_a.csv"

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
    df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
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
        peso_forma = st.slider("Forma recente", 0.0, 1.0, 0.10, 0.05,
                           help="Validato su 3 stagioni indipendenti: un peso piccolo aiuta, oltre 0.20-0.25 peggiora e diventa rumore.")
        peso_scontri = st.slider("Scontri diretti", 0.0, 0.5, 0.0, 0.05,
                           help="Su 3 stagioni di backtest non aggiunge valore misurabile una volta pesate bene le quote: default a 0.")
        peso_quote = st.slider("Quote bookmaker", 0.0, 1.0, 0.90, 0.05,
                           help="Peso delle quote storiche Bet365/Pinnacle. Pesi di default validati su 3 stagioni indipendenti di backtest.")
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

# Statistiche storiche pesate nel tempo (le partite recenti contano di più di
# quelle di 30 anni fa) invece di una media semplice su tutta la storia.
stats = stats_pesate_squadre(df, data_riferimento=df["Date"].max(), half_life_giorni=EMIVITA_GIORNI)

def stima_probabilita(df, stats, squadra_casa, squadra_trasferta,
                      peso_forma=0.10, peso_scontri=0.0, peso_quote=0.90):
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
    # Prendiamo le quote medie degli ultimi scontri diretti (se disponibili).
    # Preferiamo la quota di consenso multi-bookmaker (OddsAvg*) a Bet365 da solo:
    # meno rumore da un singolo book, copertura pressoché totale dal 2011 in poi.
    quote_presenti = False
    prob_1_quote, prob_X_quote, prob_2_quote = 0, 0, 0
    colonne_quota = ("OddsAvgH", "OddsAvgD", "OddsAvgA") if "OddsAvgH" in df.columns else ("B365H", "B365D", "B365A")

    if colonne_quota[0] in df.columns and scontri[0] is not None:
        _, _, _, _, _, tabella_scontri = scontri
        if colonne_quota[0] in tabella_scontri.columns:
            quote_valide = tabella_scontri.dropna(subset=list(colonne_quota))
            if len(quote_valide) > 0:
                # Converti quote in probabilità implicite e fai la media
                prob_1_quote = (1 / quote_valide[colonne_quota[0]]).mean()
                prob_X_quote = (1 / quote_valide[colonne_quota[1]]).mean()
                prob_2_quote = (1 / quote_valide[colonne_quota[2]]).mean()
                # Rimuovi il margine del bookmaker con la correzione di Shin
                # (1992/1993) invece della normalizzazione proporzionale
                # semplice: valida su 3 stagioni indipendenti in
                # pages/backtesting.py, migliora leggermente l'RPS (calibrazione)
                # senza peggiorare l'accuratezza. "Quote equivalenti" perché qui
                # partiamo da probabilità già mediate su più scontri diretti,
                # non dalle quote di una singola partita.
                quote_equivalenti = [1 / prob_1_quote, 1 / prob_X_quote, 1 / prob_2_quote]
                prob_1_quote, prob_X_quote, prob_2_quote = probabilita_shin(quote_equivalenti)
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

    # Combinazione pesata per i gol attesi: storico+forma+scontri vanno rinormalizzati
    # a sommare 1 tra loro, perché "quote" non entra qui (entra dopo, sulle probabilità
    # finali) — altrimenti con peso_quote alto (es. 0.85) i pesi restanti (es. 0.15 di
    # scontri) scalano l'xG verso il basso invece di usarlo per intero.
    peso_xg_totale = peso_storico_final + peso_forma + peso_scontri
    if peso_xg_totale > 0:
        xG_casa = (peso_storico_final * xG_casa_storico +
                   peso_forma * xG_casa_forma +
                   peso_scontri * xG_casa_scontri) / peso_xg_totale
        xG_trasferta = (peso_storico_final * xG_trasf_storico +
                        peso_forma * xG_trasf_forma +
                        peso_scontri * xG_trasf_scontri) / peso_xg_totale
    else:
        xG_casa, xG_trasferta = xG_casa_storico, xG_trasf_storico

    # Il vantaggio campo è già incorporato sopra (ogni componente è scalata su
    # media_gol_casa o media_gol_trasferta), quindi qui non va riapplicato: raddoppiarlo
    # schiacciava il modello su "vince sempre la casa" a prescindere dalle squadre.

    # --- DISTRIBUZIONE ESATTA DEI PUNTEGGI (Dixon-Coles) ---
    # Poisson indipendenti + correzione tau per i punteggi bassi, al posto della
    # simulazione Monte Carlo: stesso modello concettuale ma deterministico (niente
    # rumore campionario) e senza sottostimare sistematicamente i pareggi.
    matrice_punteggi = distribuzione_punteggi(xG_casa, xG_trasferta, rho=RHO_DIXON_COLES)
    esiti = esiti_da_matrice(matrice_punteggi)
    p_1_base, p_X_base, p_2_base = esiti["p_1"], esiti["p_X"], esiti["p_2"]

    # Se abbiamo le quote, facciamo un blend finale
    if quote_presenti:
        p_1 = (1 - peso_quote) * p_1_base + peso_quote * prob_1_quote
        p_X = (1 - peso_quote) * p_X_base + peso_quote * prob_X_quote
        p_2 = (1 - peso_quote) * p_2_base + peso_quote * prob_2_quote
    else:
        p_1, p_X, p_2 = p_1_base, p_X_base, p_2_base

    top_risultati = esiti["top_risultati"]
    over_25, over_15, under_25 = esiti["over_25"], esiti["over_15"], esiti["under_25"]
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