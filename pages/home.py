"""
Home di PredictA: solo la scelta fra Serie A e NBA, senza altro contenuto.
Le due sezioni sono modelli separati (Dixon-Coles con mercato 1X2 per la
Serie A, rating Elo con mercati moneyline/spread/totale per l'NBA: l'NBA non
ha pareggi, quindi non condivide il modello a tre esiti del calcio), quindi
la scelta va fatta prima di mostrare qualunque controllo.
"""
import streamlit as st

st.set_page_config(
    page_title="PredictA",
    page_icon=":material/receipt_long:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.space("large")
st.title("PredictA", text_alignment="center")
st.markdown("Scegli il campionato per iniziare.", text_alignment="center")
st.space("large")

st.markdown(
    """
    <style>
    .st-key-card_serie_a, .st-key-card_nba {
        background: #18191c;
        border: 1px solid #323439;
        border-radius: 12px;
        padding: 24px;
        transition: all 0.2s ease;
    }
    .st-key-card_serie_a:hover { border-color: #0080ff; box-shadow: 0 0 20px rgba(0, 128, 255, 0.15); }
    .st-key-card_nba:hover { border-color: #ff6b00; box-shadow: 0 0 20px rgba(255, 107, 0, 0.15); }
    </style>
    """,
    unsafe_allow_html=True,
)

_, col_a, col_b, _ = st.columns([1, 2, 2, 1], gap="large")

with col_a:
    with st.container(key="card_serie_a"):
        st.markdown("### :material/sports_soccer: Serie A")
        st.caption("Costruttore di schedina 1X2 / Over-Under, calibrato su quote di mercato e "
                   "statistiche storiche pesate nel tempo.")
        st.page_link("pages/1_Serie_A_Schedina.py", label="Entra in Serie A",
                     icon=":material/arrow_forward:", use_container_width=True)

with col_b:
    with st.container(key="card_nba"):
        st.markdown("### :material/sports_basketball: NBA")
        st.caption("Costruttore di schedina moneyline / spread / totale punti, con rating Elo "
                   "calibrato sullo storico ufficiale delle partite.")
        st.page_link("pages/2_NBA_Schedina.py", label="Entra in NBA",
                     icon=":material/arrow_forward:", use_container_width=True)
