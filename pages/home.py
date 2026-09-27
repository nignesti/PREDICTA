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

_, col_a, col_b, _ = st.columns([1, 2, 2, 1], gap="large")

with col_a:
    with st.container(border=True, height=220):
        st.markdown("### :material/sports_soccer: Serie A")
        st.caption("Costruttore di schedina 1X2 / Over-Under, calibrato su quote di mercato e "
                   "statistiche storiche pesate nel tempo.")
        st.page_link("pages/1_Serie_A_Schedina.py", label="Entra in Serie A",
                     icon=":material/arrow_forward:", use_container_width=True)

with col_b:
    with st.container(border=True, height=220):
        st.markdown("### :material/sports_basketball: NBA")
        st.caption("Costruttore di schedina moneyline / spread / totale punti, con rating Elo "
                   "calibrato sullo storico ufficiale delle partite.")
        st.page_link("pages/2_NBA_Schedina.py", label="Entra in NBA",
                     icon=":material/arrow_forward:", use_container_width=True)
