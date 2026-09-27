"""
Entrypoint di PredictA: solo la dichiarazione delle pagine e dei due gruppi di
navigazione (Serie A / NBA). Nessun contenuto qui — st.set_page_config() e
tutto il resto restano nelle singole pagine, esattamente come prima quando
ogni pagina della cartella pages/ lo faceva in autonomia: usare st.navigation()
qui disattiva la scoperta automatica di pages/, ma i file restano gli stessi.

pages/home.py e' la pagina di apertura (due pulsanti grandi, Serie A / NBA):
da li' si arriva alle sezioni con st.page_link, non c'e' altro contenuto in
home per non anticipare la scelta del campionato.
"""
import streamlit as st

home = st.Page("pages/home.py", title="Home", icon=":material/home:", default=True)

pagine_serie_a = [
    st.Page("pages/1_Serie_A_Schedina.py", title="Costruttore schedina", icon=":material/receipt_long:"),
    st.Page("pages/1_Pronostico.py", title="Pronostico singolo", icon=":material/query_stats:"),
    st.Page("pages/backtesting.py", title="Backtesting", icon=":material/bar_chart:"),
]

pagine_nba = [
    st.Page("pages/2_NBA_Schedina.py", title="Costruttore schedina", icon=":material/receipt_long:"),
]

navigazione = st.navigation({
    "": [home],
    "Serie A": pagine_serie_a,
    "NBA": pagine_nba,
})
navigazione.run()
