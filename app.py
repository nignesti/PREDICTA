"""
Entrypoint di PredictA: gate password, poi la dichiarazione delle pagine e dei
due gruppi di navigazione (Serie A / NBA). Nessun altro contenuto qui —
st.set_page_config() e tutto il resto restano nelle singole pagine, esattamente
come prima quando ogni pagina della cartella pages/ lo faceva in autonomia:
usare st.navigation() qui disattiva la scoperta automatica di pages/, ma i
file restano gli stessi.

pages/home.py e' la pagina di apertura (due pulsanti grandi, Serie A / NBA):
da li' si arriva alle sezioni con st.page_link, non c'e' altro contenuto in
home per non anticipare la scelta del campionato.

Il gate password sta qui e non in home.py perche' questo e' l'unico script
che Streamlit esegue sempre per primo: navigazione.run() e' cio' che poi
disegna la pagina scelta (home o una sezione), quindi bloccare prima di quella
chiamata blocca tutto il sito, non solo la home. Non e' un controllo di
sicurezza vero (la password sta in chiaro nel codice, come richiesto): serve
solo a evitare che chi trova il link ci finisca dentro per caso."""
import streamlit as st

PASSWORD = "bonfiga"


def _password_corretta():
    return st.session_state.get("_tentativo_password") == PASSWORD


if not st.session_state.get("autenticato", False):
    st.set_page_config(page_title="PredictA — Accesso", page_icon=":material/lock:", layout="centered")
    st.space("large")
    st.title("PredictA", text_alignment="center")
    st.markdown("Inserisci la password per continuare.", text_alignment="center")
    st.space("medium")

    _, colonna, _ = st.columns([1, 2, 1])
    with colonna:
        st.text_input("Password", type="password", key="_tentativo_password")
        if st.button("Entra", use_container_width=True):
            if _password_corretta():
                st.session_state.autenticato = True
                st.rerun()
            else:
                st.error(":material/error: Password errata.")
    st.stop()

home = st.Page("pages/home.py", title="Home", icon=":material/home:", default=True)

pagine_serie_a = [
    st.Page("pages/1_Serie_A_Schedina.py", title="Costruttore schedina", icon=":material/receipt_long:"),
    st.Page("pages/1_Pronostico.py", title="Pronostico singolo", icon=":material/query_stats:"),
    st.Page("pages/backtesting.py", title="Backtesting", icon=":material/bar_chart:"),
]

pagine_nba = [
    st.Page("pages/2_NBA_Schedina.py", title="Costruttore schedina", icon=":material/receipt_long:"),
    st.Page("pages/2_NBA_Valore.py", title="Quote live e valore", icon=":material/price_check:"),
]

navigazione = st.navigation({
    "": [home],
    "Serie A": pagine_serie_a,
    "NBA": pagine_nba,
})
navigazione.run()
