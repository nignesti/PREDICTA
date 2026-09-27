"""
PredictA — quote NBA in tempo reale e ricerca di valore (quote_live_nba.py).

La probabilita' di riferimento e' quella di Pinnacle senza margine (Shin): e'
il previsore migliore disponibile, e valida_nba.py mostra che il nostro Elo
non lo batte. Il valore si cerca quindi fra i bookmaker, non contro il
mercato: un book che paga piu' della quota equa di Pinnacle.

Ogni aggiornamento costa crediti The Odds API (mercati x regioni, default 3):
nessuna chiamata parte da sola, solo al clic, e il risultato resta in cache
per QUOTE_TTL_SECONDI a tutti gli utenti della sessione.
"""
import pandas as pd
import streamlit as st

import clv_nba
import quote_live_nba as ql

QUOTE_TTL_SECONDI = 15 * 60
NOMI_MERCATI = {"h2h": "Moneyline", "spreads": "Spread", "totals": "Totale punti"}

st.set_page_config(
    page_title="PredictA — Valore NBA",
    page_icon=":material/sports_basketball:",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=QUOTE_TTL_SECONDI, show_spinner="Scarico le quote da The Odds API...")
def _scarica(mercati):
    """Cache condivisa: due clic ravvicinati (o due utenti) non spendono due
    volte gli stessi crediti. La chiave API non e' un argomento, cosi' non
    finisce nella chiave di cache."""
    eventi, crediti = ql.scarica_quote(ql.chiave_api(), mercati=mercati)
    return eventi, crediti, pd.Timestamp.now(tz="Europe/Rome")


@st.cache_data(ttl=QUOTE_TTL_SECONDI)
def _carica_snapshot():
    return ql.carica_snapshot()


with st.sidebar:
    st.markdown("### :material/tune: Richiesta")
    with st.container(border=True):
        mercati = st.multiselect(
            "Mercati", list(NOMI_MERCATI), default=list(NOMI_MERCATI),
            format_func=NOMI_MERCATI.get,
            help="Ogni mercato costa 1 credito per aggiornamento (regione eu, dove sta Pinnacle).")
        st.caption(f"Costo per aggiornamento: **{len(mercati)} crediti**")
    st.markdown("### :material/filter_alt: Filtri")
    with st.container(border=True):
        soglia_ev = st.slider("Valore atteso minimo", 0.0, 0.10, 0.02, 0.005, format="%.3f",
                              help="EV = probabilità equa × quota − 1. Sotto il 2% il vantaggio "
                                   "è dello stesso ordine dell'errore sulla probabilità equa.")
        kelly = st.slider("Frazione di Kelly", 0.05, 1.0, 0.25, 0.05,
                          help="Il Kelly pieno presuppone probabilità esatte: con stime "
                               "si usa una frazione (1/4 è la scelta prudente più comune).")


def mostra_clv():
    """Sezione CLV: legge solo gli snapshot gia' raccolti, nessun credito."""
    with st.container(border=True):
        st.markdown("**3. Verifica: le segnalazioni battono la chiusura? (CLV)**")
        snapshot = _carica_snapshot()
        clv, linea_mossa = clv_nba.calcola_clv(snapshot, soglia_ev=soglia_ev) if not snapshot.empty else (pd.DataFrame(), 0)
        r = clv_nba.riepilogo_clv(clv)
        if r is None:
            st.info(":material/hourglass: Ancora nessuna segnalazione confrontabile con una chiusura. "
                    "Gli snapshot li raccoglie GitHub Actions due volte al giorno (raccogli_quote_nba.py) "
                    f"nella cartella quote_live/ del repo: {snapshot['istante'].nunique() if not snapshot.empty else 0} finora.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Segnalazioni verificate", r["n"], help=f"Escluse {linea_mossa} per linea mossa.")
            c2.metric("CLV medio", f"{r['clv_medio']:+.2%}",
                      help=f"IC 95%: {r['ic_95'][0]:+.2%} / {r['ic_95'][1]:+.2%}")
            c3.metric("Con CLV positivo", f"{r['quota_positivi']:.0%}")
            if r["ic_95"][0] > 0:
                st.success(":material/verified: Intervallo di confidenza tutto sopra zero: le segnalazioni "
                           "battono la chiusura di Pinnacle, il vantaggio è misurabile.")
            elif r["ic_95"][1] < 0:
                st.error(":material/trending_down: Intervallo tutto sotto zero: le segnalazioni perdono "
                         "contro la chiusura. Alza la soglia di EV o non usarle.")
            else:
                st.warning(":material/balance: Intervallo che include lo zero: ancora indistinguibile dal "
                           "caso. Servono più segnalazioni.")
        st.caption("Il **CLV** è quota presa × probabilità equa di Pinnacle all'ultimo snapshot prima "
                   "dell'inizio − 1: misura se il prezzo era buono senza aspettare migliaia di risultati. "
                   "La chiusura è approssimata (ultimo snapshot delle 23:00 UTC).")
    st.space("large")
    st.caption(
        ":material/warning: Strumento dimostrativo ed educativo. Non costituisce invito al gioco d'azzardo. "
        "Il gioco d'azzardo può causare dipendenza.",
        text_alignment="center",
    )


st.title("PredictA — Valore NBA", text_alignment="center")
st.markdown(
    "Quote di tutti i bookmaker europei confrontate con la quota equa di Pinnacle "
    "(senza margine): se un book paga di più, quella è una scommessa a valore atteso positivo.",
    text_alignment="center",
)
st.space("medium")

if not ql.chiave_api():
    st.warning(":material/key: Manca la chiave The Odds API: aggiungi `ODDS_API_KEY = \"...\"` in "
               "`.streamlit/secrets.toml` (in locale) o nei Secrets dell'app su Streamlit Cloud.")
    mostra_clv()
    st.stop()

col_a, col_b = st.columns([3, 1], vertical_alignment="bottom")
with col_b:
    aggiorna = st.button(f"Aggiorna quote ({len(mercati)} crediti)", type="primary",
                         disabled=not mercati, width="stretch")
if aggiorna:
    try:
        st.session_state.quote_live_nba = _scarica(tuple(mercati))
    except ql.ErroreQuoteLive as errore:
        st.error(f":material/error: {errore}")
        mostra_clv()
        st.stop()

if "quote_live_nba" not in st.session_state:
    with col_a:
        st.info(":material/info: Nessuna quota caricata. Premi **Aggiorna quote**: la chiamata parte "
                "solo al clic e resta in cache 15 minuti.")
    mostra_clv()
    st.stop()

eventi, crediti, istante = st.session_state.quote_live_nba
with col_a:
    st.caption(f"Quote delle {istante:%H:%M} · crediti rimanenti: **{crediti.get('rimanenti', '?')}** "
               f"(ultima chiamata: {crediti.get('ultima', '?')})")

tabella = ql.quote_in_tabella(eventi)
if tabella.empty:
    st.info(":material/event_busy: Nessuna partita NBA in programma con quote disponibili "
            "(fuori stagione non si spendono crediti).")
    mostra_clv()
    st.stop()

# --- Quote eque ---
eque = ql.probabilita_eque(tabella)
partite = tabella.drop_duplicates("id")[["id", "inizio", "Casa", "Trasferta"]]
righe = []
for p in partite.itertuples(index=False):
    riga = {"Inizio": p.inizio.tz_convert("Europe/Rome").strftime("%d/%m %H:%M"),
            "Partita": f"{p.Casa} – {p.Trasferta}"}
    fonti = set()
    for e in eque[eque["id"] == p.id].itertuples(index=False):
        sigla = {"h2h": "ML", "spreads": "SP", "totals": "TOT"}[e.mercato]
        fonti.add(f"{sigla}: {'Pinnacle' if e.fonte == ql.BOOK_SHARP else f'{e.n_book} book'}")
        if e.mercato == "h2h":
            riga["Casa vince"] = f"{e.p1:.0%} (equa {1 / e.p1:.2f})"
        elif e.mercato == "spreads":
            riga["Spread Casa"] = f"{e.linea_rif:+.1f}: {e.p1:.0%} (equa {1 / e.p1:.2f})"
        else:
            riga["Over"] = f"{e.linea_rif:g}: {e.p1:.0%} (equa {1 / e.p1:.2f})"
    riga["Fonte"] = " · ".join(sorted(fonti)) or "—"
    righe.append(riga)

with st.container(border=True):
    st.markdown("**1. Probabilità eque (senza margine)**")
    st.dataframe(pd.DataFrame(righe), hide_index=True, width="stretch")
    st.caption("La **quota equa** è la minima da accettare sul tuo bookmaker: se paga di più, "
               "la scommessa ha valore atteso positivo. Fonte preferita: Pinnacle. Se non quota la "
               "partita (apre le linee NBA a ridosso della gara), si usa la mediana dei book europei "
               f"sulla linea più quotata, con almeno {ql.MIN_BOOK_CONSENSO} book: riferimento più debole, "
               "i valori trovati così vanno presi con più cautela.")

st.space("medium")

# --- Valore ---
valore = ql.trova_valore(tabella, soglia_ev=soglia_ev, frazione_kelly=kelly)
with st.container(border=True):
    st.markdown("**2. Quote a valore fra i bookmaker europei**")
    if valore.empty:
        st.info(f":material/check: Nessuna quota sopra il {soglia_ev:.1%} di valore atteso. "
                "Normale: i book allineati a Pinnacle sono la regola, non l'eccezione.")
    else:
        st.dataframe(pd.DataFrame({
            "Inizio": valore["inizio"].dt.tz_convert("Europe/Rome").dt.strftime("%d/%m %H:%M"),
            "Partita": valore["Casa"] + " – " + valore["Trasferta"],
            "Mercato": valore["mercato"].map(NOMI_MERCATI),
            "Esito": [f"{ {'casa': c, 'trasferta': t, 'over': 'Over', 'under': 'Under'}[e] }"
                      f"{'' if pd.isna(l) else f' {l:+g}' if m == 'spreads' else f' {l:g}'}"
                      for e, l, m, c, t in zip(valore["esito"], valore["linea"], valore["mercato"],
                                               valore["Casa"], valore["Trasferta"])],
            "Book": valore["book"],
            "Quota": valore["quota"].map("{:.2f}".format),
            "Quota equa": valore["quota_equa"].map("{:.2f}".format),
            "EV": valore["EV"].map("{:+.1%}".format),
            "Puntata (Kelly)": valore["Kelly"].map("{:.1%} del bankroll".format),
        }), hide_index=True, width="stretch")
    st.caption("I bookmaker italiani (Snai, Sisal, Eurobet...) non sono coperti da The Odds API: "
               "confronta a mano le loro quote con la **quota equa** della tabella 1. "
               "Le quote live sono anche precaricabili nel Costruttore schedina NBA.")

st.space("medium")

mostra_clv()
