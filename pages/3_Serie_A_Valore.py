"""
PredictA — quote Serie A in tempo reale, ricerca di valore (quote_live_serie_a.py)
e valutazione dei bonus scommesse (bonus/valuta_bonus.py).

Stessa logica della pagina NBA equivalente (2_NBA_Valore.py): la probabilita'
di riferimento e' quella di Pinnacle senza margine (Shin), e il valore si
cerca fra i bookmaker, non contro il mercato.

La sezione bonus e' indipendente dalle quote live: legge un file di
configurazione compilato a mano (predicta/serie_a/bonus/bonus_config.json),
perche' nessuna API espone in modo affidabile le offerte bonus dei bookmaker.
"""
import pandas as pd
import streamlit as st

import predicta.serie_a.bonus.valuta_bonus as vb
import predicta.serie_a.quote.clv_serie_a as clv_serie_a
import predicta.serie_a.quote.quote_live_serie_a as ql

QUOTE_TTL_SECONDI = 15 * 60
NOMI_MERCATI = {"h2h": "1X2", "totals": "Over/Under"}

st.set_page_config(
    page_title="PredictA — Valore Serie A",
    page_icon=":material/sports_soccer:",
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
    st.logo("serie_a_logo.svg", size="large")
    st.markdown("### :material/tune: Richiesta")
    with st.container(border=True, key="depth_1"):
        mercati = st.multiselect(
            "Mercati", list(NOMI_MERCATI), default=list(NOMI_MERCATI),
            format_func=NOMI_MERCATI.get,
            help="Ogni mercato costa 1 credito per aggiornamento (regione eu, dove sta Pinnacle).")
        st.caption(f"Costo per aggiornamento: **{len(mercati)} crediti**")
    st.markdown("### :material/filter_alt: Filtri")
    with st.container(border=True, key="depth_2"):
        soglia_ev = st.slider("Valore atteso minimo", 0.0, 0.10, 0.02, 0.005, format="%.3f",
                              help="EV = probabilità equa × quota − 1. Sotto il 2% il vantaggio "
                                   "è dello stesso ordine dell'errore sulla probabilità equa.")
        kelly = st.slider("Frazione di Kelly", 0.05, 1.0, 0.25, 0.05,
                          help="Il Kelly pieno presuppone probabilità esatte: con stime "
                               "si usa una frazione (1/4 è la scelta prudente più comune).")


def mostra_clv():
    """Sezione CLV: legge solo gli snapshot gia' raccolti, nessun credito."""
    with st.container(border=True, key="depth_3"):
        st.markdown("**3. Verifica: le segnalazioni battono la chiusura? (CLV)**")
        snapshot = _carica_snapshot()
        clv, linea_mossa = clv_serie_a.calcola_clv(snapshot, soglia_ev=soglia_ev) if not snapshot.empty else (pd.DataFrame(), 0)
        r = clv_serie_a.riepilogo_clv(clv)
        if r is None:
            st.info(":material/hourglass: Ancora nessuna segnalazione confrontabile con una chiusura. "
                    "Gli snapshot li raccoglie GitHub Actions (raccogli_quote_serie_a.py) "
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
                   "dell'inizio − 1: misura se il prezzo era buono senza aspettare migliaia di risultati.")


def mostra_bonus():
    """Sezione bonus: dati a mano, formula approssimata (vedi valuta_bonus.py)."""
    with st.container(border=True, key="depth_6"):
        st.markdown("**4. Bonus scommesse**")
        bonus = vb.carica_bonus()
        if bonus.empty:
            st.info(":material/info: Nessun bonus in `predicta/serie_a/bonus/bonus_config.json`. "
                    "Aggiungi le offerte a mano (nessuna API le espone in modo affidabile): "
                    "bookmaker, tipo (`freebet` o `deposito`), importo, rollover, margine stimato.")
        else:
            classificato = vb.classifica_bonus(bonus)
            st.dataframe(pd.DataFrame({
                "Bookmaker": classificato["bookmaker"],
                "Tipo": classificato["tipo"],
                "Importo": classificato["importo"].map("{:.2f} €".format),
                "Valore netto stimato": classificato["valore_netto"].map("{:.2f} €".format),
            }), hide_index=True, width="stretch")
        st.caption("Stima approssimata del costo-opportunità del rollover (margine stimato di default "
                   f"{vb.MARGINE_DEFAULT:.0%}), non un valore atteso preciso da matched betting: serve solo "
                   "a ordinare le offerte fra loro. Modifica `bonus_config.json` per aggiornarle.")
    st.space("large")
    st.caption(
        ":material/warning: Strumento dimostrativo ed educativo. Non costituisce invito al gioco d'azzardo. "
        "Il gioco d'azzardo può causare dipendenza.",
        text_alignment="center",
    )


st.title("PredictA — Valore Serie A", text_alignment="center")
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
    mostra_bonus()
    st.stop()

col_a, col_b = st.columns([3, 1], vertical_alignment="bottom")
with col_b:
    aggiorna = st.button(f"Aggiorna quote ({len(mercati)} crediti)", type="primary",
                         disabled=not mercati, width="stretch")
if aggiorna:
    try:
        st.session_state.quote_live_serie_a = _scarica(tuple(mercati))
    except ql.ErroreQuoteLive as errore:
        st.error(f":material/error: {errore}")
        mostra_clv()
        mostra_bonus()
        st.stop()

if "quote_live_serie_a" not in st.session_state:
    with col_a:
        st.info(":material/info: Nessuna quota caricata. Premi **Aggiorna quote**: la chiamata parte "
                "solo al clic e resta in cache 15 minuti.")
    mostra_clv()
    mostra_bonus()
    st.stop()

eventi, crediti, istante = st.session_state.quote_live_serie_a
with col_a:
    st.caption(f"Quote delle {istante:%H:%M} · crediti rimanenti: **{crediti.get('rimanenti', '?')}** "
               f"(ultima chiamata: {crediti.get('ultima', '?')})")

tabella = ql.quote_in_tabella(eventi)
if tabella.empty:
    st.info(":material/event_busy: Nessuna partita di Serie A in programma con quote disponibili.")
    mostra_clv()
    mostra_bonus()
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
        sigla = {"h2h": "1X2", "totals": "TOT"}[e.mercato]
        fonti.add(f"{sigla}: {'Pinnacle' if e.fonte == ql.BOOK_SHARP else f'{e.n_book} book'}")
        if e.mercato == "h2h":
            p_casa, p_pareggio, p_trasferta = e.probabilita
            riga["1 (equa)"] = f"{p_casa:.0%} ({1 / p_casa:.2f})"
            riga["X (equa)"] = f"{p_pareggio:.0%} ({1 / p_pareggio:.2f})"
            riga["2 (equa)"] = f"{p_trasferta:.0%} ({1 / p_trasferta:.2f})"
        else:
            p_over, p_under = e.probabilita
            riga["Over (equa)"] = f"{e.linea_rif:g}: {p_over:.0%} ({1 / p_over:.2f})"
    riga["Fonte"] = " · ".join(sorted(fonti)) or "—"
    righe.append(riga)

with st.container(border=True, key="depth_4"):
    st.markdown("**1. Probabilità eque (senza margine)**")
    st.dataframe(pd.DataFrame(righe), hide_index=True, width="stretch")
    st.caption("La **quota equa** è la minima da accettare sul tuo bookmaker: se paga di più, "
               "la scommessa ha valore atteso positivo. Fonte preferita: Pinnacle. Se non quota la "
               "partita, si usa la mediana dei book europei sulla linea più quotata, con almeno "
               f"{ql.MIN_BOOK_CONSENSO} book: riferimento più debole, i valori trovati così vanno "
               "presi con più cautela.")

st.space("medium")

# --- Valore ---
valore = ql.trova_valore(tabella, soglia_ev=soglia_ev, frazione_kelly=kelly)
with st.container(border=True, key="depth_5"):
    st.markdown("**2. Quote a valore fra i bookmaker europei**")
    if valore.empty:
        st.info(f":material/check: Nessuna quota sopra il {soglia_ev:.1%} di valore atteso. "
                "Normale: i book allineati a Pinnacle sono la regola, non l'eccezione.")
    else:
        st.dataframe(pd.DataFrame({
            "Inizio": valore["inizio"].dt.tz_convert("Europe/Rome").dt.strftime("%d/%m %H:%M"),
            "Partita": valore["Casa"] + " – " + valore["Trasferta"],
            "Mercato": valore["mercato"].map(NOMI_MERCATI),
            "Esito": [f"{ {'casa': '1', 'pareggio': 'X', 'trasferta': '2', 'over': 'Over', 'under': 'Under'}[e] }"
                      f"{'' if pd.isna(l) else f' {l:g}'}"
                      for e, l in zip(valore["esito"], valore["linea"])],
            "Book": valore["book"],
            "Quota": valore["quota"].map("{:.2f}".format),
            "Quota equa": valore["quota_equa"].map("{:.2f}".format),
            "EV": valore["EV"].map("{:+.1%}".format),
            "Puntata (Kelly)": valore["Kelly"].map("{:.1%} del bankroll".format),
        }), hide_index=True, width="stretch")
    st.caption("I bookmaker italiani (Snai, Sisal, Eurobet...) non sono coperti da The Odds API: "
               "confronta a mano le loro quote con la **quota equa** della tabella 1. "
               "Le quote live sono anche precaricabili nel Costruttore schedina Serie A.")

st.space("medium")

mostra_clv()
mostra_bonus()
