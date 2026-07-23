# 🗺️ Roadmap tecnica di PredictA

Questo documento raccoglie l'analisi strategica completa su dati, feature e tecniche di modellazione per migliorare PredictA, con le fonti specifiche, la letteratura di riferimento e un piano di implementazione in fasi. Il `readme.md` principale resta snello per chi vuole solo installare e usare l'app; qui c'è il dettaglio per chi vuole continuare a svilupparla.

---

## Fase 1 — Completata ✅

Vedi la sezione "📊 Risultati Backtesting" nel [readme.md](readme.md) per i dettagli. In sintesi:

- Correzione tau di Dixon & Coles (1997) per i punteggi bassi, al posto della simulazione Monte Carlo
- Decadimento temporale esponenziale delle statistiche storiche (modulo `modello.py`)
- Fix di un bug sui dati (stagione 2009/10 duplicata, 2010/11 mancante)
- Fix di un bug sull'xG che collassava verso 0 con `peso_quote` alto
- Metriche RPS e log-loss nel backtesting, non solo accuratezza
- Pesi di default validati su tre stagioni di test indipendenti (2023, 2024, 2025)

Risultato: il modello batte il solo mercato di ~1 punto percentuale medio di accuratezza, in modo stabile su più stagioni. Un margine reale ma piccolo — le fasi successive partono da qui.

---

## Dati e feature per le fasi successive

Organizzati per priorità (impatto atteso sull'accuratezza) e fattibilità (facilità di ottenere il dato), con la fonte specifica e come entrerebbero nel modello.

### 🟢 Tier 1 — Alto impatto, bassa/media fatica

| Dato/Feature | Fonte specifica | Fattibilità | Impatto atteso | Come entra nel modello |
|---|---|---|---|---|
| **Rating Elo per club** | [clubelo.com](http://clubelo.com) — API REST gratuita (`api.clubelo.com/{Team}` restituisce CSV storico giornaliero) | Molto alta: CSV pronto, nessuno scraping aggressivo | Alto — letteratura (Hvattum & Arntzen 2010) mostra Elo competitivo con modelli molto più complessi | Sostituisce le medie storiche/forma attuali come feature di forza attacco/difesa, o come input diretto a un layer di ensemble insieme alle quote |
| **Expected Goals (xG) reali** | [understat.com](https://understat.com) (Serie A dal 2014/15, scraping HTML — dati in un tag `<script>` JSON) | Alta: formato noto, ampiamente usato in progetti open source | Alto — l'xG è molto meno rumoroso dei gol reali come proxy di qualità della squadra | Sostituisce FTHG/FTAG come target per calcolare "attacco/difesa storico", oppure feature aggiuntiva in un modello XGBoost |
| **Quote di più bookmaker già scaricate** | Colonne già presenti nei file `stagioni/*.txt` (WHH, BWH, PSH, ecc. — solo `OddsAvgH/D/A` consolidato è usato oggi) | Molto alta: dato già scaricato | Medio — più bookmaker = stima di mercato più stabile | Estendere la cascata di `unisci_dati.py` per calcolare anche una deviazione standard tra bookmaker, utile come feature di "incertezza del mercato" |
| **RPS/log-loss per configurazione** | Già implementato | — | — | Continuare a usarli come criterio primario invece della sola accuratezza quando si aggiungono nuove feature |

### 🟡 Tier 2 — Impatto medio-alto, richiede una pipeline dati nuova

| Dato/Feature | Fonte specifica | Fattibilità | Impatto atteso | Come entra nel modello |
|---|---|---|---|---|
| **Valore di mercato rosa** | [transfermarkt.com](https://www.transfermarkt.com) (scraping non ufficiale — pacchetti come `transfermarkt-scraper` su GitHub) o dataset Kaggle "Transfermarkt Football Data" | Media: nessuna API ufficiale | Medio-alto — Kuper & Szymanski (*Soccernomics*) mostrano correlazione ~0.9 tra monte ingaggi e posizione in classifica; cattura cambi di rosa che le medie storiche non vedono subito | Feature indipendente in XGBoost, oppure prior per la forza attacco/difesa nel modello Bayesiano (Fase 3) |
| **Statistiche avanzate (tiri, possesso, passaggi progressivi)** | [fbref.com](https://fbref.com) (dati StatsBomb via Sports Reference, Serie A dal 2017/18) — libreria Python `soccerdata` | Media-alta: sito pensato per essere consultato, rispettare i limiti di scraping | Alto per partite recenti, ma copre solo da 2017/18 in poi | Feature engineering per XGBoost (rolling average ultimi N tiri/xG/possesso) |
| **API-Football (fixtures, formazioni, infortuni, quote)** | [api-football.com](https://www.api-football.com) — tier gratuito 100 richieste/giorno, a pagamento per storico/infortuni completi | Media: gratis limitato | Alto per un modello "match-day" con formazione reale nota poche ore prima | Aggiustamento moltiplicativo dell'xG basato su "forza XI titolare atteso" vs "forza rosa completa" |
| **Sistema di rating pi-ratings** | Nessuna fonte esterna — solo implementazione (Constantinou & Fenton, *pi-football-ratings*, paper pubblico) | Media: formula pubblicata ma più complessa di Elo | Alto — nei paper accademici supera Elo semplice nella predizione 1X2 | Sostituisce interamente la sezione "storico" con un rating che si aggiorna partita per partita |
| **Meteo storico allo stadio** | [open-meteo.com](https://open-meteo.com) (API storica gratuita, no key richiesta) | Alta, ma serve mappare squadra→stadio→coordinate | Basso-medio — effetto reale ma piccolo su pioggia/vento e gol totali | Feature minore in XGBoost |
| **Shots/corner/cartellini già nel dataset** | Già estratti da `unisci_dati.py` (colonne HS, AS, HST, AST, HC, AC, HY, AY, HR, AR) ma non ancora usati dal modello | Molto alta: dato già presente in `serie_a.csv` | Medio — proxy di dominio del gioco più ricco dei soli gol | Feature per un modello XGBoost, o media pesata nel tempo come proxy xG-like semplificato |

### 🔴 Tier 3 — Avanzato, alto sforzo o costo

| Dato/Feature | Fonte specifica | Fattibilità | Impatto atteso | Come entra nel modello |
|---|---|---|---|---|
| **Modello Bayesiano gerarchico** (Baio & Blangiardo 2010) | Nessun dato esterno — libreria `PyMC` | Bassa: richiede competenze MCMC, training più lento | Alto soprattutto per squadre neopromosse/con pochi dati (oggi gestite con `fillna(0)`) | Sostituisce il layer statistico attuale: ogni squadra ha un parametro attacco/difesa con prior condiviso, che si restringe verso la media di lega in proporzione all'incertezza |
| **Formazioni/infortuni in tempo reale** | API-Football piano a pagamento, o scraping Transfermarkt pagina infortuni | Bassa-media: dati storici affidabili difficili da reperire gratis | Alto ma solo per previsioni "a ridosso della partita" | Feature runtime, non backtestabile facilmente sullo storico per mancanza di dati infortuni retroattivi |
| **Ensemble stacking** (Poisson-Dixon-Coles + Elo + XGBoost + quote) | Nessun dato esterno | Media: tecnica nota, serve disciplina per evitare leakage nel meta-learner | Alto — Groll et al. 2019 mostrano che modelli ibridi statistico+ML battono i singoli componenti | Meta-learner (regressione logistica) allenato out-of-fold sulle probabilità dei modelli base |
| **Multi-campionato** | football-data.co.uk copre già Premier/Liga/Bundesliga/Ligue 1 con lo stesso formato | Alta come dato, media come impatto | Medio — aiuta solo la componente ML, non il modello Poisson per-squadra | Training set esteso per XGBoost, con "lega" come feature categoriale |

---

## Tecniche di modellazione: cosa adottare

In ordine di ritorno sull'investimento:

1. **Dixon-Coles completo** — fatto in Fase 1 (tau + decadimento temporale).
2. **Rating system continuo (Elo o pi-ratings)** al posto delle medie storiche statiche — nella nostra grid search "storico" e "scontri diretti" non hanno aggiunto valore sopra le quote; un Elo con K-factor ben calibrato potrebbe catturare la stessa informazione in modo più principiato.
3. **XGBoost/LightGBM come sfidante, non sostituto** — utile con feature eterogenee (Elo diff, xG rolling, giorni di riposo, quote). Con ~11.500 partite il rischio di overfitting è reale: serve walk-forward CV rigoroso, mai k-fold casuale.
4. **Reti neurali: sconsigliate per ora.** Con questo volume di dati la letteratura (Groll, Baboota & Kaur 2019) mostra alberi/ensemble e modelli Poisson competitivi o superiori alle reti profonde.
5. **Ensemble stacking** come step finale — combinare Poisson-Dixon-Coles + Elo + XGBoost + quote con un meta-learner, dopo aver validato che ciascun componente aggiunge valore da solo.

## Errori comuni da evitare

Alcuni li abbiamo già commessi e corretti in questo progetto — inclusi qui perché facili da reintrodurre:

- **Data leakage temporale**: usare indici sbagliati o concatenare train+test senza tagliare al punto giusto.
- **Shrinkage eccessivo verso la media**: mediare ripetutamente ogni statistica con la media di lega annulla le differenze reali tra squadre.
- **Ignorare la correlazione nei bassi punteggi**: un Poisson indipendente sottostima i pareggi (da cui la correzione tau).
- **Ottimizzare i pesi su una sola stagione di test**: rischio concreto di scegliere la combinazione "fortunata" per rumore — validare sempre su più stagioni indipendenti (lezione imparata in Fase 1: il punto di massimo grezzo della grid search non reggeva su una terza stagione).
- **Trattare tutte le squadre allo stesso modo indipendentemente dal campione**: una neopromossa ha zero storia in A — serve shrinkage bayesiano proporzionale all'incertezza, non un default arbitrario.
- **Confondere "batte il benchmark ingenuo" con "batte il mercato"**: il modello batte facilmente "sempre 1", ma il vero test è battere le quote — riusciamo a farlo solo di ~1 punto percentuale.
- **Usare solo l'accuratezza come metrica**: su un problema 3-classi sbilanciato, l'accuratezza premia previsioni "decise" anche se mal calibrate — da qui RPS e log-loss già in uso.
- **Scraping fragile senza fallback**: fonti come Understat/FBref/Transfermarkt cambiano struttura HTML senza preavviso — prevedere test automatici che verifichino lo schema atteso.

---

## Fase 2 — Medio termine (settimane, nuova pipeline dati)

1. Integrare Elo da clubelo.com come feature/sostituto delle medie storiche statiche
2. Scraping Understat per xG reali (dal 2014/15) come segnale più pulito dei gol grezzi
3. Prototipo XGBoost su feature strutturate (Elo diff, xG rolling, quote), validato walk-forward su più stagioni indipendenti, confrontato onestamente contro il modello statistico attuale
4. Scraping valore di mercato Transfermarkt come proxy di qualità rosa aggiornata
5. Usare le colonne tiri/corner/cartellini già presenti in `serie_a.csv` (dal 2005 circa) come feature aggiuntive

## Fase 3 — Avanzato (mesi)

1. Modello Bayesiano gerarchico (Baio & Blangiardo) con partial pooling per gestire correttamente neopromosse/campioni piccoli
2. Ensemble stacking finale: Poisson-Dixon-Coles + Elo/pi-ratings + XGBoost + quote di mercato, meta-learner allenato out-of-fold
3. Estensione multi-campionato per aumentare il volume dati della componente ML
4. Formazioni/infortuni via API-Football (richiede piano a pagamento per storico affidabile) solo se il caso d'uso si sposta da "dashboard esplorativa" a "previsione a ridosso del match reale"

---

## Letteratura e riferimenti

- Dixon, M.J. & Coles, S.G. (1997), *Modelling Association Football Scores and Inefficiencies in the Betting Market* — il modello base implementato in Fase 1.
- Rue, H. & Salvesen, Ø. (2000) — modello Bayesiano dinamico con forza attacco/difesa che varia nel tempo.
- Baio, G. & Blangiardo, M. (2010) — modello Bayesiano gerarchico con partial pooling tra squadre.
- Karlis, D. & Ntzoufras, I. — modelli Poisson bivariati/Skellam per il calcio.
- Constantinou, A.C. & Fenton, N. — pi-ratings, sistema di rating pubblicato specificamente per la predizione 1X2.
- Hvattum, L.M. & Arntzen, H. (2010) — valutazione dei rating Elo applicati al calcio.
- Groll, A., Ley, C., Schauberger, G., Van Eetvelde, H. (2019) — modelli ibridi statistico+ML per le previsioni calcistiche.
- Metodologia pubblica SPI (Soccer Power Index) di FiveThirtyEight — combina Elo con differenziale reti aggiustato e valori di mercato.
