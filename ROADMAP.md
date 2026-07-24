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
| **Quote di chiusura e movimento apertura→chiusura** | Colonne `B365CH/CD/CA`, `AvgCH/CD/CA` già presenti nei file grezzi dal 2019 (7 stagioni) ma scartate da `unisci_dati.py`, che estrae solo le quote "correnti" | Molto alta: dato già scaricato, verificato con valori diversi dalla quota non-chiusura | Da validare — il differenziale apertura/chiusura è letteratura nota come segnale su informazione privata/movimenti di mercato | Nuove colonne in `unisci_dati.py` (`OddsCloseH/D/A` + dispersione tra bookmaker); copertura solo 2019+, sufficiente per le 3 stagioni di validazione |
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

1. ~~Integrare Elo da clubelo.com~~ — **fatto, risultato negativo.** Storico Elo delle 53 squadre scaricato (`scarica_elo.py` → `elo_storico.csv`), lookup point-in-time in `modello.py` (`elo_asof_batch`), integrato come componente pesata in `pages/backtesting.py` e calibrato con una regressione di Poisson vera (`calibra_regressione_elo`, non una costante indovinata). Validato sulle stesse 3 stagioni indipendenti della Fase 1: **nessuna combinazione batte la configurazione attuale** (54.87% medio senza Elo vs 54.79% nella miglior combinazione con Elo). Codice tenuto e testato (`tests/test_model.py`), `peso_elo` di default a 0. Probabile causa: ClubElo cattura informazione che il mercato incorpora già, stessa storia di storico/scontri diretti in Fase 1.
2. **Correzione di Shin per le quote (de-overrounding)** — *da valutare, priorità alta.* Oggi la quota di consenso viene convertita in probabilità con `1/quota` normalizzato in proporzione (metodo "basic"), che non corregge la favorite-longshot bias (i bookmaker mettono più margine sulle quote alte che su quelle basse). Il modello di Shin (1992/1993) stima il margine in modo non uniforme tra gli esiti ed è lo standard alternativo in letteratura. Interessante perché non richiede dati nuovi ed è a monte del blend attuale (pesato 90% sulle quote): se il segnale di mercato in ingresso è più pulito, tutto il blend beneficia. Da validare con lo stesso protocollo a 3 stagioni prima di adottarlo, senza dare per scontato l'impatto.
3. ~~Prototipo gradient boosting su feature strutturate~~ — **fatto, risultato negativo.** `prototipo_gradient_boosting.py`: `HistGradientBoostingClassifier` di scikit-learn (non XGBoost, che richiede la libreria di sistema `libomp` non disponibile in questo ambiente) allenato sulle stesse componenti già validate (xG storico/forma/scontri diretti, Elo, probabilità implicite delle quote) come feature. Con pochi dati di training (5 stagioni, ~1900 partite) overfit severo (85% accuratezza su training, 48% su test). Con 10 stagioni (~3800 partite) e forte regolarizzazione (early stopping, `l2_regularization`) il divario si riduce ma il risultato resta **leggermente sotto** il modello statistico su tutte e 3 le stagioni indipendenti: 54.17% di accuratezza media e RPS 0.1896, contro 54.87%/0.1889 del modello attuale. Causa probabile: dati Serie A troppo limitati (~380 partite/stagione) perché un modello ad alberi trovi interazioni non lineari utili oltre a quelle già catturate dal blend statistico calibrato.
4. Scraping Understat per xG reali (dal 2014/15) come segnale più pulito dei gol grezzi. Non usarlo solo come media storica: calcolare anche lo scarto xG − gol reali nelle ultime partite come segnale di "regressione alla media" (una squadra con xG alto ma pochi gol recenti è probabilmente sottovalutata dai risultati grezzi, e viceversa).
5. Scraping valore di mercato Transfermarkt come proxy di qualità rosa aggiornata — usarlo come **variazione temporale** (slope 30/90/180gg) e come prior "shrinked" per un modello gerarchico, non come feature statica diretta (raffinamento da una revisione esterna).
6. **Quote di chiusura e movimento apertura→chiusura** (idea da una revisione esterna, verificata fattibile) — `B365CH/CD/CA` e `AvgCH/CD/CA` sono già nei file grezzi dal 2019 ma scartati da `unisci_dati.py`. Da estrarre ed usare come: differenziale apertura/chiusura (segnale di informazione privata/movimento di mercato) e dispersione tra bookmaker (incertezza di mercato). Non ancora implementato.
7. Feature "quasi gratis" da dati già presenti, nessuno scraping nuovo:
   - ~~Tiri in porta/corner/cartellini~~ — **fatto, risultato negativo.** `prototipo_gradient_boosting.py --con-tiri` aggiunge le medie recenti (tiri in porta, corner) al prototipo gradient boosting. Risultato misto tra stagioni (2025: 54.4%→55.4%, migliora; 2024: 54.2%→52.6%, peggiora; 2023: stabile), ma in media **leggermente peggio** del gradient boosting senza queste feature: 54.00% di accuratezza e RPS 0.1898, contro 54.17%/0.1896 senza tiri e 54.87%/0.1889 del modello statistico. Nessun beneficio netto, coerente col tema di questa fase: più feature non aiutano se il volume dati resta lo stesso.
   - Giorni di riposo tra una partita e la precedente (differenza sulla colonna `Date`, già disponibile): non ancora implementato. Elaborazione da una revisione esterna: oltre al semplice differenziale, anche `partite_ultimi_14_giorni` e `trasferta_dopo_trasferta` (entrambe calcolabili dai dati che abbiamo); scartato `partita_di_coppa_in_5_giorni` perché richiederebbe il calendario delle coppe europee, dato che non abbiamo.
   - ~~Tendenza dell'arbitro (falli/rigori/cartellini)~~ — **scartato, verificato.** La colonna `Referee` è presente nei file grezzi solo per le stagioni 2005 e 2006 (2 su 33), poi mai più: dato insufficiente per costruire una feature storica utilizzabile su tutto il periodo di test. Confermato da una seconda revisione esterna, che infatti raccomanda la stessa cautela.
   - `Squad_Rotation_Index`, distanza di viaggio, infortuni/XI atteso in forma probabilistica: richiedono dati che non abbiamo (formazioni/minutaggio, geocoding stadi, rating giocatori) — restano Tier 3, non "quasi gratis" come proposto da alcune revisioni esterne.

## Fase 3 — Avanzato (mesi)

1. Modello Bayesiano gerarchico (Baio & Blangiardo) con partial pooling per gestire correttamente neopromosse/campioni piccoli
2. Ensemble stacking finale: Poisson-Dixon-Coles + Elo/pi-ratings + XGBoost + quote di mercato, meta-learner allenato out-of-fold. Raffinamento da una revisione esterna: non un peso fisso (es. "90% quote sempre"), ma un blend condizionato al contesto (più peso al modello statistico quando ci sono segnali forti di mismatch — fatica, differenziale di rosa — più peso alle quote nelle partite "normali").
3. Estensione multi-campionato per aumentare il volume dati della componente ML
4. Formazioni/infortuni via API-Football (richiede piano a pagamento per storico affidabile) solo se il caso d'uso si sposta da "dashboard esplorativa" a "previsione a ridosso del match reale"
5. **Indice motivazionale di fine stagione** (proposto indipendentemente da due revisioni esterne, segno che è un'idea solida): nelle ultime 8-10 giornate, squadre senza obiettivi reali (salve e senza coppe) affrontano squadre in lotta salvezza — un Poisson puro sovrastima la squadra "più forte sulla carta" in questi casi. Richiede ricostruire la classifica punto-per-punto nel tempo (walk-forward, non banale ma fattibile con i dati che abbiamo) per calcolare la distanza dai margini di salvezza/Europa giornata per giornata. Non ancora implementato.

### Cosa abbiamo scartato da revisioni esterne, e perché

Tre revisioni esterne (Deepseek, Gemini, Perplexity) hanno proposto integrazioni aggiuntive. Tenuto quanto sopra (Shin, quote di chiusura, giorni di riposo elaborati, indice motivazionale, blend condizionato); scartato:
- **Feature legate all'arbitro**: una revisione affermava che il dato fosse "già presente nei CSV di Football-Data" — verificato e smentito, la colonna `Referee` copre solo 2 stagioni su 33. Una seconda revisione, indipendentemente, raccomandava la stessa cautela.
- **`Squad_Rotation_Index`, distanza di viaggio, turnover pre-Champions, infortuni/XI atteso in forma probabilistica**: richiedono dati che non abbiamo (formazioni/minutaggio, calendario coppe europee, geocoding stadi, rating giocatori) nonostante fossero proposti come "quasi gratis" o "fattibilità alta" — restano Tier 3.
- **Pannello di "value betting"** (con soglia di EV+ e alert, o proposto come "watchlist di sottovalutazione" con linguaggio più prudente): scartato per scelta di merito, non tecnica, indipendentemente da come viene presentato nell'interfaccia. Il progetto ha un disclaimer esplicito (scopo dimostrativo, non invito al gioco d'azzardo) ed è coerente non costruire strumenti che segnalano attivamente scommesse da piazzare. La correzione di Shin resta comunque utile come miglioramento della calibrazione interna del modello, indipendentemente dall'uso per scommettere.

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
- Shin, H.S. (1992, 1993) — modello per stimare il margine del bookmaker in modo non uniforme tra gli esiti ("favorite-longshot bias"), alternativa più accurata alla normalizzazione proporzionale semplice di `1/quota`.
