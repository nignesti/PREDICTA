# 🗺️ Roadmap tecnica di PredictA

Questo documento raccoglie l'analisi strategica completa su dati, feature e tecniche di modellazione per migliorare PredictA, con le fonti specifiche, la letteratura di riferimento e un piano di implementazione in fasi. Il `readme.md` principale resta snello per chi vuole solo installare e usare l'app; qui c'è il dettaglio per chi vuole continuare a svilupparla — incluso il **registro completo degli esperimenti falliti**, con i numeri e la causa probabile: sapere cosa non funziona e perché vale quanto sapere cosa funziona.

---

## 📍 Stato attuale (aggiornato al 25 luglio 2026)

**Fasi 1, 2, 3 e 0 chiuse.** 15 voci testate in totale, **3 adottate in produzione** (Shin, quote di chiusura, medie di lega pesate).

### Configurazione di produzione

Pesi: `forma=0.10`, `scontri=0`, `quote=0.90` (quindi `storico=0` per costruzione). Quote di **chiusura** convertite con la correzione di **Shin**, medie di lega **pesate nel tempo**. Dixon-Coles con emivita 730 giorni e correzione tau.

**55.02% di accuratezza / 0.1909 RPS** su 7 stagioni di test (2019-2025, 2.659 partite), contro **54.68% / 0.1904** del solo mercato.

> ⚠️ **Quella differenza non è statisticamente significativa** (McNemar p = 0.28; sull'RPS il modello è anzi leggermente peggiore). Il progetto ha a lungo dichiarato "~1 punto percentuale sopra il mercato" sulla base di 3 stagioni: allargando a 7 il vantaggio si dimezza, il che indica che era in buona parte rumore. Vedi la sezione "problema di misura" più sotto — è la conclusione più importante di tutto il lavoro fatto finora.

### Registro completo degli esperimenti

Tutti validati con lo stesso protocollo: 3 stagioni di test indipendenti (2023, 2024, 2025), walk-forward, nessun dato futuro. Ordinati per esito.

| # | Esperimento | Fase | Acc. media | RPS | Esito |
|---|---|---|---:|---:|---|
| — | *Solo mercato (benchmark)* | — | 53.90% | — | *riferimento* |
| — | *Fine Fase 1 (apertura, proporzionale)* | 1 | 54.87% | 0.1889 | *riferimento* |
| 1 | **Correzione di Shin** | 2 | 54.79% | **0.1886** | ✅ **adottato** — RPS meglio in 3/3 stagioni, costo zero |
| 2 | **Quote di chiusura** | 2 | **55.22%** | **0.1885** | ✅ **adottato** — +0.43 pp, il risultato più forte |
| 3 | Rating Elo (clubelo.com) | 2 | 54.79% | — | ❌ nessuna combinazione batte il default |
| 4 | Gradient boosting | 2 | 54.17% | 0.1896 | ❌ sotto il modello statistico |
| 5 | + tiri in porta / corner | 2 | 54.00% | 0.1898 | ❌ peggiora anche il gradient boosting |
| 6 | + giorni di riposo / congestione | 2 | 54.17% | 0.1893 | ❌ nessun beneficio misurabile |
| 7 | xG reali di Understat | 2 | 53.87%¹ | 0.1919¹ | ⚠️ debolmente positivo, **non adottabile** (fonte ferma a set. 2024) |
| 8 | Tendenza dell'arbitro | 2 | — | — | ❌ scartato: `Referee` copre 2 stagioni su 33 |
| 9 | Modello Bayesiano gerarchico | 3 | 54.35% | 0.1883 | ❌ peggiore in accuratezza |
| 10 | Ensemble stacking | 3 | 54.35% | 0.1880² | ➖ **rivalutato in Fase 0: indistinguibile**, non adottato |
| 11 | Training multi-campionato | 3 | 54.08% | 0.1884² | ➖ **rivalutato in Fase 0: indistinguibile ma il più promettente** (p = 0.077) |
| 12 | Indice motivazionale fine stagione | 3 | 53.91% | 0.1897 | ❌ negativo e coerente in 3/3 stagioni |
| 13 | Formazioni/infortuni (API-Football) | 3 | — | — | ⛔ valutato, non implementato (a pagamento) |

*² RPS misurato con la configurazione delle quote sbagliata (pre-Fase 2): con quella corretta l'ordinamento si ribalta, vedi Fase 0.*

*¹ Understat è validato su 2021/2022/2023 e non sulle 3 stagioni standard, per limiti di copertura del dataset: il confronto onesto è contro 53.43%/0.1928 dello stesso modello ricalcolato su quella finestra, non contro i numeri di questa tabella.*

### La lezione: cosa funziona e cosa no

Il pattern è netto e coerente su 12 esperimenti misurati:

> **Aggiungere feature nuove non paga. Estrarre meglio l'informazione già contenuta nelle quote sì.**

Gli unici due miglioramenti adottati (Shin, quote di chiusura) non introducono **nessun dato nuovo**: cambiano solo come si converte in probabilità un'informazione che avevamo già. Tutto ciò che porta segnale esterno — Elo, xG, tiri, riposo, motivazione, multi-lega, rosa — o non batte il mercato, o lo batte solo sulla calibrazione (RPS) e non sull'accuratezza.

La spiegazione più probabile è la stessa in tutti i casi: **il mercato incorpora già quell'informazione**, e in modo più efficiente di quanto possiamo fare noi con medie sui gol.

Con una precisazione importante aggiunta dopo la verifica di significatività: **anche i due "successi" non sono statisticamente distinguibili dal rumore.** Shin e quote di chiusura restano adottati — sono gratuiti, teoricamente fondati e vanno nella direzione giusta su entrambe le metriche — ma vanno descritti come "coerenti con l'attesa", non come miglioramenti misurati.

Una nota secondaria ma ricorrente: **RPS e accuratezza si muovono spesso in direzioni opposte**. Ensemble stacking e multi-campionato sembravano avere il miglior RPS mai misurato, e il criterio dell'epoca ("guadagno su entrambe le metriche") li aveva esclusi. La **Fase 0 li ha rivalutati**: quel primato era un artefatto di un baseline sbagliato, e con la configurazione corretta sono entrambi *indistinguibili* dal modello statistico. La conclusione pratica non cambia, la motivazione sì — vedi la sezione Fase 0.

### Correzioni di bug (25 luglio 2026)

Una revisione del codice di produzione ha trovato sei difetti, tutti verificati empiricamente e corretti. Nessuno era coperto dai test esistenti; ora lo sono tutti.

| # | Difetto | File | Impatto misurato |
|---|---|---|---|
| 1 | **Quote degli scontri diretti non orientate casa/trasferta**: le colonne `Odds*H`/`Odds*A` si riferiscono alla squadra di casa *di quella riga*, non a quella per cui si sta prevedendo | `app.py` | Su Milan–Inter: `p_1` 0.406 invece di 0.270, `p_2` 0.350 invece di 0.485. ~14 punti percentuali di errore su una componente pesata 0.90. Appiattiva ogni previsione verso la parità e cancellava il vantaggio campo |
| 2 | **Vincolo di validità di Dixon-Coles non imposto** (`rho >= -1/lambda`) | `modello.py` | Con `rho=-0.30` e xG ≥ 3.34 la cella 0-1 diventava negativa (−0.00054): probabilità negative mascherate dalla normalizzazione. xG oltre 3.34 è raggiungibile col default, dove l'xG viene interamente dalla forma |
| 3 | **Medie di lega su 33 stagioni contro statistiche decadute a 2 anni** | `app.py`, `pages/backtesting.py` | Bias pro-casa del ~15% sul rapporto degli xG. Vedi la voce dedicata sotto |
| 4 | **Fallback delle quote di chiusura dichiarato ma non implementato**: `"OddsAvgCH" in riga.index` è sempre vero, è un nome di colonna | `pages/backtesting.py` | Le 8.874 partite pre-2019 perdevano le quote del tutto e cadevano sul modello storico (~44%) invece di ricadere sull'apertura. Fuori dalla portata della UI, ma raggiungibile dagli script |
| 5 | **Finestra forma 5 nella dashboard, 3 validata nel backtest** | `app.py` | Con `peso_storico = 0` per costruzione, è l'unico iperparametro che determina l'xG mostrato: la dashboard non girava sulla configurazione validata |
| 6 | **Dashboard e backtest usano quote di natura diversa** | `app.py` | Il backtest legge la quota della partita da prevedere, la dashboard la media dei precedenti. Inerente al fatto che la dashboard è una simulazione ipotetica, ma va detto: il 55.22% validato non descrive ciò che la dashboard calcola |

### Medie di lega pesate nel tempo — testato, positivo sul modello puro, neutro in produzione

Le formule dell'xG rapportano la forza di una squadra alla media di campionato, ma le statistiche di squadra decadono con emivita 730 giorni (descrivono gli ultimi 2-4 anni) mentre la media di lega era una media semplice su ~30 stagioni. Su questo dataset: 1.5135/1.1433 gol casa/trasferta su tutto lo storico contro 1.3932/1.2121 dal 2021. Semplificando algebricamente, `media_gol_casa` sparisce dalla formula dell'xG di casa e resta solo il divisore `media_gol_trasferta`, più basso del 6% del dovuto; simmetricamente l'xG di trasferta è sgonfiato dell'8%. **Il rapporto fra i due è distorto di circa il 15% a favore della casa su ogni partita** — la stessa modalità di errore già corretta una volta in Fase 1.

Validato con `valida_medie_lega.py` sul protocollo standard a 3 stagioni:

| Configurazione | Medie storiche | Medie pesate | Δ |
|---|---:|---:|---|
| Modello statistico puro (`peso_quote=0`) | 48.55% / 0.2067 | **49.96% / 0.2003** | **+1.41 pp**, coerente in 3/3 |
| Previsioni "1" (misura del bias pro-casa) | 76.9% | **72.2%** | il bias si riduce come previsto |
| Blend di produzione (`peso_quote=0.90`) | 55.22% / 0.1885 | 55.05% / 0.1884 | dentro il rumore |

La correzione migliora nettamente il modello statistico da solo, ma il guadagno **non si propaga al blend**: con il 90% del peso sul mercato, il 10% residuo non lo trasporta. Una grid search rifatta con le medie pesate (`valida_pesi_medie_pesate.py`) non trova un ottimo migliore di quello attuale (il massimo è 55.14% con forma=0.20/quote=0.80, contro 55.22% attuale).

**Adottato come default** (in `app.py` e nella sidebar del backtesting, dove resta selezionabile "storiche" per confronto). Il criterio è quello del progetto applicato con la metrica giusta: sull'accuratezza le due versioni sono indistinguibili (appena **4 partite discordanti su 2.659**), mentre sull'RPS — l'unica metrica con abbastanza potenza a questo campione, vedi sotto — la versione pesata è migliore in modo **statisticamente significativo** su 7 stagioni. Ed è la formulazione corretta a prescindere.

### ⚠️ Il problema di misura: nessun risultato di Fase 2 e 3 è statisticamente distinguibile

Questo è il risultato più importante emerso finora, e riguarda **tutte** le conclusioni del progetto, positive e negative.

Tutti gli esperimenti sono stati giudicati su differenze di accuratezza fra 0.1 e 0.9 punti percentuali. `valida_significativita.py` verifica se quelle differenze siano distinguibili dal rumore, con il test appropriato per confronti sugli stessi match (McNemar sulle predizioni discordanti, più bootstrap appaiato sull'RPS) — cioè il confronto **più favorevole possibile** alle differenze osservate, perché le due configurazioni sbagliano in gran parte le stesse partite.

Misurato su **7 stagioni** (2019-2025, 2.659 partite: l'intera finestra coperta dalle quote di chiusura, più del doppio del protocollo storico a 3 stagioni):

| Confronto | Δ accuratezza | Discordanti | McNemar | Δ RPS (IC95%) | Verdetto |
|---|---:|---:|---:|---|---|
| Quote chiusura vs apertura (*"il risultato più forte finora"*) | +0.38 pp | 62 / 2659 | p = 0.25 | −0.00054 [−0.00126, +0.00020] | non distinguibile |
| Medie pesate vs storiche | −0.15 pp | 4 / 2659 | p = 0.13 | **−0.00011 [−0.00021, −0.00002]** | **distinguibile su RPS** |
| **Modello (forma+quote) vs solo mercato** | **+0.34 pp** | **55 / 2659** | **p = 0.28** | **+0.00053 [−0.00007, +0.00113]** | **non distinguibile** |

Due osservazioni pesanti:

1. **Il modello non batte il mercato in modo misurabile.** Sull'accuratezza il vantaggio non è significativo; sull'RPS il modello è anzi leggermente *peggiore* del solo mercato. E il vantaggio **si dimezza allargando il campione**: +0.79 pp su 3 stagioni, +0.34 pp su 7. È la firma tipica di un risultato dovuto al caso — le 3 stagioni su cui i pesi sono stati cercati erano semplicemente favorevoli.
2. **L'accuratezza non ha abbastanza potenza per questo problema.** Solo il 2% delle partite cambia previsione fra due configurazioni: tutto il resto del campione non porta informazione sulla differenza. L'RPS usa invece ogni partita, ed è infatti l'unica metrica che ha prodotto un verdetto significativo.

**Conseguenza: l'affermazione centrale del progetto — "il modello batte il mercato di ~1 punto percentuale" — non è supportata dai dati.** Non è dimostrata falsa: è indistinguibile dal rumore, e si assottiglia quando il campione cresce. Simmetricamente, lo stesso vale per i risultati negativi: Elo, gradient boosting, indice motivazionale e gli altri sono stati scartati su differenze altrettanto piccole, che con quel campione non erano misurabili **in nessuna direzione**.

**Implicazione metodologica per il futuro**: usare l'RPS con intervallo di confidenza bootstrap come criterio primario e l'accuratezza solo come metrica descrittiva; usare la finestra a 7 stagioni invece che 3 (nessun dato nuovo necessario); e dichiarare esplicitamente "non distinguibile" invece di "leggermente meglio/peggio" quando è il caso. Diversi esperimenti archiviati come negativi meriterebbero di essere rivalutati con questo protocollo.

## Fase 0 — Protocollo di misura ✅

Introdotta dopo la scoperta che nessun risultato di Fase 2 e 3 fosse statisticamente distinguibile. Non aggiunge feature: rende interpretabili gli esperimenti.

### `protocollo.py` — il modulo di misura

Impone quattro regole a ogni confronto futuro:

1. **RPS come criterio primario**, con intervallo di confidenza bootstrap appaiato. Usa ogni partita, non solo quelle che cambiano previsione.
2. **Accuratezza come metrica descrittiva**, testata con McNemar e mai riportata come "meglio/peggio" senza il test.
3. **Finestra di test a 7 stagioni** (2019-2025), l'intera copertura delle quote di chiusura.
4. **Verdetto esplicito a tre valori**: `MEGLIO` / `PEGGIO` / `INDISTINGUIBILE`. Quest'ultimo è un risultato legittimo, non da arrotondare.

Include il calcolo di potenza: dato un confronto indistinguibile, dice quante partite servirebbero per risolverlo. Coperto da 11 test propri (`tests/test_protocollo.py`) che verificano entrambi i lati dell'errore — che riconosca un effetto vero **e** che dichiari indistinguibile il rumore.

### Un difetto che invalidava due conclusioni di Fase 3

`prototipo_gradient_boosting.py` chiamava `bt.precompute_componente` senza passare la configurazione delle quote, ereditandone i default `proporzionale`/`apertura`/`storiche`: cioè il modello **pre-Fase 2**. Sia il baseline statistico sia le feature `prob_1_quote`/`prob_X_quote`/`prob_2_quote` — 3 delle 12 feature del gradient boosting — erano calcolate col metodo vecchio, nonostante i docstring dichiarassero "quote di chiusura con Shin". Corretto con costanti esplicite (`METODO_QUOTE`, `FONTE_QUOTE`, `MEDIE_LEGA`).

### Risultati della rivalutazione (7 stagioni, 2.659 partite)

`valida_fase0_rivalutazione.py` e `valida_fase0_multiliga.py`:

| Confronto | Δ RPS | Δ acc | Discordanti | McNemar | Verdetto |
|---|---:|---:|---:|---:|---|
| Ensemble stacking vs statistico | +0.00042 | −0.15 pp | 44 | 0.65 | ➖ indistinguibile |
| Gradient boosting vs statistico | +0.00039 | +0.11 pp | 83 | 0.83 | ➖ indistinguibile |
| Stacking vs gradient boosting | +0.00004 | −0.26 pp | 97 | 0.54 | ➖ indistinguibile |
| **GB multi-campionato vs GB solo Serie A** | **−0.00065** | −0.56 pp | 63 | 0.08 | ➖ indistinguibile *(il più vicino)* |

**L'ipotesi di partenza è smentita.** La Fase 0 era nata dal sospetto che stacking e multi-campionato fossero risultati positivi archiviati per errore, perché avevano l'RPS migliore mai misurato (0.1880 e 0.1884 contro 0.1889). Con il baseline corretto l'ordinamento **si ribalta**: il modello statistico passa a 0.1908 e lo stacking a 0.1912. Quel primato era un artefatto del confronto con la configurazione pre-Fase 2. Non erano risultati positivi: non erano niente.

**Il calcolo di potenza è la parte che cambia la strategia:**

| Confronto | Partite per risolverlo su RPS | Su accuratezza |
|---|---:|---:|
| Stacking vs statistico | 13.540 (~36 stagioni) | 28.645 (~75 stagioni) |
| GB vs statistico | 24.573 (~65 stagioni) | 94.891 (~250 stagioni) |
| Stacking vs GB | 2.308.901 (~6.076 stagioni) | 20.532 (~54 stagioni) |
| **GB multi-campionato vs solo Serie A** | **5.521 (~15 stagioni)** | **3.039 (~8 stagioni)** |

La Serie A produce 380 partite l'anno: **queste domande non sono rispondibili con un solo campionato.** Non è un limite del protocollo, è la dimensione reale degli effetti.

### Conseguenze

1. **La decisione di non adottare stacking e gradient boosting resta valida, ma per un motivo diverso.** Non "sono peggiori" — non lo sono — bensì "non sono migliori, e costano due modelli in più da mantenere e riaddestrare". Fra configurazioni equivalenti si tiene la più semplice. Questa motivazione è difendibile; quella scritta in precedenza no.

2. **Il multi-campionato è l'unico esperimento ancora vivo.** È il più vicino alla significatività (p = 0.077, IC dell'RPS quasi interamente sotto lo zero) e l'unico che richiede una quantità di dati raggiungibile: ~5.500 partite contro le 2.659 attuali, cioè poco più del doppio.

3. **La via d'uscita dal regime "tutto indistinguibile"**: le altre 4 leghe sono già scaricate e usate solo come *training*. Usarle anche come **test** porterebbe il campione a ~13.000 partite su 7 stagioni — esattamente l'ordine di grandezza necessario a risolvere sia il multi-campionato sia lo stacking. Richiede di verificare che le quote di chiusura siano coperte anche lì e che il modello statistico sia calcolabile per-lega (entrambe le cose già fatte in `prototipo_gradient_boosting_multiliga.py`), ma nessun dato nuovo.

### Cosa resta aperto

**La priorità non è una feature: è il campione.** La Fase 0 ha mostrato che gli effetti in gioco richiedono 15-65 stagioni di test per essere misurati, mentre la Serie A ne produce una all'anno. Finché il campione resta questo, aggiungere feature significa scegliere a caso quali tenere.

| Priorità | Voce | Perché | Riferimento |
|---|---|---|---|
| ~~0~~ | ~~Protocollo di misura~~ | ✅ **fatto**, vedi la sezione Fase 0 sopra | `protocollo.py` |
| **1** | **Altre 4 leghe come TEST, non solo come training** | L'unico modo per uscire dal regime in cui ogni esperimento è indistinguibile: porta il campione da 2.659 a ~13.000 partite, l'ordine di grandezza che serve. Dati già scaricati, nessuna fonte nuova. Sblocca anche il multi-campionato, l'unico esperimento ancora vivo (p = 0.077) | Fase 0, conseguenza 3 |
| **2** | **Movimento apertura→chiusura e dispersione tra bookmaker** come feature separate | L'unica idea rimasta che segue la logica dei due miglioramenti adottati: nessun dato nuovo, solo informazione già nel dataset estratta meglio. Il delta apertura→chiusura è letteratura nota come proxy di denaro informato | Fase 2, punto 6 (in coda) |
| 3 | **pi-ratings** (Constantinou & Fenton) | L'unico sistema di rating continuo mai provato; nei paper supera Elo semplice sulla predizione 1X2 — ma Elo qui è risultato negativo, quindi l'aspettativa va tarata al ribasso | Tier 2 |
| 4 | **Valore di mercato Transfermarkt** come slope temporale (30/90/180gg) e prior shrinked | Cattura cambi di rosa che le medie storiche vedono in ritardo. Da usare come variazione, non come feature statica | Fase 2, punto 5 |
| 5 | **Fonte xG aggiornabile** | Il segnale Understat è debolmente positivo; serve una fonte che non si fermi a settembre 2024. Lo scarto xG − gol reali come proxy di regressione alla media resta l'idea più interessante | Fase 2, punto 4 |

Una nota di realismo su tutte e quattro: il mercato delle scommesse su Serie A è liquido ed efficiente, e sette anni di dati dicono che il margine sfruttabile sopra la quota di chiusura, se esiste, è inferiore a mezzo punto percentuale. È un risultato in sé, non un fallimento — ed è coerente con la letteratura sull'efficienza dei mercati di scommesse.

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
| **Rating Elo per club** | [clubelo.com](http://clubelo.com) — API REST gratuita (`api.clubelo.com/{Team}` restituisce CSV storico giornaliero) | Molto alta: CSV pronto, nessuno scraping aggressivo | Atteso alto (Hvattum & Arntzen 2010) — **testato: negativo**, vedi Fase 2 punto 1 | Sostituisce le medie storiche/forma attuali come feature di forza attacco/difesa, o come input diretto a un layer di ensemble insieme alle quote |
| **Expected Goals (xG) reali** | [understat.com](https://understat.com) (Serie A dal 2014/15, scraping HTML — dati in un tag `<script>` JSON) | Alta in teoria, **bassa in pratica**: scraping bloccato, usato dataset Kaggle fermo a set. 2024 | Atteso alto — **testato: debolmente positivo ma non adottabile**, vedi Fase 2 punto 4 | Sostituisce FTHG/FTAG come target per calcolare "attacco/difesa storico", oppure feature aggiuntiva in un modello XGBoost |
| **Quote di più bookmaker già scaricate** | Colonne già presenti nei file `stagioni/*.txt` (WHH, BWH, PSH, ecc. — solo `OddsAvgH/D/A` consolidato è usato oggi) | Molto alta: dato già scaricato | Medio — **non ancora testato**, è la voce aperta a priorità più alta | Estendere la cascata di `unisci_dati.py` per calcolare anche una deviazione standard tra bookmaker, utile come feature di "incertezza del mercato" |
| **Quote di chiusura** | Colonne `B365CH/CD/CA`, `AvgCH/CD/CA` già presenti nei file grezzi dal 2019 (7 stagioni) ma scartate da `unisci_dati.py`, che estraeva solo le quote "correnti" | Molto alta: dato già scaricato, verificato con valori diversi dalla quota non-chiusura | **Testato: positivo, il più forte finora** (+0.43 pp) — vedi Fase 2 punto 6, **adottato** | Colonne `OddsAvgCH/CD/CA` in `unisci_dati.py`; copertura solo 2019+, sufficiente per le 3 stagioni di validazione |
| **Movimento apertura→chiusura e dispersione tra bookmaker** | Stesse colonne, ma usate come *differenza* invece che come sostituzione | Molto alta: dato già in `serie_a.csv` | Da validare — il differenziale apertura/chiusura è letteratura nota come segnale su informazione privata/movimenti di mercato | Feature aggiuntive (`odds_move_*`, `odds_dispersion`), non sostituzione della quota nel blend |
| **Correzione di Shin per le quote** | Nessun dato nuovo — solo matematica su `OddsAvgH/D/A` | Molto alta | **Testato: positivo**, vedi Fase 2 punto 2, **adottato** | Sostituisce la normalizzazione proporzionale `1/quota` prima del blend |
| **RPS/log-loss per configurazione** | Già implementato | — | — | Continuare a usarli come criterio primario invece della sola accuratezza quando si aggiungono nuove feature |

### 🟡 Tier 2 — Impatto medio-alto, richiede una pipeline dati nuova

| Dato/Feature | Fonte specifica | Fattibilità | Impatto atteso | Come entra nel modello |
|---|---|---|---|---|
| **Valore di mercato rosa** | [transfermarkt.com](https://www.transfermarkt.com) (scraping non ufficiale — pacchetti come `transfermarkt-scraper` su GitHub) o dataset Kaggle "Transfermarkt Football Data" | Media: nessuna API ufficiale | Medio-alto — Kuper & Szymanski (*Soccernomics*) mostrano correlazione ~0.9 tra monte ingaggi e posizione in classifica; cattura cambi di rosa che le medie storiche non vedono subito. **Non testato** | Feature indipendente in XGBoost, oppure prior per la forza attacco/difesa nel modello Bayesiano (Fase 3) |
| **Statistiche avanzate (tiri, possesso, passaggi progressivi)** | [fbref.com](https://fbref.com) (dati StatsBomb via Sports Reference, Serie A dal 2017/18) — libreria Python `soccerdata` | Bassa in pratica: FBref risponde 403 in questo ambiente (verificato in Fase 2 punto 4) | Alto per partite recenti, ma copre solo da 2017/18 in poi. **Non testato** — i tiri già nel dataset sono però risultati negativi | Feature engineering per XGBoost (rolling average ultimi N tiri/xG/possesso) |
| **API-Football (fixtures, formazioni, infortuni, quote)** | [api-football.com](https://www.api-football.com) — tier gratuito 100 richieste/giorno, a pagamento per storico/infortuni completi | Media: gratis limitato | Alto per un modello "match-day" — **valutato e non implementato**, vedi Fase 3 punto 4 | Aggiustamento moltiplicativo dell'xG basato su "forza XI titolare atteso" vs "forza rosa completa" |
| **Sistema di rating pi-ratings** | Nessuna fonte esterna — solo implementazione (Constantinou & Fenton, *pi-football-ratings*, paper pubblico) | Media: formula pubblicata ma più complessa di Elo | Atteso alto (nei paper supera Elo semplice sulla predizione 1X2), ma **Elo qui è risultato negativo**: aspettativa da tarare al ribasso. Non testato | Sostituisce interamente la sezione "storico" con un rating che si aggiorna partita per partita |
| **Meteo storico allo stadio** | [open-meteo.com](https://open-meteo.com) (API storica gratuita, no key richiesta) | Alta, ma serve mappare squadra→stadio→coordinate | Basso-medio — effetto reale ma piccolo su pioggia/vento e gol totali. **Non testato** | Feature minore in XGBoost |
| **Shots/corner/cartellini già nel dataset** | Già estratti da `unisci_dati.py` (colonne HS, AS, HST, AST, HC, AC, HY, AY, HR, AR) | Molto alta: dato già presente in `serie_a.csv` | **Testato: negativo**, vedi Fase 2 punto 7 | Feature per un modello XGBoost, o media pesata nel tempo come proxy xG-like semplificato |
| **Giorni di riposo / congestione di calendario** | Colonna `Date`, già disponibile | Molto alta | **Testato: negativo**, vedi Fase 2 punto 7 | Proxy di fatica come feature del gradient boosting |

### 🔴 Tier 3 — Avanzato, alto sforzo o costo

| Dato/Feature | Fonte specifica | Fattibilità | Impatto atteso | Come entra nel modello |
|---|---|---|---|---|
| **Modello Bayesiano gerarchico** (Baio & Blangiardo 2010) | Nessun dato esterno — `PyMC` non installabile (llvmlite), sostituito con MAP via scipy | Bassa | Testato: **negativo**, vedi Fase 3 | Sostituisce il layer statistico attuale: ogni squadra ha un parametro attacco/difesa con prior condiviso, che si restringe verso la media di lega in proporzione all'incertezza |
| **Formazioni/infortuni in tempo reale** | API-Football piano a pagamento, o scraping Transfermarkt pagina infortuni | Bassa-media: dati storici affidabili difficili da reperire gratis | Alto ma solo per previsioni "a ridosso della partita" — **valutato e non implementato**, vedi Fase 3 punto 4 | Feature runtime, non backtestabile facilmente sullo storico per mancanza di dati infortuni retroattivi |
| **Ensemble stacking** (Poisson-Dixon-Coles + Elo + XGBoost + quote) | Nessun dato esterno | Media: tecnica nota, serve disciplina per evitare leakage nel meta-learner | Atteso alto (Groll et al. 2019) — **testato: misto, non adottato**, vedi Fase 3 punto 2 | Meta-learner allenato out-of-fold sulle probabilità dei modelli base |
| **Multi-campionato** | football-data.co.uk copre già Premier/Liga/Bundesliga/Ligue 1 con lo stesso formato | Alta come dato, media come impatto | **Testato: positivo su RPS, neutro su accuratezza, non adottato**, vedi Fase 3 punto 3 | Training set esteso per il gradient boosting, mai come test |
| **Indice motivazionale di fine stagione** | Nessun dato nuovo — ricostruzione classifica dai risultati | Media | **Testato: negativo**, vedi Fase 3 punto 5 | Feature del gradient boosting: distanza dalle soglie salvezza/Europa |
| **`Squad_Rotation_Index`, distanza di viaggio, turnover pre-coppe** | Richiedono formazioni/minutaggio, calendario coppe, geocoding | Bassa | Non verificato — dati non disponibili | Scartati, vedi "Cosa abbiamo scartato da revisioni esterne" |

---

## Tecniche di modellazione: cosa adottare

In ordine di ritorno sull'investimento:

1. **Dixon-Coles completo** — ✅ fatto in Fase 1 (tau + decadimento temporale).
2. **Rating system continuo (Elo o pi-ratings)** al posto delle medie storiche statiche — Elo **testato, negativo** (Fase 2 punto 1): cattura informazione che il mercato incorpora già. I pi-ratings restano l'unica variante non provata, ma il risultato di Elo consiglia di tarare al ribasso l'aspettativa.
3. **XGBoost/LightGBM come sfidante, non sostituto** — **testato, negativo** (Fase 2 punto 3, con `HistGradientBoostingClassifier`: XGBoost richiede `libomp`, non disponibile in questo ambiente). Il rischio di overfitting con ~11.500 partite si è materializzato: 85% in training contro 48% in test con 5 stagioni. Nemmeno con forte regolarizzazione, feature aggiuntive o training multi-lega ha mai battuto il modello statistico in accuratezza.
4. **Reti neurali: sconsigliate.** Con questo volume di dati la letteratura (Groll, Baboota & Kaur 2019) mostra alberi/ensemble e modelli Poisson competitivi o superiori alle reti profonde — e qui nemmeno gli alberi hanno funzionato.
5. **Ensemble stacking** come step finale — **testato, misto** (Fase 3 punto 2): miglior RPS mai misurato (0.1880), ma accuratezza sotto il modello statistico. Non adottato per il rapporto complessità/beneficio.

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

## Fase 2 — Completata ✅ (6 voci su 7 chiuse, 2 adottate)

Esito in sintesi: **Shin** e **quote di chiusura** adottati in produzione; Elo, gradient boosting, tiri/corner, giorni di riposo e arbitro negativi; xG Understat positivo ma non adottabile per limiti della fonte. Resta aperto solo il punto 5 (Transfermarkt).


1. ~~Integrare Elo da clubelo.com~~ — **fatto, risultato negativo.** Storico Elo delle 53 squadre scaricato (`scarica_elo.py` → `elo_storico.csv`), lookup point-in-time in `modello.py` (`elo_asof_batch`), integrato come componente pesata in `pages/backtesting.py` e calibrato con una regressione di Poisson vera (`calibra_regressione_elo`, non una costante indovinata). Validato sulle stesse 3 stagioni indipendenti della Fase 1: **nessuna combinazione batte la configurazione attuale** (54.87% medio senza Elo vs 54.79% nella miglior combinazione con Elo). Codice tenuto e testato (`tests/test_model.py`), `peso_elo` di default a 0. Probabile causa: ClubElo cattura informazione che il mercato incorpora già, stessa storia di storico/scontri diretti in Fase 1. **Nota deploy:** `elo_storico.csv` è gitignored (non presente su Streamlit Cloud), quindi `pages/backtesting.py` carica Elo in modo tollerante all'assenza del file (`ELO_DISPONIBILE`): se manca, lo slider Elo si disabilita invece di far crashare l'intera pagina.
2. ~~Correzione di Shin per le quote (de-overrounding)~~ — **fatto, primo risultato positivo della Fase 2.** La quota di consenso veniva convertita in probabilità con `1/quota` normalizzato in proporzione (metodo "basic"), che non corregge la favorite-longshot bias (i bookmaker mettono più margine sulle quote alte che su quelle basse). Implementata la formula chiusa di Shin (1992/1993, `probabilita_shin` in `modello.py`, derivazione di Štrumbelj 2014) come alternativa, integrata in `pages/backtesting.py` (selettore "Metodo conversione quote") e in `app.py`. Validata con lo stesso protocollo a 3 stagioni indipendenti usato per Elo e gradient boosting, pesi fissi alla configurazione già ottimale (forma=0.10, scontri=0, quote=0.90): Shin migliora l'RPS in **tutte e 3** le stagioni (media 0.1886 contro 0.1889 proporzionale), con accuratezza sostanzialmente invariata (54.79% contro 54.87%, -0.08 punti percentuali in media, spiegato da poche partite quasi in parità nella sola stagione 2025 — non un vero peggioramento). A differenza di Elo e gradient boosting, non richiede dati nuovi né addestramento: cambia solo come si toglie il margine dalle quote già in uso. Adottato come default in `app.py` e in `pages/backtesting.py` (resta selezionabile il metodo proporzionale per confronto). Script di validazione: `valida_shin.py`.
3. ~~Prototipo gradient boosting su feature strutturate~~ — **fatto, risultato negativo.** `prototipo_gradient_boosting.py`: `HistGradientBoostingClassifier` di scikit-learn (non XGBoost, che richiede la libreria di sistema `libomp` non disponibile in questo ambiente) allenato sulle stesse componenti già validate (xG storico/forma/scontri diretti, Elo, probabilità implicite delle quote) come feature. Con pochi dati di training (5 stagioni, ~1900 partite) overfit severo (85% accuratezza su training, 48% su test). Con 10 stagioni (~3800 partite) e forte regolarizzazione (early stopping, `l2_regularization`) il divario si riduce ma il risultato resta **leggermente sotto** il modello statistico su tutte e 3 le stagioni indipendenti: 54.17% di accuratezza media e RPS 0.1896, contro 54.87%/0.1889 del modello attuale. Causa probabile: dati Serie A troppo limitati (~380 partite/stagione) perché un modello ad alberi trovi interazioni non lineari utili oltre a quelle già catturate dal blend statistico calibrato.
4. ~~xG reali di Understat al posto dei gol grezzi~~ — **fatto, risultato cautamente positivo ma NON adottabile in produzione.** Lo scraping diretto di understat.com e fbref.com è risultato bloccato in questo ambiente (Understat non incorpora più i dati nell'HTML per richieste non-browser, FBref risponde 403): usato invece un dataset già raccolto (Kaggle, `codytipton/understat-data`, scaricato manualmente dall'utente in `archive/`, gitignored per dimensione ~21MB). Il dataset non ha un id di partita condiviso tra le due squadre (una riga per squadra per partita, senza nome avversario): le partite sono ricostruite abbinando ogni riga alla lista ufficiale di `serie_a.csv` per (squadra, data). Copertura Serie A: stagioni 2014-2023 complete, 2024 parziale (59 partite, l'esportazione Kaggle si ferma a fine settembre 2024), 2025 assente — le 3 stagioni di test standard (2023/2024/2025) non sono quindi utilizzabili. Validato invece su 2021/2022/2023 (`valida_understat.py`), sostituendo storico/forma calcolati sui gol reali con la stessa media pesata nel tempo calcolata sull'xG di Understat, **ricalcolando anche il modello attuale sulla stessa finestra** per un confronto onesto (il riferimento 54.87%/0.1889 è su 2023-2025, non comparabile qui): xG Understat batte i gol reali in media (53.87% acc / 0.1919 RPS contro 53.43%/0.1928), ma con risultati incoerenti tra stagioni (2021: +2.1 punti percentuali; 2022: pari; 2023: -0.8 punti percentuali) — un segnale reale ma debole, non un miglioramento netto come Shin o le quote di chiusura. **Non integrato in `app.py`/`pages/backtesting.py`**: anche accettando il segnale, il dataset disponibile si ferma a settembre 2024, quindi la feature andrebbe in stallo esattamente per le partite più recenti che l'app deve prevedere oggi — servirebbe una fonte aggiornabile (scraping funzionante o un abbonamento a un'API a pagamento) prima di poterlo usare in produzione. Lo scarto xG − gol reali come segnale di "regressione alla media" resta un'idea da provare se in futuro si risolve il problema della fonte dati.
5. Scraping valore di mercato Transfermarkt come proxy di qualità rosa aggiornata — usarlo come **variazione temporale** (slope 30/90/180gg) e come prior "shrinked" per un modello gerarchico, non come feature statica diretta (raffinamento da una revisione esterna).
6. ~~Quote di chiusura~~ — **fatto, secondo risultato positivo della Fase 2 (il più forte finora).** `B365CH/CD/CA` e `AvgCH/CD/CA` erano già nei file grezzi dal 2019 ma scartati da `unisci_dati.py`; ora estratte con la stessa logica a cascata delle quote di apertura (`OddsAvgCH/CD/CA` in `serie_a.csv`, 2.660 partite coperte su 11.534, cioè le 7 stagioni dal 2019). Validato con lo stesso protocollo a 3 stagioni indipendenti (tutte con copertura piena, 2019+), pesi fissi alla configurazione ottimale e metodo Shin già adottato: la quota di chiusura (a ridosso del fischio d'inizio, incorpora più informazione di mercato dell'apertura) batte l'apertura in 2 stagioni su 3 (2024: 53.68%→54.74%; 2023: 54.74%→55.53%; 2025: lieve calo 55.94%→55.41%), con **accuratezza media +0.43 punti percentuali** (55.22% contro 54.79%) e RPS leggermente migliore (0.1885 contro 0.1886). A differenza di Shin, migliora anche l'accuratezza pura, non solo la calibrazione. Adottato come default in `app.py` (con fallback per-partita su apertura per gli scontri diretti pre-2019) e in `pages/backtesting.py` (selettore "Fonte quote"). Script di validazione: `valida_quote_chiusura.py`. Il **movimento apertura→chiusura** e la **dispersione tra bookmaker** come feature separate (segnale di informazione privata / incertezza di mercato, non sostituzione della quota) restano da testare, probabilmente via il prototipo gradient boosting.
7. Feature "quasi gratis" da dati già presenti, nessuno scraping nuovo:
   - ~~Tiri in porta/corner/cartellini~~ — **fatto, risultato negativo.** `prototipo_gradient_boosting.py --con-tiri` aggiunge le medie recenti (tiri in porta, corner) al prototipo gradient boosting. Risultato misto tra stagioni (2025: 54.4%→55.4%, migliora; 2024: 54.2%→52.6%, peggiora; 2023: stabile), ma in media **leggermente peggio** del gradient boosting senza queste feature: 54.00% di accuratezza e RPS 0.1898, contro 54.17%/0.1896 senza tiri e 54.87%/0.1889 del modello statistico. Nessun beneficio netto, coerente col tema di questa fase: più feature non aiutano se il volume dati resta lo stesso.
   - ~~Giorni di riposo, congestione di calendario, trasferta dopo trasferta~~ — **fatto, risultato negativo.** `prototipo_gradient_boosting.py --con-riposo` aggiunge giorni dall'ultima partita, numero di partite negli ultimi 14 giorni e se la partita precedente della squadra era in trasferta (elaborazione da una revisione esterna; scartato `partita_di_coppa_in_5_giorni` perché richiederebbe il calendario delle coppe europee, dato che non abbiamo). Validato sulle stesse 3 stagioni indipendenti: 54.17% di accuratezza media e RPS 0.1893, sostanzialmente identico al gradient boosting senza queste feature (54.17%/0.1896) e sotto il modello statistico (54.87%/0.1889). Nessun beneficio: il calendario di Serie A (turni settimanali regolari, poche squadre impegnate in coppe europee) probabilmente non genera abbastanza variazione di riposo da essere un segnale utile oltre a quanto già catturato dalle quote.
   - ~~Tendenza dell'arbitro (falli/rigori/cartellini)~~ — **scartato, verificato.** La colonna `Referee` è presente nei file grezzi solo per le stagioni 2005 e 2006 (2 su 33), poi mai più: dato insufficiente per costruire una feature storica utilizzabile su tutto il periodo di test. Confermato da una seconda revisione esterna, che infatti raccomanda la stessa cautela.
   - `Squad_Rotation_Index`, distanza di viaggio, infortuni/XI atteso in forma probabilistica: richiedono dati che non abbiamo (formazioni/minutaggio, geocoding stadi, rating giocatori) — restano Tier 3, non "quasi gratis" come proposto da alcune revisioni esterne.

## Fase 3 — Completata ✅ (5 voci su 5 chiuse, 0 adottate)

Esito in sintesi: **nessuna delle cinque voci è stata adottata**. Bayesiano gerarchico e indice motivazionale negativi; ensemble stacking e multi-campionato migliorano l'RPS ma non l'accuratezza; API-Football valutato e non implementato per scelta. È la fase che ha prodotto la lezione centrale del progetto — vedi "📍 Stato attuale" in cima.


1. ~~Modello Bayesiano gerarchico (Baio & Blangiardo) con partial pooling~~ — **fatto, risultato negativo.** `pip install pymc` fallisce in questo ambiente: `llvmlite` (dipendenza di `numba`, il motore JIT di PyMC) non ha ancora una wheel precompilata per Python 3.14 e non compila da sorgente (stessa natura del problema `libomp`/XGBoost in Fase 2). Sostituito con una stima puntuale equivalente (`prototipo_bayesiano_gerarchico.py`): massima verosimiglianza penalizzata (Poisson, log-link, vantaggio-casa comune, penalità L2 su attacco/difesa per squadra), che corrisponde matematicamente alla moda a posteriori (MAP) dello stesso modello gerarchico con prior Normali — stesso effetto di partial pooling per le squadre con pochi dati, ma solo la stima puntuale, senza l'incertezza a posteriori completa che darebbe l'MCMC. Ri-adattato ogni ~10 partite (walk-forward). **Scoperta metodologica importante nel percorso**: ai pesi ottimali già validati (forma=0.10, quote=0.90, che sommano esattamente a 1) il peso della sola componente "storico" nel blend è 0 per costruzione (`peso_storico = 1 - pf - ps - pe - pq` in `valuta_componente`), quindi sostituire solo lo storico è un no-op indipendentemente dal suo valore; il test corretto sostituisce sia storico sia forma con la stessa stima gerarchica. Validato sulle 3 stagioni indipendenti: 54.35% di accuratezza media e RPS 0.1883, contro 55.22%/0.1885 sostituendo solo con lo storico/forma da gol reali (stessa configurazione Shin+chiusura) — **peggiore in accuratezza**, RPS sostanzialmente invariato. Probabile causa: il refit periodico (ogni ~10 partite) con forte regolarizzazione produce una stima di forza squadra che si muove lentamente nel tempo, perdendo il segnale di "forma recente" (ultime 3 partite) che la media mobile attuale cattura e che si è già dimostrato utile da solo.
2. ~~Ensemble stacking finale~~ (Poisson-Dixon-Coles + gradient boosting, meta-learner allenato out-of-fold) — **fatto, risultato misto, non adottato.** `prototipo_ensemble_stacking.py`: le probabilità del modello statistico (storico+forma+quote di chiusura con Shin) e del prototipo gradient boosting (Fase 2, punto 3) vengono combinate da un secondo `HistGradientBoostingClassifier` poco profondo usato come meta-learner, allenato su predizioni **out-of-fold** (5-fold CV sul training set, mai sulle stesse righe usate per allenare i modelli di base, altrimenti il meta-learner vedrebbe predizioni "facili" e il beneficio sarebbe sovrastimato). Il meta-learner ad alberi realizza di per se' il "blend condizionato al contesto" suggerito da una revisione esterna (impara da solo quando fidarsi di più dell'uno o dell'altro dalle 6 probabilità di input), senza bisogno di specificare a mano quali segnali di mismatch usare. Elo non incluso (già negativo, vedi Fase 2). Validato sulle 3 stagioni indipendenti: RPS leggermente migliore di entrambi i modelli di base (0.1880 contro 0.1889 statistico e 0.1896 gradient boosting), ma **accuratezza sotto il modello statistico puro** (54.35% contro 54.87%, sebbene sopra il gradient boosting da solo, 54.17%). A differenza di Shin e delle quote di chiusura (guadagno pulito su entrambe le metriche o quasi), qui il compromesso è ambiguo: un piccolo guadagno di calibrazione a fronte di una perdita di accuratezza più marcata, con in più il costo di mantenere in produzione due modelli aggiuntivi (gradient boosting + meta-learner, entrambi da ri-allenare periodicamente) per un beneficio modesto. **Non adottato in produzione**: il rapporto complessità/beneficio non lo giustifica, soprattutto perché il gradient boosting alla base non ha mai battuto il modello statistico in nessun test di questa fase.
3. ~~Estensione multi-campionato~~ per aumentare il volume dati della componente ML — **fatto, risultato positivo sulla calibrazione, neutro sull'accuratezza.** `scarica_altre_leghe.py` scarica lo storico di Premier League, Liga, Bundesliga e Ligue 1 da football-data.co.uk (stessa fonte e formato di `stagioni/*.txt`, cartella `altre_leghe/` gitignored); `prototipo_gradient_boosting_multiliga.py` calcola le componenti (storico/forma/scontri/quote, senza Elo) walk-forward per ciascuna lega separatamente (le medie gol si confrontano solo all'interno della stessa lega) e le usa come righe di training AGGIUNTIVE per il gradient boosting, mai come test (il test resta sempre e solo Serie A). Per tenere il tempo di calcolo gestibile, lo storico passato a `stats_pesate_squadre`/`calcola_forma_bt` è troncato a una finestra scorrevole (altrimenti il costo cresce quadraticamente con le partite totali della lega) e ogni lega è limitata alle ultime ~6.000 partite (~15-16 stagioni). Validato sulle 3 stagioni indipendenti (~22.400 partite aggiuntive nel pool, training totale 26.200 righe contro 3.799 di sola Serie A): l'RPS migliora in **tutte e 3** le stagioni (media 0.1884 contro 0.1896 solo Serie A — meglio persino del modello statistico, 0.1889), confermando che il volume dati era un fattore limitante per la calibrazione del gradient boosting; l'accuratezza resta sostanzialmente invariata (54.08% contro 54.17%, differenza dentro il rumore, entrambe comunque sotto il modello statistico, 54.87%). **Non adottato in produzione**: anche col miglioramento, il gradient boosting non supera il modello statistico in accuratezza, e mantenere una pipeline di 4 leghe aggiuntive (download, parsing, ricalcolo walk-forward) non è giustificato per un guadagno che resta comunque inferiore al modello statistico già in uso.
4. **Formazioni/infortuni via API-Football** — *valutato, non implementato per scelta.* Richiede un piano a pagamento per uno storico affidabile, e la condizione posta da questa stessa voce ("solo se il caso d'uso si sposta da dashboard esplorativa a previsione a ridosso del match reale") non si è verificata: PredictA resta una dashboard esplorativa. Non implementato per non impegnare una spesa ricorrente senza un cambio di scopo del progetto che la giustifichi.
5. ~~Indice motivazionale di fine stagione~~ (proposto indipendentemente da due revisioni esterne) — **fatto, risultato negativo.** `prototipo_gradient_boosting.py --con-motivazione` ricostruisce la classifica punto-per-punto in modo walk-forward (`costruisci_classifiche_progressive`, nessun dato futuro) e calcola per ogni partita la distanza di ciascuna squadra dalla soglia salvezza (17° posto) e dalla soglia Europa (6° posto), oltre alle partite giocate in stagione, come feature aggiuntive del prototipo gradient boosting. Validato sulle 3 stagioni indipendenti: 53.91% di accuratezza media e RPS 0.1897, **sotto** sia il gradient boosting senza questa feature (54.17%/0.1896) sia il modello statistico (54.87%/0.1889), coerente in tutte e 3 le stagioni (nessuna in controtendenza). Probabile causa: le partite genuinamente "senza obiettivi" nelle ultime giornate sono poche per stagione (tipicamente una manciata di squadre di centroclassifica sicura), un volume insufficiente perché il modello impari un pattern affidabile oltre al rumore.

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
