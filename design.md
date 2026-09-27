# Brand & Design Guidelines — Linear Aesthetic

Usa questo file come guida vincolante per l'UI e la stilizzazione di tutte le componenti web dell'applicazione.

## 1. Color Palette (Dark Theme Focus)

- **Background Principale (`bg-app`):** `#08090a` (Nero/Grigio profondo)
- **Superfici / Card (`bg-surface`):** `#18191c` / `#1e2022`
- **Superfici Hover / Secondary:** `#24262a` / `#27282c`
- **Bordi sottili (`border-subtle`):** `#323439` (1px solid)
- **Accento Primario (Linear Purple):** `#5e69d1` (Hover: `#6a76e3` / Active: `#535ebf`)
- **Testo Primario:** `#f7f8f8`
- **Testo Secondario / Muted:** `#8f9197` / `#a8aab0`

### Colori di Stato e Metric (Badge/Quote/Valore)
- **Success / Green (Quote / Value Positive):** `#26a544` (Bg soft: `#1f2c24`, testo: `#3de261`)
- **Warning / Orange (Quote Medie):** `#ff7235`
- **Danger / Red (Loss / Value Negativo):** `#ff5d5e`
- **Info / Cyan (NBA / Dati Tattici):** `#00b8cb`

---

## 2. Tipografia

- **Font Family:** `Inter`, `-apple-system`, `BlinkMacSystemFont`, `sans-serif`
- **Pesi:**
  - Regular (`400`) per testi correnti e dati tabella.
  - Medium (`500`) per bottoni, label e intestazioni di colonna.
  - Semi-Bold (`600`) per titoli, quote e numeri chiave.

---

## 3. Componenti & Stili di Layout

### Cards & Contenitori
- Usa uno sfondo leggermente staccato (`#18191c`) con un bordo sottile da 1px (`border-[#323439]`).
- **Border-radius:** `8px` (`rounded-lg`) per card e riquadri modulari.
- Evita ombre invadenti: prediligi un leggero bagliore interno o bordi definiti.

### Bottoni & Input
- **Radius Bottoni Primari/Pillole:** `9999px` (`rounded-full`) o `6px` (`rounded-md`).
- **Input / Form:** Sfondo scuro (`#1b1e25`), bordo 1px (`#323439`), focus ring con colore d'accento `#5e69d1`.

### Tabelle & Griglie Dati (Essenziale per Schedine/Quote)
- Evita lo stile griglia classica "foglio di calcolo".
- Usa righe flessibili con separatori sottili `border-b border-[#323439]/50`.
- Applica stati di `:hover` sulle righe delle partite con sfondo `#24262a`.

---

## 4. Direttive per la Scrittura del Codice UI

1. Mantieni un'estetica minimale, compatta e ad elevata densità informativa (data-dense).
2. Ogni pannello o sezione deve avere confini chiari delimitati da bordi `1px` anziché spaziature vuote ampie.
3. Riduci al minimo le animazioni, preferendo micro-transizioni veloci (`150ms ease-in-out`).