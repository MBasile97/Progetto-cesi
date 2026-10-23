from langchain_core.tools import tool

# ==========================================
# STATO DEL MONDO CONDIVISO
# ==========================================

# La dispensa
dispensa = {
    "lacrima_di_fenice": 3,
    "polvere_di_stelle": 2,
    "radice_di_mandragora": 4,
    "zanna_di_serpente": 1
}

# Il calderone
calderone = {
    "fuoco": "spento",        # Valori ammessi: "spento", "basso", "medio", "alto"
    "ingredienti": {},        # es. {"lacrima_di_fenice": 2}
    "mescolato": 0,           # Contatore di quante volte è stato mescolato
    "stato": "in_preparazione"
}


# ==========================================
# TOOL A DISPOSIZIONE DELL'AGENTE
# ==========================================

@tool
def ispeziona_dispensa() -> str:
    """Ispeziona la dispensa e restituisce la lista di tutti gli ingredienti disponibili con le relative quantità."""
    if not dispensa:
        return "La dispensa è completamente vuota."
    
    report = "Scorte disponibili in dispensa:\n"
    for ing, qta in dispensa.items():
        report += f"- {ing}: {qta} unità\n"
    return report.strip()


@tool
def guarda_calderone() -> str:
    """Controlla lo stato attuale del calderone: livello del fuoco, ingredienti già inseriti e quante volte è stato mescolato."""
    ing_str = ", ".join([f"{k} ({v})" for k, v in calderone["ingredienti"].items()]) if calderone["ingredienti"] else "nessuno"
    return (
        f"Stato Calderone:\n"
        f"- Livello del fuoco: {calderone['fuoco']}\n"
        f"- Ingredienti contenuti: {ing_str}\n"
        f"- Volte mescolato: {calderone['mescolato']}\n"
        f"- Stato pozione: {calderone['stato']}"
    )


@tool
def regola_fuoco(livello: str) -> str:
    """Regola l'intensità del fuoco sotto il calderone.
    Argomenti:
        livello: una tra le seguenti opzioni: 'spento', 'basso', 'medio', 'alto'.
    """
    opzioni_valide = ["spento", "basso", "medio", "alto"]
    livello_pulito = livello.strip().lower()
    
    if livello_pulito not in opzioni_valide:
        return f"Errore: livello '{livello}' non valido. Le opzioni valide sono: {', '.join(opzioni_valide)}."
    
    calderone["fuoco"] = livello_pulito
    return f"Fatto! Il fuoco sotto il calderone ora è regolato su '{livello_pulito}'."


@tool
def aggiungi_ingrediente(nome_ingrediente: str, quantita: int) -> str:
    """Preleva un ingrediente dalla dispensa e lo inserisce dentro il calderone.
    Argomenti:
        nome_ingrediente: il nome esatto dell'ingrediente da prendere dalla dispensa (es. 'lacrima_di_fenice').
        quantita: numero intero di unità da inserire.
    """
    nome_pulito = nome_ingrediente.strip().lower().replace(" ", "_")
    
    if nome_pulito not in dispensa:
        return f"Errore: '{nome_ingrediente}' non esiste nella dispensa."
    
    if quantita <= 0:
        return "Errore: la quantità deve essere un numero positivo maggiore di zero."
    
    if dispensa[nome_pulito] < quantita:
        return (
            f"Errore: quantità insufficiente in dispensa. "
            f"Hai richiesto {quantita} di {nome_pulito}, ma ne abbiamo solo {dispensa[nome_pulito]}."
        )
    
    # Scala dalla dispensa
    dispensa[nome_pulito] -= quantita
    
    # Aggiungi al calderone
    if nome_pulito in calderone["ingredienti"]:
        calderone["ingredienti"][nome_pulito] += quantita
    else:
        calderone["ingredienti"][nome_pulito] = quantita
        
    return f"Aggiunto con successo: {quantita} unità di '{nome_pulito}' al calderone. Rimanenti in dispensa: {dispensa[nome_pulito]}."


@tool
def mescola_calderone(volte: int = 1) -> str:
    """Mescola il contenuto del calderone per un certo numero di volte.
    Argomenti:
        volte: quante volte girare il mestolo (default 1).
    """
    if volte <= 0:
        return "Errore: devi mescolare almeno una volta."
    
    calderone["mescolato"] += volte
    return f"Hai mescolato il calderone per {volte} volta/e. Conteggio totale mescolate: {calderone['mescolato']}."


@tool
def verifica_pozione_completata() -> str:
    """Verifica se la pozione nel calderone è pronta e corretta rispetto alle condizioni alchemiche."""
    # Condizioni per la pozione 'Visione Astrale':
    # 1. Fuoco medio
    # 2. 2 lacrime di fenice
    # 3. 1 radice di mandragora
    # 4. Mescolato esattamente 3 volte in totale (1 prima + 2 dopo)
    
    errori = []
    if calderone["fuoco"] != "medio":
        errori.append(f"Il fuoco è su '{calderone['fuoco']}', ma doveva essere su 'medio'.")
        
    ing = calderone["ingredienti"]
    if ing.get("lacrima_di_fenice", 0) != 2:
        errori.append(f"Nel calderone ci sono {ing.get('lacrima_di_fenice', 0)} lacrime di fenice (ne servivano 2).")
        
    if ing.get("radice_di_mandragora", 0) != 1:
        errori.append(f"Nel calderone ci sono {ing.get('radice_di_mandragora', 0)} radici di mandragora (ne serviva 1).")
        
    # Controlliamo che non ci siano ingredienti estranei
    altri_ing = [k for k in ing.keys() if k not in ["lacrima_di_fenice", "radice_di_mandragora"]]
    if altri_ing:
        errori.append(f"Ci sono ingredienti non richiesti: {', '.join(altri_ing)}.")
        
    if calderone["mescolato"] < 3:
        errori.append(f"La pozione è stata mescolata solo {calderone['mescolato']} volte (servono 3 mescolate).")
    elif calderone["mescolato"] > 3:
        errori.append(f"La pozione è stata mescolata troppo ({calderone['mescolato']} volte invece di 3) ed è instabile!")
        
    if errori:
        calderone["stato"] = "fallita"
        return "Pozione Fallita! Motivi:\n- " + "\n- ".join(errori)
    else:
        calderone["stato"] = "riuscita"
        return "Successo Straordinario! La Pozione della Visione Astrale brilla di una luce dorata ed è completata alla perfezione!"