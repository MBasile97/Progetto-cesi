import json
import re
import select
import sys
import uuid
from typing import List, Dict, Any

from llama_cpp import Llama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# Importiamo i tool e lo stato del mondo
from world import (
    ispeziona_dispensa,
    guarda_calderone,
    regola_fuoco,
    aggiungi_ingrediente,
    mescola_calderone,
    verifica_pozione_completata
)

# ==========================================
# ⚙️ CONFIGURAZIONE PARAMETRI ESPERIMENTO
# ==========================================
TEMPO_INATTIVITA_SECONDI = 20  # Secondi di silenzio prima dell'intervento

# ==========================================
# 1. SETUP TOOL
# ==========================================
tools = [
    ispeziona_dispensa,
    guarda_calderone,
    regola_fuoco,
    aggiungi_ingrediente,
    mescola_calderone,
    verifica_pozione_completata
]

tools_schema = [convert_to_openai_tool(t) for t in tools]

# ==========================================
# 2. CARICAMENTO DEL MODELLO
# ==========================================
print("⏳ Caricamento di Qwen 3.8 27B sulla RTX A4000...")
llm = Llama(
    model_path="./Qwen3.8-27B-UD-IQ4_XS.gguf",
    n_gpu_layers=-1,
    n_ctx=2048,
    verbose=False
)
print("✅ Modello caricato con successo!")

SYSTEM_PROMPT = """Sei un alchimista esperto e collaborativo, compagno di laboratorio dell'utente.
Lavorate INSIEME per preparare pozioni alchemiche.
TU gestisci operativamente la dispensa e il calderone usando i tuoi tool.
IMPORTANTE: TU NON CONOSCI LA RICETTA DELLA POZIONE! Solo il tuo compagno umano ha il ricettario.
Quando il compagno ti dà indicazioni, esegui i tool necessari.
Se noti esitazione o silenzio, intervieni con tatto e spirito di squadra per aiutare a sbloccare la situazione.
Rispondi SEMPRE in italiano cordiale, collaborativo e sintetico."""


# ==========================================
# 3. PARSER PARAMETRI QWEN
# ==========================================
def parse_qwen_parameters(args_raw: str) -> Dict[str, Any]:
    args_raw = args_raw.strip()
    if not args_raw:
        return {}

    try:
        parsed = json.loads(args_raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    param_matches = re.findall(
        r"<parameter=([a-zA-Z0-9_]+)>\s*(.*?)\s*</parameter>",
        args_raw,
        re.DOTALL
    )
    if param_matches:
        result = {}
        for param_name, param_val in param_matches:
            val_clean = param_val.strip()
            try:
                result[param_name] = json.loads(val_clean)
            except Exception:
                result[param_name] = val_clean
        return result

    return {}


# ==========================================
# 4. NODO CALL MODEL
# ==========================================
def call_model(state: MessagesState):
    all_messages = state["messages"]
    recent_messages = all_messages[-6:] if len(all_messages) > 6 else all_messages

    formatted_messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for msg in recent_messages:
        if isinstance(msg, HumanMessage):
            formatted_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, ToolMessage):
            formatted_messages.append({
                "role": "tool",
                "content": str(msg.content),
                "tool_call_id": msg.tool_call_id,
                "name": msg.name
            })
        elif isinstance(msg, AIMessage):
            ai_dict = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                ai_dict["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["args"] if isinstance(tc["args"], dict) else {}
                        }
                    }
                    for tc in msg.tool_calls
                ]
            formatted_messages.append(ai_dict)

    response = llm.create_chat_completion(
        messages=formatted_messages,
        tools=tools_schema,
        temperature=0.4,
        max_tokens=1024
    )

    choice = response["choices"][0]
    msg_data = choice["message"]
    raw_content = msg_data.get("content") or ""
    tool_calls_data = msg_data.get("tool_calls") or []

    # Diagnostica raw
    print("\n" + "─"*25 + " [DIAGNOSTICA RAW LLM] " + "─"*25)
    print(f"📌 Motivo di stop: {choice.get('finish_reason')}")
    print(f"📌 Token usati: {response.get('usage')}")
    print(f"📌 Testo grezzo generato:\n{repr(raw_content)}")
    print("─"*73)

    if not tool_calls_data:
        qwen_matches = re.findall(
            r"<tool_call>\s*<function=([a-zA-Z0-9_]+)>\s*(.*?)\s*</function>\s*</tool_call>",
            raw_content,
            re.DOTALL
        )
        for fn_name, fn_args in qwen_matches:
            parsed_args = parse_qwen_parameters(fn_args)
            tool_calls_data.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": fn_name.strip(),
                    "arguments": parsed_args
                }
            })

    lc_tool_calls = []
    for tc in tool_calls_data:
        args = tc["function"]["arguments"]
        if isinstance(args, str):
            args = parse_qwen_parameters(args)
        lc_tool_calls.append({
            "name": tc["function"]["name"],
            "args": args,
            "id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}")
        })

    thought_text = ""
    reply_text = raw_content

    if "</think>" in raw_content:
        parts = raw_content.split("</think>", 1)
        thought_text = parts[0].replace("<think>", "").strip()
        reply_text = parts[1].strip()

    reply_text = re.sub(r"<tool_call>.*?</tool_call>", "", reply_text, flags=re.DOTALL).strip()

    ai_message = AIMessage(
        content=reply_text,
        tool_calls=lc_tool_calls,
        additional_kwargs={"thought": thought_text}
    )
    return {"messages": [ai_message]}


# ==========================================
# 5. GRAFO LANGGRAPH
# ==========================================
builder = StateGraph(MessagesState)

builder.add_node("chatbot", call_model)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", tools_condition)
builder.add_edge("tools", "chatbot")

memory = MemorySaver()
app = builder.compile(checkpointer=memory)


# ==========================================
# 6. GESTIONE INPUT CON CONTEGGIO SECONDI A VIDEO
# ==========================================
def input_con_timeout(prompt: str, timeout_sec: int):
    """Stampa i secondi che passano [1s] [2s] finché l'utente non scrive o scade il timer."""
    print(prompt, end="", flush=True)
    try:
        for sec in range(1, timeout_sec + 1):
            rlist, _, _ = select.select([sys.stdin], [], [], 1.0)
            if rlist:
                line = sys.stdin.readline()
                if not line:
                    return None, False
                return line.strip(), False
            print(f" [{sec}s]", end="", flush=True)
        return None, True  # Sono passati tutti i secondi del timeout
    except (KeyboardInterrupt, Exception):
        return None, False


# ==========================================
# 7. LOOP DI GIOCO INTERATTIVO
# ==========================================
def main():
    print("\n" + "="*65)
    print("📜 RICETTARIO SEGRETO DELL'ALCHIMISTA (Visibile SOLO a te!)")
    print("="*65)
    print("Obiettivo: 'Pozione della Visione Astrale'")
    print("  1. Regola il fuoco su 'medio'")
    print("  2. Inserisci 2x 'lacrima_di_fenice'")
    print("  3. Mescola 1 volta")
    print("  4. Inserisci 1x 'radice_di_mandragora'")
    print("  5. Mescola 2 volte")
    print("  6. Verifica se la pozione è completata!")
    print("="*65)
    print(f"⏱️  Modalità Proattiva: l'agente interviene dopo {TEMPO_INATTIVITA_SECONDI}s di inattività.")
    print("   (Scrivi 'esci' per terminare la sessione)\n")

    config = {"configurable": {"thread_id": "sessione_proattiva_1"}}

    while True:
        try:
            testo_utente, timeout_scattato = input_con_timeout("\n🧙‍♂️ Tu: ", TEMPO_INATTIVITA_SECONDI)

            if timeout_scattato:
                print(f"\n⏳ [Scaduti i {TEMPO_INATTIVITA_SECONDI}s: interviene l'alchimista proattivamente]")
                stato_attuale = guarda_calderone.invoke({})
                messaggio_evento = (
                    f"[EVENTO DI SISTEMA - INATTIVITÀ DELL'UTENTE]\n"
                    f"L'utente è rimasto in silenzio per {TEMPO_INATTIVITA_SECONDI} secondi senza darti istruzioni.\n"
                    f"Ecco lo stato attuale del calderone:\n{stato_attuale}\n"
                    f"Intervieni spontaneamente da compagno di squadra attento: fai un'osservazione "
                    f"breve, costruttiva e cordiale basata su questo stato per sbloccarlo o chiedergli "
                    f"cosa dice il ricettario per il prossimo passo. Non essere pressante."
                )
                messaggio_da_inviare = HumanMessage(content=messaggio_evento)

            else:
                if testo_utente is None or testo_utente.lower() in ["esci", "exit", "quit"]:
                    print("\nChiusura del laboratorio alchemico. Arrivederci!")
                    break
                if not testo_utente:
                    continue
                messaggio_da_inviare = HumanMessage(content=testo_utente)

            # Esecuzione del grafo LangGraph
            for step in app.stream(
                {"messages": [messaggio_da_inviare]},
                config=config,
                stream_mode="updates"
            ):
                if "chatbot" in step:
                    msg = step["chatbot"]["messages"][-1]
                    thought = msg.additional_kwargs.get("thought", "")

                    if thought:
                        print(f"\n💭 [Pensiero Agente]: {thought}")

                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            print(f"\n⚙️  [Azione]: Invocazione tool -> {tc['name']}({tc['args']})")
                    elif msg.content:
                        print(f"\n🤖 Compagno: {msg.content}")

                if "tools" in step:
                    for tool_msg in step["tools"]["messages"]:
                        print(f"📋 [Risultato dal Mondo]:\n{tool_msg.content}")

        except KeyboardInterrupt:
            print("\nUscita.")
            break
        except Exception as e:
            print(f"\n❌ Errore durante l'esecuzione: {e}")


if __name__ == "__main__":
    main()