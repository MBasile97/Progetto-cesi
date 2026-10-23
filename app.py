import json
import re
import uuid
from typing import List, Dict, Any

from llama_cpp import Llama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# Importiamo i tool del mondo alchemico
from world import (
    ispeziona_dispensa,
    guarda_calderone,
    regola_fuoco,
    aggiungi_ingrediente,
    mescola_calderone,
    verifica_pozione_completata
)

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
Rispondi SEMPRE in italiano cordiale, collaborativo e sintetico."""


# ==========================================
# 3. PARSER SPECIFICO DEI PARAMETRI XML DI QWEN
# ==========================================
def parse_qwen_parameters(args_raw: str) -> Dict[str, Any]:
    """Estrae i parametri da formato JSON oppure dai tag XML <parameter=nome>valore</parameter>"""
    args_raw = args_raw.strip()
    if not args_raw:
        return {}

    # 1. Prova prima come JSON valido
    try:
        parsed = json.loads(args_raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # 2. Prova come tag XML di Qwen: <parameter=livello>\nmedio\n</parameter>
    param_matches = re.findall(
        r"<parameter=([a-zA-Z0-9_]+)>\s*(.*?)\s*</parameter>",
        args_raw,
        re.DOTALL
    )
    if param_matches:
        result = {}
        for param_name, param_val in param_matches:
            val_clean = param_val.strip()
            # Converte numeri interi se possibile (es. quantità)
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

    # ========================================================
    # 🔍 DIAGNOSTICA IN TEMPO REALE
    # ========================================================
    print("\n" + "─"*25 + " [DIAGNOSTICA RAW LLM] " + "─"*25)
    print(f"📌 Motivo di stop: {choice.get('finish_reason')}")
    print(f"📌 Token usati: {response.get('usage')}")
    print(f"📌 Testo grezzo generato:\n{repr(raw_content)}")
    print("─"*73)
    # ========================================================

    # Se llama-cpp non ha popolato tool_calls_data, facciamo il parsing manuale
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

    # Creazione della lista tool_calls per LangGraph
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

    # Estrazione pensiero e risposta per l'umano
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
# 5. GRAFO LANGGRAPH CON MEMORIA
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
# 6. LOOP DI GIOCO INTERATTIVO
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
    print("💡 L'agente NON conosce questa ricetta. Collabora parlando con lui.")
    print("   (Scrivi 'esci' per terminare la sessione)\n")

    config = {"configurable": {"thread_id": "sessione_alchimia_2"}}

    while True:
        try:
            user_input = input("\n🧙‍♂️ Tu: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["esci", "exit", "quit"]:
                print("Chiusura del laboratorio alchemico. Arrivederci!")
                break

            for step in app.stream(
                {"messages": [HumanMessage(content=user_input)]},
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