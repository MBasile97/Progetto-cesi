from llama_cpp import Llama

# 1. Carica il modello assegnando TUTTI i layer alla GPU
print("⏳ Caricamento di Qwen 3.8 27B sulla RTX A4000...")
llm = Llama(
    model_path="./Qwen3.8-27B-UD-IQ4_XS.gguf",
    n_gpu_layers=-1,  # -1 significa: carica tutti i layer sulla GPU
    n_ctx=2048,       # Contesto moderato (2048 token) per non sforare i 16GB di VRAM
    verbose=False
)

# 2. Prepariamo il prompt con il formato chat di Qwen
prompt = "<|im_start|>user\nCiao Qwen, presentati brevemente e dimmi cosa sai fare!<|im_end|>\n<|im_start|>assistant\n"

print("🤖 Generazione risposta in corso...")
output = llm(
    prompt,
    max_tokens=1024,
    stop=["<|im_end|>"],
    temperature=0.7
)

print("\n--- RISPOSTA ---")
print(output["choices"][0]["text"])