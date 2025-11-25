from transformers import AutoTokenizer

# Modelos a analizar
model_names = {
    "Gemma 2B": "google/gemma-2b",
    "Phi-3 Mini": "microsoft/Phi-3-mini-128k-instruct",
    "Mistral 7B": "mistralai/Mistral-7B-v0.1",
    "Falcon 7B": "tiiuae/falcon-7b-instruct",
    "LLaMA 3 8B": "meta-llama/Meta-Llama-3-8B-Instruct",
    "LLaMA 3.2 3B": "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen 2.5 7B": "Qwen/Qwen2.5-7B-Instruct"
}

# Secuencias simbólicas de ejemplo
sequences = {
    "Brady": "....|...:..|....", # 3
    "Normal": "..|..:.|:|:|..|.:", # 9
    "Tachy": "|:|:|:|:|:|:|:|:" # 16
}

def count_true_peaks(seq: str) -> int:
    """Cuenta los picos reales (| o :) en la secuencia original."""
    return sum(1 for ch in seq if ch in ['|', ':'])

def count_peaks_in_tokens(tokens: list) -> int:
    """Cuenta cuántos símbolos de pico (| o :) aparecen en los tokens."""
    return sum(t.count('|') + t.count(':') for t in tokens)

print("\n Picos reales por clase:")
for clase, seq in sequences.items():
    print(f"  {clase}: {count_true_peaks(seq)}")

print("\n=== COMPROBACIÓN DE TOKENIZACIÓN Y CONTEO DE PICOS ===\n")

for model_name, model_path in model_names.items():
    print(f"\n{'='*80}")
    print(f" Modelo: {model_name}")
    print(f"{'='*80}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        for clase, seq in sequences.items():
            true_peaks = count_true_peaks(seq)

            # Tokenización normal
            tokens = tokenizer.tokenize(seq)
            counted_peaks = count_peaks_in_tokens(tokens)

            # Tokenización con separador ';'
            separated_seq = ";".join(list(seq))
            tokens_sep = tokenizer.tokenize(separated_seq)
            counted_peaks_sep = count_peaks_in_tokens(tokens_sep)

            print(f"\n{clase} ({seq})")
            print(f"Tokens: {tokens}")
            print(f" Total tokens: {len(tokens)}")

            print(f"\nTokens (con ';' separador): {tokens_sep}")
            print(f"Total tokens (con ';'): {len(tokens_sep)}")

            print(f"\nPicos reales: {true_peaks}")
            print(f"Sin separador → Picos detectados: {counted_peaks}")
            print(f"Con ';' separador → Picos detectados: {counted_peaks_sep}")

            if counted_peaks_sep == true_peaks and counted_peaks != true_peaks:
                print("Mejora con separador")
            elif counted_peaks_sep == true_peaks and counted_peaks == true_peaks:
                print("✅ Ambas tokenizaciones correctas")
            elif counted_peaks_sep != true_peaks and counted_peaks == true_peaks:
                print("Peor con separador")
            else:
                print("❌ Ninguna coincide exactamente")

            print("—" * 40)

    except Exception as e:
        print(f"⚠️ Error cargando {model_name}: {e}")
