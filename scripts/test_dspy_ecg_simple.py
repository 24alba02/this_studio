import dspy
import json
import re

# Función de conteo de picos
def count_peaks(seq: str) -> int:
    """Cuenta los picos '|' y ':' en una secuencia."""
    return seq.count('|') + seq.count(':')

# Configurar el modelo
ollama_lm = dspy.LM(
    model="ollama/mistral:7b",  # probamos con mistral:7b
    api_base="http://localhost:11434"
)
dspy.configure(lm=ollama_lm)

# Definir la tarea (Signature)
class ECGClassifier(dspy.Signature):
    """
    You are an expert in analyzing symbolic ECG sequences composed of '.', '|', ':', '_', and '~'.

    Each '|' or ':' counts as one peak.
    total_peaks = count('|') + count(':')

    Classify based on total_peaks:
      - Brady: 4–7 peaks (<60 bpm)
      - Normal: 8–11 peaks (60–100 bpm)
      - Tachy: 12–18 peaks (>100 bpm)

    Examples:
    Input: ".|:|:|:|"     → total_peaks = 7  → classification = "Brady"
    Input: ".|:|:|:|:|:" → total_peaks = 10 → classification = "Normal"
    Input: ".|:|:|:|:|:|:|:" → total_peaks = 14 → classification = "Tachy"

    Respond ONLY in JSON format:
    {"total_peaks": <number>, "classification": "Brady" | "Normal" | "Tachy"}
    """
    sequence = dspy.InputField(desc="Symbolic ECG sequence (. | : _ ~)")
    result = dspy.OutputField(desc="JSON with total_peaks and classification")

# Crear el predictor
classifier = dspy.Predict(ECGClassifier)

# Ejemplos de prueba de cada clase
examples = {
    "Brady_7": ".|:|:|:|",               # 7 picos
    "Normal_10": ".|:|:|:|:|:",          # 10 picos
    "Tachy_14": ".|:|:|:|:|:|:|:",       # 14 picos
}

# Función de parsing robusto
def extract_json(text: str):
    """Extrae el JSON desde texto con formato irregular."""
    if not text:
        return None
    # Buscar bloque JSON
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None

# Evaluar cada caso
print("\n⚙️ Clasificación ECG con DSPy (verificación automática):\n")

for label, seq in examples.items():
    true_peaks = count_peaks(seq)
    # Determinar la clase esperada
    if 4 <= true_peaks <= 7:
        expected_class = "Brady"
    elif 8 <= true_peaks <= 11:
        expected_class = "Normal"
    elif 12 <= true_peaks <= 18:
        expected_class = "Tachy"
    else:
        expected_class = "OutOfRange"

    # Ejecutar el modelo DSPy
    result = classifier(sequence=seq)
    parsed = extract_json(result.result)

    if parsed:
        predicted_class = parsed.get("classification", "").strip()
        predicted_peaks = parsed.get("total_peaks", "?")
    else:
        predicted_class = str(result.result).strip()
        predicted_peaks = "?"

    correct = predicted_class.lower() == expected_class.lower()
    mark = "✅" if correct else "❌"
    comment = "✅ Nº picos correcto" if true_peaks == predicted_peaks else "❌ Nº de picos incorrecto"

    print(f"🫀 {label:<10} | Picos reales: {true_peaks:<2} | Esperado: {expected_class:<7} "
          f"| Modelo → {predicted_class:<7} ({predicted_peaks}) {mark} {comment}")
