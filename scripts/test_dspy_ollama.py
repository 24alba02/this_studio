import dspy

# 1️⃣ Configurar el modelo local de Ollama
ollama_lm = dspy.LM(
    model="ollama/mistral:7b",        
    api_base="http://localhost:11434"
)
dspy.configure(lm=ollama_lm)

# 2️⃣ Definir la tarea (Signature)
class WeatherClassifier(dspy.Signature):
    """
    You are a weather expert.
    Classify the weather condition based on a short description.
    Possible outputs: "Sunny", "Rainy", or "Cloudy".
    """
    description = dspy.InputField(desc="Short text describing the weather")
    category = dspy.OutputField(desc="Sunny, Rainy, or Cloudy")

# 3️⃣ Crear el predictor
classifier = dspy.Predict(WeatherClassifier)

# 4️⃣ Mostrar el prompt que DSPy genera
print("\n🧩 Prompt generado automáticamente por DSPy:")
print("---------------------------------------------")
print(WeatherClassifier.__doc__)
print("---------------------------------------------\n")

# 5️⃣ Probar con ejemplos simples
examples = [
    "The sun is shining and the sky is clear.",
    "Dark clouds are gathering and it starts to rain.",
    "The sky is grey and there is no sunlight.",
]

print("⚙️ Clasificación directa:\n")
for text in examples:
    result = classifier(description=text)
    print(f"🌦️  '{text}' → 🧠 {result.category}")
