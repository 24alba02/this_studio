""" Evaluación de un modelo fine-tuneado por etiquetas (Task 1) """

from pathlib import Path
import sys
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluación de un modelo fine-tuneado por etiquetas"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Modelo base usado en el fine-tuning (ej: google/gemma-2b)"
    )
    return parser.parse_args()

# Añadir el directorio src al path
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from data.loader import DataLoader 

def predict(model, tokenizer, sequence: str) -> str:
    """
    Genera la clase predicha por el modelo fine-tuneado
    usando el mismo prompt que en el entrenamiento.
    """
    prompt = (
        "Classify the ECG sequence into one of the following classes:\n"
        "Brady, Normal, Tachy.\n\n"
        f"ECG sequence:\n{sequence}\n\n"
        "Answer:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Extraer SOLO la etiqueta
    if "Answer:" in decoded:
        predicted = decoded.split("Answer:")[-1].strip()
    else:
        predicted = decoded

    # Normalización defensiva
    predicted = predicted.split()[0]
    predicted = predicted.replace(".", "").replace(",", "")

    return predicted


def main():
    args = parse_args()

    safe_model_name = args.model.replace("/", "_")
    model_dir = (
        project_root
        / "outputs"
        / "fine_tuning"
        / "labels"
        / safe_model_name
    )

    print("📊 Evaluación del modelo fine-tuneado por etiquetas (Task 1)")
    print("=" * 60)
    print(f"🤖 Modelo evaluado: {args.model}")
    print(f"📂 Ruta del modelo: {model_dir}\n")

    loader = DataLoader(project_root / "data" / "task1")
    test_data = loader.load_test_data(ood=False)

    print(f"🧪 Número de ejemplos de test: {len(test_data)}")

    print("🤖 Cargando modelo fine-tuneado...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto"
    )
    model.eval()

    correct = 0
    total = 0

    # Métricas por clase
    results_by_class = {
        "Brady": {"tp": 0, "total": 0},
        "Normal": {"tp": 0, "total": 0},
        "Tachy": {"tp": 0, "total": 0},
    }

    # Evaluación
    for sequence, true_class in test_data:
        predicted_class = predict(model, tokenizer, sequence)

        total += 1
        results_by_class[true_class]["total"] += 1

        if predicted_class == true_class:
            correct += 1
            results_by_class[true_class]["tp"] += 1

    accuracy = correct / total if total > 0 else 0.0

    print(f"\n📊 Resultados:")
    print(f"Accuracy general: {accuracy:.3f} ({correct}/{total})\n")

    # Accuracy por clase
    for fc_class in ['Brady', 'Normal', 'Tachy']:
        class_acc = (results_by_class[fc_class]['tp'] / 
                    results_by_class[fc_class]['total'] 
                    if results_by_class[fc_class]['total'] > 0 else 0)
        tp = results_by_class[fc_class]['tp']
        total_cls = results_by_class[fc_class]['total']
        print(f"  {fc_class}: {class_acc:.3f} ({tp}/{total_cls})")

if __name__ == "__main__":
    main()
