import dspy
from dspy.teleprompt import LabeledFewShot
from dspy.teleprompt import LabeledFewShot

import argparse
from pathlib import Path
import csv
from datetime import datetime

import sys
# Añadir el directorio src al path
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from data.loader import DataLoader

def count_peaks(seq: str) -> int:
    """Cuenta los picos '|' y ':' en una secuencia."""
    return seq.count('|') + seq.count(':')

def show_progress(i, total_samples, true_class, true_peaks, predicted_class, predicted_peaks, is_correct, comment):
    # Mostrar algunos ejemplos
    if i < 5:
        status = "✅" if is_correct else "❌"
        print(f"🫀 {true_class:<10} | Picos reales: {true_peaks:<2} | Esperado: {true_class:<7} "
        f"| Predicción → {predicted_class:<7} ({predicted_peaks:<2}) {status} {comment}")
    
    # Mostrar progreso cada 5 muestras
    if (i + 1) % 5 == 0:
        print(f"  ⏳ {i + 1}/{total_samples}")

def save_predictions(predictions_dir, predictions, model_name, n_shots):
    predictions_file = predictions_dir / f"{model_name}_n_shots_{n_shots}_results.csv"
    print(f"\n💾 Guardando predicciones en {predictions_file}...")
    
    with open(predictions_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'sequence', 
            'true_class', 
            'true_peaks',
            'predicted_class', 
            'predicted_peaks',
            'correct',
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)
    
    print(f"✅ Predicciones guardadas ({len(predictions)} ejemplos)")

def save_metrics(results_dir, predictions, model_name, n_shots):
    # Calcular métricas
    correct = sum(1 for p in predictions if p['correct'])
    accuracy = correct / len(predictions)
    
    # Métricas por clase
    results_by_class = {'Brady': {'tp': 0, 'total': 0}, 
                       'Normal': {'tp': 0, 'total': 0}, 
                       'Tachy': {'tp': 0, 'total': 0}}
    
    for pred in predictions:
        true_class = pred['true_class']
        results_by_class[true_class]['total'] += 1
        if pred['correct']:
            results_by_class[true_class]['tp'] += 1
    
    print(f"\n📊 Resultados:")
    print(f" Accuracy general: {accuracy:.3f} ({correct}/{len(predictions)})")
    
    # Accuracy por clase
    for fc_class in ['Brady', 'Normal', 'Tachy']:
        class_acc = (results_by_class[fc_class]['tp'] / 
                    results_by_class[fc_class]['total'] 
                    if results_by_class[fc_class]['total'] > 0 else 0)
        tp = results_by_class[fc_class]['tp']
        total = results_by_class[fc_class]['total']
        print(f"  {fc_class}: {class_acc:.3f} ({tp}/{total})")
    
    # Guardar métricas a CSV
    metrics_file = results_dir / f"{model_name}_n_shots_{n_shots}_metrics.csv"
    print(f"\n💾 Guardando métricas en {metrics_file}...")
    
    with open(metrics_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['metric', 'value'])
        writer.writeheader()
        writer.writerow({'metric': 'Overall Accuracy', 'value': accuracy})
        writer.writerow({'metric': 'Correct Predictions', 'value': correct})
        writer.writerow({'metric': 'Total Predictions', 'value': len(predictions)})
        writer.writerow({'metric': 'N-shots', 'value': n_shots})
        writer.writerow({'metric': 'Model', 'value': model_name})
        writer.writerow({'metric': 'Timestamp', 'value': datetime.now().isoformat()})
        
        for fc_class in ['Brady', 'Normal', 'Tachy']:
            class_acc = (results_by_class[fc_class]['tp'] / 
                        results_by_class[fc_class]['total'] 
                        if results_by_class[fc_class]['total'] > 0 else 0)
            writer.writerow({'metric': f'{fc_class} Accuracy', 'value': class_acc})
            writer.writerow({'metric': f'{fc_class} TP', 'value': results_by_class[fc_class]['tp']})
            writer.writerow({'metric': f'{fc_class} Total', 'value': results_by_class[fc_class]['total']})
    
    print(f"✅ Métricas guardadas")

def evaluate_icl_classification(output_dir, classifier, test_sample, model_name: str, n_shots: int):
    """
    Evalúa ICL en la tarea de clasificación de FC
    
    Args:
        output_dir: Directorio de salida
        classifier: Predictor compilado con ejemplos ICL
        test_sample: Lista de tuplas (secuencia, clase verdadera)
        model_name: Nombre del modelo
        n_shots: Número de ejemplos ICL
    """
    
    print(f"🧪 Evaluando ICL con {n_shots} shots - Modelo: {model_name}")
    
    # Crear directorios de output
    predictions_dir = output_dir / "predictions" / "task1"
    results_dir = output_dir / "results" / "task1"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    total_samples = len(test_sample)
    predictions = []
    
    # Iterar sobre ejemplos de test
    for i, (test_sequence, true_class) in enumerate(test_sample):
        
        true_peaks = count_peaks(test_sequence)

        # Obtener predicción
        result = classifier(sequence=test_sequence)
        predicted_peaks = int(result.total_peaks)
        predicted_class = result.classification
        
        is_correct = predicted_class.lower() == true_class.lower()
        comment = "Nº picos correcto ✅ " if true_peaks == predicted_peaks else "Nº de picos incorrecto ❌"

        predictions.append({
            'sequence': test_sequence,
            'true_class': true_class,
            'true_peaks': true_peaks,
            'predicted_class': predicted_class,
            'predicted_peaks': predicted_peaks,
            'correct': is_correct,
        })
        
        show_progress(
            i, 
            total_samples,
            true_class, 
            true_peaks, 
            predicted_class, 
            predicted_peaks, 
            is_correct, 
            comment
        )

    # Guardar predicciones a CSV
    save_predictions(predictions_dir, predictions, model_name, n_shots)
    save_metrics(results_dir, predictions, model_name, n_shots)

# Definir la tarea (Signature)
class ECGClassifier(dspy.Signature):
    """
    You are an expert in analyzing symbolic ECG sequences composed of '.', '|', ':', '_', and '~'.

    Each '|' or ':' counts as one peak.
    total_peaks = count('|') + count(':')

    Classify the label for the ECG based on total_peaks:
        - Brady: 4–7 peaks (<60 bpm)
        - Normal: 8–11 peaks (60–100 bpm)
        - Tachy: 12–18 peaks (>100 bpm)
    """
    # Datos de entrada
    sequence = dspy.InputField(desc="Symbolic ECG sequence (. | : _ ~)")
    # Campos de salida
    total_peaks = dspy.OutputField(desc="Total number of peaks in the ECG sequence")
    classification = dspy.OutputField(desc="Label: Brady, Normal, or Tachy")

def main(data_dir: str, output_dir: str, n_shots: int, model_name: str, engine: str, api_base: str):
    """    
    Args:
        data_dir: Directorio de datos
        output_dir: Directorio de salida
        n_shots: Número de ejemplos ICL (1, 2, 4, 8)
        model_name: Nombre del modelo
        engine: Framework que sirve LLM
        api_base: Base URL de la API del modelo
    """

    # Modelo
    lm = dspy.LM(
        model=engine + "/" + model_name,
        api_base=api_base,
        temperature=0.1,
    )
    dspy.configure(lm=lm)

    # Tarea
    task = dspy.Predict(ECGClassifier)

    # Ejemplos ICL
    loader = DataLoader(data_dir=data_dir)
    icl_examples = loader.load_icl_examples(n_shots)
    icl_examples = [
        dspy.Example(
            sequence=seq, 
            total_peaks=str(count_peaks(seq)),
            classification=cls
        ).with_inputs('sequence') 
        for seq, cls in icl_examples
    ]

    # Ejemplos de test
    test_samples = loader.load_test_data(ood=False)

    # Predictor
    teleprompter = LabeledFewShot(k=n_shots)
    classifier = teleprompter.compile(student=task, trainset=icl_examples)

    # Evaluación
    evaluate_icl_classification(output_dir,classifier, test_samples, model_name, n_shots)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluar ICL en diferentes tareas")
    parser.add_argument("--data_dir", type=str, default=str(project_root / "data" / "task1"), help="Directorio de datos")
    parser.add_argument("--output_dir", type=str, default=str(project_root / "outputs"), help="Directorio de salida")
    parser.add_argument("--n_shots", type=int, default=4, help="Número de ejemplos ICL (1, 2, 4, 8)")
    parser.add_argument("--model_name", type=str, default="medgemma-4b-it", help="Nombre del modelo")
    parser.add_argument("--api_base", type=str, default="http://localhost:1234/v1", help="Base URL de la API del modelo")
    parser.add_argument("--engine", type=str, default="lm_studio", help="Framework que sirve LLM")
    
    args = parser.parse_args()
    
    main(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        n_shots=args.n_shots, 
        model_name=args.model_name,
        engine=args.engine,
        api_base=args.api_base
    )