from pathlib import Path
import sys
import csv
import json
from datetime import datetime

# Añadir el directorio src al path
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from data.loader import DataLoader
from utils.prompts import PromptManager
from typing import List, Dict
from openai import OpenAI


class LocalLLM:
    """Cliente para LLM servido localmente usando OpenAI API"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:11434/v1", 
                 model: str = "local-model",
                 temperature: float = 0.1):
        """
        Args:
            base_url: URL del servidor local
            model: Nombre del modelo (puede ser cualquiera para servidores locales)
            temperature: Temperatura para sampling (baja para más determinismo)
        """
        self.client = OpenAI(
            base_url=base_url,
            api_key="dummy-key"  # Muchos servidores locales no requieren key real
        )
        self.model = model
        self.temperature = temperature
    
    def predict(self, messages: List[Dict[str, str]]) -> str:
        """
        Obtener predicciones
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=500,
                timeout=30
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"❌ Error en la predicción: {e}")


def evaluate_icl_classification(task: int = 1, n_shots: int = 4, model_name: str = "local-model"):
    """
    Evalúa ICL en la tarea de clasificación de FC
    
    Args:
        task: Número de tarea (1, 2, o 3)
        n_shots: Número de ejemplos ICL (1, 2, 4, 8)
        model_name: Nombre del modelo para guardar resultados
    """
    
    print(f"🧪 Evaluando Task {task} - ICL con {n_shots} shots - Modelo: {model_name}")
    
    # Cargar datos y prompts basado en la tarea
    data_dir = project_root / "data" / f"task{task}"
    prompts_dir = project_root / "src" / "prompts" / f"task{task}"
    loader = DataLoader(data_dir=data_dir)
    prompt_manager = PromptManager(prompts_dir=prompts_dir)
    
    # Crear directorios de output
    predictions_dir = project_root / "outputs" / "predictions" / f"task{task}"
    results_dir = project_root / "outputs" / "results" / f"task{task}"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Ejemplos ICL
    #icl_examples = loader.load_icl_examples(n_shots)
    icl_examples = []
    
    # Datos de test - usar TODOS los ejemplos
    test_data = loader.load_test_data(ood=False)
    test_sample = test_data
    
    # Crear modelo
    llm = LocalLLM(model=model_name)
    
    # Preparar ejemplos ICL en formato para prompts
    icl_for_prompts = []
    for sequence, fc_class in icl_examples:
        icl_for_prompts.append({
            'sequence': sequence,
            'fc_class': fc_class
        })
    
    # Lista para almacenar predicciones
    predictions = []
    
    print(f"\n🔄 Evaluando ejemplos...")
    total_samples = len(test_sample)
    
    # Mostrar ejemplo de prompt para el primer caso
    show_prompt_example = True
    
    # Iterar sobre ejemplos de test
    for i, (test_sequence, true_class) in enumerate(test_sample):
        
        # Construir prompt
        messages = prompt_manager.build_messages(
            sequence=test_sequence,
            task=task,
            approach="regular",  # Clasificación simple
            #examples=icl_for_prompts
            examples=[]
        )
        
        # Mostrar ejemplo de prompt completo en el primer caso
        if show_prompt_example:
            print(f"\n📄 Ejemplo de prompt enviado al modelo:")
            print("=" * 80)
            print(f"🔹 SYSTEM MESSAGE:")
            print(f"{messages[0]['content']}")
            print(f"\n🔹 USER MESSAGE:")
            print(f"{messages[1]['content']}")
            print("=" * 80)
            show_prompt_example = False  # Solo mostrar una vez
        
        # Obtener predicción
        raw_response = llm.predict(messages)
        #if llm:
        #    raw_response = llm.predict(messages)
        #    predicted_class = DataFormatter.parse_regular_response(raw_response)
            
            # Si no se puede parsear, usar fallback
        #    if predicted_class is None:
        #        predicted_class = "Unknown"
        #else:
            # Lanzar error si no hay LLM
        #    raise ValueError("No se proporcionó LLM para la predicción.")
        
        # extraemos de la respuesta del modelo el nº de picos
        try:
            data = json.loads(raw_response)
            data = {k.lower(): v for k, v in json.loads(raw_response).items()}

            narrow = int(data.get("narrow_peaks", 0))
            wide = int(data.get("wide_peaks", 0))
            total = int(data.get("total_peaks", narrow + wide))
        except:
            narrow, wide, total = 0, 0, 0

        # clasificamos según el total de picos
        if total < 8:
            predicted_class = "Brady"
        elif total <= 11:
            predicted_class = "Normal"
        else:
            predicted_class = "Tachy"

        # Almacenar predicción
        is_correct = (predicted_class == true_class)
        #predictions.append({
         #   'sequence': test_sequence,
         #   'true_class': true_class,
         #   'predicted_class': predicted_class,
         #   'correct': is_correct,
         #   'raw_response': raw_response if llm else ""
        #})

        predictions.append({
            'sequence': test_sequence,
            'true_class': true_class,
            'predicted_class': predicted_class,
            'correct': is_correct,
            'narrow_peaks': narrow,
            'wide_peaks': wide,
            'total_peaks': total,
            'raw_response': raw_response
        })
        
        # Mostrar algunos ejemplos
        if i < 5:
            status = "✅" if is_correct else "❌"
            if llm:
                print(f"  {i+1}. {status} {true_class} → {predicted_class} | {test_sequence[:30]}... | Raw: '{raw_response[:20]}...'")
            else:
                print(f"  {i+1}. {status} {true_class} → {predicted_class} | {test_sequence[:30]}...")
        
        # Mostrar progreso cada 5 muestras
        if (i + 1) % 5 == 0:
            print(f"  ⏳ {i + 1}/{total_samples}")

    # Guardar predicciones a CSV
    predictions_file = predictions_dir / f"{model_name}_n_shots_{n_shots}_results.csv"
    print(f"\n💾 Guardando predicciones en {predictions_file}...")
    
    with open(predictions_file, 'w', newline='', encoding='utf-8') as f:
        #fieldnames = ['sequence', 'true_class', 'predicted_class', 'correct', 'raw_response']
        fieldnames = [
            'sequence', 'true_class', 'predicted_class', 'correct',
            'narrow_peaks', 'wide_peaks', 'total_peaks',
            'raw_response'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)
    
    print(f"✅ Predicciones guardadas ({len(predictions)} ejemplos)")
    
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
        writer.writerow({'metric': 'Task', 'value': task})
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
    
    return accuracy, results_by_class


def main(task: int = 1, n_shots: int = 4, model_name: str = "local-model"):
    """Función principal de evaluación
    
    Args:
        task: Número de tarea a evaluar (1, 2, o 3)
        n_shots: Número de ejemplos ICL (1, 2, 4, 8)
        model_name: Nombre del modelo
    """
    
    task_names = {
        1: "Contar picos en texto",
        2: "Señal 1D",
        3: "ECGs reales"
    }
    
    if task not in task_names:
        print(f"❌ Tarea {task} no válida. Usa 1, 2 o 3.")
        return
    
    print(f"Tarea {task}: {task_names[task]}")
    print("=" * 45)
    
    # Evaluación
    print(f"\n1️⃣ Evaluación con {n_shots}-shot:")
    evaluate_icl_classification(task=task, n_shots=n_shots, model_name=model_name)
    
    # TO-DO:
    #   - Prueba datos OOD: loader.load_test_data(ood=True)
    #   - Experimentar con diferentes temperaturas y prompts

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluar ICL en diferentes tareas")
    parser.add_argument("--task", type=int, default=1, help="Número de tarea (1, 2, o 3)")
    parser.add_argument("--n-shots", type=int, default=4, help="Número de ejemplos ICL (1, 2, 4, 8)")
    parser.add_argument("--model-name", type=str, default="local-model", help="Nombre del modelo")
    args = parser.parse_args()
    
    main(task=args.task, n_shots=args.n_shots, model_name=args.model_name)
