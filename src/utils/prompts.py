"""
Utilidades para manejar prompts y formateo de datos
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import json


class PromptManager:
    """Maneja la carga y formateo de prompts"""
    
    # Mapeo de tareas a modalidades
    TASK_TO_MODALITY = {
        1: "text",      # Tarea 1: Texto simbólico
        2: "vision",    # Tarea 2: Imagen 1D
        3: "vision"     # Tarea 3: ECGs reales
    }
    
    def __init__(self, prompts_dir: Optional[Path] = None):
        if prompts_dir is None:
            # Ruta por defecto relativa al archivo actual
            self.prompts_dir = Path(__file__).parent.parent / "prompts"
        else:
            self.prompts_dir = Path(prompts_dir)
    
    def load_prompt(self, approach: str, prompt_type: str) -> str:
        """
        Carga un prompt específico
        
        Args:
            approach: 'regular' o 'cbm'  
            prompt_type: 'system_prompt', 'user_prompt_zero_shot', 'user_prompt_few_shot'
        """
        prompt_path = self.prompts_dir / approach / f"{prompt_type}.txt"
        
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt no encontrado: {prompt_path}")
        
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    
    def format_example(self, sample: Dict[str, Any], approach: str = "regular", example_num: int = 1) -> str:
        """
        Formatea un ejemplo para usar en few-shot learning
        
        Args:
            sample: Diccionario con 'sequence', 'fc_class' (y 'concepts' para CBM)
            approach: 'regular' o 'cbm'
            example_num: Número del ejemplo (para referencia)
        """
        if approach == "regular":
            sequence = sample['sequence']
            fc_class = sample['fc_class']
            
            # Contar picos
            narrow_peaks = sequence.count('|')
            wide_peaks = sequence.count(':')
            total_peaks = narrow_peaks + wide_peaks
            
            return f"""Example {example_num}:
Sequence: {sequence}
Narrow peaks (|): {narrow_peaks}
Wide peaks (:): {wide_peaks}
Total peaks: {total_peaks}
Classification: {fc_class}"""
        
        elif approach == "cbm":
            sequence = sample['sequence']
            concepts = sample.get('concepts', {})
            fc_class = sample['fc_class']
            
            concepts_json = json.dumps(concepts, ensure_ascii=False, indent=2)
            return f"""Example {example_num}:
Sequence: {sequence}
Concepts:
{concepts_json}
Classification: {fc_class}"""
        
        else:
            raise ValueError(f"Approach no soportado: {approach}")
    
    def format_few_shot_examples(self, examples: List[Dict[str, Any]], 
                                approach: str = "regular") -> str:
        """
        Formatea múltiples ejemplos para few-shot learning
        """
        formatted_examples = []
        
        for i, example in enumerate(examples, 1):
            formatted = self.format_example(example, approach, example_num=i)
            formatted_examples.append(formatted)
        
        return "\n\n".join(formatted_examples)
    
    def build_messages(self, 
                      sequence: str,
                      task: int = 1,
                      approach: str = "regular", 
                      examples: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, str]]:
        """
        Construye los mensajes completos para el modelo
        
        Args:
            sequence: Secuencia a analizar
            task: Número de tarea (1, 2, o 3) - se mapea automáticamente a modalidad
            approach: 'regular' o 'cbm'
            examples: Lista de ejemplos para few-shot (None para zero-shot)
        """
        
        # Mapear tarea a modalidad
        if task not in self.TASK_TO_MODALITY:
            raise ValueError(f"Tarea {task} no válida. Use 1, 2 o 3.")
        
        # System prompt
        system_prompt = self.load_prompt(approach, "system_prompt")
        
        # User prompt
        if examples is None or len(examples) == 0:
            # Zero-shot
            user_template = self.load_prompt(approach, "user_prompt_zero_shot")
            user_prompt = user_template.format(sequence=sequence)
        else:
            # Few-shot
            user_template = self.load_prompt(approach, "user_prompt_few_shot")
            formatted_examples = self.format_few_shot_examples(examples, approach)
            user_prompt = user_template.format(
                examples=formatted_examples,
                sequence=sequence
            )
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]


class DataFormatter:
    """Utilidades para formatear datos de entrada y salida"""
    
    @staticmethod
    def parse_cbm_response(response: str) -> Dict[str, Any]:
        """
        Parsea la respuesta JSON del modelo en formato CBM
        
        Returns:
            Dict con 'conceptos' y 'fc_clase', o None si hay error
        """
        try:
            # Limpiar la respuesta (remover markdown si existe)
            cleaned = response.strip()
            if "```json" in cleaned:
                # Extraer JSON del bloque de código
                start = cleaned.find("```json") + 7
                end = cleaned.find("```", start)
                cleaned = cleaned[start:end].strip()
            elif "```" in cleaned:
                # Manejar bloques sin especificar lenguaje
                start = cleaned.find("```") + 3
                end = cleaned.rfind("```")
                cleaned = cleaned[start:end].strip()
            
            # Parsear JSON
            parsed = json.loads(cleaned)
            
            # Validar estructura esperada
            if "conceptos" not in parsed or "fc_clase" not in parsed:
                return None
            
            return parsed
            
        except (json.JSONDecodeError, ValueError, KeyError):
            return None
    
    @staticmethod
    def parse_regular_response(response: str) -> Optional[str]:
        """
        Parsea la respuesta del modelo en formato regular (solo clasificación)
        
        Returns:
            Clasificación FC o None si hay error
        """
        cleaned = response.strip()
        
        # Buscar una de las clasificaciones válidas
        valid_classes = ["Brady", "Normal", "Tachy"]
        
        for fc_class in valid_classes:
            if fc_class.lower() in cleaned.lower():
                return fc_class
        
        return None
    
    @staticmethod
    def extract_concepts_from_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extrae conceptos de una muestra para usar en ejemplos few-shot
        """
        if 'concepts' in sample:
            return sample['concepts']
        
        # Si no están pre-calculados, calcular desde la secuencia
        # (esto debería ser raro si usamos el generador correctamente)
        sequence = sample['sequence']
        
        n_narrow = sequence.count('|')
        n_wide = sequence.count(':')
        n_noise = sequence.count('~') + sequence.count('_')
        noise_pct = (n_noise / len(sequence)) * 100
        
        return {
            'C1_picos_estrechos': n_narrow,
            'C2_picos_anchos': n_wide,
            'C3_pct_ruido': round(noise_pct, 2),
            'C4_irregularidad': 0.05,  # Valor por defecto
            'C5_artefactos_significativos': sequence.count('_') > len(sequence) * 0.05
        }
