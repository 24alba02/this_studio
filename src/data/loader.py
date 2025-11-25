"""
Utilidades simples para cargar y evaluar datos
"""

from pathlib import Path
from typing import List, Tuple, Dict, Optional
import random
import pandas as pd


class DataLoader:
    """Carga datos"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir)

    def load_icl_examples(self, n_shots: Optional[int] = None) -> List[Tuple[str, str]]:
        """
        Carga ejemplos para ICL desde CSV
        
        Args:
            n_shots: Número de ejemplos por clase a cargar. Si None, carga todos.
        
        Returns:
            Lista de tuplas (secuencia, clase)
        """
        file_path = self.data_dir / "icl" / "metadata.csv"
        all_examples = self._load_csv_file(file_path)
        
        if n_shots is None:
            return all_examples
        
        # Seleccionar n_shots por clase
        return self.sample_balanced_examples(all_examples, n_shots, seed=42)
    
    def load_test_data(self, ood: bool = False) -> List[Tuple[str, str]]:
        """
        Carga datos de test desde CSV
        
        Args:
            ood: Si True, carga test_ood_metadata.csv, sino test_metadata.csv
        
        Returns:
            Lista de tuplas (secuencia, clase)
        """
        filename = "test_ood_metadata.csv" if ood else "test_metadata.csv"
        file_path = self.data_dir / "test" / filename
        return self._load_csv_file(file_path)
    
    def _load_csv_file(self, file_path: Path) -> List[Tuple[str, str]]:
        """Carga un archivo CSV con columnas id,sequence,label,total_picos"""
        if not file_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Verificar que tenga las columnas necesarias
        required_cols = ['sequence', 'label']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columnas faltantes en {file_path}: {missing_cols}")
        
        # Convertir a lista de tuplas
        data = [(row['sequence'], row['label']) for _, row in df.iterrows()]
        return data
    
    def get_examples_by_class(self, data: List[Tuple[str, str]]) -> Dict[str, List[Tuple[str, str]]]:
        """Agrupa ejemplos por clase"""
        by_class = {'Brady': [], 'Normal': [], 'Tachy': []}
        
        for sequence, fc_class in data:
            if fc_class in by_class:
                by_class[fc_class].append((sequence, fc_class))
        
        return by_class
    
    def sample_balanced_examples(self, data: List[Tuple[str, str]], 
                                n_per_class: int, seed: int = 42) -> List[Tuple[str, str]]:
        """
        Muestrea N ejemplos por clase de forma balanceada
        
        Args:
            data: Lista de ejemplos
            n_per_class: Número de ejemplos por clase
            seed: Semilla para reproducibilidad
        """
        random.seed(seed)
        
        by_class = self.get_examples_by_class(data)
        sampled = []
        
        for fc_class in ['Brady', 'Normal', 'Tachy']:
            class_examples = by_class[fc_class]
            if len(class_examples) < n_per_class:
                print(f"Advertencia: solo {len(class_examples)} ejemplos disponibles para clase {fc_class}")
                sampled.extend(class_examples)
            else:
                sampled.extend(random.sample(class_examples, n_per_class))
        
        # Mezclar ejemplos finales
        random.shuffle(sampled)
        return sampled
    
    def print_stats(self, data: List[Tuple[str, str]], title: str = "Dataset"):
        """Imprime estadísticas básicas de un dataset"""
        print(f"\n=== {title} ===")
        print(f"Total ejemplos: {len(data)}")
        
        by_class = self.get_examples_by_class(data)
        for fc_class in ['Brady', 'Normal', 'Tachy']:
            count = len(by_class[fc_class])
            pct = (count / len(data) * 100) if data else 0
            print(f"  {fc_class}: {count} ({pct:.1f}%)")
        
        if data:
            # Estadísticas de secuencias
            sequences = [seq for seq, _ in data]
            seq_lengths = [len(seq) for seq in sequences]
            avg_length = sum(seq_lengths) / len(seq_lengths)
            print(f"Longitud promedio: {avg_length:.1f} caracteres")
            
            # Contar símbolos
            all_sequences = ''.join(sequences)
            symbol_counts = {
                'picos estrechos (|)': all_sequences.count('|'),
                'picos anchos (:)': all_sequences.count(':'),
                'baseline (.)': all_sequences.count('.'),
                'ruido (~)': all_sequences.count('~'),
                'artefactos (_)': all_sequences.count('_')
            }
            
            print("Símbolos por tipo:")
            for symbol, count in symbol_counts.items():
                pct = (count / len(all_sequences) * 100) if all_sequences else 0
                print(f"  {symbol}: {count} ({pct:.1f}%)")
