"""Script para generar los datasets de la tarea 1 (texto).

Usa `src/data/generate_data.py` como backend de generación.
"""
from pathlib import Path
import sys
import argparse

# Añadir el directorio src al path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "src"))

from data.generate_data import TextGenerator, TextConfig
import json


def generate_task1_datasets(n_test_samples: int = 999, n_ood_samples: int = 300):
    config = TextConfig(
        sequence_length=100,
        brady_range=(4, 7),
        normal_range=(8, 11),
        tachy_range=(12, 16),
        noise_prob_range=(0.05, 0.25),
        artifact_prob_range=(0.02, 0.10),
        wide_peak_ratio_range=(0.2, 0.8)
    )
    generator = TextGenerator(config)
    base_dir = project_root / "data" / "task1"
    icl_dir = base_dir / "icl"
    test_dir = base_dir / "test"

    import shutil
    if base_dir.exists():
        shutil.rmtree(base_dir)
    icl_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    print("  🎯 ICL examples (24 total)...")
    generator.set_seed(42)
    icl_dataset = generator.generate_dataset(n_samples_per_class=8)
    generator.save_metadata_csv(icl_dataset, icl_dir / "metadata.csv", "icl")
    print(f"  🧪 Test dataset ({n_test_samples} total)...")
    generator.set_seed(200)
    test_dataset = generator.generate_dataset(n_samples_per_class=n_test_samples//3)
    generator.save_metadata_csv(test_dataset, test_dir / "test_metadata.csv", "test")
    print(f"  🔀 OOD dataset ({n_ood_samples} total)...")

    ood_config = TextConfig(
        sequence_length=100,
        brady_range=(2, 5),
        normal_range=(9, 10),
        tachy_range=(14, 20),
        noise_prob_range=(0.20, 0.40),
        artifact_prob_range=(0.08, 0.18),
    )
    ood_generator = TextGenerator(ood_config)
    ood_generator.set_seed(300)
    ood_dataset = ood_generator.generate_dataset(n_samples_per_class=n_ood_samples//3)
    ood_generator.save_metadata_csv(ood_dataset, test_dir / "test_ood_metadata.csv", "test_ood")
    config_dict = {
        'sequence_length': 100,
        'brady_range': config.brady_range,
        'normal_range': config.normal_range,
        'tachy_range': config.tachy_range,
        'ood_brady_range': ood_config.brady_range,
        'ood_normal_range': ood_config.normal_range,
        'ood_tachy_range': ood_config.tachy_range,
        'n_test_samples': n_test_samples,
        'n_ood_samples': n_ood_samples,
    }
    with open(base_dir / "config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"\n✅ Estructura creada: {base_dir}")
    return base_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generar datasets para Task 1")
    parser.add_argument("--n-test-samples", type=int, default=999, help="Número de ejemplos de test (default: 999)")
    parser.add_argument("--n-ood-samples", type=int, default=300, help="Número de ejemplos OOD (default: 300)")
    args = parser.parse_args()
    
    generate_task1_datasets(n_test_samples=args.n_test_samples, n_ood_samples=args.n_ood_samples)
