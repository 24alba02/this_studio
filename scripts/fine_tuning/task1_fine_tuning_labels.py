""" Fine-tuning por etiquetas (Task 1) usando LoRA (PEFT) """

from pathlib import Path
import argparse
import random
import numpy as np
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

from peft import LoraConfig, get_peft_model

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning por etiquetas con LoRA")
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-2b",
        help="Modelo base a ajustar (Hugging Face hub)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Número de épocas de entrenamiento"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    project_root = Path(__file__).parent.parent.parent

    data_dir = project_root / "data" / "task1" / "fine_tuning"
    train_file = data_dir / "train_labels.jsonl"

    model_name = args.model.replace("/", "_")
    output_dir = (
        project_root
        / "outputs"
        / "fine_tuning"
        / "labels"
        / model_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("📂 Cargando dataset de entrenamiento...")
    dataset = load_dataset("json", data_files=str(train_file))

    # Tokenizar
    print(f" Cargando tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    def format_example(example):
        """
        Construye el texto de entrenamiento (prompt + respuesta).
        """
        label = example["output"].strip()

        prompt = (
            "Classify the ECG sequence into one of the following classes:\n"
            "Brady, Normal, Tachy.\n\n"
            f"ECG sequence:\n{example['input']}\n\n"
            "Answer:"
        )

        return tokenizer(
            prompt + " " + label,
            truncation=True,
            padding=False,
            max_length=64
        )

    print("🧩 Tokenizando ejemplos...")
    tokenized_dataset = dataset.map(
        format_example,
        remove_columns=dataset["train"].column_names
    )

    print(f"Cargando modelo base: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto"
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # LoRA
    print("🧠 Aplicando LoRA...")
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Entrenamiento
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        fp16=True,
        save_strategy="no",
        logging_steps=10,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
    )

    print("⏳ Comenzando fine-tuning...")
    trainer.train()

    print("💾 Guardando modelo ajustado...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("✅ Fine-tuning por etiquetas")
    print(f"Modelo guardado en: {output_dir}")


if __name__ == "__main__":
    main()


