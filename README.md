# ECG Classification with In-Context Learning

## 📋 Descripción
Proyecto de evaluación de modelos de lenguaje (LLMs) para clasificación de ECG simbólicas mediante In-Context Learning (ICL).

Se plantean 3 tareas.

**Tarea 1**

Evalúa la capacidad de Large Language Models para clasificar secuencias ECG representadas simbólicamente en tres categorías de frecuencia cardíaca:
- **Brady** (Bradicardia): 4-7 picos (<60 bpm)
- **Normal**: 8-11 picos (60-100 bpm)
- **Tachy** (Taquicardia): 12-18 picos (>100 bpm)

Las secuencias ECG se representan usando los símbolos: `.`, `|`, `:`, `_`, `~`, donde los picos se cuentan como `|` + `:`.

## 🗂️ Estructura del Proyecto

```
this_studio/
├── README.md                          # Este archivo
├── requirements.txt                   # Dependencias del proyecto
├── data/
│   └── task1/                        # Datos para la tarea de clasificación
│       ├── config.json               # Configuración de rangos y parámetros
│       ├── icl/
│       │   └── metadata.csv          # Ejemplos para In-Context Learning
│       └── test/
│           ├── test_metadata.csv     # Datos de test (distribución ID)
│           └── test_ood_metadata.csv # Datos de test (distribución OOD)
├── outputs/
│   ├── predictions/                  # Predicciones del modelo (CSV)
│   │   └── task1/
│   └── results/                      # Métricas de evaluación (CSV)
│       └── task1/
├── scripts/
│   ├── eval/
│   │   ├── task1_evaluation.py       # Script principal de evaluación -> con dspy
│   │   └── task1_evaluation_raw.py   # Versión alternativa -> llamando al LLM directamente
│   └── generate/
│       └── generate_task1.py         # Generación de datos sintéticos
├── src/
│   ├── data/
│   │   ├── loader.py                 # Utilidades para cargar datos
│   │   └── generate_data.py          # Generación de secuencias ECG
│   ├── prompts/
│   │   └── task1/                    # Plantillas de prompts
│   └── utils/
│       └── prompts.py                # Utilidades para prompts
└── env/                              # Entorno virtual Python
```

## 🚀 Instalación

### 1. Clonar el repositorio
```bash
cd this_studio
```

### 2. Crear y activar entorno virtual
```bash
conda create --name env python=3.12
conda activate env
```
Si se utiliza anaconda o:
```bash
uv venv --python 3.12 env
source env/bin/activate
```

Si se utiliza uv

[Nada si se utiliza lightning]

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```
```bash
uv pip install -r requirements.txt
```

**Dependencias principales:**
- `dspy` - Framework para programación con LLMs
- `pandas` - Manipulación de datos
- `openai` - Cliente API OpenAI
- `transformers` - Modelos de Hugging Face

## 🧪 Ejecutar Experimentos

### Task 1: Clasificación ECG con ICL

El script `task1_evaluation.py` evalúa modelos LLM en la tarea de clasificación de secuencias ECG usando In-Context Learning.

#### Uso Básico

```bash
cd scripts/eval
python task1_evaluation.py
```

#### Parámetros Disponibles

```bash
python task1_evaluation.py \
  --data_dir ../../data/task1 \
  --output_dir ../../outputs \
  --n_shots 4 \
  --model_name medgemma-4b-it \
  --api_base http://localhost:1234/v1 \
  --engine ollama|lm_studio
```

**Argumentos:**

| Parámetro | Descripción | Valor por defecto | Opciones |
|-----------|-------------|-------------------|----------|
| `--data_dir` | Directorio con datos de la tarea | `../../data/task1` | Ruta al directorio |
| `--output_dir` | Directorio para guardar resultados | `../../outputs` | Ruta al directorio |
| `--n_shots` | Número de ejemplos ICL por clase | `4` | `1`, `2`, `4`, `8` |
| `--model_name` | Nombre del modelo a evaluar | `medgemma-4b-it` | Cualquier modelo disponible |
| `--api_base` | URL base de la API del modelo | `http://localhost:1234/v1` | URL del servidor |
| `--engine` | Framework que sirve el LLM | `lm_studio` | `lm_studio`, `ollama`, `openai` |

#### Ejemplos de Uso

**1. Evaluar con diferentes números de shots:**
```bash
# 1-shot learning
python task1_evaluation.py --n_shots 1

# 8-shot learning
python task1_evaluation.py --n_shots 8
```

**2. Usar modelos diferentes:**
```bash
# Con modelo local en LM Studio
python task1_evaluation.py --model_name llama-3.2-3b-instruct

# Con Ollama
python task1_evaluation.py \
  --model_name llama3.2 \
  --engine ollama \
  --api_base http://localhost:11434/v1
```

### Configuración del Servidor LLM

Antes de ejecutar los experimentos, asegúrate de tener un servidor LLM corriendo:

**LM Studio:**
1. Abre LM Studio
2. Carga el modelo deseado
3. Inicia el servidor local (por defecto en `http://localhost:1234`)

**Ollama:**
```bash
ollama serve  # Inicia el servidor en http://localhost:11434
ollama pull llama3.2  # Descarga el modelo
```

## 📊 Resultados

Los resultados se guardan automáticamente en dos formatos:

### 1. Predicciones Individuales
`outputs/predictions/task1/{model_name}_n_shots_{n}_results.csv`

Contiene:
- `sequence`: Secuencia ECG analizada
- `true_class`: Clase verdadera
- `true_peaks`: Número real de picos
- `predicted_class`: Clase predicha por el modelo
- `predicted_peaks`: Número de picos detectados por el modelo
- `correct`: Booleano indicando si la predicción fue correcta

### 2. Métricas Agregadas
`outputs/results/task1/{model_name}_n_shots_{n}_metrics.csv`

Incluye:
- Accuracy general
- Accuracy por clase (Brady, Normal, Tachy)
- Total de predicciones correctas
- Timestamp de la evaluación

### Ejemplo de Salida en Consola

```
🧪 Evaluando ICL con 4 shots - Modelo: medgemma-4b-it
🫀 Normal     | Picos reales: 9  | Esperado: Normal  | Predicción → Normal  (9 ) ✅ Nº picos correcto ✅ 
🫀 Brady      | Picos reales: 6  | Esperado: Brady   | Predicción → Brady   (6 ) ✅ Nº picos correcto ✅ 
  ⏳ 5/100
  ⏳ 10/100
...

📊 Resultados:
 Accuracy general: 0.850 (85/100)
  Brady: 0.800 (24/30)
  Normal: 0.886 (31/35)
  Tachy: 0.857 (30/35)

💾 Guardando predicciones en outputs/predictions/task1/medgemma-4b-it_n_shots_4_results.csv...
✅ Predicciones guardadas (100 ejemplos)

💾 Guardando métricas en outputs/results/task1/medgemma-4b-it_n_shots_4_metrics.csv...
✅ Métricas guardadas
```