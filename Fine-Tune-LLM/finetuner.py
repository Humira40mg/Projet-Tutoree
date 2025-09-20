"""
Ce script est fait pour entrainer un llm sur un dataset jsonl (format question / reponse)
"""
import os
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# -----------------------------
# 1️⃣ Configuration
# -----------------------------
print("Loading Configuration")
MODEL_NAME = "Qwen/Qwen3-0.6B" #Model a fine-tune VOIR Hugging Face pour les modèles
JSONL_FILE = "dataset.jsonl"
OUTPUT_DIR = "./lora_output"
MAX_LENGTH = 1024  # tokens max

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# 2️⃣ Charger datasets
# -----------------------------
jsonl_ds = load_dataset("json", data_files=JSONL_FILE)["train"]

def convert_convo(example):
    dialogue = ""
    for msg in example["messages"]:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        dialogue += f"{role}: {content}\n"
    return {"text": dialogue}

dataset = jsonl_ds.map(convert_convo)

# -----------------------------
# 3️⃣ Tokenizer
# -----------------------------
print("Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# -----------------------------
# 4️⃣ Model & QLoRA setup
# -----------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

print("Loading model")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

# ⚡ Préparer le modèle pour int8/4-bit + LoRA
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj","v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# -----------------------------
# 5️⃣ Tokenization dataset
# -----------------------------
print("Tokenization of the dataset")
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# -----------------------------
# 6️⃣ Training
# -----------------------------
print("Training model")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_checkpointing=True,
    fp16=True,            # float16
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    optim="paged_adamw_32bit",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()
print("Model trained")

# -----------------------------
# 7️⃣ Sauvegarder le modèle
# -----------------------------
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Fine-tune terminé et sauvegardé dans: {OUTPUT_DIR}")