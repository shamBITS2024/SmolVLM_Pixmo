import os
import torch
from hashlib import sha256
from PIL import Image
from io import BytesIO
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    Idefics3ForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from peft import prepare_model_for_kbit_training

# Logger Setup
import logging
def setup_logger():
    logger = logging.getLogger("FineTuningLogger")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler("invalid_images.log", mode="a")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.ERROR)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    return logger

logger = setup_logger()

# CUDA Setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

# Directories
images_dir = "batch_1"
base_model_path = "HuggingFaceTB/SmolVLM-Base"
model_save_dir = "finetune_models"
current_model_id = base_model_path

# Load Model and Processor
processor = AutoProcessor.from_pretrained(current_model_id)
model = Idefics3ForConditionalGeneration.from_pretrained(
    current_model_id,
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2",
).to(DEVICE)

# Freeze vision model parameters
for param in model.model.vision_model.parameters():
    param.requires_grad = False

# Training Arguments
training_args = TrainingArguments(
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=50,
    learning_rate=1e-4,
    weight_decay=0.01,
    logging_steps=10,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=1,
    output_dir=model_save_dir,
    report_to="tensorboard",
    logging_dir="./logs",
    resume_from_checkpoint=True,
    gradient_checkpointing=True,
    bf16=True,
)

# Dataset
data = load_dataset("allenai/pixmo-points", split="train")

# Collate Function
def collate_fn(examples, image_dir):
    texts = []
    images = []
    for example in examples:
        try:
            image_sha = example["image_sha256"]
            image_path = os.path.join(image_dir, f"{image_sha}.jpg")  # Adjust extension
            if not os.path.exists(image_path):
                logger.error(f"Missing Image: {image_path}")
                continue

            image = Image.open(image_path).convert("RGB")
            user_prompt = f"Point to the {example['label']} in the image."
            answer = example["points"]
            messages = [
                {"role": "system", "content": "System prompt text here."},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": answer},
            ]
            text = processor.apply_chat_template(messages, add_generation_prompt=False)
            texts.append(text.strip())
            images.append(image)

        except Exception as e:
            logger.error(f"Error processing example: {e}")
            continue

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels
    return batch

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=lambda examples: collate_fn(examples, images_dir),
    train_dataset=data,
)

# Fine-Tuning Workflow
def fine_tune(images_dir, model, trainer, batch_number, save_dir):
    print(f"Starting fine-tuning for batch {batch_number}...")
    batch_save_dir = os.path.join(save_dir, f"batch_{batch_number}")
    os.makedirs(batch_save_dir, exist_ok=True)

    trainer.train()
    model.save_pretrained(batch_save_dir)
    processor.save_pretrained(batch_save_dir)

    print(f"Model for batch {batch_number} saved at {batch_save_dir}")
    return batch_save_dir

# Run Fine-Tuning
for batch_number in range(1, 21):  # Assuming 20 batches
    batch_dir = os.path.join(images_dir, f"batch_{batch_number}")
    print(batch_dir)
    if not os.path.exists(batch_dir):
        print(f"Batch {batch_number} directory not found. Skipping...")
        continue

    current_model_id = fine_tune(batch_dir, model, trainer, batch_number, model_save_dir)
