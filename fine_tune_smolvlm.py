import torch


from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoProcessor, BitsAndBytesConfig, Idefics3ForConditionalGeneration,TrainingArguments, Trainer
import logging
def setup_logger():
    # Create a logger instance
    logger = logging.getLogger("BasicLogger")
    # logger.setLevel(logging.DEBUG)  # Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL

    # Formatter: Customize log output format
    formatter = logging.Formatter("%(message)s")

    # # # Console Handler: Logs to the console
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.INFO)  # Logs only INFO and higher to the console
    # console_handler.setFormatter(formatter)

    # File Handler: Logs to a file
    

    file_handler=logging.FileHandler("Could_not_access_url.txt","a")
    file_handler.setLevel(logging.ERROR)  # Logs DEBUG and higher to a file
    file_handler.setFormatter(formatter)


    # Add handlers to the logger
    # logger.addHandler(console_handler)
    
    logger.addHandler(file_handler)

    return logger

logger = setup_logger()
USE_LORA = False
USE_QLORA = False
SMOL = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
       print(f"Using GPU: {torch.cuda.get_device_name(0)}")
       print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0)} bytes")
       print(f"GPU Memory Cached: {torch.cuda.memory_reserved(0)} bytes")
else:
       print("Using CPU")

model_id = "HuggingFaceTB/SmolVLM-Base" if SMOL else "HuggingFaceM4/Idefics3-8B-Llama3"
model_name= "SMOL_MOLMO"


processor = AutoProcessor.from_pretrained(
    model_id
)

if USE_QLORA or USE_LORA:
    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=['down_proj','o_proj','k_proj','q_proj','gate_proj','up_proj','v_proj'],
        use_dora=False if USE_QLORA else True,
        init_lora_weights="gaussian"
    )
    lora_config.inference_mode = False
    if USE_QLORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    model = Idefics3ForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config if USE_QLORA else None,
        _attn_implementation="flash_attention_2",
        device_map="auto"
    )
    model.add_adapter(lora_config)
    model.enable_adapters()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    print(model.get_nb_trainable_parameters())
else:
    model = Idefics3ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2",
    ).to("cuda")

    # if you'd like to only fine-tune LLM
    for param in model.model.vision_model.parameters():
        param.requires_grad = False
model= model.to("cuda")
### training arguments for Trainer Class 

early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=3,
    early_stopping_threshold=0.001
)

training_args = TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=50,
    learning_rate=1e-4,
    weight_decay=0.01,
    logging_steps=10,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=1,
    optim="paged_adamw_8bit", # for 8-bit, keep this, else adamw_hf
    bf16=True, # underlying precision for 8bit
    output_dir=f"./{model_name}-vqav2",
    hub_model_id=f"{model_name}-vqav2",
    report_to="tensorboard",
    remove_unused_columns=False,
    gradient_checkpointing=True,
    resume_from_checkpoint=True,
    logging_dir="./logs",         # Directory to save logs
    logging_first_step=True,      # Log the first step of training
    report_to="tensorboard",      # Log to TensorBoard (or "wandb", "comet_ml", etc.)
    log_level="info",             # Level of detail to log (info, warning, debug)
    log_level_replica="warning",  # For distributed setups
)


import requests
from hashlib import sha256
import time
from PIL import Image
from io import BytesIO
# ====================================================================================
# download dataset:::
from datasets import load_dataset
data = load_dataset("allenai/pixmo-points", split="train")

   


# =======================================================================================
from requests.exceptions import RequestException
def download_image(url, retries=3):
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.content
        except RequestException as e:
            logger.error(f"Download failed (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)  # Exponential backoff
    return None

image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<image>")]
i=0
def collate_fn(examples):
  global i  
  texts = []
  images = []
  
  
  
  for example in examples:
    image_url=example["image_url"]
    try:
      image_bytes = download_image(image_url)
      if image_bytes is None:
        logger.error(f"Failed to download image from URL: {image_url}")
        continue
      
      byte_hash = sha256(image_bytes).hexdigest()
      try:
        assert byte_hash == example["image_sha256"]
      except:
        logger.error(f"Image hash mismatch for URL: {image_url}")
        continue


      try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        with open("good_url.txt","a") as f:
           f.write(f"{image_url}\n")
      except Exception as e:
        logger.error(f"Invalid Image format {e}")
        continue



      input_prompt = f"""I will ask you to point on something, and you need to give me the output in the form of coordinates as percentages. According to the prompt, there can be a single or multiple objects in the image, and you have to return me the coordinates accordingly.
  For example, single object prompts can be like: "point to the black bottle of beer on the bottom shelf" or "point to the wall with visible fixtures" and the output should be somewhat like: <point x=\"79.9\" y=\"69.0\" alt=\"black bottle of beer on the bottom shelf\">black bottle of beer on the bottom shelf</point> or <point x=\"33.9\" y=\"58.6\" alt=\"wall with visible fixtures\">wall with visible fixtures</point>.
  Multiple objects prompts can be like: "point to the reflection of plants in the glass" or "point to the streetlights illuminating the road" and the output should be somewhat like: <points x1=\"38.0\" y1=\"10.0\" x2=\"46.5\" y2=\"10.0\" x3=\"52.9\" y3=\"10.0\" x4=\"65.0\" y4=\"10.0\" x5=\"71.0\" y5=\"10.0\" x6=\"92.0\" y6=\"10.0\" alt=\"reflection of plants in the glass\">reflection of plants in the glass</points> or " <points x1=\"38.0\" y1=\"14.0\" x2=\"53.5\" y2=\"10.1\" x3=\"54.2\" y3=\"19.6\" x4=\"60.0\" y4=\"17.5\" x5=\"60.0\" y5=\"24.0\" x6=\"64.0\" y6=\"23.6\" x7=\"64.9\" y7=\"25.6\" x8=\"65.9\" y8=\"26.7\" x9=\"66.5\" y9=\"27.7\" x10=\"67.0\" y10=\"28.6\" x11=\"67.5\" y11=\"29.3\" x12=\"68.0\" y12=\"30.0\""

  Make sure the number of points is correct.
  """
      answer= example['points']
      messages = [
            {
                "role":"system"
                "content":[
                    {"type":"text",
                     "text":input_prompt   
                    }
                ]
            }
            {
                "role": "user",
                "content": [
                    {"type": "text", "text":"Point to the {example["label"]} in the image" },
                    {"type": "image"}

                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer}
                ]
            }
        ]
      text = processor.apply_chat_template(messages, add_generation_prompt=False)
      texts.append(text.strip())
      images.append([image])
      i=i+1
      print(f"Processed Image {i}")
      
      
    except Exception as e:
      logger.error(f"Error processing example: {e}")
      continue


  batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
  labels = batch["input_ids"].clone()
  labels[labels == processor.tokenizer.pad_token_id] = -100
  labels[labels == image_token_id] = -100
  batch["labels"] = labels

  return batch






from transformers import Trainer




trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=data,
        callbacks=[early_stopping_callback]
    )
trainer.train()

final_output_dir = f"./{model_name}-final-checkpoint"
model.save_pretrained(final_output_dir)
processor.save_pretrained(final_output_dir)
print(f"Final checkpoint saved to {final_output_dir}")

with open(f"{final_output_dir}/metadata.txt", "w") as f:
    f.write(f"Training finished at step: {trainer.state.global_step}\n")
    f.write(f"Final Epoch: {trainer.state.epoch}\n")
    f.write(f"Final Loss: {trainer.state.log_history[-1]['loss'] if 'loss' in trainer.state.log_history[-1] else 'N/A'}\n")





    

