import torch
from datasets import load_dataset 

data = load_dataset("shambhuDATA/Pixmo_dataset_16800")

from transformers import AutoModelForCausalLM, AutoProcessor
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-large-ft",####change this to largee
    trust_remote_code=True,
    
).to(device) 
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large-ft", 
    trust_remote_code=True)

for param in model.vision_tower.parameters():
  param.is_trainable = False
torch.cuda.empty_cache()

# Function to run the model on an example
def run_example(task_prompt, text_input, image):
    prompt = task_prompt + text_input

    # Ensure the image is in RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))
    return parsed_answer
def convert_to_xml(points,annot):
  n=len(points)
  if n==0:
    xml_string="Not present"
    return xml_string
  elif n==1:
    p="point"
  else:
    p="points"
  xml_string=f"<{p}"
  for i,point in enumerate(points):
    x=point["x"]
    y=point["y"]
    xml_string+=f' x{i+1}="{x}" y{i+1}="{y}"'
  xml_string+=f' alt="{annot}">{annot}</{p}>'  
  return xml_string

import os
from pathlib import Path

from torch.utils.data import Dataset

class PointObjectDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
      example = self.data[idx]
      prompt = "<PointObject> Point to " + example['label']
      first_answer = convert_to_xml(example['points'],example['label'])
      
      url=f"https://school-student-app.s3.ap-south-1.amazonaws.com/pixmo_images/{example['image_path']}"

      import io
      response = requests.get(url, stream=True,timeout=1).content
      image = Image.open(io.BytesIO(response))

      if image.mode != "RGB":
          image = image.convert("RGB")
      return prompt, first_answer, image
      
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AdamW, AutoProcessor, get_scheduler)

def collate_fn(batch):
    questions, answers, images = zip(*batch)
    print(f"Prompts: {questions}  \n Answers : {answers}  \n images : {images}")
    inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True).to(device)
    return inputs, answers

# Create datasets
train_dataset = PointObjectDataset(data['train'])
# val_dataset = DocVQADataset(data['validation'])

# Create DataLoader
batch_size = 1
num_workers = 0

train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers)
# 
# 
def train_model(train_loader, model, processor, val_loader=None, epochs=10, lr=1e-6):
    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        i = -1
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
            i += 1
            inputs, answers = batch

            input_ids = inputs["input_ids"]
            pixel_values = inputs["pixel_values"]
            labels = processor.tokenizer(text=answers, return_tensors="pt", padding=True, return_token_type_ids=False,truncation=True).input_ids.to(device)

            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss}")

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"):
                inputs, answers = batch

                input_ids = inputs["input_ids"]
                pixel_values = inputs["pixel_values"]
               
                labels = processor.tokenizer(text=answers_str, return_tensors="pt", padding=True, return_token_type_ids=False).input_ids.to(device)

                outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
                loss = outputs.loss

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Average Validation Loss: {avg_val_loss}")

        # Save model checkpoint
        output_dir = f"./model_checkpoints/epoch_{epoch+1}"
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
from huggingface_hub import notebook_login

notebook_login()

for param in model.vision_tower.parameters():
  param.is_trainable = False

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

train_model(train_loader, model, processor, epochs=2)

model.push_to_hub("HuggingFaceM4/Florence-2-FT-DocVQA")
processor.push_to_hub("HuggingFaceM4/Florence-2-FT-DocVQA")