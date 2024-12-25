 # Change requierements.txt
from transformers import Idefics3ForConditionalGeneration, AutoProcessor
from datasets import load_dataset

import peft
from trl import SFTConfig, SFTTrainer
from PIL import Image
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
torch.cuda.empty_cache()

#add trl , accelerate and bits and bytes


def get_image(image_sha25):
    image=Image.open(f"batch_1/{image_sha25}.jpg")
    return image
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
    



system_message="""I will ask you to point on a thing or things in a given image, and you have to generate the output in the form of coordinates as percentages. Follow the below guidelines strictly -

1) According to the prompt, there can be a single or multiple objects in the image, and you have to return me the pixel coordinates as percentages of image dimensions accordingly.

For example,
- single object prompts can be like: "point to the black bottle of beer on the bottom shelf" or "point to the wall with visible fixtures" and the output should be somewhat like: <point x=\"79.9\" y=\"69.0\" alt=\"black bottle of beer on the bottom shelf\">black bottle of beer on the bottom shelf</point> or <point x=\"33.9\" y=\"58.6\" alt=\"wall with visible fixtures\">wall with visible fixtures</point>.

- Multiple objects prompts can be like: "point to the reflection of plants in the glass" or "point to the streetlights illuminating the road" and the output should be somewhat like: <points x1=\"38.0\" y1=\"10.0\" x2=\"46.5\" y2=\"10.0\" x3=\"52.9\" y3=\"10.0\" x4=\"65.0\" y4=\"10.0\" x5=\"71.0\" y5=\"10.0\" x6=\"92.0\" y6=\"10.0\" alt=\"reflection of plants in the glass\">reflection of plants in the glass</points> or " <points x1=\"38.0\" y1=\"14.0\" x2=\"53.5\" y2=\"10.1\" x3=\"54.2\" y3=\"19.6\" x4=\"60.0\" y4=\"17.5\" x5=\"60.0\" y5=\"24.0\" x6=\"64.0\" y6=\"23.6\" x7=\"64.9\" y7=\"25.6\" x8=\"65.9\" y8=\"26.7\" x9=\"66.5\" y9=\"27.7\" x10=\"67.0\" y10=\"28.6\" x11=\"67.5\" y11=\"29.3\" x12=\"68.0\" y12=\"30.0\""
2) Make sure the number of instances of objects match with the number of coordinates in your output
3) Generate only the xml and nothing else"""



def format_data(sample):
    user_message=f"point to the {sample['annotation']}"
    img=get_image(sample['image_sha256'])
    xml_output=convert_to_xml(sample['points'],sample['annotation'])

    
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_message
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img,
                },
                {
                    "type": "text",
                    "text": user_message,
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": xml_output
                }
            ],
        },
    ]

from datasets import load_dataset,load_from_disk




filtered_data_path = "filtered_dataset"
if os.path.exists(filtered_data_path):
    data = load_from_disk(filtered_data_path)
else:
    images= set(os.listdir("batch_1"))
    image_sha=[i.split(".")[0] for i in images]
    data = load_dataset("allenai/pixmo-points", split="train")
    data = data.rename_column("label", "annotation")
    data = data.filter(lambda x: x['image_sha256'] in image_sha)
    data.save_to_disk(filtered_data_path)
    
print(f"the size of trainable image is{len(data)}")

train_dataset= [format_data(sample) for sample in data]
train_dataset=train_dataset[2800:]

print(f"the size of trainable image is{len(train_dataset)}")


# print(train_dataset[0])



model_id="HuggingFaceTB/SmolVLM-Instruct"

# model = Idefics3ForConditionalGeneration.from_pretrained(
#     model_id,
#     device_map="auto",
#     torch_dtype=torch.bfloat16,
#     _attn_implementation="flash_attention_2",
# )

# processor = AutoProcessor.from_pretrained(model_id)
# train_dataset[1]
# train_dataset[1][1:2]

def generate_text_from_sample(model, processor, sample, max_new_tokens=1024, device="cuda"):
    # Prepare the text input by applying the chat template
    text_input = processor.apply_chat_template(
        sample[1:2],  # Use the sample without the system message
        add_generation_prompt=True
    )

    image_inputs = []
    image = sample[1]['content'][0]['image']
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_inputs.append([image])

    # Prepare the inputs for the model
    model_inputs = processor(
        #text=[text_input],
        text=text_input,
        images=image_inputs,
        return_tensors="pt",
    ).to(device)  # Move inputs to the specified device

    # Generate text with the model
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)

    # Trim the generated ids to remove the input ids
    trimmed_generated_ids = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Decode the output text
    output_text = processor.batch_decode(
        trimmed_generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return output_text[0]  # Return the first decoded output text

# output = generate_text_from_sample(model, processor, train_dataset[1])


import gc
import time

def clear_memory():
    # Delete variables if they exist in the current global scope
    if 'inputs' in globals(): del globals()['inputs']
    if 'model' in globals(): del globals()['model']
    if 'processor' in globals(): del globals()['processor']
    if 'trainer' in globals(): del globals()['trainer']
    if 'peft_model' in globals(): del globals()['peft_model']
    if 'bnb_config' in globals(): del globals()['bnb_config']
    time.sleep(2)

    # Garbage collection and clearing CUDA memory
    gc.collect()
    time.sleep(2)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    gc.collect()
    time.sleep(2)

    print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

# clear_memory()

from transformers import BitsAndBytesConfig

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model and tokenizer
model = Idefics3ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    _attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(model_id)

from peft import LoraConfig, get_peft_model

# Configure LoRA
peft_config = LoraConfig(
    r=8,
    lora_alpha=8,
    lora_dropout=0.1,
    target_modules=['down_proj','o_proj','k_proj','q_proj','gate_proj','up_proj','v_proj'],
    use_dora=True,
    init_lora_weights="gaussian"
)

# Apply PEFT model adaptation
peft_model = get_peft_model(model, peft_config)

# Print trainable parameters
peft_model.print_trainable_parameters()

from trl import SFTConfig

# Configure training arguments using SFTConfig
training_args = SFTConfig(
    output_dir="smolvlm-instruct-trl-sft-PixMoPoints",
    # output_dir="smolvlm-instruct-trl-sft-ChartQA",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    # torch_empty_cache_steps=16,
    # torch_compile=True
    warmup_steps=50,
    learning_rate=1e-4,
    weight_decay=0.01,
    logging_steps=25,
    save_strategy="steps",
    save_steps=64,
    save_total_limit=1,
    optim="adamw_torch_fused",
    bf16=True,
    push_to_hub=True,
    # report_to="tensorboard",
    remove_unused_columns=False,
    gradient_checkpointing=True,
    dataset_text_field="",
    dataset_kwargs={"skip_prepare_dataset": True},
    
)

image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<image>")]

def collate_fn(examples):
    
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]

    image_inputs = []
    for example in examples:
      image = example[1]['content'][0]['image']
      
      if image.mode != 'RGB':
          image = image.convert('RGB')
      image_inputs.append([image])

    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels
    labels[labels == image_token_id] = -100  # Mask image token IDs in labels

    batch["labels"] = labels

    return batch


from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    # eval_dataset=eval_dataset,
    data_collator=collate_fn,
    peft_config=peft_config,
    tokenizer=processor.tokenizer,
)
trainer.train(resume_from_checkpoint=True)

trainer.save_model(training_args.output_dir)

##### for testing purpose

# clear_memory()

