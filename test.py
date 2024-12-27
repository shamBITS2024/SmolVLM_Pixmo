# # from transformers import Idefics3ForConditionalGeneration, AutoProcessor

# # # Replace with your checkpoint directory path
# # checkpoint_path = "finetune_models/checkpoint-477"
# # base_model_path = "HuggingFaceTB/SmolVLM-Base"

# # try:
# #     # Load the processor
# #     processor = AutoProcessor.from_pretrained(base_model_path)

# #     # Load the model
# #     model = Idefics3ForConditionalGeneration.from_pretrained(checkpoint_path)
# #     print("Checkpoint is valid and can be used!")
# #     model.push_to_hub("SmolVLM_2K_Pixmo")
# #     processor.push_to_hub("SmolVLM_2K_Pixmo")

# # except Exception as e:
# #     print(f"Error loading checkpoint: {e}")
# import gc
# import time
# import torch

# def clear_memory():
#     # Delete variables if they exist in the current global scope
#     if 'inputs' in globals(): del globals()['inputs']
#     if 'model' in globals(): del globals()['model']
#     if 'processor' in globals(): del globals()['processor']
#     if 'trainer' in globals(): del globals()['trainer']
#     if 'peft_model' in globals(): del globals()['peft_model']
#     if 'bnb_config' in globals(): del globals()['bnb_config']
#     time.sleep(2)

#     # Garbage collection and clearing CUDA memory
#     gc.collect()
#     time.sleep(2)
#     torch.cuda.empty_cache()
#     torch.cuda.synchronize()
#     time.sleep(2)
#     gc.collect()
#     time.sleep(2)

#     print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
#     print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

# clear_memory()
from transformers import Idefics3ForConditionalGeneration, AutoProcessor
from datasets import load_from_disk
import torch
# import peft
# from trl import SFTConfig, SFTTrainer
from PIL import Image
import os
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

model_id="HuggingFaceTB/SmolVLM-Instruct"
model = Idefics3ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2",
)
from finetune_using_trl import format_data
processor = AutoProcessor.from_pretrained(model_id)
filtered_data_path = "filtered_dataset"
data = load_from_disk(filtered_data_path)
train_dataset= [format_data(sample) for sample in data]

adapter_path = "SmolVLM_Pixmo/smolvlm-instruct-trl-sft-PixMoPoints"
model.load_adapter(adapter_path)
print(train_dataset[5000][:2])
print(f"th answer should be {train_dataset[5000][2]}")
print("\n")

# Image.imshow(train_dataset[20][1]['content'][0]['image'],"the image")
output = generate_text_from_sample(model, processor, train_dataset[5000])
print(f"the answer is {output}")
#====================================================================================
# from datasets import load_dataset, Dataset
# import os

# # Define paths
# filtered_data_path = "filtered_dataset"
# batch_path = "batch_1"

# Load dataset with streaming


#     # Load dataset in streaming mode
# data = load_dataset("allenai/pixmo-points", split="train", streaming=True)

#     # Preload valid hashes for faster lookup
# images = set(os.listdir(batch_path))  # O(n) complexity
# image_sha = {i.split(".")[0] for i in images}  # Hash lookup table (O(1) lookup)

# # Streaming filter with optimized lookup
# def fast_filter(dataset):
#     for sample in dataset:
#         if sample['image_sha256'] in image_sha:
#             sample['annotation'] = sample.get('label')
#             # Fast O(1) lookup
#             yield sample

# # Apply filter as a stream
# data = fast_filter(data)

# # Save filtered dataset to disk (optional)
# # data = data.rename_column("label", "annotation")
# # data.save_to_disk(filtered_data_path)   

# # Streaming-compatible processing
# def format_data_stream(dataset, skip_rows=3000):
#     counter = 0
#     for sample in dataset:
#         if counter >= skip_rows:
#             yield format_data(sample)  # Process each sample on-the-fly
#         counter += 1

# # Process streamed data
# data = format_data_stream(data, skip_rows=3000)
# train_dataset= [format_data(sample) for sample in data]



