# from transformers import Idefics3ForConditionalGeneration, AutoProcessor

# # Replace with your checkpoint directory path
# checkpoint_path = "finetune_models/checkpoint-477"
# base_model_path = "HuggingFaceTB/SmolVLM-Base"

# try:
#     # Load the processor
#     processor = AutoProcessor.from_pretrained(base_model_path)

#     # Load the model
#     model = Idefics3ForConditionalGeneration.from_pretrained(checkpoint_path)
#     print("Checkpoint is valid and can be used!")
#     model.push_to_hub("SmolVLM_2K_Pixmo")
#     processor.push_to_hub("SmolVLM_2K_Pixmo")

# except Exception as e:
#     print(f"Error loading checkpoint: {e}")
import gc
import time
import torch

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

clear_memory()


