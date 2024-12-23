from transformers import Idefics3ForConditionalGeneration, AutoProcessor

# Replace with your checkpoint directory path
checkpoint_path = "finetune_models/checkpoint-477"
base_model_path = "HuggingFaceTB/SmolVLM-Base"

try:
    # Load the processor
    processor = AutoProcessor.from_pretrained(base_model_path)

    # Load the model
    model = Idefics3ForConditionalGeneration.from_pretrained(checkpoint_path)
    print("Checkpoint is valid and can be used!")
    model.push_to_hub("SmolVLM_2K_Pixmo")
    processor.push_to_hub("SmolVLM_2K_Pixmo")

except Exception as e:
    print(f"Error loading checkpoint: {e}")
