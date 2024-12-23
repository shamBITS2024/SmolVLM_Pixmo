import torch
from transformers import AutoProcessor, Idefics3ForConditionalGeneration
from PIL import Image


from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(checkpoint_path: str, base_model_id: str ="HuggingFaceTB/SmolVLM-Base", device: str = "cuda"):
    # Load processor from original model
    processor = AutoProcessor.from_pretrained(base_model_id)
    if checkpoint_path:
        # Load fine-tuned model from checkpoint
        model = Idefics3ForConditionalGeneration.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
    else:
         model = Idefics3ForConditionalGeneration.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map=device
         )
    
    # Configure processor for video frames
    return model, processor

def generate_response(model, processor, image_path: str, max_frames: int = 50):
    # Extract frames
   
    image = Image.open(image_path).convert("RGB")
    logger.info(f"Loaded image ")
    
    # Create prompt with frames
#     system_prompt="""I will ask you to point on a thing or things in a given image, and you have to generate the output in the form of coordinates as percentages. Follow the below guidelines strictly -

# 1) According to the prompt, there can be a single or multiple objects in the image, and you have to return me the pixel coordinates as percentages of image dimensions accordingly.

# For example,
# - single object prompts can be like: "point to the black bottle of beer on the bottom shelf" or "point to the wall with visible fixtures" and the output should be somewhat like: <point x=\"79.9\" y=\"69.0\" alt=\"black bottle of beer on the bottom shelf\">black bottle of beer on the bottom shelf</point> or <point x=\"33.9\" y=\"58.6\" alt=\"wall with visible fixtures\">wall with visible fixtures</point>.

# - Multiple objects prompts can be like: "point to the reflection of plants in the glass" or "point to the streetlights illuminating the road" and the output should be somewhat like: <points x1=\"38.0\" y1=\"10.0\" x2=\"46.5\" y2=\"10.0\" x3=\"52.9\" y3=\"10.0\" x4=\"65.0\" y4=\"10.0\" x5=\"71.0\" y5=\"10.0\" x6=\"92.0\" y6=\"10.0\" alt=\"reflection of plants in the glass\">reflection of plants in the glass</points> or " <points x1=\"38.0\" y1=\"14.0\" x2=\"53.5\" y2=\"10.1\" x3=\"54.2\" y3=\"19.6\" x4=\"60.0\" y4=\"17.5\" x5=\"60.0\" y5=\"24.0\" x6=\"64.0\" y6=\"23.6\" x7=\"64.9\" y7=\"25.6\" x8=\"65.9\" y8=\"26.7\" x9=\"66.5\" y9=\"27.7\" x10=\"67.0\" y10=\"28.6\" x11=\"67.5\" y11=\"29.3\" x12=\"68.0\" y12=\"30.0\""
# 2) Make sure the number of instances of objects match with the number of coordinates in your output
# 3) Generate only the xml and nothing else"""        
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Point to ball"}
            ]
        }
    ]
    inputs = processor(
        text=processor.apply_chat_template(messages, add_generation_prompt=True),
        images=[image],
        return_tensors="pt"
    ).to(model.device)

    # Generate response
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        num_beams=5,
        temperature=0.7,
        do_sample=True,
        use_cache=True
    )
    
    # Decode response
    response = processor.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    # Configuration
    base_model_id="HuggingFaceTB/SmolVLM-Base"
    # checkpoint_path=base_model_id
    checkpoint_path = "shambhuDATA/SmolVLM_2K_Pixmo"
    image_path = "5bc7e10c6445b9c9408777cc6525e1ac29b84d7a17b2379cd2fef81df52d32d3.jpg"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    
    # Load model
    logger.info("Loading model...")
    model, processor = load_model(checkpoint_path, base_model_id, device)
    
    # Generate response
    logger.info("Generating response...")
    response = generate_response(model, processor, image_path)
    
    # Print results
    print("Question: Point to ball")
    print("Response:", response)

if __name__ == "__main__":
    main()