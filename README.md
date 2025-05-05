# capstone-image-recognition

Image recognition demo project that demonstrates how modular pieces of code can be picked up and repurposed elsewhere.

## Project Structure

The project is organized into several key components:

1. **Image Dataset Generation**

   - Uses SVHN (Street View House Numbers) dataset
   - Contains 50-100 images of street numbers
   - Each image is paired with its corresponding text label
   - Dataset is organized in `data/images/` with a `labels.txt` file
2. **Image Tagging**

   - Built-in tool to tag each image with its text content
   - Creates a dataset format of [image_name, image_path, image_size, image_output]
   - Labels are stored in tab-separated format in `labels.txt`
3. **Image Augmentation**

   - Located in `utils/augmentation.py`
   - Features:
     - Image flipping (horizontal/vertical)
     - Zoom in/out functionality
     - Image rotation
     - Brightness adjustment
     - Geolocation extraction from image metadata
   - All augmentations maintain image quality and aspect ratio
4. **Dataset Preparation**

   - Preprocessing pipeline in `utils/helpers.py`
   - Features:
     - Image standardization to fixed size (224x224)
     - Aspect ratio preservation with padding
     - Image quality normalization
     - Metadata extraction
   - Dataset splitting into train/test sets (80/20 split)
   - Preprocessed images are stored separately for model training
5. **Model Implementation**

   - Two model architectures:
     1. Simple Model (CNN+RNN):
        - Takes image input and returns text output
        - Baseline performance model
     2. Advanced Model (ViT+GPT2):
        - Uses Vision Transformer for image encoding
        - GPT-2 for text generation
        - Fine-tuned on the prepared dataset
        - Supports both text recognition and geolocation
6. **Hosting Strategy**

   - Model Deployment Options:
     1. HuggingFace Spaces
        - Easy deployment of the fine-tuned model
        - Interactive demo interface
     2. FastAPI Server
        - REST API for model inference
        - Scalable deployment option

## Usage

1. Install dependencies:

```bash
uv init #repository uses uv instead of pip
uv sync --all-groups #insalls all packages
```

2. Download and prepare dataset:

```bash
python download_dataset.py
```

3. Train the model:

```bash
python model.py
```

4. Make predictions:

```python
from model import predict_image, VisionEncoderDecoderModel

# Load the saved model
model = VisionEncoderDecoderModel.from_pretrained("saved_model")

# Make a prediction
result = predict_image(model, "path/to/your/image.jpg")
print(f"Text in image: {result['text']}")
print(f"Location: {result['location']}")
```

## Project Goals

- Demonstrate modular code that can be easily repurposed (for building simpler/complex models)
- Show the impact of preprocessing on model performance
- Provide a baseline for image-to-text recognition
- Include geolocation capabilities for enhanced functionality
- Make the model easily deployable through various platforms
