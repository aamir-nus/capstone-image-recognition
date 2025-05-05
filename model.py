import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import pandas as pd
from typing import Tuple, List, Dict
from utils.helpers import load_image, standardize_image, prepare_image_for_model
from utils.augmentation import get_geolocation

class ImageTextDataset(Dataset):
    def __init__(self, image_paths: List[str], texts: List[str], processor, tokenizer):
        self.image_paths = image_paths
        self.texts = texts
        self.processor = processor
        self.tokenizer = tokenizer
        # Set padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        text = self.texts[idx]
        
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        labels = self.tokenizer(text, padding="max_length", max_length=128).input_ids
        
        return {
            "pixel_values": pixel_values.squeeze(),
            "labels": torch.tensor(labels)
        }

def load_dataset(data_dir: str, test_size: float = 0.2) -> Tuple[DataLoader, DataLoader]:
    """Load and split dataset into train and test sets."""
    # Load dataset from directory
    image_paths = []
    texts = []
    
    # Assuming data is organized as: data_dir/images/*.jpg and data_dir/labels.txt
    with open(os.path.join(data_dir, 'labels.txt'), 'r') as f:
        for line in f:
            image_name, text = line.strip().split('\t')
            image_path = os.path.join(data_dir, 'images', image_name)
            if os.path.exists(image_path):
                image_paths.append(image_path)
                texts.append(text)
    
    # Initialize processor and tokenizer
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # Set padding token
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset
    dataset = ImageTextDataset(image_paths, texts, processor, tokenizer)
    
    # Split dataset
    train_size = int((1 - test_size) * len(dataset))
    train_dataset, test_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    return train_loader, test_loader

def train_model(train_loader: DataLoader, test_loader: DataLoader, num_epochs: int = 3) -> VisionEncoderDecoderModel:
    """Train the model on the dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        "google/vit-base-patch16-224", "gpt2"
    )
    
    # Configure model token IDs
    model.config.pad_token_id = model.config.decoder.eos_token_id
    model.config.decoder.pad_token_id = model.config.decoder.eos_token_id
    model.config.decoder_start_token_id = model.config.decoder.bos_token_id
    
    # Ensure the model's decoder is properly configured
    model.decoder.config.pad_token_id = model.config.decoder.eos_token_id
    model.decoder.config.bos_token_id = model.config.decoder.bos_token_id
    model.decoder.config.eos_token_id = model.config.decoder.eos_token_id
    
    model.to(device)
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    
    return model

def predict_image(model: VisionEncoderDecoderModel, image_path: str) -> Dict[str, str]:
    """Predict text from image and get geolocation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # Set padding token
    tokenizer.pad_token = tokenizer.eos_token
    
    # Get model prediction
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # Get geolocation
    location = get_geolocation(image_path)
    
    return {
        "text": generated_text,
        "location": location
    }

if __name__ == "__main__":
    # Example usage
    data_dir = "data"  # Directory containing images and labels.txt
    train_loader, test_loader = load_dataset(data_dir)
    model = train_model(train_loader, test_loader)
    
    # Save model
    model.save_pretrained("saved_model") 