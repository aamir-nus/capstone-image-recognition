import os
import scipy.io as sio
from PIL import Image
import urllib.request

def download_svhn():
    """Download and prepare SVHN dataset."""
    # Create data directory if it doesn't exist
    os.makedirs('data/images', exist_ok=True)
    
    # Download SVHN dataset
    print("Downloading SVHN dataset...")
    url = "http://ufldl.stanford.edu/housenumbers/train_32x32.mat"
    urllib.request.urlretrieve(url, "train_32x32.mat")
    
    # Load the dataset
    print("Loading dataset...")
    train_data = sio.loadmat('train_32x32.mat')
    
    # Extract images and labels
    images = train_data['X']
    labels = train_data['y']
    
    # Convert labels to 1D array
    labels = labels.flatten()
    
    # Create labels.txt file
    with open('data/labels.txt', 'w') as f:
        for i in range(len(labels)):
            # Save image
            img = Image.fromarray(images[:,:,:,i])
            img_path = f'data/images/{i:05d}.png'
            img.save(img_path)
            
            # Write label
            f.write(f'{i:05d}.png\t{labels[i]}\n')
    
    # Clean up
    os.remove('train_32x32.mat')
    print("Dataset preparation complete!")

if __name__ == "__main__":
    download_svhn() 