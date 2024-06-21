import os
import random
import matplotlib.pyplot as plt
import cv2
import numpy as np
from glob import glob

# Function to load random images from a directory
def load_random_images_from_dir(directory, num_samples=5):
    # Get all image files in the directory
    image_files = glob(os.path.join(directory, '*.jpg'))
    
    # Check if number of samples is more than available images
    if num_samples > len(image_files):
        num_samples = len(image_files)
    
    # Sample random images from the list
    selected_files = random.sample(image_files, num_samples)
    images = [cv2.imread(img_file) for img_file in selected_files]
    images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
    return images, selected_files


# Function to display images in a grid
def display_images(images, titles=None, figsize=(15, 5)):
    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=figsize)
    if num_images == 1:
        axes = [axes]
    for i, img in enumerate(images):
        axes[i].imshow(img)
        axes[i].axis('off')
        if titles:
            axes[i].set_title(titles[i])
    plt.tight_layout()
    plt.show()


# Function to get class distribution
def get_class_distribution(data_dir):
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    distribution = {cls: len(glob(os.path.join(data_dir, cls, '*.jpg'))) for cls in classes}
    return distribution


# Function to plot class distribution
def plot_class_distribution(distribution):
    plt.figure(figsize=(10, 6))
    plt.bar(distribution.keys(), distribution.values(), color=['blue', 'green'])
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.title('Class Distribution')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()


# Function to plot image histograms
def plot_image_histograms(images, bins=50):
    plt.figure(figsize=(20, 10))
    for i, img in enumerate(images):
        plt.subplot(2, len(images) // 2, i + 1)
        plt.hist(img.ravel(), bins=bins, color='gray', alpha=0.7)
        plt.title(f'Histogram {i + 1}')
    plt.tight_layout()
    plt.show()


# Define paths to directories containing images
train_dir = 'data/train'
val_dir = 'data/val'
test_dir = 'data/test'

# Main function to run exploration
def main():
    # Load and display random images from each category
    print("Loading random images from 'malnourished' category...")
    malnourished_images, _ = load_random_images_from_dir(os.path.join(train_dir, 'malnourished'), num_samples=5)
    print("Loading random images from 'healthy' category...")
    healthy_images, _ = load_random_images_from_dir(os.path.join(train_dir, 'healthy'), num_samples=5)

    print(f"Displaying {len(malnourished_images)} random images from malnourished category:")
    display_images(malnourished_images)

    print(f"Displaying {len(healthy_images)} random images from healthy category:")
    display_images(healthy_images)

    # Display class distribution
    print("Class Distribution:")
    distribution = get_class_distribution(train_dir)
    plot_class_distribution(distribution)

    # Display image histograms for a few random images
    sample_images = [cv2.imread(img) for img in random.sample(glob(os.path.join(train_dir, 'malnourished', '*.jpg')) + glob(os.path.join(train_dir, 'healthy', '*.jpg')), 6)]
    sample_images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in sample_images]
    print("Displaying image histograms:")
    plot_image_histograms(sample_images)


if __name__ == '__main__':
    main()
