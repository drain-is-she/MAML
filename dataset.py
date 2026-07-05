import os
import random

from PIL import Image
import torch 
from torch.utils.data import Dataset
from torchvision import transforms

from config import *


class MammalDataset(Dataset):

    def __init__(self, root_dir=ROOT_DIR, transform=None):

        self.root_dir = root_dir
        self.transform = transform

        self.classes = sorted([
            folder
            for folder in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, folder))
        ])

        self.class_to_images = {}

        for class_name in self.classes:

            class_path = os.path.join(self.root_dir, class_name)

            image_files = [
                os.path.join(class_path, image)
                for image in os.listdir(class_path)
                if image.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

            self.class_to_images[class_name] = image_files

        print(f"Total Classes: {len(self.classes)}")
        print(self.classes[:5])
        print(f"Horse Images: {len(self.class_to_images['horse'])}")

    def __len__(self):

        # In MAML we don't have a fixed dataset length.
        # Each call generates a completely new episode.
        return 100000

    def __getitem__(self, idx):
      

    # Randomly choose N classes for one episode
      selected_classes = random.sample(self.classes, N_WAY)

    # Lists for support set
      support_images = []
      support_labels = []

    # Lists for query set
      query_images = []
      query_labels = []

    # Process each selected class
      for label, class_name in enumerate(selected_classes):

        # Get all image paths for this class
         images = self.class_to_images[class_name]

        # Sample K_SHOT + Q_QUERY images
         sampled_images = random.sample(
            images,
            K_SHOT + Q_QUERY
        )

        # Split into support and query
         support = sampled_images[:K_SHOT]
         query = sampled_images[K_SHOT:]

        
         for image_path in support:
            

            image = Image.open(image_path).convert("RGB")

            if self.transform is not None:
                image = self.transform(image)

            support_images.append(image)
            support_labels.append(label)

        
         for image_path in query:

            image = Image.open(image_path).convert("RGB")

            if self.transform is not None:
                image = self.transform(image)

            query_images.append(image)
            query_labels.append(label)

    # Convert to tensors
      support_images = torch.stack(support_images)
      query_images = torch.stack(query_images)

      support_labels = torch.tensor(support_labels, dtype=torch.long)
      query_labels = torch.tensor(query_labels, dtype=torch.long)

      return (
        support_images,
        support_labels,
        query_images,
        query_labels
    )

if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.Resize((84,84)),
        transforms.ToTensor()
    ])

    dataset = MammalDataset(transform=transform)

    support_images, support_labels, query_images, query_labels = dataset[0]

