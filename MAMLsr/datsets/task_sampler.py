import random
import torch


class TaskSampler:

    def __init__(
        self,
        dataset,
        support_size,
        query_size
    ):

        self.dataset = dataset
        self.support_size = support_size
        self.query_size = query_size

    def sample_task(self):

        total = self.support_size + self.query_size

        
        indices = random.sample(
            range(len(self.dataset)),
            total
        )

        support_indices = indices[:self.support_size]
        query_indices = indices[self.support_size:]

        support_images = []
        query_images = []

      
        for idx in support_indices:
            image = self.dataset[idx]
            support_images.append(image)

        for idx in query_indices:
            image = self.dataset[idx]
            query_images.append(image)

        support_images = torch.stack(support_images)
        query_images = torch.stack(query_images)

        return support_images, query_images
