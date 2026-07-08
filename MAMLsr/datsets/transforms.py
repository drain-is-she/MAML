from torchvision import transforms


def get_transform(image_size=128):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
