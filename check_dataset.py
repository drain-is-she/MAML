import os

ROOT_DIR = "mammals"

print("Current Working Directory:")
print(os.getcwd())

print()

classes = sorted(os.listdir(ROOT_DIR))

print(f"Total Classes: {len(classes)}\n")

for class_name in classes:

    class_path = os.path.join(ROOT_DIR, class_name)

    images = [
        img for img in os.listdir(class_path)
        if img.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    
