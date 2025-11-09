import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.datasets import Food101
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Define transformations
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset (downloads if not present; set download=False if manually downloaded)
train_data = Food101(root='./data', split='train', transform=train_transform, download=True)
test_data = Food101(root='./data', split='test', transform=test_transform, download=True)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Calorie dictionary (from nutritional data)
calorie_dict = {
    'apple_pie': 300, 'baby_back_ribs': 250, 'baklava': 290, 'beef_carpaccio': 150, 'beef_tartare': 200,
    'beet_salad': 75, 'beignets': 95, 'bisque': 200, 'bread_pudding': 150, 'breakfast_burrito': 350,
    'bruschetta': 80, 'caesar_salad': 160, 'cannoli': 200, 'caprese_salad': 220, 'carrot_cake': 250,
    'ceviche': 200, 'cheesecake': 350, 'cheese_plate': 100, 'chicken_curry': 300, 'chicken_quesadilla': 400,
    'chicken_wings': 430, 'chocolate_cake': 250, 'chocolate_mousse': 250, 'churros': 150, 'clam_chowder': 110,
    'club_sandwich': 500, 'crab_cakes': 220, 'creme_brulee': 400, 'croque_madame': 500, 'cup_cakes': 250,
    'deviled_eggs': 80, 'donuts': 200, 'dumplings': 200, 'edamame': 190, 'eggs_benedict': 300,
    'escargots': 140, 'falafel': 350, 'filet_mignon': 180, 'fish_and_chips': 500, 'foie_gras': 100,
    'french_fries': 300, 'french_onion_soup': 110, 'french_toast': 200, 'fried_calamari': 350, 'fried_rice': 240,
    'frozen_yogurt': 100, 'garlic_bread': 150, 'gnocchi': 130, 'greek_salad': 200, 'grilled_cheese_sandwich': 350,
    'grilled_salmon': 180, 'guacamole': 120, 'gyoza': 200, 'hamburger': 250, 'hot_and_sour_soup': 100,
    'hot_dog': 250, 'huevos_rancheros': 400, 'hummus': 60, 'ice_cream': 140, 'lasagna': 400,
    'lobster_bisque': 200, 'lobster_roll_sandwich': 400, 'macaroni_and_cheese': 400, 'macarons': 150, 'miso_soup': 50,
    'mussels': 160, 'nachos': 500, 'omelette': 200, 'onion_rings': 244, 'oysters': 80,
    'pad_thai': 400, 'paella': 400, 'pancakes': 200, 'panna_cotta': 250, 'peking_duck': 200,
    'pho': 300, 'pizza': 250, 'pork_chop': 200, 'poutine': 500, 'prime_rib': 250,
    'pulled_pork_sandwich': 400, 'ramen': 400, 'ravioli': 300, 'red_velvet_cake': 300, 'risotto': 200,
    'samosa': 300, 'sashimi': 100, 'scallops': 100, 'seaweed_salad': 50, 'shrimp_and_grits': 300,
    'spaghetti_bolognese': 400, 'spaghetti_carbonara': 500, 'spring_rolls': 200, 'steak': 200, 'strawberry_shortcake': 300,
    'sushi': 50, 'tacos': 300, 'takoyaki': 200, 'tuna_tartare': 150, 'waffles': 200
}

# Load pre-trained model (updated to use weights for modern PyTorch)
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 101)  # 101 classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

# Train and evaluate
train_model(model, train_loader, criterion, optimizer)
evaluate_model(model, test_loader)

# Prediction and calorie estimation
def predict_and_estimate_calories(image_path):
    transform = test_transform
    image = transform(Image.open(image_path)).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(image)
        pred_idx = torch.argmax(output, dim=1).item()
    class_name = train_data.classes[pred_idx]
    calories = calorie_dict.get(class_name, "Unknown")
    return class_name, calories

# Example usage (replace with actual image path)
# class_name, calories = predict_and_estimate_calories('example_image.jpg')
# print(f"Recognized: {class_name}, Estimated Calories: {calories}")
