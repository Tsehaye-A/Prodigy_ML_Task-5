# Food Recognition and Calorie Estimation Model

This project develops a machine learning model to recognize food items from images and estimate their calorie content using the Food-101 dataset. It enables users to track dietary intake for informed food choices. The model uses transfer learning with EfficientNet-B0 for classification and a nutritional lookup for calorie estimates.

**Key Points:**
- Transfer learning with models like EfficientNet achieves over 80% accuracy on Food-101.
- Calorie estimates are approximations based on average servings.
- Implementation in PyTorch, with code in `food_model.py`.

## Setup Instructions
- **Requirements**: Python 3.x, PyTorch, torchvision.
- Download Food-101 via PyTorch or from [Kaggle](https://www.kaggle.com/dansbecker/food-101).
- Run `python food_model.py` for training and testing.

## Usage
Upload an image to predict the food class and calories. Example: "Pizza" as 250 kcal per slice.

## Results
Benchmarks show 85-90% accuracy. Enhance with portion detection for better calorie precision.
