import random
import matplotlib.pyplot as plt
import numpy as np

# Updated Dataset
x = [
    [160, 7], [140, 5], [120, 6], [155, 8], [135, 4], [145, 7],
    [90, 2], [110, 3], [100, 4], [95, 3], [115, 5], [130, 6],
    [180, 9], [160, 6], [170, 8], [140, 5], [150, 7], [85, 2],
    [120, 3], [100, 5], [110, 4], [105, 4], [125, 6], [140, 5],
    [155, 7], [165, 8], [135, 6], [150, 7], [145, 6]
]
y = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

# Initialize Weights with Random Small Values
weight = random.random() * 0.1
color_intensity = random.random() * 0.1
b = random.random() * 0.1
nita = 0.01  # Learning Rate
epochs = 100  # Number of Training Iterations

# Tracking weight changes
weight_history = []
color_intensity_history = []
bias_history = []

def perceptron_train():
    global weight, color_intensity, b  # Use global variables
    
    for epoch in range(epochs):
        for i in range(len(x)):
            x1, x2 = x[i]
            z = x1 * weight + x2 * color_intensity + b
            prediction = 1 if z >= 0 else 0
            
            # Store weight history before update
            weight_history.append(weight)
            color_intensity_history.append(color_intensity)
            bias_history.append(b)

            if prediction != y[i]:
                weight += nita * (y[i] - prediction) * x1
                color_intensity += nita * (y[i] - prediction) * x2
                b += nita * (y[i] - prediction)

perceptron_train()

def perceptron_predict(x_test):
    z = x_test[0] * weight + x_test[1] * color_intensity + b
    return "Apple" if z >= 0 else "Orange"

# Plot decision boundary
def plot_decision_boundary():
    x_min, x_max = min([sample[0] for sample in x]) - 10, max([sample[0] for sample in x]) + 10
    y_min, y_max = min([sample[1] for sample in x]) - 1, max([sample[1] for sample in x]) + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = np.array([perceptron_predict([i, j]) == "Apple" for i, j in zip(np.ravel(xx), np.ravel(yy))])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3)
    for i, sample in enumerate(x):
        plt.scatter(sample[0], sample[1], c='red' if y[i] == 1 else 'blue', marker='o')
    
    plt.title("Decision Boundary after Training")
    plt.xlabel("Feature 1 (e.g., Size)")
    plt.ylabel("Feature 2 (e.g., Color Intensity)")
    plt.legend(["Apple", "Orange"])
    plt.show()

def plot_weight_changes():
    plt.figure(figsize=(8, 6))
    plt.plot(weight_history, label="Weight (Size)")
    plt.plot(color_intensity_history, label="Weight (Color Intensity)")
    plt.plot(bias_history, label="Bias")
    plt.xlabel("Update Step")
    plt.ylabel("Value")
    plt.title("Weight and Bias Changes During Training")
    plt.legend()
    plt.grid()
    plt.show()

plot_decision_boundary()
plot_weight_changes()

# Function to handle user input and prediction
def user_input_and_predict():
    while True:
        try:
            x1 = float(input("Enter feature 1 (e.g., size): "))
            x2 = float(input("Enter feature 2 (e.g., color intensity): "))
            print(f"Prediction: {perceptron_predict([x1, x2])}")
            if input("Do you want to make another prediction? (yes/no): ").lower() != 'yes':
                break
        except ValueError:
            print("Please enter valid numeric values.")

user_input_and_predict()
