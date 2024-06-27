import matplotlib.pyplot as plt
import json
import os

def plotLoss(model_name):
    # Construct the path to the JSON file
    file_path = os.path.join(os.path.dirname(__file__), '..', 'Models', f"{model_name}_loss.json")
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print("Log does not exist!")
        return
    
    # Read the JSON file
    with open(file_path, 'r') as file:
        loss_data = json.load(file)
    train_loss_list = loss_data['training']
    val_loss_list = loss_data['validation']
    
    # Plot the loss data
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_list, label='Training Loss')
    plt.plot(val_loss_list, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    