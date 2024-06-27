import torch 
import os
import json


def saveModel(model, model_name):
    script_dir = os.path.dirname(__file__)
    models_dir = os.path.join(script_dir, '..', 'Models')
    model_path = os.path.join(models_dir, f"{model_name}_{getEpochs(model_name)}.pth")
    
    torch.save(model.state_dict(), model_path)
    
    print("Model saved successfully!")

def logLoss(model_name, train_loss_list, val_loss_list):
    script_dir = os.path.dirname(__file__)
    models_dir = os.path.join(script_dir, '..', 'Models')
    os.makedirs(models_dir, exist_ok=True)
    
    file_path = os.path.join(models_dir, f"{model_name}_loss.json")
    
    loss_data = {
        "training": train_loss_list,
        "validation": val_loss_list
    }
    
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            existing_data = json.load(file)
            existing_data["training"].extend(loss_data["training"])
            existing_data["validation"].extend(loss_data["validation"])
        with open(file_path, "w") as file:
            json.dump(existing_data, file)
    else:
        with open(file_path, "w") as file:
            json.dump(loss_data, file)
            
def getEpochs(model_name):
    file_path = os.path.join(os.path.dirname(__file__), '..', 'Models', f"{model_name}_loss.json")
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print("Model does not exist!")
        return
    
    # Read the JSON file
    with open(file_path, 'r') as file:
        loss_data = json.load(file)
    
    train_loss_list = loss_data['training']
    
    return len(train_loss_list)
