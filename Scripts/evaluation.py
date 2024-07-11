import torch
from sklearn.metrics import confusion_matrix

def confusionMatrix(model, dataloader, device):
    model.eval()

    # Initialize lists to store true labels and predictions
    true_labels = []
    predictions = []

    with torch.no_grad():  # No need to track gradients
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
        
            # Forward pass
            outputs = model(inputs)
        
            # Convert output probabilities to predicted class
            _, preds = torch.max(outputs, 1)
        
            # Append batch predictions and labels to lists
            predictions.extend(preds.view(-1).cpu().numpy())
            true_labels.extend(labels.view(-1).cpu().numpy())

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(true_labels, predictions)
    
    return conf_matrix

    