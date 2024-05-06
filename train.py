# Importing necessary libraries
from data import X_train, X_test, y_train, y_test, device, NUM_CLASSES, NUM_FEATURES, visualize_data
import torch
from torch import nn
from model import BlobClassifier, accuracy

# Create an instance of the model
blobClassifier =  BlobClassifier(input_features = NUM_FEATURES, output_features = NUM_CLASSES, hidden_units = 16).to(device)

# Set the hyperparameters of the model
LEARNING_RATE = 0.1
EPOCHS = 1000

# Set the loss function and the optimizer 
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params = blobClassifier.parameters(), lr = LEARNING_RATE)

# Loop through the data
for epoch in range(EPOCHS):
    ### Training loop
    
    # 1. Forward pass
    y_logits = blobClassifier(X_train)
    y_preds = torch.softmax(y_logits, dim = 1).argmax(dim = 1)

    # 2. Calculate the loss / acc
    loss = loss_function(y_logits, y_train.type(torch.LongTensor).to(device))
    acc = accuracy(y_true = y_train, y_preds = y_preds)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backward
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    ### Testing loop
    blobClassifier.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = blobClassifier(X_test)
        test_preds = torch.softmax(test_logits, dim = 1).argmax(dim = 1)

        # 2. Calculate the loss and acc
        test_loss = loss_function(test_logits, y_test.type(torch.LongTensor).to(device))
        test_acc = accuracy(y_true = y_test, y_preds = test_preds)

    # Print everything
    if epoch % 100 == 0:
        print(f"Epoch : {epoch} | Loss : {loss:.5f} | Acc : {acc:.2f}% | Test Loss : {test_loss:.5f} | Test Acc : {test_acc:.2f}%") 

# Making predictions with the model
blobClassifier.eval()
with torch.inference_mode():
    y_logits = blobClassifier(X_test)
    y_preds = torch.softmax(y_logits, dim = 1).argmax(dim = 1)

# Compare predictions with the actual data
print(f"Predicted Values For First 10 : {y_preds[:10]}")
print(f"Actual Values For First 10 : {y_test[:10]}")

# Visualize the data
visualize_data(blobClassifier, X_train, X_test, y_train, y_test)