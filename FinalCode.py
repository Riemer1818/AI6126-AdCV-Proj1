import os
import numpy as np
import torch
from transformers import ViTForImageClassification
from torchvision import transforms
from PIL import Image
import argparse
import logging
import numpy as np
import os
from PIL import Image

import torch.nn as nn
import torch.nn.functional as F
logging.basicConfig(level=logging.INFO)


import torchvision.transforms as transforms

class FashionNet_Dataset():

    def __init__(self, root, txt, dataset):
        self.img_path = []
        self.labels = [[] for _ in range(len(num_subattributes))]

        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                # make dummy label for test set
                if 'test' in txt:
                    for i in range(len(num_subattributes)):
                        self.labels[i].append(0)

        if 'test' not in txt:
            with open(txt.replace('.txt', '_attr.txt')) as f:
                for line in f:
                    attrs = line.split()
                    for i in range(len(num_subattributes)):
                        self.labels[i].append(int(attrs[i]))

    def __len__(self):
        return len(self.labels[0])

    def __getitem__(self, index):

        path = self.img_path[index]
        label = np.array([self.labels[i][index] for i in range(len(num_subattributes))])
        label = label.astype(np.float32)

        with Image.open(path) as image:
            image = image.convert('RGB')
            sample = transform(image)

        return sample, label, index

def attribute_distribution(attr, num_classes, label_count):
  for label in range(attr.shape[1]):
    for i in range(len(attr)):
      class_label = attr[i, label]
      label_count[class_label, label] += 1
  return label_count

class MultiTaskHead(nn.Module):
    def __init__(self, input_size, dropout_rate):
        super(MultiTaskHead, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(dropout_rate),
            nn.ReLU(),

            nn.Linear(1024, 1024),
            nn.Dropout(dropout_rate),
            nn.ReLU(),

            nn.Linear(1024, 1024),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
        )

        # Task-specific layers
        self.task_layers = nn.ModuleList([
            nn.Linear(1024, n) for n in num_subattributes
        ])

    def forward(self, x):
        for layer in self.shared_layers:
            x = layer(x)
        outputs = [layer(x) for layer in self.task_layers]
        return torch.cat(outputs, dim=1)

class MultiTaskPretrained(nn.Module):
    def __init__(self, layers_to_unfreeze, dropout_rate):
        super(MultiTaskPretrained, self).__init__()

        self.pretrained = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')

        input_size = self.pretrained.classifier.in_features
        self.pretrained.classifier = MultiTaskHead(input_size, dropout_rate)

        # Freeze all parameters initially
        for param in self.pretrained.parameters():
            param.requires_grad = False

        # Unfreeze the last 'layers_to_unfreeze' transformer blocks
        if layers_to_unfreeze > 0:
            for layer in self.pretrained.vit.encoder.layer[-layers_to_unfreeze:]:
                for param in layer.parameters():
                    param.requires_grad = True

        # Unfreeze classifier (head) parameters
        for param in self.pretrained.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        # Modify here to correctly access the logits
        model_output = self.pretrained(x)
        logits = model_output.logits  # This assumes the output object has a .logits attribute
        return logits

def custom_loss_adjusted_relative(outputs, labels, critList):
    start = 0
    losses = []

    for i, num_classes in enumerate(num_subattributes):

        task_labels = labels[:, i].long()  # Ensure task labels are long for indexing

        end = start + num_classes
        task_output = outputs[:, start:end]
        task_output = F.softmax(task_output, dim=1)  # Apply softmax to get probabilities

        # Initialize labelExpanded with correct dimensions
        labelExpanded = torch.zeros((task_output.shape[0], num_classes)).to(device)
        labelExpanded.scatter_(1, task_labels.unsqueeze(1), 1.0) # no flipping way this works

        # get criterion from relative weights list.
        criterion = critList[i]
        task_loss = criterion(task_output, labelExpanded)

        losses.append(task_loss)

        start = end

    # Dynamically adjust losses
    total_loss = sum(losses)
    weighted_losses = [(loss / total_loss).item() * loss for loss in losses]

    return sum(weighted_losses)

def extract_predictions(outputs):
    start = 0
    predictions = []
    for num_classes in num_subattributes:
        end = start + num_classes

        # Extract logits for the current attribute
        attr_logits = outputs[:, start:end]

        # Determine the predicted class for the current attribute
        attr_predictions = torch.argmax(attr_logits, dim=1)
        predictions.append(attr_predictions)
        start = end

    # Stack predictions along columns
    return torch.stack(predictions, dim=1)

def compute_avg_class_acc(gt_labels, pred_labels):

    per_class_acc = []
    for attr_idx in range(len(num_subattributes)):
        for idx in range(num_subattributes[attr_idx]):
            target = gt_labels[:, attr_idx]
            pred = pred_labels[:, attr_idx]
            correct = np.sum((target == pred) * (target == idx))
            total = np.sum(target == idx)
            per_class_acc.append(float(correct) / float(total))

    return sum(per_class_acc) / len(per_class_acc)

def compute_accuracy(model, val_loader, save=False):

  # Assume model, val_loader, and device are already defined
  model.eval()
  all_gt_labels = []
  all_pred_labels = []

  with torch.no_grad():
      for images, labels, _ in val_loader:
          images = images.to(device)
          labels = labels.to(device)

          outputs = model(images)
          predictions = extract_predictions(outputs)  # Adjust the list as per your attributes
          # print(predictions) # still in batch shape

          # Accumulate labels and predictions for later evaluation
          all_gt_labels.append(labels.cpu().numpy())
          all_pred_labels.append(predictions.cpu().numpy())

  # Concatenate all batches to create a single array for labels and predictions
  all_gt_labels = np.concatenate(all_gt_labels, axis=0)
  all_pred_labels = np.concatenate(all_pred_labels, axis=0)

  # Compute average class accuracy across all attributes
  avg_class_acc = compute_avg_class_acc(all_gt_labels, all_pred_labels)

  # Compute total accuracy
  total_acc = np.mean(all_gt_labels == all_pred_labels)

  # save to text file
  if save:
    np.savetxt(f'pred_labels_bestModel.txt', all_pred_labels, fmt='%d')

  return [avg_class_acc, total_acc]

def count_abundances():
    # # Initialize the array with zeros
    counts = np.zeros((7, 6))

    # Automatically calculate bin counts and fill in the array
    for i, _ in enumerate(num_subattributes):
    # binn for the labels so I can normalize over the attributes
        bincount = np.bincount(trainset.labels[i])
        length = min(len(bincount), 7)  # Ensure we don't exceed the number of rows in thisOne
        counts[:length, i] = bincount[:length]

    relative_abundances = counts / counts.sum(axis=0)
    relative_abundances = 1/relative_abundances

    return relative_abundances
    
def train(LAYERS_TO_UNFREEZE=1, EPOCHS=20, BATCH_SIZE=32, LR=0.0003, DROPOUT_RATE=0.2):
    epoch_train_losses = []
    epoch_val_losses = []
    epoch_accuracies = []  


    # make relative abundances to tensor
    relative_abundances = torch.tensor(count_abundances()).to(device)

    model = MultiTaskPretrained(LAYERS_TO_UNFREEZE, DROPOUT_RATE)
    model.to(device)

    train_losses = []
    val_losses = []

    train_loader = torch.utils.data.DataLoader(batch_size=BATCH_SIZE, shuffle=True, dataset=trainset)
    val_loader = torch.utils.data.DataLoader(batch_size=BATCH_SIZE, shuffle=True, dataset=valset)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # making a list of criterions for later calling.
    critList = []
    for i, num_classes in enumerate(num_subattributes):
        pos_weights = relative_abundances[:num_classes, i].float()

        # cast to criterion
        critList.append(torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights))


    best_acc = 0.0  # Initialize best accuracy to 0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = custom_loss_adjusted_relative(outputs, labels, critList)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        val_loss = 0.0
        for images, labels, _ in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = custom_loss_adjusted_relative(outputs, labels, critList)
            
            val_losses.append(loss.item())

        # params and metrics
        val_loss = np.mean(val_losses)
        train_loss = np.mean(train_losses)
        accuracies = compute_accuracy(model = model, val_loader = val_loader, save=False)
        metric = accuracies[0]*accuracies[1]

        epoch_train_losses.append(train_loss)
        epoch_val_losses.append(val_loss)
        epoch_accuracies.append(accuracies)  # Store accuracy tuple

        print(f"Epoch {epoch+1}/{EPOCHS}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Average Class Accuracy: {accuracies[0]:.2f}, Total Accuracy: {accuracies[1]:.2f}, Metric: {metric:.2f}")

        if metric > best_acc:
            best_acc = metric
            torch.save(model.state_dict(), f'{dir_name}/{model_name}_best_model.pth')
            
        scheduler.step()
        
    return epoch_train_losses, epoch_val_losses, epoch_accuracies


def test(LAYERS_TO_UNFREEZE=1, EPOCHS=20, BATCH_SIZE=32, LR=0.0003, DROPOUT_RATE=0.2):
    # Load the best performing model
    model = MultiTaskPretrained(layers_to_unfreeze=LAYERS_TO_UNFREEZE, dropout_rate=DROPOUT_RATE)
    model.load_state_dict(torch.load(f'{dir_name}/{model_name}_best_model.pth'))
    model.to(device)

    # Create the test data loader
    test_loader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

    model.eval()
    all_gt_labels = []
    all_pred_labels = []
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predictions = extract_predictions(outputs)
            # all_gt_labels.append(labels.cpu().numpy())
            all_pred_labels.append(predictions.cpu().numpy())
    # all_gt_labels = np.concatenate(all_gt_labels, axis=0)
    all_pred_labels = np.concatenate(all_pred_labels, axis=0)
    # avg_class_acc = compute_avg_class_acc(all_gt_labels, all_pred_labels, num_subattributes)
    # total_acc = np.mean(all_gt_labels == all_pred_labels)
    return all_pred_labels

if __name__ == '__main__':
    # system arguments 

    # Create the parser
    parser = argparse.ArgumentParser(description='model parameters.')

    # Add arguments
    parser.add_argument('layers_to_unfreeze', type=int, help='Number of layers to unfreeze')
    parser.add_argument('epochs', type=int, help='Number of epochs')
    parser.add_argument('batch_size', type=int, help='Batch size')
    parser.add_argument('learning_rate', type=float, help='Learning rate')
    parser.add_argument('dropout_rate', type=float, help='Dropout rate')

    # Parse the arguments
    args = parser.parse_args()
    
    # Assigning parsed arguments to variables
    LAYERS_TO_UNFREEZE = args.layers_to_unfreeze
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LR = args.learning_rate
    DROPOUT_RATE = args.dropout_rate
    
    global num_subattributes
    num_subattributes = [7, 3, 3, 4, 6, 3]
    global device 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using", device)
    
    global model_name
    model_name = f'{LAYERS_TO_UNFREEZE}_{EPOCHS}_{BATCH_SIZE}_{LR}_{DROPOUT_RATE}'
    global dir_name
    dir_name = f"./models/{model_name}"  # Use a relative path for the directory

    # Create the directory if it does not exist
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.RandomVerticalFlip(p=0.5)
    ])

    testlist = "./FashionDataset/split/test.txt"
    trainlist = "./FashionDataset/split/train.txt"
    vallist = "./FashionDataset/split/val.txt"

    global trainset, valset, testset
    trainset = FashionNet_Dataset("./FashionDataset", trainlist, "train")
    valset = FashionNet_Dataset("./FashionDataset", vallist, "val")
    testset = FashionNet_Dataset("./FashionDataset", testlist, "test")

  
    epoch_train_losses, epoch_val_losses, epoch_accuracies = train(LAYERS_TO_UNFREEZE=args.layers_to_unfreeze, EPOCHS=args.epochs, BATCH_SIZE=args.batch_size, LR=args.learning_rate, DROPOUT_RATE=args.dropout_rate)
    pred_labels = test(LAYERS_TO_UNFREEZE=args.layers_to_unfreeze, EPOCHS=args.epochs, BATCH_SIZE=args.batch_size, LR=args.learning_rate, DROPOUT_RATE=args.dropout_rate)



    # After the training loop, save the metrics and model
    np.savetxt(f'{dir_name}/pred_labels_{model_name}.txt', pred_labels, fmt='%d')
    np.savetxt(f'{dir_name}/epoch_train_losses_{model_name}.txt', epoch_train_losses, fmt='%f')
    np.savetxt(f'{dir_name}/epoch_val_losses_{model_name}.txt', epoch_val_losses, fmt='%f')
    np.savetxt(f'{dir_name}/epoch_accuracies_{model_name}.txt', epoch_accuracies, fmt='%f')

    # For epoch_accuracies, assuming it's a list of tuples, you might want to unpack it first
    avg_class_acc, total_acc = zip(*epoch_accuracies)  # This separates the tuple into two lists
    np.savetxt(f'{dir_name}/avg_class_acc_{model_name}.txt', avg_class_acc, fmt='%f')
    np.savetxt(f'{dir_name}/total_acc_{model_name}.txt', total_acc, fmt='%f')


