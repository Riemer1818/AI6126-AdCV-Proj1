import torch
from transformers import ViTForImageClassification

import torch.nn as nn

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

if __name__ == '__main__':
    num_subattributes = [7, 3, 3, 4, 6, 3]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using", device)

    model = MultiTaskPretrained(layers_to_unfreeze=1, dropout_rate=0.2)
    print(model)
