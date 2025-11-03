self.classifier = nn.Linear(num_features, num_classes)

def supported_hyperparameters():
    return {'lr','momentum'}
