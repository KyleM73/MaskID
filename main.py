from util import *
from config import *

def main(verbose=False):
    # Initialize Model
    model, input_size = initialize_model(feature_extract=True)

    if verbose:
        from torchsummary import summary
        summary(model,(3,H,W))
        #from torchstat import stat
        #stat(model,(3,H,W))

    # Set Optimizer
    optimizer = optim.SGD(set_update_params(model), lr=LR, momentum=M)

    # Set Loss Function
    criterion = MaskLoss()

    # Train
    model, hist = train_model(model, get_datasets(), criterion, optimizer, num_epochs=E)

    # Evaluate
    evaluate(model,get_eval_dataset(),hist)

if __name__ == "__main__":
    main()