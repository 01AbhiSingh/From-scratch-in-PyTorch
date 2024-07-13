from alexnet import create_alexnet
from utils import print_model_summary

def main():
    # Create an instance of the model
    model = create_alexnet(num_classes=200, dropout_rate=0.5)

    # Print model summary
    print_model_summary(model)

if __name__ == "__main__":
    main()