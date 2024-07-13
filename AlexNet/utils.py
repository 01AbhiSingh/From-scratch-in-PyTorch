def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def print_model_summary(model):
    print(model)
    print(f"Total number of parameters: {count_parameters(model):,}")