from collections import defaultdict

def count_parameters(model):
    params = defaultdict(int)
    for name, param in model.named_parameters():
        name = name.split('.')[0]  # get the name of the submodule
        params[name] += param.numel()
    for name, count in params.items():
        print(f'{name}: {count}')
