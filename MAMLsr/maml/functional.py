from torch.func import functional_call


def forward_with_params(model, params, inputs):

    return functional_call(model, params, inputs)
