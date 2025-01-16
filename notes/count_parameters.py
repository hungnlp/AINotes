from torch import nn


def count_number_parameters(model: nn.Module):
    total_parameters = 0
    trainable_parameters = 0
    for _, parameter in model.named_parameters():
        total_parameters += parameter.numel()

        if parameter.requires_grad:
            trainable_parameters += parameter.numel()

    print(f"Total parameters: {total_parameters} || Trainable parameters: {trainable_parameters}")


if __name__ == '__main__':
    from transformers import AutoModelForSequenceClassification

    model_id = "clapAI/modernBERT-base-multilingual-sentiment"

    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    count_number_parameters(model)
