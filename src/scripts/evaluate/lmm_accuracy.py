from functools import partial

import torch

from src.architecture.phi_4_multimodal import (
    load_lmm, get_prompt, generate_responses
)
from src.data import cifar10
from src.data.batch import batch

BATCH_SIZE = 4


def classify(images, model, prompts):
    responses = generate_responses(images, prompts, model)

    predictions = []
    for response in responses:
        prediction = response.strip()
        class_id = cifar10.CLASS_ID.get(prediction, -1)
        if class_id == -1:
            print(f"❌ Unexpected response: \"{response}\"")
        predictions.append(class_id)

    return predictions


def main():
    lmm = load_lmm()
    gb_used = torch.cuda.memory_allocated(lmm.device) / 1e9
    print(f"\nModel size: {gb_used:.2f} GB")  # 11.89 GB

    prompt = get_prompt()
    prompts = [prompt] * BATCH_SIZE

    classify_images = partial(classify, model=lmm, prompts=prompts)

    _, validation_data = cifar10.get_data(sizes=(49_000, 1_000))

    correct_count, total_count = 0, 0
    for images, labels in batch(validation_data, BATCH_SIZE):
        with torch.no_grad():
            predictions = classify_images(images)

        correct_count += sum(1 for p, l in zip(predictions, labels) if p == l)
        total_count += len(labels)

        accuracy = correct_count / total_count
        log_str = f"Current {correct_count}/{total_count} ({accuracy:.1%})"
        print(log_str, end='\r')

    print(f"Validation accuracy: {correct_count / total_count:.1%}")  # 86.9%


if __name__ == "__main__":
    main()
