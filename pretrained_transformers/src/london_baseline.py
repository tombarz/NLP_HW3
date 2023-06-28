# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.
from pretrained_transformers.src.dataset import CharCorruptionDataset
from pretrained_transformers.src.utils import evaluate_places

dev_dataset = CharCorruptionDataset(open("birth_dev.tsv",encoding='utf-8').read(), 128)
predictions = ['London'] * (len(dev_dataset) -1)
total, correct = evaluate_places("birth_dev.tsv", predictions)
accuracy = correct / total * 100

print(f"Baseline Accuracy: {accuracy:.2f}%")
