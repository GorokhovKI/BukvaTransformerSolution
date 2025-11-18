CLASSES = [
    "А", "Б", "В", "Г", "Д", "Е", "Ё", "Ж", "З", "И", "Й",
    "К", "Л", "М", "Н", "О", "П", "Р", "С", "Т", "У", "Ф",
    "Х", "Ц", "Ч", "Ш", "Щ", "Ъ", "Ы", "Ь", "Э", "Ю", "Я"
]

CLASS_TO_IDX = {label: idx for idx, label in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: label for idx, label in enumerate(CLASSES)}

NUM_CLASSES = len(CLASSES)


SEQUENCE_LENGTH = 80
NUM_LANDMARKS = 21
NUM_HANDS = 2
INPUT_SIZE = NUM_HANDS * NUM_LANDMARKS * 3

HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_HEADS = 4
DROPOUT = 0.1
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 50

TRIMMED_VIDEOS_DIR = "data/trimmed/"
ANNOTATIONS_PATH = "data/annotations.tsv"
FEATURES_DIR = "features_data"
CHECKPOINT_PATH = "best_model_bukva.pth"