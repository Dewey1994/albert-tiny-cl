import transformers

from torch.cuda.amp import GradScaler


class config:
    MODEL_NAME = 'chinese-bert-wwm-ext-layer4'
    NB_EPOCHS = 10
    LR = 2e-5
    MAX_LEN = 32
    N_SPLITS = 5
    TRAIN_BS = 32
    VALID_BS = 32
    HIDDEN_SIZE = 768
    TEMP = 0.05
    HARD_NEGATIVE_WEIGHT = 1
    FILE_NAME = 'data/data_all.csv'
    MODEL_PATH = "./chinese-bert-wwm-ext"
    TOKENIZER = transformers.AutoTokenizer.from_pretrained(MODEL_PATH, do_lower_case=True)
    scaler = GradScaler()
