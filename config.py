import transformers

from torch.cuda.amp import GradScaler


class config:
    MODEL_NAME = 'bert-base-chinese-layer4'
    NB_EPOCHS = 10
    LR = 2e-5
    MAX_LEN = 32
    N_SPLITS = 5
    TRAIN_BS = 32
    VALID_BS = 32
    HIDDEN_SIZE = 768
    TEMP = 0.05
    HARD_NEGATIVE_WEIGHT = 1
    FILE_NAME = './data_all.csv'
    MODEL_PATH = "./bert-base-chinese"
    TOKENIZER = transformers.AutoTokenizer.from_pretrained("./bert-base-chinese", do_lower_case=True)
    scaler = GradScaler()
