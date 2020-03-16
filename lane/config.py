
# model config
OUTPUT_STRIDE = 16
ASPP_OUTDIM = 256
SHORTCUT_DIM = 48
SHORTCUT_KERNEL = 1
NUM_CLASSES = 8

# train config
EPOCHS = 200
WEIGHT_DECAY = 1.0e-4
SAVE_PATH = "logs"
BASE_LR = 0.0006
batch_size= 16

Root_dir = '/root/lane/data_list'
train_csv_dir = 'train.csv'
val_csv_dir = 'val.csv'




