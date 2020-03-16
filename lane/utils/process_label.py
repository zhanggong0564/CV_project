import numpy as np

def encode_labels(color_mask):
    encode_mask = np.zeros((color_mask.shape[0],color_mask.shape[1]))
    id_train = {
        0: [0, 249, 255, 213, 206, 207, 211, 208, 216, 215, 218, 219, 232, 202, 231, 230, 228, 229, 233, 212, 223],
        1: [200, 204, 209], 2: [201, 203], 3: [217], 4: [210], 5: [214],
        6: [220, 221, 222, 224, 225, 226], 7: [205, 227, 250]}
    for i in range(8):
        for item in id_train[i]:
            encode_mask[color_mask==item] = i
    return encode_mask
def decode_labels(labels):
    deocde_mask = np.zeros((labels.shape[0], labels.shape[1]), dtype='uint8')
    # 0
    deocde_mask[labels == 0] = 0
    # 1
    deocde_mask[labels == 1] = 204
    # 2
    deocde_mask[labels == 2] = 203
    # 3
    deocde_mask[labels == 3] = 217
    # 4
    deocde_mask[labels == 4] = 210
    # 5
    deocde_mask[labels == 5] = 214
    # 6
    deocde_mask[labels == 6] = 224
    # 7
    deocde_mask[labels == 7] = 227

    return deocde_mask