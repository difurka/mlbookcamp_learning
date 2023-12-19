
import torch
import torch.nn as nn


from src.model import SegNet
from src.utils import get_datasets, train, score_model, iou_pytorch

if __name__ == '__main__':

    EPOCHS = 1
    LEARNING_RATE = 0.003
    BATCH_SIZE = 32

    data_tr, data_val, data_ts = get_datasets(BATCH_SIZE)
    model_seg = SegNet()
    loss = nn.BCEWithLogitsLoss()
    optim = torch.optim.AdamW(model_seg.parameters(), lr=LEARNING_RATE)

    res_bce, acc_bce = train(model_seg, optim, loss, EPOCHS, LEARNING_RATE, data_tr, data_val)
    print('score: ', score_model(model_seg, iou_pytorch, data_val))

    torch.save(model_seg.state_dict(), 'best_model_seg.pth')