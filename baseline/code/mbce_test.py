import torch
import torch.nn.functional as F

def masked_binary_cross_entropy(pred, gold):

    pred = pred.flatten()
    gold = gold.flatten()
    mask = (gold >= 0)

    return F.binary_cross_entropy(pred[mask], gold[mask], reduction='sum')

def test_mbce():

    pred_lin_1 = torch.tensor([0.9, 0.1]).float()
    gold_lin_1 = torch.tensor([0, 1]).float()

    pred_lin_2 = torch.tensor([0.9, 0.1, 0.5]).float()
    gold_lin_2 = torch.tensor([0, 1, -1]).float()

    pred_multi_1 = torch.tensor([[0.9, 0.1, 0.5, 0.5], [0.1, 0.9, 0.1, 0.5]]).float()
    gold_multi_1 = torch.tensor([[0, 1, -1, -1], [1, 0, 1, -1]]).float()

    mbce_lin_1 = masked_binary_cross_entropy(pred_lin_1, gold_lin_1)
    mbce_lin_2 = masked_binary_cross_entropy(pred_lin_2, gold_lin_2)
    mbce_multi_1 = masked_binary_cross_entropy(pred_multi_1, gold_multi_1)

    print(mbce_lin_1)
    print(mbce_lin_2)
    print(mbce_multi_1)

    assert(mbce_lin_1 == mbce_lin_2)
    assert(mbce_lin_1 == mbce_lin_2)
    assert(mbce_lin_2 == mbce_multi_1)

if __name__ == "__main__":
    test_mbce()
