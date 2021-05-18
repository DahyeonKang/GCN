import argparse
import torch
from datasets import load_data
from models.gcn import GCN
from models.utils import get_accuracy

parser = argparse.ArgumentParser()  # 인자값을 받을 수 있는 ArgumentParser 객체 생성
# ArgumentParser에 프로그램 인자에 대한 정보를 채우기 위해 add_argumert 메소드 호출
parser.add_argument('--dataset', type=str, default='citeseer', help='Dataset to train')
parser.add_argument('--hidden_dim', type=list, default=16, help='Dimensions of hidden layers')
parser.add_argument('--checkpoint', type=str, help='Directory to save checkpoints')
args = parser.parse_args() # 위에서 정의한 3개의 객체를 반환한다.(type:namespace)


adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(args.dataset)
model = GCN(features.shape[1], args.hidden_dim, y_train.shape[1], 0)

#check = model.load_state_dict(torch.load(args.checkpoint))

def evaluate(checkpoint):
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    outputs = model(adj, features)
    accuracy = get_accuracy(outputs, y_test, test_mask)
    print("Accuracy on test set is %f" %accuracy)


if __name__ == '__main__':
    evaluate(args.checkpoint)