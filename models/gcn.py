import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Func


def preprocess_adj(A):
    '''
    Pre-process adjacency matrix
    :param A: adjacency matrix
    :return:
    '''
    I = np.eye(A.shape[0])
    A_hat = A + I # add self-loops
    D_hat_diag = np.sum(A_hat, axis=1)
    D_hat_diag_inv_sqrt = np.power(D_hat_diag, -0.5)
    D_hat_diag_inv_sqrt[np.isinf(D_hat_diag_inv_sqrt)] = 0.
    D_hat_inv_sqrt = np.diag(D_hat_diag_inv_sqrt)
    return np.dot(np.dot(D_hat_inv_sqrt, A_hat), D_hat_inv_sqrt)


class GCNLayer(nn.Module):
    # __init__과 forward로 구성된다.
    def __init__(self, in_dim, out_dim, acti=True): # acti: boolean Type
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim) # bias = False is also ok.
        if acti:
            self.acti = nn.ReLU(inplace=True)
            # inplace=True는 input으로 들어온 것 자체를 수정한다는 뜻
            # 한마디로, inplace 결과를 새로운 변수에 저장하는 것이 아니라, 기존의 데이터를 대체한다는 것
            # ->메모리 usage 좋아짐->input을 없앤다.
        else:
            self.acti = None
    def forward(self, F):
        output = self.linear(F)
        if not self.acti:
            return output
        return self.acti(output)


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, p):
        super(GCN, self).__init__()
        self.gcn_layer1 = GCNLayer(input_dim, hidden_dim)
        self.gcn_layer2 = GCNLayer(hidden_dim, num_classes, acti=False)
        self.dropout = nn.Dropout(p)

    def forward(self, A, X):
        A = torch.from_numpy(preprocess_adj(A)).float()
        X = self.dropout(X.float())
        F = torch.mm(A, X)
        F = self.gcn_layer1(F)
        F = self.dropout(F)
        F = torch.mm(A, F)
        output = self.gcn_layer2(F)
        return output