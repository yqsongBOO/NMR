import torch
import torch.nn as nn

# from src.models.model.nmr_encoder import NMR_encoder
from src.models.model.nmr_encoder_onlyHorC import NMR_encoder
class ConditionT(nn.Module):

    def __init__(self, dim_enc_H, dimff_enc_H, dim_enc_C, dimff_enc_C, ffn_hidden, n_head, n_layers_TE, drop_prob, device):
        super().__init__()

        self.NMR_encoder = NMR_encoder(device=device, dim_H=dim_enc_H, dimff_H=dimff_enc_H, dim_C=dim_enc_C,
                                       dimff_C=dimff_enc_C,
                                       hidden_dim=ffn_hidden, n_head=n_head, num_layers=n_layers_TE,
                                       drop_prob=drop_prob)

        self.device = device

        checkpoint = torch.load('/public/home/ustc_yangqs/molecular2molecular/src/step2_15_H_withFormular.ckpt')
        # 获取模型的 state_dict
        state_dict = checkpoint['state_dict']
        # 从 state_dict 中提取 conditionEn 部分的权重
        NMR_encoder_state_dict = {k[len('model.NMR_encoder.'):]: v for k, v in state_dict.items() if
                                  k.startswith('model.NMR_encoder.')}
        print(len(NMR_encoder_state_dict['H1nmr_encoder.embed.shift_emb.centers']))
        # 加载到模型的 conditionEn 部分
        self.NMR_encoder.load_state_dict(NMR_encoder_state_dict)

    def forward(self, condition):
         H,C = self.NMR_encoder(condition)  # 2维张量 （batch, dim）
         # nmr = self.NMR_encoder(condition)  # 2维张量 （batch, dim）
         return H

dim_enc_H = 1024
dimff_enc_H = 2048
dim_enc_C = 256
dimff_enc_C = 512

ffn_hidden = 512
n_head = 8
n_layers_TE = 3
drop_prob = 0.
device = torch.device("cuda")

nmr_vec_test = ConditionT(dim_enc_H, dimff_enc_H, dim_enc_C, dimff_enc_C, ffn_hidden, n_head, n_layers_TE, drop_prob, device)
nmr_vec_test = nmr_vec_test.to('cuda')
nmr_vec_test.eval()

def nmrVec(conditionVec):
    nmr = nmr_vec_test(conditionVec).to('cuda')
    return nmr