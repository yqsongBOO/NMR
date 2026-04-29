
import torch
import torch.nn as nn

from .transformer_model import GraphTransformer
from src.models.model.encoder import Encoder

class ConditionGT(nn.Module):

    def __init__(self, n_layers_GT: int, input_dims: dict, hidden_mlp_dims: dict, hidden_dims: dict,
                 output_dims: dict, act_fn_in: nn.ReLU(), act_fn_out: nn.ReLU(), enc_voc_size, max_len, d_model, ffn_hidden,
                 n_head, n_layers_TE, drop_prob, device):
        super().__init__()
        self.GT = GraphTransformer(n_layers=n_layers_GT,
                                      input_dims=input_dims,
                                      hidden_mlp_dims=hidden_mlp_dims,
                                      hidden_dims=hidden_dims,
                                      output_dims=output_dims,
                                      act_fn_in=act_fn_in,
                                      act_fn_out=act_fn_out)

        self.transEn = Encoder(enc_voc_size=enc_voc_size, max_len=max_len, d_model=d_model, ffn_hidden=ffn_hidden, n_head=n_head,
                               n_layers=n_layers_TE, drop_prob=drop_prob, device=device)
        self.linear_layer = nn.Linear(max_len * d_model, 512)
        self.device = device

        # checkpoint = torch.load('/public/home/ustc_yangqs/molecular2molecular/src/epoch=438.ckpt')
        # # 获取模型的 state_dict
        # state_dict = checkpoint['state_dict']
        # # 从 state_dict 中提取 conditionEn 部分的权重
        # GT_state_dict = {k[len('model.GT.'):]: v for k, v in state_dict.items() if
        #                           k.startswith('model.GT.')}
        # # 加载到模型的 conditionEn 部分
        # self.GT.load_state_dict(GT_state_dict)
        #
        # checkpoint = torch.load('/public/home/ustc_yangqs/molecular2molecular/src/epoch=35.ckpt')
        # # 获取模型的 state_dict
        # state_dict = checkpoint['state_dict']
        # # 从 state_dict 中提取 conditionEn 部分的权重
        # linear_layer_state_dict = {k[len('model.linear_layer.'):]: v for k, v in state_dict.items() if
        #                           k.startswith('model.linear_layer.')}
        # # 加载到模型的 conditionEn 部分
        # self.linear_layer.load_state_dict(linear_layer_state_dict)
        #
        # checkpoint = torch.load('/public/home/ustc_yangqs/molecular2molecular/src/epoch=35.ckpt')
        # # 获取模型的 state_dict
        # state_dict = checkpoint['state_dict']
        # # 从 state_dict 中提取 conditionEn 部分的权重
        # transEn_state_dict = {k[len('model.transEn.'):]: v for k, v in state_dict.items() if
        #                       k.startswith('model.transEn.')}
        # # 加载到模型的 conditionEn 部分
        # self.transEn.load_state_dict(transEn_state_dict)
        ####################################################################################
        #
        # for param in self.transEn.parameters():
        #     param.requires_grad = False
        # for param in self.GT.parameters():
        #     param.requires_grad = False
        # for param in self.linear_layer.parameters():
        #     param.requires_grad = False

    def make_src_mask(self, src):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        return src_mask

    def forward(self, X, E, y, node_mask, conditionVec):
        assert isinstance(conditionVec, torch.Tensor), "conditionVec should be a tensor, but got type {}".format(
            type(conditionVec))

        srcMask = self.make_src_mask(conditionVec).to(self.device)
        conditionVec = self.transEn(conditionVec, srcMask)
        conditionVec = conditionVec.view(conditionVec.size(0), -1)
        conditionVec = self.linear_layer(conditionVec)

        y = torch.hstack((y, conditionVec)).float()

        return self.GT(X, E, y, node_mask)