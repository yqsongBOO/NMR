
import torch
import torch.nn as nn
from src.models.model.encoder import Encoder

class ConditionGT(nn.Module):

    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers_TE, drop_prob, device):
        super().__init__()


        self.transEn = Encoder(enc_voc_size=enc_voc_size, max_len=max_len, d_model=d_model, ffn_hidden=ffn_hidden, n_head=n_head, n_layers=n_layers_TE, drop_prob=drop_prob, device=device)
        self.linear_layer = nn.Linear(max_len * d_model, 512)
        self.device = device

        checkpoint = torch.load('/public/home/ustc_yangqs/molecular2molecular/src/epoch=23.ckpt')
        # 获取模型的 state_dict
        state_dict = checkpoint['state_dict']
        # 从 state_dict 中提取 conditionEn 部分的权重
        linear_layer_state_dict = {k[len('model.linear_layer.'):]: v for k, v in state_dict.items() if
                                  k.startswith('model.linear_layer.')}
        # 加载到模型的 conditionEn 部分
        self.linear_layer.load_state_dict(linear_layer_state_dict)

        checkpoint = torch.load('/public/home/ustc_yangqs/molecular2molecular/src/epoch=23.ckpt')
        # 获取模型的 state_dict
        state_dict = checkpoint['state_dict']
        # 从 state_dict 中提取 conditionEn 部分的权重
        transEn_state_dict = {k[len('model.transEn.'):]: v for k, v in state_dict.items() if
                              k.startswith('model.transEn.')}
        # 加载到模型的 conditionEn 部分
        self.transEn.load_state_dict(transEn_state_dict)
        #
        for param in self.transEn.parameters():
            param.requires_grad = False
        for param in self.linear_layer.parameters():
            param.requires_grad = False

    def make_src_mask(self, src):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        return src_mask

    def forward(self, conditionVec):
        assert isinstance(conditionVec, torch.Tensor), "conditionVec should be a tensor, but got type {}".format(
            type(conditionVec))

        srcMask = self.make_src_mask(conditionVec).to(self.device)
        conditionVec = self.transEn(conditionVec, srcMask)
        conditionVec = conditionVec.view(conditionVec.size(0), -1)
        conditionVec = self.linear_layer(conditionVec)
        return conditionVec

def numericalize_text(text, vocab_to_id, dim):
    """
    将给定的文本转换为对应的 token id 。

    参数:
        text (str): 输入文本为一个字符串，单词以空格分隔。
        vocab_to_id (dict): 词汇表字典，将单词映射为唯一的 id。
        dim (int): 返回的每个 token id 列表的长度。

    返回:
        list of list: 对应的数值化 token id ，文本对应一个长度为 dim 的 token id 。
    """

    # 如果输入文本为空，则返回长度为 dim 且全为 0 的列表
    if not text:
        token_ids = [0] * dim
    else:
        # 将文本按空格进行分割，生成单词列表
        words = text.split(" ")

        # 使用词汇表字典将每个单词转换为对应的 id，如果不在词汇表中则使用 <unk> 的 id
        token_ids = [vocab_to_id.get(word, vocab_to_id["<unk>"]) for word in words]

        # 如果 token_ids 长度小于 dim，则在后面补充 0
        if len(token_ids) < dim:
            token_ids += [0] * (dim - len(token_ids))
        # 如果 token_ids 长度超过 dim，则截断到 dim 长度
        else:
            token_ids = token_ids[:dim]

    return token_ids

enc_voc_size = 5450
max_len = 256
d_model = 256
ffn_hidden = 1024
n_head = 8
n_layers_TE = 3
drop_prob = 0.
device = torch.device("cpu")

model = ConditionGT(enc_voc_size=enc_voc_size,
                         max_len=max_len,
                         d_model=d_model,
                         ffn_hidden=ffn_hidden,
                         n_head=n_head,
                         n_layers_TE=n_layers_TE,
                         drop_prob=drop_prob,
                         device=device)

# 确保模型处于评估模式（evaluation mode）
model.eval()

# 随机生成一个 conditionVec，用作输入。假设它的大小为 (batch_size, max_len)
batch_size = 1
vocab_to_id = {
            "<blank>": 0,
            "<unk>": 1
        }

with open("/public/home/ustc_yangqs/molecular2molecular/src/vocab.src", "r", encoding="utf-8") as vocab_file:
    # 从编号 2 开始，因为 0 和 1 分配给了特殊的 token
    current_id = 2
    for line in vocab_file:
        # 每行分割为单词和它的频次
        word, _ = line.strip().split("\t")
        vocab_to_id[word] = current_id
        current_id += 1

nmr_smiles = 'Cc1csc(CBr)c1'
tokenized_input = '1HNMR 6.80 6.77 m 1H | 6.66 6.63 q 1H J 1.02 | 4.65 4.62 d 2H J 1.01 | 2.26 2.23 d 3H J 0.87 13CNMR 139.6 137.8 125.9 122.4 28.3 15.5'
conditionVec = torch.tensor(numericalize_text(text=tokenized_input, vocab_to_id=vocab_to_id, dim=256), dtype=torch.long)

# 增加一个 batch 维度，变为 [1, max_len]
conditionVec = conditionVec.unsqueeze(0)  # 形状变为 [1, max_len]

# 如果模型在 GPU 上运行，需要将 conditionVec 转移到相同的设备
conditionVec = conditionVec.to(model.device)

# 禁用梯度计算
with torch.no_grad():
    # 将 conditionVec 输入模型
    output = model(conditionVec)
#
print("Nmr smiles:", nmr_smiles)
# print("Output:", output)

import pandas as pd
import torch
import numpy as np
from torch.nn.functional import cosine_similarity

# 加载 CSV 文件
csv_path = "/public/home/ustc_yangqs/molecular2molecular/src/mol_rep.csv"  # 替换为你的 CSV 文件路径
data = pd.read_csv(csv_path)

# 假设分子向量存储在 "molecularVec" 列中，SMILES 存储在 "smiles" 列中
# 将 molecularVec 转换为 NumPy 数组
data["molecularRep"] = data["molecularRep"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))

# 转换为 PyTorch 张量
molecular_vectors = torch.tensor(np.stack(data["molecularRep"].values), dtype=torch.float32)
smiles_list = data["smiles"].tolist()

# 确保模型的输出向量也是一个 PyTorch 张量
output = output.squeeze(0)  # 模型输出维度 [1, 512] -> [512]

# 计算余弦相似度
similarities = cosine_similarity(output.unsqueeze(0), molecular_vectors).squeeze(0)  # [1, N] -> [N]

# 获取相似度最高的三个分子的索引
top_k = 3
top_k_indices = torch.topk(similarities, k=top_k).indices

# 打印对应的 SMILES 和相似度
print("Top 3 most similar molecules:")
for idx in top_k_indices:
    print(f"SMILES: {smiles_list[idx]}, Similarity: {similarities[idx].item():.4f}")