'''将原始数据中谱的序列变换成字典格式，同时将1HNMR中化学位移的最大值和最小值变成均值和差值'''
import json
import pandas as pd

def parse_nmr(nmr_str):
    # 初始化结果字典
    result = {"1HNMR": [], "13CNMR": []}
    # hnmr = []
    # cnmr = []

    # 分割1H NMR和13C NMR部分
    parts = nmr_str.strip().split("13CNMR")
    if len(parts) < 2:
        print('***')# 处理缺少13C NMR的情况
        # return result

    # 解析1H NMR部分
    h1_part = parts[0].replace("1HNMR", "").strip()
    for peak in h1_part.split("|"):
        peak = peak.strip()
        if not peak:
            continue

        # 分割每个峰的组成部分
        tokens = [t for t in peak.split() if t]
        j_index = None

        try:
            # 查找耦合常数位置
            j_index = tokens.index("J")
            coupling = list(map(float, tokens[j_index + 1:]))
        except ValueError:
            coupling = []
            j_index = len(tokens)

        # 提取主要信息部分
        main_part = tokens[:j_index]
        if len(main_part) < 4:  # 无效数据
            continue

        try:
            # 解析化学位移、裂分模式、积分值
            a, b = map(float, main_part[:2])  # 解包两个值
            mean_value = round((a + b) / 2, 2)  # 计算均值并保留2位小数
            delta = round(abs(a - b), 2)  # 计算差值绝对值并保留2位小数
            chem_shift = [mean_value, delta]

            split_type = main_part[2]
            integral = main_part[3]
            result['1HNMR'].append([*chem_shift, split_type, integral, coupling])
        except:
            continue

    # 解析13C NMR部分
    c13_part = parts[1].strip()
    if c13_part:
        try:
            result['13CNMR'] = list(map(float, c13_part.split()))
        except:
            pass

    return result


# 示例用法
if __name__ == "__main__":
    # 读取CSV文件
    df = pd.read_csv('../raw/tokenized_dataset_N.csv')

    # 应用解析函数
    df["tokenized_input"] = df["tokenized_input"].apply(parse_nmr)
    df["tokenized_input"] = df["tokenized_input"].apply(json.dumps)
    # 查看结果
    # print(df["tokenized_input"].iloc[0])  # 打印第一条解析结果

    df.to_csv("tokenized_dataset_N_new.csv", index=False)
    # df = pd.read_csv('../raw/tokenized_dataset_N.csv')
    # df_new = pd.DataFrame(columns=['smiles','1HNMR','13CNMR', 'atom_count'])
    #
    # nmr_results = df["tokenized_input"].apply(parse_nmr)
    #
    #
    # hnmr, cnmr = list(zip(*nmr_results))
    # df_new['1HNMR'] = hnmr
    # df_new['13CNMR'] = cnmr
    #
    # df_new['smiles'] = df['smiles']
    # df_new['atom_count'] = df['atom_count']
    #
    # df_new.to_csv("tokenized_dataset_N_new.csv", index=False)
    #
    # ddd = pd.read_csv('tokenized_dataset_N_new.csv')['1HNMR'][2]
    # print()
