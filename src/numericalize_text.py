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


