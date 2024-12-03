import os
import re
import tqdm
import hashlib
import multiprocessing

vocab = []

with open("./dataset/vocab.txt", "r", encoding="utf-8") as f:
    vocab = f.readlines()

# 注意，模型不会将语法作用的大括号作为token.

token_lut = {}
pattern = r"[ \n]"

filtered_vocab = []
for index, token in enumerate(vocab):
    token = re.sub(pattern, "", token)
    filtered_vocab.append(token)
    token_lut[token] = index
vocab = filtered_vocab


def FMM_func(vocab_list: list[str], sentence: str):
    """
    正向最大匹配（FMM）
    :param user_dict: 词典
    :param sentence: 句子
    """
    # 词典中最长词长度
    max_len = len(max(vocab_list, key=len))
    start = 0
    token_list = []
    while start != len(sentence):
        index = min(start + max_len, len(sentence))
        for _ in range(max_len):
            token = sentence[start:index]
            if (token in vocab_list) or (len(token) == 1):
                token_list.append(token)
                start = index
                break
            index -= 1
    return token_list


rootdir = "dataset/labels"
files = os.listdir(rootdir)


mod_prime = 9817130993
mul_prime = 9547


def findTokenHashWorker(filename: str) -> tuple[list, int]:
    filePath = rootdir + "/" + filename
    with open(filePath, "r", encoding="UTF-8") as f:
        label_string = " ".join(f.readlines())

        token_string = FMM_func(vocab, label_string)
        ret = []
        key = 0
        for token in token_string:
            if token in token_lut:
                num = token_lut[token]
                key = (key * mul_prime + num) % mod_prime

                ret.append(token)

    return (ret, key)


stat_dict = {}
key2list = {}
pool = multiprocessing.Pool(6)


keys = []
for e in tqdm.tqdm(pool.imap_unordered(findTokenHashWorker, files)):
    keys.append(e)

for raw_list, key in tqdm.tqdm(keys):
    if key not in stat_dict:
        stat_dict[key] = 1
        key2list[key] = raw_list
    else:
        stat_dict[key] += 1


# 按 stat_dict 的值降序排序
sorted_stat = sorted(stat_dict.items(), key=lambda x: x[1], reverse=True)

# 打印排序后的结果
for key, value in sorted_stat:
    print(f"{value} Occurance | {" ".join(key2list[key])}")
