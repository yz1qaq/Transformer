# 导入open方法
from io import open
# 用于字符规范化
import unicodedata
# 导入正则表达式的包
import re
# 导入随机处理数据的包
import random
# 导入torch相关的包
import torch
import torch.nn as nn
import torch.nn.functional as F
# 导入优化方法的工具包
from torch import optim
import time
import math
import matplotlib.pyplot as plt

# 设备的选择, 可以选择在GPU上运行或者在CPU上运行
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义起始标志
SOS_token = 0
# 定义结束标志
EOS_token = 1

PAD_token = 2

# 明确一下数据文件的存放地址
data_path = 'data/eng-fra.txt'


class Lang():
    def __init__(self, name):
        # name: 参数代表传入某种语言的名字
        self.name = name
        # 初始化单词到索引的映射字典
        self.word2index = {}
        # 初始化索引到单词的映射字典, 其中0, 1对应的SOS, EOS已经在字典中了
        self.index2word = {0: "SOS", 1: "EOS", 2: "PAD"}
        # 初始化词汇对应的数字索引, 从2开始, 因为0, 1已经被开始字符和结束字符占用了
        self.n_words = 3

    def addSentence(self, sentence):
        # 添加句子的函数, 将整个句子中所有的单词依次添加到字典中
        # 因为英文, 法文都是空格进行分割的语言, 直接进行分词就可以
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        # 添加单词到类内字典中, 将单词转换为数字
        # 首先判断word是否已经在self.word2index字典的key中
        if word not in self.word2index:
            # 添加的时候, 索引值取当前类中单词的总数量
            self.word2index[word] = self.n_words
            # 再添加翻转的字典
            self.index2word[self.n_words] = word
            # 第三步更新类内的单词总数量
            self.n_words += 1


# 将unicode字符串转换为ASCII字符串, 主要用于将法文的重音符号去除掉
def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


# 定义字符串规范化函数
def normalizeString(s):
    # 第一步使字符转变为小写并去除掉两侧的空白符, 再调用上面的函数转换为ASCII字符串
    s = unicodeToAscii(s.lower().strip())
    # 在.!?前面加一个空格
    s = re.sub(r"([.!?])", r" \1", s)
    # 使用正则表达式将字符串中不是大小写字符和正常标点符号的全部替换成空格
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


# 读取原始数据并实例化源语言+目标语言的类对象
def readLangs(lang1, lang2):
    # lang1: 代表源语言的名字
    # lang2: 代表目标语言的名字
    # 整个函数返回对应的两个类对象, 以及语言对的列表
    lines = open(data_path, encoding='utf-8').read().strip().split('\n')
    # 对lines列表中的句子进行标准化处理, 并以\t进行再次划分, 形成子列表
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    # 直接初始化两个类对象
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)
    return input_lang, output_lang, pairs


# 设置组成句子中单词或标点的最多个数
MAX_LENGTH = 10

# 选择带有指定前缀的英文源语言的语句数据作为训练数据
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


# 过滤语言对的具体逻辑函数
def filterPair(pair):
    # 当前传入的pair是一个语言对的形式
    # pair[0]代表英文源语句, 长度应小于MAX_LENGTH， 并且以指定前缀开始
    # pair[1]代表法文源语句, 长度应小于MAX_LENGTH
    return len(pair[0].split(' ')) < MAX_LENGTH and \
           pair[0].startswith(eng_prefixes) and \
           len(pair[1].split(' ')) < MAX_LENGTH


# 过滤语言对的函数
def filterPairs(pairs):
    # 函数直接遍历列表中的每个语言字符串并调用filterPair()函数即可
    return [pair for pair in pairs if filterPair(pair)]


# 整合数据预处理的函数
def prepareData(lang1, lang2):
    # lang1: 代表源语言的名字, 英文
    # lang2: 代表目标语言的名字, 法文
    # 第一步通过调用readLangs()函数得到两个类对象, 并得到字符串类型的语言对的列表
    input_lang, output_lang, pairs = readLangs(lang1, lang2)
    # 第二步对字符串类型的列表进行过滤操作
    pairs = filterPairs(pairs)
    # 对过滤后的语言对列表进行遍历操作, 添加进类对象中
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    # 返回数值映射后的类对象, 以及过滤后的语言对列表
    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData('eng', 'fra')


def tensorFromSentence(lang, sentence):
    # lang: 代表是Lang类的实例化对象
    # sentence: 代表传入的语句
    indexes = [lang.word2index[word] for word in sentence.split(' ')]

    # 使用torch.tensor对列表进行封装, 并将其形状改变成 n*1
    return torch.tensor(indexes, dtype=torch.long, device=device).unsqueeze(0)


def tensorsFromPair(pair):
    # pair: 一个语言对 (英文, 法文)
    intput_indexes = [input_lang.word2index[word] for word in pair[0].split(' ')]+[EOS_token]

    output_indexes = [SOS_token]+[output_lang.word2index[word] for word in pair[1].split(' ')]+[EOS_token]

    # 转为tensor，shape: (1, seq_len)
    input_tensor = torch.tensor(intput_indexes, dtype=torch.long, device=device).unsqueeze(0)
    output_tensor = torch.tensor(output_indexes, dtype=torch.long, device=device).unsqueeze(0)

    return (input_tensor, output_tensor)



# 构建时间计算的辅助函数
def timeSince(since):
    # since: 代表模型训练的开始时间
    # 首先获取当前时间
    now = time.time()
    # 计算得到时间差
    s = now - since
    # 将s转换为分钟, 秒的形式
    m = math.floor(s / 60)
    # 计算余数的秒
    s -= m * 60
    # 按照指定的格式返回时间差
    return '%dm %ds' % (m, s)






class BatchLoader:
    def __init__(self, pairs, batch_size):
        self.pairs = pairs
        self.batch_size = batch_size
        self.pad_token = PAD_token
        self.index = 0

    def __len__(self):
        return (len(self.pairs) + self.batch_size - 1) // self.batch_size

    def pad_sequence_before_eos(self, seq, max_len):
        assert EOS_token in seq, "序列中必须包含EOS_token"
        eos_pos = seq.index(EOS_token)
        seq_len = len(seq)
        pad_len = max_len - seq_len
        assert pad_len >= 0, f"目标长度小于序列长度，pad_len={pad_len}"
        assert eos_pos == seq_len - 1, "EOS_token必须是序列最后一个token"
        return seq[:eos_pos] + [self.pad_token] * pad_len + [EOS_token]

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self.pairs):
            raise StopIteration

        batch_pairs = self.pairs[self.index:self.index + self.batch_size]
        self.index += self.batch_size

        input_seqs = []
        output_seqs = []

        for pair in batch_pairs:
            input_idx = [input_lang.word2index[word] for word in pair[0].split(' ')] + [EOS_token]
            output_idx = [SOS_token] + [output_lang.word2index[word] for word in pair[1].split(' ')] + [EOS_token]
            input_seqs.append(input_idx)
            output_seqs.append(output_idx)

        max_input_len = max(len(seq) for seq in input_seqs)
        max_output_len = max(len(seq) for seq in output_seqs)

        input_padded = [self.pad_sequence_before_eos(seq, max_input_len) for seq in input_seqs]
        output_padded = [self.pad_sequence_before_eos(seq, max_output_len) for seq in output_seqs]

        input_tensor = torch.tensor(input_padded, dtype=torch.long, device=device)
        output_tensor = torch.tensor(output_padded, dtype=torch.long, device=device)

        return input_tensor, output_tensor



import torch.optim as optim
from transformer import make_model, create_padding_mask, subsequent_mask, PAD_token


def train_model(model, loader, num_epochs, lr=1e-4, print_every=100):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.NLLLoss(ignore_index=PAD_token)

    step = 0
    for epoch in range(1, num_epochs + 1):
        total_loss = 0
        for batch_idx, (src, tgt) in enumerate(loader):

            # 构造输入
            src_mask = create_padding_mask(src, PAD_token).to(src.device)  # (batch, 1, 1, src_len)

            tgt_input = tgt[:, :-1]  # 去掉最后一个 EOS
            tgt_y = tgt[:, 1:]       # 去掉第一个 SOS，作为目标
            tgt_mask = create_padding_mask(tgt_input, PAD_token).to(tgt.device) & subsequent_mask(tgt_input.size(1)).to(tgt.device)

            # 前向传播
            preds = model(src, tgt_input, src_mask, tgt_mask)

            # 预测输出维度: (batch, tgt_len, vocab_size) → reshape 成 (batch*tgt_len, vocab_size)
            preds_flat = preds.reshape(-1, preds.size(-1))
            tgt_y_flat = tgt_y.reshape(-1)

            loss = criterion(preds_flat, tgt_y_flat)

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            step += 1

            if step % print_every == 0:
                avg_loss = total_loss / print_every
                print(f"Epoch {epoch} | Step {step} | Loss: {avg_loss:.4f}")
                total_loss = 0
    torch.save(model.state_dict(), 'model.pth')


def prepare_sentence(sentence, lang):
    # 按训练时的流程做分词和索引映射，添加EOS
    indexes = [lang.word2index[word] for word in sentence.split(' ')]
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).unsqueeze(0)  # (1, seq_len)


def translate_sentence(model, sentence, input_lang, output_lang, max_len=20):
    model.eval()
    src = prepare_sentence(sentence, input_lang)
    src_mask = create_padding_mask(src, PAD_token)

    # 解码器输入，起始是SOS_token
    tgt_indices = [SOS_token]
    for i in range(max_len):
        tgt_input = torch.tensor(tgt_indices, dtype=torch.long, device=device).unsqueeze(0)  # (1, len)
        tgt_mask = create_padding_mask(tgt_input, PAD_token) & subsequent_mask(tgt_input.size(1)).to(device)

        out = model(src, tgt_input, src_mask, tgt_mask)  # (1, seq_len, vocab_size)
        prob = out[:, -1, :]  # 取最后一步的预测 (1, vocab_size)
        next_word = torch.argmax(prob, dim=-1).item()  # 贪心选择概率最高的词

        tgt_indices.append(next_word)
        if next_word == EOS_token:
            break

    # 将索引转回单词
    output_words = [output_lang.index2word[idx] for idx in tgt_indices[1:]]  # 去掉起始符SOS
    #去掉EOS
    output_words = output_words[:-1]
    return ' '.join(output_words)



if __name__ == '__main__':
    # 构建模型
    # model = make_model(
    #     source_vocab=input_lang.n_words,
    #     target_vocab=output_lang.n_words,
    #     N=6,
    #     d_model=512,
    #     d_ff=2048,
    #     head=8,
    #     dropout_rate=0.1
    # ).to(device)

    # 构建 loader
    loader = BatchLoader(pairs, batch_size=64)

    # 开始训练
    #train_model(model, loader, num_epochs=15, lr=1e-4, print_every=50)

    model = make_model(source_vocab=input_lang.n_words, target_vocab=output_lang.n_words)
    model.load_state_dict(torch.load('model.pth', map_location=device))
    model.to(device)
    model.eval()  # 切换到评估模式

    for i in range(5):
        # 随机选择一个英文句子
        pair = random.choice(pairs)
        eng_sentence = pair[0]
        fra_sentence = pair[1]
         # 翻译成法文
        fra_sentence = translate_sentence(model, eng_sentence, input_lang, output_lang)
        print(f"英文:{eng_sentence},实际翻译:{fra_sentence},预测翻译:{fra_sentence}")
        print("--------------------------------")




# from transformer import *
# loader = BatchLoader(pairs, batch_size=64)
#
# src, tgt  = next(iter(loader))
# d_model = 512
# src_vocab = input_lang.n_words
# tgt_vocab = output_lang.n_words
# print("src_vocab:", src_vocab)
# print("tgt_vocab:", tgt_vocab)
# tgt_y = tgt[:, 1:].reshape(-1)
# print(tgt_y)
# src_embed = nn.Sequential(Embeddings(src_vocab, d_model), PositionalEncoding(d_model)).to(device)
# tgt_embed = nn.Sequential(Embeddings(tgt_vocab, d_model), PositionalEncoding(d_model)).to(device)
#
# # 多头注意力 & 前馈层
# self_attn = MultiHeadAttention(head_num=8, d_model=d_model).to(device)
# src_attn = MultiHeadAttention(head_num=8, d_model=d_model).to(device)
# ff = PositionwiseFeedForward(d_model, 2048).to(device)
#
# # 子层
# enc_layer = EncoderLayer(d_model, copy.deepcopy(self_attn), copy.deepcopy(ff)).to(device)
# dec_layer = DecoderLayer(d_model, copy.deepcopy(self_attn), copy.deepcopy(src_attn), copy.deepcopy(ff), dropout_rate=0.1).to(device)
#
# # 输出层
# generator = Generator(d_model, tgt_vocab).to(device)
#
# src_mask = create_padding_mask(src, pad_token_id=PAD_token)  # shape: (batch, 1, 1, src_len)
#
# # 词嵌入 + 位置编码
# src_emb = src_embed(src)  # (batch, src_len, d_model)
# print("src_emb.shape:", src_emb.shape)
#
# #传入三层编码器
# enc_output1 = enc_layer(src_emb, src_mask)
# enc_output2 = enc_layer(enc_output1, src_mask)
# enc_output3 = enc_layer(enc_output2, src_mask)
# enc_output = enc_output3
# print("enc_output.shape:", enc_output.shape)
#
#
# tgt_input = tgt[:, :-1]  # 去除最后 EOS，作为 decoder 输入
# tgt_y = tgt[:, 1:]       # 去除第一个 SOS，作为目标
#
# tgt_mask = create_padding_mask(tgt_input, PAD_token) & subsequent_mask(tgt_input.size(1)).to(device)
#
# tgt_emb = tgt_embed(tgt_input)  # (batch, tgt_len-1, d_model)
# print("tgt_emb.shape:", tgt_emb.shape)
#
# dec_output1 = dec_layer(tgt_emb, enc_output, src_mask, tgt_mask)
# dec_output2 = dec_layer(dec_output1, enc_output, src_mask, tgt_mask)
# dec_output3 = dec_layer(dec_output2, enc_output, src_mask, tgt_mask)
# dec_output = dec_layer(dec_output3, enc_output, src_mask, tgt_mask)
# print("dec_output.shape:", dec_output.shape)
#
# logits = generator(dec_output)  # (batch, tgt_len-1, tgt_vocab)
# print("logits.shape:", logits.shape)

# predicted = logits.argmax(-1)
# print("predicted:", predicted)
# print("predicted.shape:",predicted.shape)
