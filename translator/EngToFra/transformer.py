import torch
import torch.nn as nn
import math
from torch.nn import functional as F
import copy

PAD_token = 2
class Embeddings(nn.Module):
    def __init__(self,vocab,d_model):
        super(Embeddings,self).__init__()
        self.lut = nn.Embedding(vocab,d_model,padding_idx=PAD_token)
        self.d_model = d_model
    def forward(self,x):
        return self.lut(x)*math.sqrt(self.d_model) #和位置编码相加，为了不要太大过位置编码，进行缩放


class PositionalEncoding(nn.Module):  #一个字对应一行向量
    def __init__(self,d_model,dropout_rate=0.2,max_len=5000):
        super(PositionalEncoding,self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        pe = torch.zeros(max_len,d_model) #位置编码矩阵
        position = torch.arange(0,max_len).unsqueeze(1) #生成一个shape=(max_len,1)的矩阵
        div_term = torch.exp(torch.arange(0,d_model,2)*-(math.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(position*div_term) #对偶数列用sin填充
        pe[:,1::2]=torch.cos(position*div_term) #对奇数列用cos填充
        pe = pe.unsqueeze(0)  #shape = (1,max_len,d_model) 为了个词嵌入后的向量相加，维度要相同
        self.register_buffer('pe',pe)  #这个位置矩阵是固定的，注册一个
    def forward(self,x):
        x = x+self.pe[:,:x.size(1)]  #x.shape=(bs,seq_len,d_model) 只需要加到对应列即可
        return self.dropout(x)



def subsequent_mask(size):
    return torch.triu(torch.ones((1, size, size), dtype=torch.bool), diagonal=1) == 0


def create_padding_mask(x, pad_token_id):
    mask = (x != pad_token_id).unsqueeze(1).unsqueeze(2)
    return mask


#注意力计算规则
def attention(q,k,v,mask=None,dropout=None):
    d_k = q.size(-1)
    scores=torch.matmul(q,k.transpose(-2,-1))/math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask==0,-1e9)
    p_attn = F.softmax(scores,dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn,v),p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self,head_num,d_model,dropout_rate=0.2):
        super(MultiHeadAttention,self).__init__()
        assert d_model % head_num ==0
        self.d_k = d_model //head_num  #每个头的维度
        self.head_num = head_num
        self.W_q = nn.Linear(d_model, d_model) #4个线形层，对应了四个参数矩阵，因为输入的x*W_Q矩阵，变为Q，其它同理
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)
    def forward(self,q,k,v,mask=None):
        if mask is not None:
            mask = mask.unsqueeze(0)  #因为下面我们Q,K,V矩阵维度变化了，在attention中，为了匹配维度
        batch_size = q.size(0)
        Q = self.W_q(q)  #得到输入矩阵对应的Q,K,V矩阵，也就是每个字的q拼起来的矩阵 (batch, seq_len, embed_dim)
        K = self.W_k(k) #同Q
        V = self.W_v(v)
        #view:在原来数据存储的顺序上，变为4维,transpose:会改变原来第二维和第三维的数据存储顺序
        Q = Q.view(batch_size, -1, self.head_num, self.d_k).transpose(1,2).contiguous() #(batch_size, head, seq_len, d_k)，为了多个头并行计算
        K = K.view(batch_size, -1, self.head_num, self.d_k).transpose(1,2).contiguous()
        V = V.view(batch_size, -1, self.head_num, self.d_k).transpose(1,2).contiguous()
        x,self.attn = attention(Q,K,V,mask=mask,dropout=self.dropout) #这里的Q,K,V其实就是多个q，k，v进行并行计算
        x = x.transpose(1,2).contiguous().view(batch_size,-1,self.head_num * self.d_k) #把多个头拼接起来
        result = self.W_o(x) #再经过一个线性变换
        return result

#规范化层
class LayerNorm(nn.Module):
    def __init__(self,d_model,eps=1e-6):
        super(LayerNorm,self).__init__()
        #a2,b2是模型可以更新的参数
        self.a2 = nn.Parameter(torch.ones(d_model))
        self.b2 = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    def forward(self,x):
        mean = x.mean(-1,keepdim=True)
        std = x.std(-1,keepdim=True)
        return self.a2 *(x-mean)/(std+self.eps)+self.b2  #分母会加一个特别小的数防止分母为0


#前馈全连接层ADD
class PositionwiseFeedForward(nn.Module):
    def __init__(self,d_model,linaer_dim,dropout_rate=0.2):
        super(PositionwiseFeedForward,self).__init__()
        self.Linear1 = nn.Linear(d_model,linaer_dim)
        self.Linear2 =nn.Linear(linaer_dim,d_model)
        self.dropout=nn.Dropout(p=dropout_rate)
    def forward(self,x):
        x = F.relu(self.Linear1(x))
        x = self.dropout(x)
        x = self.Linear2(x)
        return x

#实现子层连接的类
class SublayerConnection(nn.Module):
    def __init__(self,d_model,dropout_rate=0.2):
        super(SublayerConnection,self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout_rate)
    def forward(self,x,sublayer):
        out1 = self.norm(x)
        out2 = sublayer(out1)
        result = x+self.dropout(out2)
        return result

class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout_rate=0.2):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model

        self.self_attn = self_attn  # 注意力层
        self.feed_forward = feed_forward  # 前馈全连接层
        self.sublayer1 = SublayerConnection(d_model, dropout_rate)  # 用于连接第一个子层
        self.sublayer2 = SublayerConnection(d_model, dropout_rate)  # 用于连接第二个子层

    def forward(self, x, mask=None):
        # 连接注意力层和norm层
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, mask))
        # 连接前馈全连接层和norm层
        x = self.sublayer2(x, self.feed_forward)
        return x


def clones(module, N):
    # module: 代表要克隆的目标网络层
    # N: 将module克隆几个
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    def __init__(self,layer,N):
        super(Encoder,self).__init__()
        self.layers = clones(layer,N)
        self.norm = LayerNorm(layer.d_model)
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout_rate):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model

        self.self_attn = self_attn  # 注意力层
        self.src_attn = src_attn  # 跨注意力
        self.feed_forward = feed_forward  # 前馈全连接层
        self.sublayers = clones(SublayerConnection(d_model, dropout_rate), 3)  # 用于连接第一个子层

    def forward(self, x, memory, source_mask, target_mask):
        # 来自编码器
        m = memory
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, target_mask))
        x = self.sublayers[1](x, lambda x: self.src_attn(x, m, m, source_mask))
        x = self.sublayers[2](x, self.feed_forward)
        return x

class Decoder(nn.Module):
    def __init__(self,layer,N):
        super(Decoder,self).__init__()
        self.layers = clones(layer,N)
        self.norm = LayerNorm(layer.d_model)
    def forward(self, x,memory,source_mask,target_mask):
        for layer in self.layers:
            x = layer(x, memory,source_mask,target_mask)
        return self.norm(x)

class Generator(nn.Module):
    def __init__(self,d_model,vocab_size):
        super(Generator,self).__init__()
        self.Linear = nn.Linear(d_model,vocab_size)
    def forward(self,x):
        result = F.log_softmax(self.Linear(x),dim=-1)
        return result


# 构建编码器-解码器结构类
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, source, target, source_mask, target_mask):
        encode_result = self.encode(source, source_mask)
        decode_result = self.decode(encode_result, source_mask, target, target_mask)
        return self.generator(decode_result)

    def encode(self, source, source_mask):
        embed_result = self.src_embed(source)
        encoder_result = self.encoder(embed_result, source_mask)
        return encoder_result

    def decode(self, memory, source_mask, target, target_mask):
        # memory: 代表经历编码器编码后的输出张量
        embed_result = self.tgt_embed(target)
        decoder_result = self.decoder(embed_result, memory, source_mask, target_mask)
        return decoder_result


def make_model(source_vocab, target_vocab, N=6, d_model=512, d_ff=2048, head=8, dropout_rate=0.1):
    c = copy.deepcopy
    attn = MultiHeadAttention(head, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout_rate)
    position = PositionalEncoding(d_model, dropout_rate)

    encoderLayer = EncoderLayer(d_model, c(attn), c(ff), dropout_rate)
    decoderLayer = DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout_rate)
    encoder = Encoder(encoderLayer, N)
    decoder = Decoder(decoderLayer, N)
    src_embed = nn.Sequential(Embeddings(source_vocab, d_model), c(position))
    tgt_embed = nn.Sequential(Embeddings(target_vocab, d_model), c(position))
    gen = Generator(d_model, target_vocab)

    model = EncoderDecoder(encoder, decoder, src_embed, tgt_embed, gen)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model






