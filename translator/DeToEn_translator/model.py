from transformers import BertTokenizer,BertModel
import torch

import torch.nn as nn
import math
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





#实现解码器
# ==================== 第一步：位置编码 ====================
class PositionalEncoding(nn.Module):
    """位置编码层"""

    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        return x + self.pe[:x.size(0), :]


# ==================== 第二步：多头注意力机制 ====================
class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Q, K, V: [batch_size, n_heads, seq_len, d_k]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)

        return output, attention_weights

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 线性变换并分割为多头
        # query: [batch_size, seq_len, d_model]
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 计算注意力
        attention, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # 合并多头
        attention = attention.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)

        output = self.w_o(attention)
        return output

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 线性变换并分割为多头
        # query: [batch_size, seq_len, d_model]
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 计算注意力
        attention, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # 合并多头
        attention = attention.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)

        output = self.w_o(attention)
        return output


# ==================== 第三步：前馈神经网络 ====================
class PositionwiseFeedForward(nn.Module):
    """位置前馈神经网络"""

    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class DecoderLayer(nn.Module):
    """单个解码器层"""

    def __init__(self, d_model, n_heads, d_ff):
        super(DecoderLayer, self).__init__()

        # 三个子层
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.cross_attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)

        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x, encoder_output, self_attn_mask=None, cross_attn_mask=None):
        """
        x: 解码器输入 [batch_size, tgt_len, d_model]
        encoder_output: 编码器输出 [batch_size, src_len, d_model]
        self_attn_mask: 自注意力掩码 (因果掩码)
        cross_attn_mask: 交叉注意力掩码 (padding掩码)
        """

        # 第一个子层：带掩码的自注意力
        self_attn_output = self.self_attention(x, x, x, self_attn_mask)
        x = self.norm1(x + self.dropout(self_attn_output))

        # 第二个子层：编码器-解码器注意力
        cross_attn_output = self.cross_attention(x, encoder_output, encoder_output, cross_attn_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))

        # 第三个子层：前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x


# ==================== 第五步：完整的解码器 ====================
class TransformerDecoder(nn.Module):
    """完整的Transformer解码器"""

    def __init__(self, vocab_size, d_model=768, n_heads=8, n_layers=6, d_ff=2048, max_len=512):
        super(TransformerDecoder, self).__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # 堆叠的解码器层
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])

        # 输出投影层
        self.linear = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(0.2)

    def create_causal_mask(self, size):
        """创建因果掩码，防止解码器看到未来的token"""
        mask = torch.tril(torch.ones(size, size, dtype=torch.bool))
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, size, size]

    def create_padding_mask(self, x, pad_idx=0):
        """创建padding掩码"""
        return (x != pad_idx).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]

    def forward(self, tgt, encoder_output, encoder_mask=None, tgt_mask=None):
        """
        tgt: 目标序列 [batch_size, tgt_len]
        encoder_output: 编码器输出 [batch_size, src_len, d_model]
        encoder_mask: 编码器掩码 [batch_size, src_len]
        tgt_mask: 目标序列掩码 [batch_size, tgt_len]
        """
        batch_size, tgt_len = tgt.size()

        # 词嵌入 + 位置编码
        x = self.embedding(tgt) * math.sqrt(self.d_model)  # [batch_size, tgt_len, d_model]
        x = x.transpose(0, 1)  # [tgt_len, batch_size, d_model] for positional encoding
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # 转回 [batch_size, tgt_len, d_model]
        x = self.dropout(x)

        # 创建掩码
        # 1. 因果掩码 (防止看到未来)
        causal_mask = self.create_causal_mask(tgt_len).to(tgt.device)

        # 2. Padding掩码 (目标序列)
        if tgt_mask is None:
            tgt_padding_mask = self.create_padding_mask(tgt)
        else:
            tgt_padding_mask = tgt_mask.unsqueeze(1).unsqueeze(2)

        # 3. 组合自注意力掩码
        self_attn_mask = causal_mask & tgt_padding_mask

        # 4. 交叉注意力掩码 (编码器padding)
        if encoder_mask is not None:
            cross_attn_mask = encoder_mask.unsqueeze(1).unsqueeze(2)
        else:
            cross_attn_mask = None

        # 通过解码器层
        for layer in self.layers:
            x = layer(x, encoder_output, self_attn_mask, cross_attn_mask)

        # 输出投影
        output = self.linear(x)  # [batch_size, tgt_len, vocab_size]

        return output


# ==================== 第六步：完整的翻译模型 ====================
class German2EnglishTranslator(nn.Module):
    """德语到英语的翻译模型"""

    def __init__(self, encoder_model, en_vocab_size, d_model=768, n_heads=8, n_layers=6, d_ff=2048):
        super(German2EnglishTranslator, self).__init__()

        # 使用预训练的BERT作为编码器
        self.encoder = encoder_model

        # 自定义解码器
        self.decoder = TransformerDecoder(
            vocab_size=en_vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff
        )

        # 冻结编码器参数 (可选)
        # for param in self.encoder.parameters():
        #     param.requires_grad = False

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        src: 德语输入 [batch_size, src_len]
        tgt: 英语目标 [batch_size, tgt_len]
        src_mask: 德语掩码 [batch_size, src_len]
        tgt_mask: 英语掩码 [batch_size, tgt_len]
        """

        # 编码器
        encoder_outputs = self.encoder(src, attention_mask=src_mask)
        encoder_hidden = encoder_outputs.last_hidden_state  # [batch_size, src_len, 768]

        # 解码器
        decoder_output = self.decoder(tgt, encoder_hidden, src_mask, tgt_mask)

        return decoder_output

    def generate(self, src, src_mask, max_len=64, sos_token=None, eos_token=None, pad_token=0):
        """贪婪解码生成翻译"""
        self.eval()
        with torch.no_grad():
            batch_size = src.size(0)
            device = src.device

            # 编码
            encoder_outputs = self.encoder(src, attention_mask=src_mask)
            encoder_hidden = encoder_outputs.last_hidden_state

            # 初始化解码序列
            if sos_token is None:
                sos_token = 1  # 默认SOS token

            generated = torch.full((batch_size, 1), sos_token, device=device, dtype=torch.long)

            for _ in range(max_len - 1):
                # 解码当前序列
                decoder_output = self.decoder(generated, encoder_hidden, src_mask)

                # 获取下一个token
                next_token_logits = decoder_output[:, -1, :]  # [batch_size, vocab_size]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # 添加到生成序列
                generated = torch.cat([generated, next_token], dim=1)

                # 检查是否所有序列都结束了
                if eos_token is not None and (next_token == eos_token).all():
                    break

            return generated