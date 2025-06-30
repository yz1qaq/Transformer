import torch
import torch.nn as nn
from transformers import BertTokenizer,BertModel
from sklearn.utils import shuffle
from model import German2EnglishTranslator
from load_data import load_data_list


test_data_path_de = "Data/test_2016_flickr.de"
test_data_path_en = "Data/test_2016_flickr.en"

train_data_path_de = "Data/train.de"
train_data_path_en = "Data/train.en"

valid_data_path_de = "Data/val.de"
valid_data_path_en = "Data/val.en"

MAX_LEN = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("device:",device)


tokenizer_en = BertTokenizer.from_pretrained("bert-base-uncased")
tokenizer_en.add_special_tokens({'bos_token': '<sos>', 'eos_token': '<eos>'})
tokenizer_en.pad_token = '[PAD]'

en_vocab_size = len(tokenizer_en)


encoder_model = BertModel.from_pretrained("bert-base-german-cased").to(device)

tokenizer_de = BertTokenizer.from_pretrained("bert-base-german-cased")



#填充英语编码
def pad_sequence(seq, max_len, pad_id=0):
    seq_len = len(seq)
    if seq_len > max_len:
        # 截断
        seq = seq[:max_len]
    else:
        # 填充到max_len，PAD放在EOS后面
        seq = seq + [pad_id] * (max_len - seq_len)
    return seq


def train_data_loader(batch_size=32,tokenizer_de=None,tokenizer_en=None):

    if tokenizer_de is None or tokenizer_en is None :
        raise ValueError("tokenizer must be provided")

    def _loader_generator(shuffle_flag):
        # 转换为列表并打乱
        data_list = list(zip(train_de, train_en))
        if shuffle_flag:
            data_list = shuffle(data_list)

        for i in range(0, len(data_list), batch_size):
            batch_de, batch_en = zip(*data_list[i:i + batch_size])
            # 批量编码
            encoded_de = tokenizer_de(
                batch_de,
                return_tensors="pt",
                padding="max_length",
                max_length=MAX_LEN,
                truncation=True
            )

            batch_en_input_ids = []
            for en in batch_en:
                en_tokens_en = tokenizer_en.tokenize(en)
                encoded_en = ["<sos>"] + en_tokens_en + ["<eos>"]
                en_input_ids = tokenizer_en.convert_tokens_to_ids(encoded_en)
                en_input_ids = pad_sequence(en_input_ids, max_len=MAX_LEN, pad_id=tokenizer_en.pad_token_id)
                batch_en_input_ids.append(en_input_ids)

            batch_en_input_ids = torch.tensor(batch_en_input_ids,dtype=torch.long).to(device)


            batch_de_input_ids = encoded_de["input_ids"].to(device)
            mask_de = encoded_de["attention_mask"].to(device)



            yield batch_de_input_ids,mask_de,batch_en_input_ids

    return _loader_generator(shuffle_flag=True)


# ==================== 创建验证数据加载器 ====================
def valid_data_loader(batch_size=32):
    """验证数据加载器"""

    def _loader_generator():
        data_list = list(zip(valid_de, valid_en))

        for i in range(0, len(data_list), batch_size):
            batch_de, batch_en = zip(*data_list[i:i + batch_size])

            # 编码德语
            encoded_de = tokenizer_de(
                batch_de,
                return_tensors="pt",
                padding="max_length",
                max_length=MAX_LEN,
                truncation=True
            )

            # 编码英语
            batch_en_input_ids = []
            for en in batch_en:
                en_tokens = tokenizer_en.tokenize(en)
                encoded_en = ["<sos>"] + en_tokens + ["<eos>"]
                en_input_ids = tokenizer_en.convert_tokens_to_ids(encoded_en)
                en_input_ids = pad_sequence(en_input_ids, max_len=MAX_LEN, pad_id=pad_token_id)
                batch_en_input_ids.append(en_input_ids)

            batch_en_input_ids = torch.tensor(batch_en_input_ids, dtype=torch.long).to(device)
            batch_de_input_ids = encoded_de["input_ids"].to(device)
            mask_de = encoded_de["attention_mask"].to(device)

            yield batch_de_input_ids, mask_de, batch_en_input_ids

    return _loader_generator()



# ==================== 训练函数 ====================
def train_epoch(model, data_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_de_input_ids, mask_de, batch_en_input_ids in data_loader:
        # 准备解码器输入和目标
        # 解码器输入：去掉最后一个token (通常是EOS)
        decoder_input = batch_en_input_ids[:, :-1]
        # 目标：去掉第一个token (SOS)
        target = batch_en_input_ids[:, 1:]

        # 创建目标掩码 (非padding位置为1)
        tgt_mask = (decoder_input != pad_token_id)

        # 前向传播
        output = model(batch_de_input_ids, decoder_input, mask_de, tgt_mask)

        # 计算损失
        loss = criterion(output.reshape(-1, output.size(-1)), target.reshape(-1))

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪 (可选)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if (num_batches+1) % 50 == 0:
            print(f"Batch {num_batches+1}, Loss: {loss.item():.4f}")

    return total_loss / num_batches


def validate(model, data_loader, criterion, device):
    """验证函数"""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch_de_input_ids, mask_de, batch_en_input_ids in data_loader:
            decoder_input = batch_en_input_ids[:, :-1]
            target = batch_en_input_ids[:, 1:]
            tgt_mask = (decoder_input != pad_token_id)

            output = model(batch_de_input_ids, decoder_input, mask_de, tgt_mask)
            loss = criterion(output.reshape(-1, output.size(-1)), target.reshape(-1))

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


# ==================== 翻译生成函数 ====================
def translate_sentence(model, sentence, tokenizer_de, tokenizer_en, device, max_len=32):
    """翻译单个句子"""
    model.eval()

    with torch.no_grad():
        # 编码德语句子
        encoded = tokenizer_de(
            sentence,
            return_tensors="pt",
            padding="max_length",
            max_length=MAX_LEN,
            truncation=True
        )
        src = encoded["input_ids"].to(device)
        src_mask = encoded["attention_mask"].to(device)

        # 生成翻译
        generated = model.generate(
            src, src_mask, max_len=max_len+10,
            sos_token=sos_token_id,
            eos_token=eos_token_id,
            pad_token=pad_token_id
        )

        # 解码生成的token
        translated = tokenizer_en.decode(generated[0], skip_special_tokens=True)
        return translated


def train_model(num_epochs=5, batch_size=16):
    """完整的训练流程"""
    print("开始训练...")

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)

        # 训练
        train_loader = train_data_loader(batch_size=batch_size,
                                         tokenizer_de=tokenizer_de,
                                         tokenizer_en=tokenizer_en)
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # 验证
        valid_loader = valid_data_loader(batch_size=batch_size)
        val_loss = validate(model, valid_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Valid Loss: {val_loss:.4f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_translator.pth')
            print("保存最佳模型!")

        # 测试翻译
        test_sentence = "Hallo, wie geht es dir?"
        translation = translate_sentence(model, test_sentence, tokenizer_de, tokenizer_en, device)
        print(f"测试翻译: '{test_sentence}' -> '{translation}'")


def calculate_bleu_score(model, test_sentences_de, test_sentences_en, tokenizer_de, tokenizer_en, device):
    """计算BLEU分数"""
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    nltk.download('punkt', quiet=True)

    model.eval()
    bleu_scores = []

    for de_sent, en_sent in zip(test_sentences_de[:100], test_sentences_en[:100]):  # 测试前100个
        # 生成翻译
        translation = translate_sentence(model, de_sent, tokenizer_de, tokenizer_en, device)

        # 计算BLEU
        reference = [en_sent.split()]
        candidate = translation.split()

        smoothie = SmoothingFunction().method4
        score = sentence_bleu(reference, candidate, smoothing_function=smoothie)
        bleu_scores.append(score)

    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    print(f"平均BLEU分数: {avg_bleu:.4f}")
    return avg_bleu




if __name__ == '__main__':

    train_de, train_en, valid_de, valid_en, test_de, test_en = load_data_list(train_data_path_de, train_data_path_en,valid_data_path_de, valid_data_path_en, test_data_path_de, test_data_path_en)


    model = German2EnglishTranslator(
        encoder_model=encoder_model,
        en_vocab_size=en_vocab_size,
        d_model=768,  # BERT的hidden size
        n_heads=8,
        n_layers=6,
        d_ff=2048
    ).to(device)

    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer_en.pad_token_id)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

    # 获取特殊token的ID
    sos_token_id = tokenizer_en.convert_tokens_to_ids('<sos>')
    eos_token_id = tokenizer_en.convert_tokens_to_ids('<eos>')
    pad_token_id = tokenizer_en.pad_token_id

    print(f"SOS token ID: {sos_token_id}")
    print(f"EOS token ID: {eos_token_id}")
    print(f"PAD token ID: {pad_token_id}")

    # 开始训练
    #train_model(num_epochs=10, batch_size=32)  # 根据GPU内存调整batch_size


    # 加载最佳模型
    model.load_state_dict(torch.load('best_translator.pth'))

    # 测试几个翻译
    test_sentences = [
        "Ein Mann mit einem orangefarbenen Hut, der etwas anstarrt.",
        "Ein Boston Terrier läuft über saftig - grünes Gras vor einem weißen Zaun.",
        "Ein Mädchen in einem Karateanzug bricht ein Brett mit einem Tritt.",
        "Fünf Leute in Winterjacken und mit Helmen stehen im Schnee mit Schneemobilen im Hintergrund."
    ]

    print("\n" + "=" * 50)
    print("翻译测试:")
    print("=" * 50)

    for sentence in test_sentences:
        translation = translate_sentence(model, sentence, tokenizer_de, tokenizer_en, device)
        print(f"DE: {sentence}")
        print(f"EN: {translation}")
        print("-" * 30)

    # 计算BLEU分数
    #print("计算BLEU分数...")
    #bleu_score = calculate_bleu_score(model, test_de, test_en, tokenizer_de, tokenizer_en, device)
