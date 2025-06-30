


# 函数：从文件中读取数据
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = file.readlines()
    # 移除每行的换行符
    data = [line.strip() for line in data]
    return data


def load_data_list(train_data_path_de,train_data_path_en,valid_data_path_de,valid_data_path_en,test_data_path_de,test_data_path_en):
    print("Loading data...")
    # 读取数据
    test_de = load_data(test_data_path_de)
    test_en = load_data(test_data_path_en)

    train_de = load_data(train_data_path_de)
    train_en = load_data(train_data_path_en)

    valid_de = load_data(valid_data_path_de)
    valid_en = load_data(valid_data_path_en)
    print("Data loaded.")
    print("train_de num:", len(train_de))
    print("train_en num:", len(train_en))
    print("valid_de num:", len(valid_de))
    print("valid_en num:", len(valid_en))
    print("test_de  num:", len(test_de))
    print("test_en  num:", len(test_en))
    return train_de,train_en,valid_de,valid_en,test_de,test_en