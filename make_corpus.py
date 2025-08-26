import underthesea
import torch
from datasets import load_dataset
import json
from datasets import load_dataset
from multiprocessing import Pool, cpu_count
from collections import defaultdict
def data_split(ds):
    split_train = ds['train'].train_test_split(test_size=0.1, seed=42)
    train_ds = split_train['train']
    val_ds = split_train['test']
    test_ds = ds['test']
    return train_ds, val_ds, test_ds

def tokenize(text):
    tokens = underthesea.word_tokenize(text)
    return tokens

# def remove_underscore(text):
#     return text.replace('_', " ")

 #remove word that have one char
def remove_w(text):
    words = text.split(" ")
    words = [word for word in words if len(words)> 1] 
    text = " ".join(words)
    return text

stopwords_vietnamese = [
    "a", "ai", "anh", "ba", "bác", "bạn", "bây", "bà", "bị", "bên", "bằng", "bởi", "bấy", "bất",
    "cái", "có", "các", "cần", "cùng", "của", "cũng", "cứ", "càng", "chỉ", "cho", "chưa", "chúng", 
    "chị", "chiếc", "chính", "chứ", "chẳng", "chưa", "chưa_từng", "chưa_hề",
    "đã", "đang", "để", "đến", "đi", "được", "dù", "do", "dưới", "đó", "đây", "đấy", "điều",
    "gì", "gần", "gồm", "giữa", "hay", "hơn", "hoặc", "hết", "hãy", "họ", "hoàn_toàn",
    "khi", "không", "khoảng", "kể", "kém", "làm", "lại", "lên", "lúc", "là", "lại", "lấy",
    "mà", "mỗi", "mọi", "một", "mấy", "mình", "muốn", "mang", "mọi_người",
    "nào", "này", "nên", "nếu", "những", "như", "nhưng", "nhận", "nơi", "nó", "nói",
    "ở", "ông", "rất", "rằng", "ra", "riêng", "rồi", "sau", "sẽ", "so", "sự", "song", "sao", 
    "sang", "sẵn", "số", "sao", "sao_cho", "sao_mà",
    "tại", "theo", "thành", "thì", "trong", "trên", "trước", "từ", "tới", "từng", "ta", "tôi",
    "tớ", "tự", "tuy", "tuy_nhiên", "thật", "thế", "thường", "toàn", "tất_cả", "từng", 
    "vào", "vì", "với", "vừa", "vẫn", "vậy", "về", "vài", "vẫn", "việc",
    "xem", "xa", "xin", "xảy", "ý"
]

def remove_stopword(text):
    text = [w for w in text if w.lower() not in stopwords_vietnamese]
    
    return "".join(text)

# Tokenize and preprocess the text
def processing_text(text):
    # text = remove_underscore(text)
    
    #remove word that have one char
    text = remove_w(text)

    #remove stopword
    text = remove_stopword(text)
    
    tokens = tokenize(text)
    return tokens

corpus ={}
idx = 1
# Trả về set từ vựng trong 1 tập văn bản
def extract_vocab(data):
    local_vocab = set()
    for text in data:
        tokens = processing_text(text['text'])
        local_vocab.update(tokens)
    return local_vocab

if __name__ == "__main__":
    ds = load_dataset("hiuman/vietnamese_classification")
    train_ds = ds["train"]
    test_ds = ds["test"]

    # Gộp tất cả dữ liệu
    all_data = list(train_ds) + list(test_ds)

    # Chia nhỏ dữ liệu theo số CPU
    n_cpu = cpu_count()
    chunk_size = len(all_data) // n_cpu
    chunks = [all_data[i:i+chunk_size] for i in range(0, len(all_data), chunk_size)]

    # Dùng multiprocessing
    with Pool(n_cpu) as p:
        results = p.map(extract_vocab, chunks)

    # Merge kết quả từ các tiến trình
    vocab = set().union(*results)

    # Tạo corpus {word: idx}
    corpus = {w: idx for idx, w in enumerate(vocab, start=1)}

    print("Số từ vựng:", len(corpus))
    print("Ví dụ:", list(corpus.items())[:10])