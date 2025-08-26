import underthesea
import torch
from datasets import load_dataset
import json 

class ProcessingText:
    def __init__(self):
        self.stopwords_vietnamese = [
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
        

    def data_split(self, ds):
        split_train = ds['train'].train_test_split(test_size=0.1, seed=42)
        train_ds = split_train['train']
        val_ds = split_train['test']
        test_ds = ds['test']
        return train_ds, val_ds, test_ds

    def tokenize(self, text):
        tokens = underthesea.word_tokenize(text)
        return tokens

    # def remove_underscore(text):
    #     return text.replace('_', " ")

    #remove word that have one char
    def remove_w(self, text):
        words = text.split(" ")
        words = [word for word in words if len(words)> 1] 
        text = " ".join(words)
        return text


    def remove_stopword(self, text):
        text = [w for w in text if w.lower() not in self.stopwords_vietnamese]
        
        return "".join(text)

    # Tokenize and preprocess the text
    def processing_text(self, text):
        # text = remove_underscore(text)
        
        #remove word that have one char
        text = self.remove_w(text)

        #remove stopword
        text = self.remove_stopword(text)
        
        tokens = self.tokenize(text)
        return tokens


if __name__ == "__main__":

    ds = load_dataset("hiuman/vietnamese_classification")
    processingText = ProcessingText()
    train_ds, val_ds, test_ds = processingText.data_split(ds)  
    # with open("corpus.json", "r", encoding="utf-8") as f:
    #     corpus = json.load(f)

    # print(type(corpus))
    # print(len(corpus))
    labels = set()
    for data in ds['train']:
        labels.add(data['label'])
    labels2 = set()
    for data in ds['test']:
        labels.add(data['label'])

    print(labels)
    print(len(labels))
    print(labels2)
    print(len(labels2))