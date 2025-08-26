import numpy as np
from preprocessing import ProcessingText
from gensim.models import Word2Vec
from datasets import load_dataset
import pickle


def texts_to_vectors(texts, w2v_model, max_len=100):
    """
    Chuyển danh sách văn bản (đã tokenized) thành ma trận embedding cố định.
    - max_len: padding/truncate độ dài câu
    """
    vector_size = w2v_model.vector_size
    all_vecs = []

    for tokens in texts:
        vecs = []
        for token in tokens:
            if token in w2v_model.wv:
                vecs.append(w2v_model.wv[token])
        # Padding/truncate
        if len(vecs) < max_len:
            vecs.extend([np.zeros(vector_size)] * (max_len - len(vecs)))
        else:
            vecs = vecs[:max_len]
        all_vecs.append(np.array(vecs))
    
    return np.array(all_vecs)  # shape: (num_texts, max_len, vector_size)

if __name__ == "__main__":
    processingText = ProcessingText()
    ds = load_dataset("hiuman/vietnamese_classification")
    train_ds, val_ds, test_ds = processingText.data_split(ds)
    
    w2v = Word2Vec.load("D:/TextEmotionClassify/w2v.model")

    train_texts = [processingText.processing_text(sample['text']) for sample in train_ds]
    valid_texts = [processingText.processing_text(sample['text']) for sample in val_ds]
    test_texts = [processingText.processing_text(sample['text']) for sample in test_ds]

    train_vectors = texts_to_vectors(train_texts, w2v, max_len=120)
    valid_vectors = texts_to_vectors(valid_texts, w2v, max_len=120)
    test_vectors = texts_to_vectors(test_texts, w2v, max_len=120)

    print("Embedding shape:", train_vectors.shape)  
    print("Embedding shape:", valid_vectors.shape)  
    print("Embedding shape:", test_vectors.shape)  



    # Save lại thành file .npy
    np.save("embedded_vector/train_embeddings.npy", train_vectors)
    print("✅ Saved embeddings to train_embeddings.npy")
    np.save("embedded_vector/valid_embeddings.npy", valid_vectors)
    print("✅ Saved embeddings to valid_embeddings.npy")
    np.save("embedded_vector/test_embeddings.npy", test_vectors)
    print("✅ Saved embeddings to test_embeddings.npy")
