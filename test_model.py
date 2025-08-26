import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from LSTM import MyLstmClassifier   
from dataset import MyDataset    
import numpy as np
from preprocessing import ProcessingText
from datasets import load_dataset
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "lstm_w2v_best.pth"
OUTPUT_CSV = "test_metrics.csv"

if __name__ == 'main':
    list_labels = ['__label__kinh_doanh', '__label__pháp_luật', '__label__giáo_dục', '__label__công_nghệ', '__label__sống_trẻ', '__label__thời_sự',
                    '__label__sức_khỏe', '__label__xuất_bản', '__label__xe_360', '__label__thế_giới', '__label__nhịp_sống', '__label__âm_nhạc', 
                    '__label__du_lịch', '__label__giải_trí', '__label__thể_thao', '__label__phim_ảnh', '__label__thời_trang', '__label__ẩm_thực']

    X_test  = np.load("/embedded_vector/test_embeddings.npy", allow_pickle=True)
    processingText = ProcessingText()
    ds = load_dataset("hiuman/vietnamese_classification")
    train_ds, val_ds, test_ds = processingText.data_split(ds)
    y_test  = np.array([list_labels.index(sample["label"])  for sample in test_ds])

    test_dataset = MyDataset(X_test,y_test)
    test_dataloader = DataLoader(dataset=test_dataset,
                                batch_size=16,
                                shuffle=True)

    input_dim = 100    
    hidden_dim = 128
    num_classes = 18  
    model = MyLstmClassifier(input_dim, hidden_dim, num_classes).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()


    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_dataloader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            outputs = model(X_batch)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="weighted")
    rec = recall_score(all_labels, all_preds, average="weighted")
    f1 = f1_score(all_labels, all_preds, average="weighted")

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

    results = pd.DataFrame([{
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1
    }])

    results.to_csv(OUTPUT_CSV, index=False)
    print(f"Metrics saved to {OUTPUT_CSV}")
