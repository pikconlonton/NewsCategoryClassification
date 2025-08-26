import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from preprocessing import ProcessingText
from dataset import MyDataset
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

class MyLstmClassifier(nn.Module):
    def __init__(self, input_dim,hidden_dim,num_classes, num_layers=1,dropout= 0.3):
        super(MyLstmClassifier,self).__init__()
        self.lstm = nn.LSTM(input_dim,hidden_dim,num_layers=num_layers,
                            batch_first=True,dropout= dropout,bidirectional=False)
        
        self.fc1 = nn.Linear(hidden_dim,64)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64,num_classes)

    def forward(self,X):
        out, (h_n,c_n) = self.lstm(X)
        h_last = h_n[-1]
        x = self.fc1(h_last)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x
    

# ==============================
# Train Model
# ==============================
if __name__ == "__main__":
    list_labels = ['__label__kinh_doanh', '__label__pháp_luật', '__label__giáo_dục', '__label__công_nghệ', '__label__sống_trẻ', '__label__thời_sự',
                    '__label__sức_khỏe', '__label__xuất_bản', '__label__xe_360', '__label__thế_giới', '__label__nhịp_sống', '__label__âm_nhạc', 
                    '__label__du_lịch', '__label__giải_trí', '__label__thể_thao', '__label__phim_ảnh', '__label__thời_trang', '__label__ẩm_thực']

    X_train = np.load("embedded_vector/train_embeddings.npy", allow_pickle=True)
    X_val   = np.load("embedded_vector/valid_embeddings.npy", allow_pickle=True)
    X_test  = np.load("embedded_vector/test_embeddings.npy", allow_pickle=True)

    processingText = ProcessingText()
    ds = load_dataset("hiuman/vietnamese_classification")
    train_ds, val_ds, test_ds = processingText.data_split(ds)

    y_train = np.array([list_labels.index(sample["label"]) for sample in train_ds])
    y_val   = np.array([list_labels.index(sample["label"])  for sample in val_ds])
    y_test  = np.array([list_labels.index(sample["label"])  for sample in test_ds])

    train_dataset = MyDataset(X_train,y_train)
    val_dataset =   MyDataset(X_val,y_val)
    test_dataset = MyDataset(X_test,y_test)

    train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=16,
                                shuffle=True)
    valid_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=16,
                                shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset,
                                batch_size=16,
                                shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = X_train.shape[2]
    hiden_dim = 128
    num_classes = len(set(y_train))

    model = MyLstmClassifier(input_dim=input_dim,
                            hidden_dim= hiden_dim,
                            num_classes=num_classes
                            ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=1e-3)

    EPOCHS = 5
    best_val_acc = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0

        train_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)
        for X_batch, y_batch in train_bar:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

            # update tqdm bar
            train_bar.set_postfix({
                "Loss": f"{total_loss/(total/ y_batch.size(0)):.4f}",
                "Acc": f"{correct/total:.4f}"
            })

        train_acc = correct / total

        # Validation loop
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            val_bar = tqdm(valid_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", leave=False)
            for X_batch, y_batch in val_bar:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == y_batch).sum().item()
                val_total += y_batch.size(0)
                val_bar.set_postfix({"Acc": f"{val_correct/val_total:.4f}"})

        val_acc = val_correct / val_total

        print(f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train Loss: {total_loss/len(train_dataloader):.4f} "
            f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "lstm_w2v_best.pth")


    model.load_state_dict(torch.load("lstm_w2v_best.pth"))
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        test_bar = tqdm(test_dataloader, desc="Testing", leave=True)
        for X_batch, y_batch in test_bar:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, preds = torch.max(outputs, 1)
            test_correct += (preds == y_batch).sum().item()
            test_total += y_batch.size(0)
            test_bar.set_postfix({"Acc": f"{test_correct/test_total:.4f}"})

    print(f"Test Accuracy: {test_correct/test_total:.4f}")