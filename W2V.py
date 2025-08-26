from preprocessing import ProcessingText
from gensim.models import Word2Vec
from datasets import load_dataset
def train_and_save_w2v(sentences):
    w2v_model = Word2Vec(
        sentences= sentences,
        vector_size=100,
        window=5,
        min_count=2,
        sg=1,
        workers=8
    )
    w2v_model.save('w2v.model')
    print('Save model Sucessfully !!!')


if __name__ == "__main__":

    processingText = ProcessingText()
    ds = load_dataset("hiuman/vietnamese_classification")
    train_ds, val_ds, test_ds = processingText.data_split(ds)
    sentences = []
    for ds in (train_ds,val_ds,test_ds): 
        for sentence in ds:
            sentences.append(processingText.processing_text(sentence['text']))
            
    
    train_and_save_w2v(sentences=sentences)
    # w2v = Word2Vec.load("D:/TextEmotionClassify/w2v.model")
    # vec = w2v.wv['âu_việt']
    # print(vec.shape)
    # print(vec[:10])
    # print(w2v.wv.most_similar('âu_việt',topn=4))