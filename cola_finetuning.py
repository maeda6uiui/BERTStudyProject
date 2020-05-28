import wget
import os
import zipfile
import time
import datetime
import random
import torch
import numpy as np
import pandas as pd
from transformers import(
    BertTokenizer,
    BertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from torch.utils.data import(
    TensorDataset,
    random_split,
    DataLoader,
    RandomSampler,
    SequentialSampler
)

def flat_accuracy(preds,labels):
    pred_flat=np.argmax(preds,axis=1).flatten()
    labels_flat=labels.flatten()

    return (pred_flat==labels_flat).mean()

def format_time(elapsed):
    elapsed_rounded=int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def extract_dataset():
    print("データセットの準備中......")

    zip_filename="./cola_public_1.1.zip"

    if not os.path.exists(zip_filename):
        wget.download(
            "https://nyu-mll.github.io/CoLA/cola_public_1.1.zip",
            zip_filename)
    
    if not os.path.exists("./cola_public/"):
        with zipfile.ZipFile(zip_filename) as existing_zip:
            existing_zip.extractall(".")

    df=pd.read_csv(
        "./cola_public/raw/in_domain_train.tsv",
        delimiter="\t",header=None,
        names=["sentence_source","label","label_notes","sentence"])

    sentences=df.sentence.values
    labels=df.label.values

    return sentences,labels

def create_tokenizer_and_model():
    print("tokenizerとモデルの作成中......")

    tokenizer=BertTokenizer.from_pretrained("bert-large-uncased",do_lower_case=True)

    model=BertForSequenceClassification.from_pretrained(
      "bert-large-uncased",num_labels=2,
      output_attentions=False,output_hidden_states=False)
    model.cuda()

    return tokenizer,model

def create_dataset(tokenizer,sentences,labels):
    print("TensorDatasetの作成中......")

    input_ids=[]
    attention_masks=[]

    for sentence in sentences:
        encoded_dict=tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=64,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids.append(encoded_dict["input_ids"])
        attention_masks.append(encoded_dict["attention_mask"])

    input_ids=torch.cat(input_ids,dim=0)
    attention_masks=torch.cat(attention_masks,dim=0)
    labels=torch.tensor(labels)

    dataset=TensorDataset(input_ids,attention_masks,labels)

    return dataset

def create_dataloaders(dataset):
    print("データローダの作成中......")

    train_size=int(0.9*len(dataset))
    val_size=len(dataset)-train_size

    train_dataset,val_dataset=random_split(dataset,[train_size,val_size])

    batch_size=32
    train_dataloader=DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size
    )
    val_dataloader=DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=batch_size
    )

    return train_dataloader,val_dataloader

def set_seed():
    seed_val=100
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

def train(model,train_dataloader):
    print("訓練中......")

    optimizer=AdamW(model.parameters(),lr=2e-5,eps=1e-8)

    epochs=4
    total_steps=len(train_dataloader)*epochs
    scheduler=get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps)

    for i in range(epochs):
        print("========== Epoch {}/{} ==========".format(i+1,epochs))

        t0=time.time()
        total_train_loss=0
        model.train()

        for step,batch in enumerate(train_dataloader):
            if step%40==0 and not step==0:
                elapsed=format_time(time.time()-t0)
                print("Batch {} of {}. 経過時間: {}.".format(step,len(train_dataloader),elapsed))
            
            b_input_ids=batch[0].cuda()
            b_input_mask=batch[1].cuda()
            b_labels=batch[2].cuda()

            model.zero_grad()

            loss,logits=model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels)
            
            total_train_loss+=loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss=total_train_loss/len(train_dataloader)
        training_time=format_time(time.time()-t0)

        print("train lossの平均値: {:.2f}".format(avg_train_loss))
        print("このepochの経過時間: {}".format(training_time))

    print("訓練終了")
    
    print("モデルを保存中......")
    torch.save(model.state_dict(),"pytorch_model.bin")

def validation(model,val_dataloader):
    print("検証中......")

    t0=time.time()

    model.eval()

    total_eval_accuracy=0
    total_eval_loss=0
    nb_eval_steps=0

    for batch in val_dataloader:
        b_input_ids=batch[0].cuda()
        b_input_mask=batch[1].cuda()
        b_labels=batch[2].cuda()

        with torch.no_grad():
            loss,logits=model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels)
        
        total_eval_loss+=loss.item()

        logits=logits.detach().cpu().numpy()
        label_ids=b_labels.cpu().numpy()

        total_eval_accuracy+=flat_accuracy(logits,label_ids)

    avg_val_accuracy=total_eval_accuracy/len(val_dataloader)
    print("正解率: {:.2f}".format(avg_val_accuracy))

    avg_val_loss=total_eval_loss/len(val_dataloader)
    validation_time=format_time(time.time()-t0)

    print("validation lossの平均値: {:.2f}".format(avg_val_loss))
    print("検証にかかった時間: {}".format(validation_time))

    print("検証終了")

#def test(model,test_dataloader):
#   テストデータを用いてテストを行う。
#
    
if __name__=="__main__":
    #データセットの準備
    sentences,labels=extract_dataset()
    #tokenizerとmodelの作成
    tokenizer,model=create_tokenizer_and_model()
    #TensorDatasetの作成
    dataset=create_dataset(tokenizer,sentences,labels)
    #訓練用データローダと評価用データローダの作成
    train_dataloader,val_dataloader=create_dataloaders(dataset)
    #乱数のシードを設定
    set_seed()
    #訓練
    train(model,train_dataloader)
    #検証
    validation(model,val_dataloader)
    #テスト
    #test(model,test_dataloader)
