import torch
import numpy as np
import io
from transformers import(
    BertConfig,
    BertTokenizer,
    BertForSequenceClassification,
)
from torch.utils.data import(
    TensorDataset,
    DataLoader,
)

tokenizer=None
model=None

def init():
    global tokenizer
    global model

    config=BertConfig.from_json_file("./Model/bert_config.json")
    tokenizer=BertTokenizer.from_pretrained(
        "./Model/vocab.txt",
        do_lower_case=True,num_labels=2)
    model=BertForSequenceClassification.from_pretrained(
        "./Model/pytorch_model.bin",config=config)
    model.cuda()
    model.eval()

#一文のみを判定する。
def is_acceptable_sentence(sentence):
    encoded_dict=tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        max_length=64,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    input_ids=encoded_dict["input_ids"].cuda()
    attention_mask=encoded_dict["attention_mask"].cuda()
    labels=torch.tensor([1]).cuda()

    with torch.no_grad():
        loss,logits=model(
            input_ids,
            token_type_ids=None,
            attention_mask=attention_mask,
            labels=labels)

    logits=logits.detach().cpu().numpy()
    preds=np.argmax(logits,axis=1).flatten()

    if preds[0]==0:
        return False
    else:
        return True

#複数の文章をまとめて判定する。
def analyze(sentences):
    input_ids=[]
    attention_masks=[]
    labels=[]

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
        labels.append(1)

    input_ids=torch.cat(input_ids,dim=0)
    attention_masks=torch.cat(attention_masks,dim=0)
    labels=torch.tensor(labels)

    dataset=TensorDataset(input_ids,attention_masks,labels)
    dataloader=DataLoader(
        dataset,
        batch_size=len(dataset),
        shuffle=False
    )

    for batch in dataloader:
        b_input_ids=batch[0].cuda()
        b_input_mask=batch[1].cuda()
        b_input_labels=batch[2].cuda()

        with torch.no_grad():
            loss,logits=model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_input_labels)

        logits=logits.detach().cpu().numpy()
        
        preds=np.argmax(logits,axis=1).flatten()

    for i,sentence in enumerate(sentences):
        acceptability="unacceptable" if preds[i]==0 else "acceptable"
        print(sentence,"->",acceptability)
    
if __name__=="__main__":
    init()

    while True:
        sentence=input()
        if sentence=="exit":
            break

        if is_acceptable_sentence(sentence):
            print("->","acceptable")
        else:
            print("->","unacceptable")

    """
    sentences=[]
    while True:
        sentence=input()
        if sentence=="exit":
            break

        sentences.append(sentence)

    analyze(sentences)
    """
