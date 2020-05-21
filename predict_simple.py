import sys
import torch
from transformers import BertTokenizer,BertForMaskedLM,BertConfig
from pyknp import Juman

juman=None

config=None
model=None
bert_tokenizer=None

def init():
    global juman
    juman=Juman(jumanpp=True)

    global config
    global model
    global bert_tokenizer
    config=BertConfig.from_json_file(
        "/home/maeda/Documents/Laboratory/Japanese_L-24_H-1024_A-16_E-30_BPE_WWM/bert_config.json")
    model=BertForMaskedLM.from_pretrained(
        "/home/maeda/Documents/Laboratory/Japanese_L-24_H-1024_A-16_E-30_BPE_WWM/pytorch_model.bin",config=config)
    bert_tokenizer=BertTokenizer.from_pretrained(
        "/home/maeda/Documents/Laboratory/Japanese_L-24_H-1024_A-16_E-30_BPE_WWM/vocab.txt",
        do_lower_case=False,do_basic_tokenize=False)

    model.eval()

def predict(text):
    result=juman.analysis(text)

    mrph_list=result.mrph_list()
    masked_index=-1

    for i,mrph in enumerate(mrph_list):
        if mrph.midasi=='*':
            masked_index=i+1
            break

    if masked_index<0:
        print("Error: マスクされている形態素がありません。")
        return

    tokenized_text=[mrph.midasi for mrph in mrph_list]
    tokenized_text.insert(0,"[CLS]")
    tokenized_text.append("[SEP]")
    tokenized_text[masked_index]="[MASK]"

    input_ids=bert_tokenizer.convert_tokens_to_ids(tokenized_text)
    input_ids_tensor=torch.tensor([input_ids])

    input_ids_tensor=input_ids_tensor.to("cpu")
    model.to("cpu")

    with torch.no_grad():
        outputs=model(input_ids_tensor)
        predictions=outputs[0]

    _,predicted_indexes=torch.topk(predictions[0,masked_index],k=5)
    predicted_tokens=bert_tokenizer.convert_ids_to_tokens(predicted_indexes.tolist())
    print(predicted_tokens)

if __name__=="__main__":
    init()

    text=sys.stdin.readline()
    text=text.rstrip()
    predict(text)
