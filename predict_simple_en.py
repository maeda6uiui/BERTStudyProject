import sys
import torch
from transformers import BertTokenizer,BertForMaskedLM

model=None
bert_tokenizer=None

def init():
  global model
  global bert_tokenizer
  model=BertForMaskedLM.from_pretrained("bert-base-uncased")
  bert_tokenizer=BertTokenizer.from_pretrained("bert-base-uncased",do_lower_case=True)

def predict(text):
  tokenized_text=bert_tokenizer.tokenize(text)

  masked_index=-1

  for i,word in enumerate(tokenized_text):
    if word=="*":
      masked_index=i+1
      break

  if masked_index<0:
    print("No masked word.")
    sys.exit(0)

  tokenized_text.insert(0,"[CLS]")
  tokenized_text.append("[SEP]")
  tokenized_text[masked_index]="[MASK]"

  input_ids=bert_tokenizer.convert_tokens_to_ids(tokenized_text)
  input_ids_tensor=torch.tensor([input_ids])

  input_ids_tensor=input_ids_tensor.to("cuda")
  model.to("cuda")

  with torch.no_grad():
    outputs=model(input_ids_tensor)
    predictions=outputs[0]

  _,predicted_indexes=torch.topk(predictions[0,masked_index],k=5)
  predicted_tokens=bert_tokenizer.convert_ids_to_tokens(predicted_indexes.tolist())
  print(predicted_tokens)

if __name__=="__main__":
  init()

  text="* is the capital city of Japan."
  predict(text)
