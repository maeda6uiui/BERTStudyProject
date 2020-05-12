import sys
import torch
from transformers import BertTokenizer,BertForQuestionAnswering

model=None
bert_tokenizer=None

def init():
  global model
  global bert_tokenizer
  model=BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
  bert_tokenizer=BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

  model.to("cuda")

def qa(question,text):
  input_ids=bert_tokenizer.encode(question,text)
  input_ids_tensor=torch.tensor([input_ids])
  input_ids_tensor=input_ids_tensor.to("cuda")

  #質問文:0 テキスト:1
  token_type_ids=[0 if i<=input_ids.index(102) else 1 for i in range(len(input_ids))]
  token_type_ids_tensor=torch.tensor([token_type_ids])
  token_type_ids_tensor=token_type_ids_tensor.to("cuda")

  with torch.no_grad():
    start_scores,end_scores=model(input_ids_tensor,token_type_ids=token_type_ids_tensor)

  answer_start=torch.argmax(start_scores)
  answer_end=torch.argmax(end_scores)

  answer=input_ids[answer_start:answer_end+1]
  answer=bert_tokenizer.convert_ids_to_tokens(answer)

  concat_answer=""
  for word in answer:
    concat_answer+=word
    concat_answer+=" "

  print(concat_answer)

if __name__=="__main__":
  init()

  print("text: ",end="")
  text=input()
  print("question: ",end="")
  question=input()

  qa(question,text)
