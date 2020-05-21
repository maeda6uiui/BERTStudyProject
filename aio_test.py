import torch
from transformers import BertConfig,BertForMultipleChoice,BertJapaneseTokenizer

class InputExample(object):
    def __init__(self,example_id,question,contexts,endings,label=None):
        self.example_id=example_id
        self.question=question
        self.contexts=contexts
        self.endings=endings
        self.label=label

class InputFeatures(object):
    def __init__(self,example_id,choices_features,label):
        self.example_id=example_id
        self.choices_features=[
            {
                "input_ids":input_ids,
                "input_mask":input_mask,
                "segment_ids":segment_ids
            }
            for input_ids,input_mask,segment_ids in choices_features
        ]
        self.label=label

class DataProcessor(object):
    def get_examples(self,mode,data_dir,fname,entities_fname):
        raise NotImplementedError()

    def get_labels(self):
        raise NotImplementedError()

class JaqketProcessor(DataProcessor):
    

config=None
model=None
tokenizer=None

def init():
    global config
    global model
    global tokenizer
    config=BertConfig.from_json_file("./AIO_FineTuning/config.json")
    model=BertForMultipleChoice.from_pretrained("./AIO_FineTuning/pytorch_model.bin",config=config)
    tokenizer=BertJapaneseTokenizer.from_pretrained("./AIO_FineTuning/vocab.txt",do_lower_case=False,do_basic_tokenize=False)

    model.eval()
    model.to("cpu")

