import json
import os
import torch
from transformers import BertConfig,BertForMultipleChoice,BertJapaneseTokenizer,PreTrainedTokenizer

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
    def get_entities(self,data_dir,entities_fname):
        entities=dict()

        for line in self._read_json_gzip(os.path.join(data_dir,entities_fname)):
            entity=json.loads(line.strip())
            entities[entity["title"]]=entity["text"]

        return entities

    def get_examples(self,mode,data_dir,fname,entities_fname,num_options=20):
        entities=self._get_entities(data_dir,entities_fname)
        return self._create_examples(
            self._read_json(os.path.join(data_dir,fname)),
            mode,
            entities,
            num_options,
        )

    def get_labels(self):
        return [
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
            "19",
        ]
    
    def _read_json(self,input_file):
        with open(input_file,"r",encoding="utf-8") as fin:
            lines=fin.readlines()
            return lines

    def _create_examples(self,lines,t_type,entities,num_options):
        examples=[]
        skip_examples=0

        for line in lines:
            data_raw=json.loads(line.strip("\n"))

            id=data_raw["qid"]
            question=data_raw["question"].replace("_","")
            options=data_raw["answer_candidates"][:num_options]
            answer=data_raw["answer_entity"]

            if answer not in options:
                continue

            if len(options)!=num_options:
                skip_examples+=1
                continue

            contexts=[entities[options[i]] for i in range(num_options)]
            truth=str(options.index(answer))

            if len(options)==num_options:
                examples.append(
                    InputExample(
                        example_id=id,
                        question=question,
                        contexts=contexts,
                        endings=options,
                        label=truth
                    )
                )

            return examples

def convert_examples_to_features(
    examples:List[InputExample],
    label_list:List[str],
    max_length:int
    tokenizer:PreTrainedTokenizer,
    pad_token_segment_id=0,
    pad_on_left=False,
    pad_token=0,
    mask_padding_with_zero=True,
)->List[InputFeatures]:
    #TODO
   


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

if __name__=="__main__":
    init()

    