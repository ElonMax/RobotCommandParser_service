import logging

import torch
import numpy as np
from simpletransformers.ner import NERModel, NERArgs
from transformers import BertConfig, BertForTokenClassification, BertTokenizer

class SplitterWrapper:
    def __init__(self, config):
        self.config = config
        self.tokenizer = None
        self.model = None
        if config['use_gpu']:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                raise ValueError("В конфиге указан флаг использования гпу, но torch не обнаружил доступных гпу")
        else:
            self.device = "cpu"

    def load_model(self):
        model_args = NERArgs(labels_list=self.config["Model"]["labels"])
        #self.model = NERModel(
            #self.config["Model"]["model_type"],
            #self.config["Model"]["model_path"],
            #args=model_args,
            #labels=self.config["Model"]["labels"],
            #use_cuda=self.config['use_gpu']
        #)

        bertconfig = BertConfig.from_pretrained(
            self.config["Model"]["model_path"], **model_args.config)
        self.tokenizer = BertTokenizer.from_pretrained(
            self.config["Model"]["model_path"],
            do_lower_case=model_args.do_lower_case)
        self.model = BertForTokenClassification.from_pretrained(
            self.config["Model"]["model_path"],
            config=bertconfig
        )
        self.model.eval()
        self.model.to(self.device)

        logging.info("Модель загружена успешно")

    def _convert_tokens_to_word_logits(
            input_ids, label_ids, attention_mask, logits, tokenizer
    ):

        ignore_ids = [
            tokenizer.convert_tokens_to_ids(tokenizer.pad_token),
            tokenizer.convert_tokens_to_ids(tokenizer.sep_token),
            tokenizer.convert_tokens_to_ids(tokenizer.cls_token),
        ]

        # Remove unuseful positions
        masked_ids = input_ids[(1 == attention_mask)]
        masked_labels = label_ids[(1 == attention_mask)]
        masked_logits = logits[(1 == attention_mask)]
        for id in ignore_ids:
            masked_labels = masked_labels[(id != masked_ids)]
            masked_logits = masked_logits[(id != masked_ids)]
            masked_ids = masked_ids[(id != masked_ids)]

        # Map to word logits
        word_logits = []
        tmp = []
        for n, lab in enumerate(masked_labels):
            if lab != -100:  # pad_token_label_id
                if n != 0:
                    word_logits.append(tmp)
                tmp = [list(masked_logits[n])]
            else:
                tmp.append(list(masked_logits[n]))
        word_logits.append(tmp)

        return word_logits
    def predict(self, phrases):
        """

        :param phrases: list - список строк (фраз комманд)
        :return:
            subcommands: list - список списков строк подкоманд
        """
        if type(phrases) == str:
            phrases = [phrases]
        if self.model is None:
            raise ValueError("Model is not loaded. Call ClassifierWrapper().load_model() before making any predictions")
        with torch.no_grad():
            model_inputs = self.tokenizer.batch_encode_plus(
                phrases, return_tensors="pt", padding=True, truncation=True
            )
            model_inputs = {key: value.to(self.device) for key, value in model_inputs.items()}
            logits = self.model(**model_inputs)
            logits = logits[0].cpu().numpy()
            inputs = model_inputs['input_ids'].cpu().numpy()
            att_mask = model_inputs['attention_mask'].cpu().numpy()
        predictions = []
        for i in range(len(phrases)):
            labels = [self.tokenizer.ids_to_tokens[x] for x in inputs[i]]
            labels = np.array([-100 if x[:2] == "##" else 0 for x in labels])  # pad_token_label_id=-100
            word_logits = SplitterWrapper._convert_tokens_to_word_logits(inputs[i], labels, att_mask[i], logits[i], self.tokenizer)
            assert len(phrases[i].split()) == len(word_logits)
            predictions.append([])
            for w_i, word in enumerate(phrases[i].split()):
                predictions[-1].append({word: self.config["Model"]["labels"][np.argmax(word_logits[w_i][0])]})

        subcommands = []
        current_command = []
        for token in predictions[0]:
            token, cls = list(token.items())[0]
            if cls == "O":
                continue
            current_command.append(token)
            if cls == "[SEP]":
                subcommands.append(" ".join(current_command))
                current_command = []
        return subcommands
