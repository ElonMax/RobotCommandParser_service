import json
import logging

import torch
import numpy as np
import joblib
from transformers import (
    BertConfig,
    BertTokenizer
)

from RobotCommandParser.ClassificationUtils.MyMultilabel import MyMultiLabelClassificationArgs, MyBertForMultiLabelSequenceClassification


class ClassifierWrapper:
    def __init__(self, config):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.onehotenc = None
        with open("Data/labels_names.json", "r") as f:
            self.labels_names = json.load(f)
        if config['use_gpu']:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                raise ValueError("В конфиге указан флаг использования гпу, но torch не обнаружил доступных гпу")
        else:
            self.device = "cpu"

    def load_model(self):
        model_args = MyMultiLabelClassificationArgs()
        num_labels = sum(self.config["Model"]['num_sublabels_per_biglabel'])
        bertconfig = BertConfig.from_pretrained(
            self.config["Model"]["model_path"],
            num_labels=num_labels, **model_args.config)
        self.tokenizer = BertTokenizer.from_pretrained(
            self.config["Model"]["model_path"],
            do_lower_case=model_args.do_lower_case)
        self.model = MyBertForMultiLabelSequenceClassification.from_pretrained(
            self.config["Model"]["model_path"],
            bertconfig,
            num_sublabels_per_biglabel=self.config["Model"]["num_sublabels_per_biglabel"],
            add_attention_for_labels=self.config["Model"]["add_attention_for_labels"]
        )
        self.model.eval()
        self.onehotenc = joblib.load(self.config["Model"]["model_path"] + "/onehotenc.joblib")
        self.model.to(self.device)

        logging.info("Модель загружена успешно")

    def predict(self, phrases):
        """

        :param phrases: list - список строк (фраз комманд)
        :return:
            parse_output_list: list - список списков пар лейбл-класс
        """
        if self.model is None:
            raise ValueError("Model is not loaded. Call ClassifierWrapper().load_model() before making any predictions")
        with torch.no_grad():
            model_inputs = self.tokenizer.batch_encode_plus(
                phrases, return_tensors="pt", padding=True, truncation=True
            )
            model_inputs = {key: value.to(self.device) for key, value in model_inputs.items()}
            logits = self.model(**model_inputs)
        logits = logits[0].cpu()
        # превращаем логитсы для one-hot кодированных лейблов в дормальный мальтилейбл+мультикласс
        predictions = np.zeros((logits.shape[0], len(self.onehotenc.categories_)), dtype=np.int16)
        for i in range(predictions.shape[0]):
            shift = 0
            for j in range(len(self.onehotenc.categories_)):
                predictions[i, j] = np.argmax(logits[i, shift:shift + len(self.onehotenc.categories_[j])])
                shift += len(self.onehotenc.categories_[j])
            assert shift == logits.shape[1]

        # превращаем в пары тюплов
        parse_output_list = []
        for phrase_i in range(predictions.shape[0]):
            parse_output_list.append([])
            for label_i in range(predictions.shape[1]):
                label_name = self.config["Model"]["target_labels"][label_i]
                class_name = self.labels_names[label_name][predictions[phrase_i, label_i]]
                if class_name=="":
                    continue
                parse_output_list[-1].append((label_name, class_name))
        return parse_output_list
