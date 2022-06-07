import re
import json
import logging

import torch
import pandas as pd
import numpy as np
import joblib
from transformers import (
    BertConfig,
    BertTokenizer
)

from RobotCommandParser.ClassificationUtils.MyMultilabel import MyMultiLabelClassificationArgs, \
    MyBertForMultiLabelSequenceClassification


def softmax(x):
    # e_x = x - np.expand_dims(np.max(x, axis=1), axis=1)
    # return e_x / np.expand_dims(e_x.sum(axis=1), axis=1)
    e_x = np.exp(x)
    return e_x / np.expand_dims(np.sum(e_x, axis=1), axis=1)

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

        self.check_possible_combinations = config["check_possible_combinations"]
        if config["check_possible_combinations"]:
            self.possible_combinations_arr = pd.read_csv(config["possible_combinations_table"]).loc[:,
                                             config["Model"]["target_labels"]].values

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
        if type(phrases) == str:
            phrases = [phrases]
        # модель обучалась на ловеркейсе скорее всего
        phrases = [x.lower() for x in phrases]

        if self.model is None:
            raise ValueError("Model is not loaded. Call ClassifierWrapper().load_model() before making any predictions")
        with torch.no_grad():
            model_inputs = self.tokenizer.batch_encode_plus(
                phrases, return_tensors="pt", padding=True, truncation=True
            )
            model_inputs = {key: value.to(self.device) for key, value in model_inputs.items()}
            logits = self.model(**model_inputs)
        logits = logits[0].cpu()
        # превращаем логитсы для one-hot кодированных лейблов в нормальный мальтилейбл+мультикласс
        if self.check_possible_combinations:
            predictions = self.postprocess_with_rule_for_possible_combinations(logits.numpy())
        else:
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
                if class_name == "":
                    continue
                parse_output_list[-1].append((label_name, class_name))
        return parse_output_list

    def postprocess_with_rule_for_possible_combinations(self, raw_outputs):
        softmax_outputs = np.zeros_like(raw_outputs, dtype=np.float32)
        shift = 0
        for num_sublabels in self.config['Model']['num_sublabels_per_biglabel']:
            softmax_outputs[:, shift:shift + num_sublabels] = softmax(raw_outputs[:, shift:shift + num_sublabels])
            shift += num_sublabels

        predictions = []
        for i in range(len(softmax_outputs)):
            shift = 0
            probs_for_combinations = np.zeros_like(self.possible_combinations_arr, dtype=np.float32)
            # предполагается, что первые значения в выходном векторе из модели - классы экшена
            assert self.config["Model"]["target_labels"][0] == "action"
            probs_for_combinations[:, 0] = np.take(
                softmax_outputs[i, shift:shift + self.config['Model']['num_sublabels_per_biglabel'][0]],
                self.possible_combinations_arr[:, 0])
            shift += self.config['Model']['num_sublabels_per_biglabel'][0]
            # print(probs_for_combinations[:,0])
            maxprob_attribute_classes = [-1]  # -1 for action
            for attribute_i in range(1, len(self.config['Model']['num_sublabels_per_biglabel'])):
                # есть вариант ставить 0 для нулевых классов или наоборот - обратное от максимального класса
                probs = softmax_outputs[i,
                        shift:shift + self.config['Model']['num_sublabels_per_biglabel'][attribute_i]]
                assert np.round(sum(probs), 5) == 1
                zerocls_prob = probs[0]
                nonzerocls_max_prob = np.max(probs[1:])
                # print(zerocls_prob, nonzerocls_max_prob)
                maxprob_attribute_classes.append(np.argmax(probs[1:]) + 1)
                probs_for_combinations[self.possible_combinations_arr[:, attribute_i] == 0, attribute_i] = zerocls_prob
                probs_for_combinations[
                    self.possible_combinations_arr[:, attribute_i] == 1, attribute_i] = nonzerocls_max_prob
                shift += self.config['Model']['num_sublabels_per_biglabel'][attribute_i]

            best_template_i = np.argmax(np.sum(probs_for_combinations, axis=1))
            sample_prediction = self.possible_combinations_arr[best_template_i].copy()
            for i in range(1, len(self.config['Model']['num_sublabels_per_biglabel'])):
                if sample_prediction[i] != 0:
                    sample_prediction[i] = maxprob_attribute_classes[i]
            predictions.append(sample_prediction)
        return np.array(predictions)
