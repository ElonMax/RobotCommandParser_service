import logging

import torch
from simpletransformers.ner import NERModel, NERArgs


class SplitterWrapper:
    def __init__(self, config):
        self.config = config
        self.tokenizer = None
        self.model = None
        if config['use_gpu'] and not torch.cuda.is_available():
            raise ValueError("В конфиге указан флаг использования гпу, но torch не обнаружил доступных гпу")

    def load_model(self):
        model_args = NERArgs()

        self.model = NERModel(
            self.config["Model"]["model_type"],
            self.config["Model"]["model_path"],
            args=model_args,
            labels=self.config["Model"]["labels"],
            use_cuda=self.config['use_gpu']
        )
        #self.model.evaluate()
        #self.model.to(self.device)

        logging.info("Модель загружена успешно")

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
            predictions, raw_outputs = self.model.predict(phrases)
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
