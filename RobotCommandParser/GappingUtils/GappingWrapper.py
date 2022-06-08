import torch
from RobotCommandParser.GappingUtils.agrr.stuff import get_model, get_tokenizer
from RobotCommandParser.GappingUtils.agrr.data_utils import to_examples, load_csv, to_result, to_tensor_data
from RobotCommandParser.GappingUtils.agrr.tokenization import BertTokenizer
from copy import deepcopy


class GappingWrapper:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        if config['use_gpu']:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                raise ValueError("В конфиге указан флаг использования гпу, но torch не обнаружил доступных гпу")
        else:
            self.device = "cpu"

    def load_model(self):
        self.model = get_model(self.config)
        self.model.load_state_dict(
            torch.load(self.config["checkpoint_path"], map_location=self.device))
        self.model.eval()
        self.tokenizer = get_tokenizer(self.config)

    def predict(self, commands):
        if type(commands) == str:
            commands = [commands]
        commands_with_fillers = []
        for command in commands:
            commands_with_fillers.append({"text": command,
                                          'class': 0, 'cV': '', 'cR1': '', 'cR2': '', 'V': '', 'R1': '', 'R2': ''})
        examples = to_examples(commands_with_fillers, self.tokenizer, self.config["max_seq_length"])
        token_ids, masks, tag_ids, missed_ids, label_ids, idxs = to_tensor_data(examples)
        input_dict = {"input_ids": token_ids.to(self.model.device), "attention_mask": masks.to(self.model.device)}
        with torch.no_grad():
            logits = self.model(**input_dict)
        sentence_logits, gap_resolution_logits, full_annotation_logits = logits
        sentence_logits = sentence_logits.detach().cpu().numpy()
        gap_resolution_logits = gap_resolution_logits.detach().cpu().numpy()
        full_annotation_logits = full_annotation_logits.detach().cpu().numpy()

        results = []
        for sl, gl, fl, i in zip(sentence_logits, gap_resolution_logits, full_annotation_logits, idxs):
            results.append(to_result(sl, gl, fl, i))

        for i in range(len(commands)):
            if results[i]["class"] == '0':
                continue
            # не уверен, может ли cV иметь несколько
            verbphrase = []
            for pair in results[i]["cV"].split(" "):
                start, end = pair.split(":")
                verbphrase.append(commands[i][int(start):int(end)])
            verbphrase = " ".join(verbphrase)
            Vpositions = []
            for pair in set(results[i]["V"].split(" ")):
                Vpositions.append(int(pair.split(":")[0]))
            newtext = deepcopy(commands[i])
            for vpos in sorted(Vpositions, reverse=True):
                newtext = newtext[:vpos] + " " + verbphrase + " " + newtext[vpos:]
            commands[i] = newtext
        return commands