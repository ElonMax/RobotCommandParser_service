# https://github.com/ivbelkin/AGRR_2019
import logging
import torch

from RobotCommandParser.GappingUtils.agrr.models import BertAgrrModel

from transformers.models.bert.modeling_bert import BertForSequenceClassification
from transformers.models.bert.modeling_bert import BertForTokenClassification
from RobotCommandParser.GappingUtils.agrr.tokenization import BertTokenizer
# from transformers.optimization import AdamW
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE


def get_tokenizer(conf):
    tokenizer = BertTokenizer.from_pretrained(
        conf["bert_model"],
        do_lower_case=conf["do_lower_case"]
    )
    return tokenizer

def get_device(conf):
    if not conf["use_gpu"]:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_model(conf):
    if conf["task_name"] == "classification":
        model = BertForSequenceClassification.from_pretrained(
            conf["bert_model"],
            cache_dir=PYTORCH_PRETRAINED_BERT_CACHE,
            num_labels=2
        )
    elif conf["task_name"] == "tagging":
        model = BertForTokenClassification.from_pretrained(
            conf["bert_model"],
            cache_dir=PYTORCH_PRETRAINED_BERT_CACHE,
            num_labels=7
        )
    elif conf["task_name"] == "agrr":
        model = BertAgrrModel.from_pretrained(
            conf["bert_model"],
            cache_dir=PYTORCH_PRETRAINED_BERT_CACHE
        )
    else:
        raise NotImplementedError
    device = get_device(conf)
    logger = logging.getLogger("console")
    logger.info("Using " + device.upper() + " device")
    logger.info(
        f"Number of parameters: {sum([p.nelement() for p in model.parameters()])}",
    )
    model.to(device)
    return model

