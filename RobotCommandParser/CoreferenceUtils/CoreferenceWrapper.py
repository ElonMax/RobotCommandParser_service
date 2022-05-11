import re
import logging
import tensorflow as tf
import ufal.udpipe as udpipe
from TextCorporaReaders import utils as tcrutils
from TextCorporaReaders.Readers.jsons.CorefJsonlines import KentonCorefJsonline, FBCorefJsonline
import TextCorporaReaders.Readers.TSV as TSV
from RobotCommandParser.CoreferenceUtils import CorefModel


class CoreferenceWrapper:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.session = None
        self.fb_reader = FBCorefJsonline()
        self.tsv_reader = TSV.TSVReader(TSV.COLUMNS_CONLL, TSV.DTYPES_CONLL)

    def load_model(self):
        self.model = CorefModel.CorefModel(self.config["Model"])
        session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        session_config.gpu_options.allow_growth = True
        self.session = tf.Session(config=session_config)
        self.model.restore(self.session)

        self.udpipe_model = udpipe.Model.load(self.config["udpipe_model_path"])
        self.udpipe_pipeline = udpipe.Pipeline(self.udpipe_model, "tokenize", "tag", "parse", "")

    def predict(self, phrases):
        """

        :param phrases:
        :return:
        """
        if type(phrases) == str:
            phrases = [phrases]
        logging.debug("received phrases:"+str(phrases))
        phrases_preprocessed = self.preprocess_phrases(phrases)
        logging.debug("preprocessed phrases:" + str(phrases_preprocessed))
        for phraseData in phrases_preprocessed:
            tensorized_example = self.model.tensorize_example(phraseData, is_training=False)
            feed_dict = {i: t for i, t in zip(self.model.input_tensors, tensorized_example)}
            top_span_starts = []
            top_span_ends = []
            top_antecedents = []
            top_antecedent_scores = [[0]]
            try:
                _, _, _, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores = self.session.run(
                    self.model.predictions, feed_dict=feed_dict)
            except ValueError as ve:
                logging.error("Can't process example {}".format(str(phraseData)))

            predicted_antecedents = self.model.get_predicted_antecedents(top_antecedents, top_antecedent_scores)
            phraseData["clusters"], _ = self.model.get_predicted_clusters(top_span_starts, top_span_ends,
                                                                                predicted_antecedents)
            phraseData["top_spans"] = list(zip((int(i) for i in top_span_starts), (int(i) for i in top_span_ends)))
            phraseData['head_scores'] = []
        phrases_postprocessed = self.postprocess_phrases(phrases_preprocessed, phrases)
        logging.debug("postprocessed phrases:" + str(phrases_postprocessed))
        output_clusters = []
        for p_i in range(len(phrases)):
            output_clusters.append([])
            for cluster in phrases_postprocessed[p_i]["coreference"]["clusters"]:
                output_clusters[-1].append([])
                for m_i in cluster:
                    output_clusters[-1][-1].append({
                        "startPos": phrases_postprocessed[p_i]["coreference"]["mentions"][m_i]["startPos"],
                        "endPos": phrases_postprocessed[p_i]["coreference"]["mentions"][m_i]["endPos"]
                    })
        return output_clusters

    def preprocess_phrases(self, phrases):
        preprocessed_phrases = []
        for p_i, phrase in enumerate(phrases):
            docData = {
                "meta": {"fileName": "doc_{}".format(p_i), "docName": "doc_{}".format(p_i)},
                "raw": phrase,
                "coreference": {
                    "clusters": [],
                    "mentions": []
                }
            }
            text_extraSpaces = re.sub(r"([!?*/)(.,;\"\-'«»])", " \\1 ", phrase)
            conllu = self.udpipe_pipeline.process(text_extraSpaces)
            udpipeData = self.tsv_reader.read(conllu)
            docData["sentences"] = udpipeData["sentences"]
            tcrutils.addTokensPositions(docData)
            tcrutils.projectCorefPosToTokens(docData)

            docData["sentences"] = [s["tokens"] for s in docData["sentences"]]
            docData["objects"] = {"NER": []}
            for sent in docData["sentences"]:
                for t_i, t in enumerate(sent):
                    if t_i == 0:
                        t["constituent"] = "(TOP*"
                    elif t_i == (len(sent) - 1):
                        t["constituent"] = "*)"
                    else:
                        t["constituent"] = "*"
                    t["speaker"] = ""
                    t["UPOS"] = t["upos"]
                    if t["UPOS"] == "PRON":
                        t["UPOS"] = "PRP"

            docData, subtoken_map = self.fb_reader.makeSubtokens(docData, self.model.tokenizer)
            docData, subtoken_map, sentence_map = self.fb_reader.independent_segmentation(docData, subtoken_map,
                                                                                     self.config["Model"]['max_segment_len'])
            fb_jsonline = self.fb_reader.sag_to_fb(docData, subtoken_map, sentence_map)
            preprocessed_phrases.append(fb_jsonline)
        return preprocessed_phrases

    def postprocess_phrases(self, phrases_preprocessed, phrases):
        postprocessed_phrases = []
        for p_i, phraseData in enumerate(phrases_preprocessed):
            phraseData, subtoken_map, sentence_map = self.fb_reader.fb_to_sag(phraseData)
            phraseData = self.fb_reader.DeMapTokens(phraseData, subtoken_map, sentence_map)
            phraseData["raw"] = phrases[p_i]
            for s_i, sent in enumerate(phraseData["sentences"]):
                phraseData["sentences"][s_i] = {"tokens": sent}
            tcrutils.addTokensPositions(phraseData)
            tcrutils.projectCorefTokensToPos(phraseData)
            postprocessed_phrases.append(phraseData)
        return postprocessed_phrases

    def __del__(self):
        self.session.close()
