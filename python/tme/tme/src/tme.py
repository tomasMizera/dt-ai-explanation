import logging

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from sumy.summarizers.text_rank import TextRankSummarizer
from lime import lime_text
from tme.src.helper import summary_to_string


class TextModelsExplainer:

    def __init__(
            self,
            modelfn=None,
            classnames=None,
            language="english",
            explainer=None,
            summarizer=None,
            fm=962,
            topfeaturescount=100,
            sentencescount=6,
            logger=None
    ):
        self.fm = fm
        self.modelfn = modelfn
        self.classnames = classnames
        self.topfeaturescount = topfeaturescount
        self.language = language
        self.sentencescount = sentencescount

        if explainer is not None:
            self.explainer = explainer
        else:
            self.explainer = lime_text.LimeTextExplainer(class_names=self.classnames)

        if summarizer is not None:
            self.summarizer = summarizer
        else:
            self.summarizer = TextRankSummarizer(Stemmer(self.language))
            self.summarizer.stop_words = get_stop_words(self.language)

        if logger is not None:
            self.log = logger
        else:
            self.log = logging.getLogger()

    def explanation_summaries(self, instances, fm=None, precomputed_explanations=None, summary_type='str'):
        """
        Creates explanation summaries for all elements in instances
        :param instances: list of instances (1 instace = 1 string) or map of instanceId:instace
        :param fm: factor multiplier to use when boosting specific sentences in summary
        :param precomputed_explanations: optionally, if one already has precomputed explanations
        :return: list or map of summaries (based on input type of instances)
        """

        if fm is not None:
            self.fm = fm
            self._logi(f'Factor Multiplier changed to {self.fm}')

        if type(instances) == dict:
            summaries = {}

            for instance in instances:
                summaries[instance] = self._summarize_doc_custom(instances[instance], summary_type=summary_type)

            return summaries

        elif type(instances) == list:
            summaries = []

            for i, instance in enumerate(instances):
                if precomputed_explanations:
                    e = precomputed_explanations[i]
                    if type(e) != list:
                        e = e.as_list()
                    summaries.append(self._summarize_doc_custom(instance, e, summary_type=summary_type))
                else:
                    summaries.append(self._summarize_doc_custom(instance, summary_type=summary_type))

            return summaries
        else:
            raise ValueError("Unknown data input type for instances in create_e_summaries")

    def simple_summaries(self, instances):
        """
        Creates TextRank summaries for all instances
        :param instances: list of instances or map of type instanceID:instance
        :return: list or map of summaries (based on input type of instances)
        """

        if type(instances) == dict:
            summaries = {}

            for instance in instances:
                summaries[instance] = self._summarize_doc_simple(instances[instance])

            return summaries

        elif type(instances) == list:
            summaries = []

            for instance in instances:
                summaries.append(self._summarize_doc_simple(instance))

            return summaries

    def _summarize_doc_simple(self, instance):
        """
        Creates TextRank summary from instance
        :param instance:
        :return:
        """
        parser = PlaintextParser.from_string(instance, Tokenizer(self.language))
        return summary_to_string(self.summarizer(parser.document, self.sentencescount))

    def _summarize_doc_custom(self, instance, explanation=None, summary_type='str'):
        """
        Creates summary with altered weights based on explanation
        :param instance: text of instance
        :return: tupple of (summary, explanation words list)
        """
        parser = PlaintextParser.from_string(instance, Tokenizer(self.language))
        # generates graph with weights
        graph = self.summarizer.rate_sentences(parser.document)

        # generate explanation
        if not explanation:
            explanation = self.explainer.explain_instance(
                instance,
                self.modelfn,
                num_features=self.topfeaturescount
            ).as_list()

        # iterate over each sentence in textrank graph
        for sentence in graph.keys():
            factor = self.__compute_factor(sentence, explanation)
            graph[sentence] = graph[sentence] * factor
        # noinspection PyProtectedMember
        resulting_summary = self.summarizer._get_best_sentences(parser.document.sentences, self.sentencescount, graph)

        if summary_type == 'raw':
            return resulting_summary, explanation

        return summary_to_string(resulting_summary, '\n'), explanation

    def _logw(self, msg):
        if self.log is not None:
            self.log.warning(msg)

    def _logi(self, msg):
        if self.log is not None:
            self.log.info(msg)

    def _logd(self, msg):
        if self.log is not None:
            self.log.debug(msg)

    def __compute_factor(self, sentence, explanation_words_weight):
        factor = 1.0
        exp_map = dict(explanation_words_weight)
        for word in sentence.words:  # for each word in sentence
            if word in exp_map:  # check if word is in important words list from LIME
                factor += self.fm * abs(exp_map[word])
        return factor
