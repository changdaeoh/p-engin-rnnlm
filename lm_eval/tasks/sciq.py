"""
Crowdsourcing Multiple Choice Science Questions
https://aclanthology.org/W17-4413.pdf

The SciQ dataset contains 13,679 crowdsourced science exam questions about Physics,
Chemistry and Biology, among others. The questions are in multiple-choice format
with 4 answer options each. For the majority of the questions, an additional paragraph
with supporting evidence for the correct answer is provided.

Homepage: https://allenai.org/data/sciq
"""
from lm_eval.base import MultipleChoiceTask


_CITATION = """
@inproceedings{Welbl2017CrowdsourcingMC,
    title={Crowdsourcing Multiple Choice Science Questions},
    author={Johannes Welbl and Nelson F. Liu and Matt Gardner},
    booktitle={NUT@EMNLP},
    year={2017}
}
"""


class SciQ(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "sciq"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        choices = [
            doc["distractor1"],
            doc["distractor2"],
            doc["distractor3"],
            doc["correct_answer"],
        ]
        src = doc["support"]
        out_doc = {
            "source": src,
            "query": doc["question"],
            "choices": choices,
            "gold": 3,
        }
        return out_doc

    def doc_to_text(self, doc):
        cand_str = ', '.join([ f'{i+1}) {c}' for i, c in enumerate(doc['choices'])])
        #! default prompt (D/Q/A)
        if self.prompt_type == 0: return f"{doc['source']}\nQuestion: {doc['query']}\nAnswer:".strip()
        #* junyong's original
        # prompt = f"In the earth Which is, natural? Description:{doc['support']}, {doc['question']}, Choices:{doc['choices']}"
        # prompt = f"{doc['support']}, What is the correct answer to this question: '{question}'? Choices: {answer_candidates}."
        # prompt = f"I am very good student who study science. I am solving the quiz problem {doc['support']} \n Question: {doc['question']} \n Choices: {doc[’answer_choice’]}"
        # prompt = f"I am graduate student who writing paper. I am arguing the logical proble. Help Me. {doc['support']} \n Question: {doc['question']} \n Choices: {doc[’answer_choice’]}"
        # Question - Document - Answer
        # prompt = f"Based on descriptions, which choice is the best fit. Question:{doc['question']}, Description:{doc['support']}, Choices:{doc['choices']}"
        # Question - Document - Question - Answer
        # prompt = f"Based on descriptions, which choice is the best fit. Question:{doc['question']}, Description:{doc['support']},Question:{doc['question']}, Choices:{doc['choices']}"
        # Question - Answer - Docuemnt -Question - Answer
        # f"Based on descriptions, which choice is the best fit. Question:{doc['question']}, Choices:{doc['choices']}, Description:{doc['support']},Question:{doc['question']}, Choices:{doc['choices']}"
        # Docuement - Docuemnt - Question - Answer
        # f"Based on descriptions, which choice is the best fit. Description:{doc['support']},Description:{doc['support']}, Question:{doc['question']}, Choices:{doc['choices']}"
        # Question - Docuement - Docuemnt - Question - Answer
        # f"Based on descriptions, which choice is the best fit. Question:{doc['question']}, Description:{doc['support']}, Description:{doc['support']}, Question:{doc['question']}, Choices:{doc['choices']}"
        #* changdae's
        #! D/Q/C/A prompt
        elif self.prompt_type == 1: return f"{doc['source']}\nQuestion: {doc['query']}\nAnswer candidates: {cand_str[:-2]}\nAnswer:"
        #! P/D/Q/A prompt
        elif self.prompt_type == 2: return f"I am a high school student who has a good background in science. I will answer a given question referring to a description from now.\nDescription: {doc['source']}\nQuestion: {doc['query']}\nAnswer:"
        #! P/D/Q/C/A prompt
        elif self.prompt_type == 3: return f"I am a high school student who has a good background in science. I will answer a given question referring to a description from now.\nDescription: {doc['source']}\nQuestion: {doc['query']}\nAnswer candidates: {cand_str[:-2]}\nAnswer:"
        #! T/D/Q/A prompt
        elif self.prompt_type == 4: return f"For a given science exam question like \'{doc['query']}\', you can refer a description like \'{doc['source']}\' and should answer like this, Answer:".strip()
        #! T/D/Q/C/A prompt
        elif self.prompt_type == 5: return f"For a given science exam question like \'{doc['query']}\', you can refer a description like \'{doc['source']}\' and should pick an answer among candidates {cand_str[:-2]} like this, Answer:".strip()
        #* junyong's order/iteration-based investigation will be addressed in the future work
        else: raise ValueError(f'invalid prompt specification: ptype {self.prompt_type}')

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["source"] + " " + doc["query"]