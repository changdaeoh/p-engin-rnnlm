"""
Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge
https://arxiv.org/pdf/1803.05457.pdf

The ARC dataset consists of 7,787 science exam questions drawn from a variety
of sources, including science questions provided under license by a research
partner affiliated with AI2. These are text-only, English language exam questions
that span several grade levels as indicated in the files. Each question has a
multiple choice structure (typically 4 answer options). The questions are sorted
into a Challenge Set of 2,590 “hard” questions (those that both a retrieval and
a co-occurrence method fail to answer correctly) and an Easy Set of 5,197 questions.

Homepage: https://allenai.org/data/arc
"""
from lm_eval.base import MultipleChoiceTask


_CITATION = """
@article{Clark2018ThinkYH,
  title={Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge},
  author={Peter Clark and Isaac Cowhey and Oren Etzioni and Tushar Khot and Ashish Sabharwal and Carissa Schoenick and Oyvind Tafjord},
  journal={ArXiv},
  year={2018},
  volume={abs/1803.05457}
}
"""


class ARCEasy(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "ai2_arc"
    DATASET_NAME = "ARC-Easy"

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
        # NOTE: Some `doc["answerKey"]`s are in numeric string format being one
        # of {'1', '2', '3', '4', '5'}. We map them back to letters.
        self.num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
        doc["answerKey"] = self.num_to_letter.get(doc["answerKey"], doc["answerKey"])
        out_doc = {
            "id": doc["id"],
            #! we should modify this part
            #"query": "Question: " + doc["question"] + "\nAnswer:",
            "question": doc["question"],
            #!
            "choices": doc["choices"]["text"],
            "gold": ["A", "B", "C", "D", "E"].index(doc["answerKey"]),
        }
        return out_doc

    def doc_to_text(self, doc):
        cand_str = ', '.join([ f'{self.num_to_letter[str(i+1)]}) {c}' for i, c in enumerate(doc['choices'])])
        #! vanilla Q/A prompt
        if   self.prompt_type == 0: return "Question: " + doc["question"] + "\nAnswer:"
        #* sangyoon's original
        # prompts = [
        #         f"As an experienced science teacher, I am evaluating the following question: \nQuestion: {doc['question']} \nWhich answer is correct? \nOptions: {doc['answer_candidates']} \nAnswer: {doc['answer_choice']}",
        #         f"As a seasoned researcher, I am analyzing this challenging question: \nQuestion: {doc['question']} \nBased on scientific evidence, what would be the right answer? \nOptions: {doc['answer_candidates']} \nAnswer: {doc['answer_choice']}",
        #         f"As a participant in a scientific dialogue, I'm reflecting on this query: \nQuestion: {doc['question']} \nConsidering the given options, which one is accurate? \nOptions: {doc['answer_candidates']} \nAnswer: {doc['answer_choice']}",
        #         f"During a science tutoring session, I encountered this question: \nQuestion: {doc['question']} \nTo educate effectively, which answer should be chosen? \nOptions: {doc['answer_candidates']} \nAnswer: {doc['answer_choice']}",
        #         f"In a science study group, we've been asked to consider this question: \nQuestion: {doc['question']} \nAfter thorough discussion, which answer seems most plausible? \nOptions: {doc['answer_candidates']} \nAnswer: {doc['answer_choice']}",
        #         f"Hosting a quiz on scientific facts, here's a question for the contestants: \nQuestion: {doc['question']} \nCan you identify the correct answer? \nOptions: {doc['answer_candidates']} \nAnswer: {doc['answer_choice']}",
        #         f"As a quizmaster for today's science round, I'm asking this: \nQuestion: {doc['question']} \nWhat do you think is the right choice? \nOptions: {doc['answer_candidates']} \nAnswer: {doc['answer_choice']}",
        #         f"As a judge in a scientific knowledge competition, I present this question: \nQuestion: {doc['question']} \nWhich of these answers is the winner? \nOptions: {doc['answer_candidates']} \nAnswer: {doc['answer_choice']}",
        #         f"As a collaborator in a scientific inquiry, let's explore the answer to: \nQuestion: {doc['question']} \nGiven our collective expertise, which answer is correct? \nOptions: {doc['answer_candidates']} \nAnswer: {doc['answer_choice']}",
        #         f"Approaching this question as a critical thinker: \nQuestion: {doc['question']} \nWhich of these options holds up to logical scrutiny? \nOptions: {doc['answer_candidates']} \nAnswer: {doc['answer_choice']}",
        #         f"As a physics professor with a very good understanding of physical phenomena, I am answering a given physics question. \nQuestion: {doc['question']} \nOptions: {doc['answer_candidates']} \nAnswer: {doc['answer_choice']}"
        #         ]
        #* changdae
        #! Q/C/A prompt
        elif self.prompt_type == 1: return f"Question: {doc['question']}. \nAnswer candidates: {cand_str[:-2]}. \nAnswer:"
        #! P/Q/A prompt
        elif self.prompt_type == 2: return f"I am an elementary school teacher who has a strong background in science and society. I will answer a given question from now.\nQuestion: {doc['question']}\nAnswer:"
        #! P/Q/C/A prompt
        elif self.prompt_type == 3: return f"I am an elementary school teacher who has a strong background in science and society. I will answer a given question from now.\nQuestion: {doc['question']}.\nAnswer candidates: {cand_str[:-2]}.\nAnswer:"
        #! T/Q/A prompt
        elif self.prompt_type == 4: return f"For a given grade-school level science question like \'{doc['question']}\', you should answer like: "
        #! T/Q/C/A prompt
        elif self.prompt_type == 5: return f"For a given grade-school level science question like \'Question: {doc['question']}\', you should pick an answer across some candidates {cand_str[:-2]} like this, Answer:"
        else: raise ValueError(f'invalid prompt specification: ptype {self.prompt_type}')

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]


class ARCChallenge(ARCEasy):
    DATASET_PATH = "ai2_arc"
    DATASET_NAME = "ARC-Challenge"
