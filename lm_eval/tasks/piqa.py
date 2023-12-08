"""
PIQA: Reasoning about Physical Commonsense in Natural Language
https://arxiv.org/pdf/1911.11641.pdf

Physical Interaction: Question Answering (PIQA) is a physical commonsense
reasoning and a corresponding benchmark dataset. PIQA was designed to investigate
the physical knowledge of existing models. To what extent are current approaches
actually learning about the world?

Homepage: https://yonatanbisk.com/piqa/
"""
from lm_eval.base import MultipleChoiceTask


_CITATION = """
@inproceedings{Bisk2020,
    author = {Yonatan Bisk and Rowan Zellers and
            Ronan Le Bras and Jianfeng Gao
            and Yejin Choi},
    title = {PIQA: Reasoning about Physical Commonsense in
           Natural Language},
    booktitle = {Thirty-Fourth AAAI Conference on
               Artificial Intelligence},
    year = {2020},
}
"""


class PiQA(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "piqa"
    DATASET_NAME = None
    # prompt_type is define at root class Task() in lm_eval > base.py

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def _process_doc(self, doc):
        out_doc = {
            "goal": doc["goal"],
            "choices": [doc["sol1"], doc["sol2"]],
            "gold": doc["label"],
        }
        return out_doc
    
    #! For prompt engineering, we need to modify this classmethod.
    #! 1) we can directly modify this part (on here, or on calling scripts)
    #! 2) we can define descriptions (json file, specified via run-argument) for each task
    # def doc_to_text(self, doc):
    #     return "Question: " + doc["goal"] + "\nAnswer:"

    def doc_to_text(self, doc):
        #! vanilla Q/A prompt
        if   self.prompt_type == 0: return "Question: " + doc["goal"] + "\nAnswer:"
        #* jooeon's original
        #! propmt1: 0.642 -> 0.577 (accruacy drop)
        # elif self.prompt_type == 1: return f"Goal: {doc['goal']}. Solutions: 1) {doc['choices'][0]}, 2) {doc['choices'][1]}. Choose the best solution and explain why. Answer:"
        # elif self.prompt_type == 2: return f"What is the best way to {doc['goal']}? Is it 1) {doc['sol1']} or 2) {doc['sol2']}? Justify your choice. Answer:"
        # elif self.prompt_type == 3: return f"For the goal of '{doc['goal']}', which solution is more effective? Option 1: {doc['sol1']} or Option 2: {doc['sol2']}? Provide your reasoning. Answer:"
        # elif self.prompt_type == 4: return f"Considering the goal to '{doc['goal']}', which of these solutions would work better? A) {doc['sol1']}, B) {doc['sol2']}. Explain your selection. Answer:"
        # elif self.prompt_type == 5: return f"Goal: {doc['goal']}. Which solution is more practical and why? 1) {doc['sol1']}, 2) {doc['sol2']}. Answer:"
        # elif self.prompt_type == 6: return f"To achieve the goal of '{doc['goal']}', should one opt for 1) {doc['sol1']} or 2) {doc['sol2']}? State your arguments. Answer:"
        # elif self.prompt_type == 7: return f"With the objective of '{doc['goal']}', which approach is advisable? First: {doc['sol1']}, Second: {doc['sol2']}. Discuss your preference. Answer:"
        # elif self.prompt_type == 8: return f"Given the goal of '{doc['goal']}', which solution is more appropriate: 1) {doc['sol1']} or 2) {doc['sol2']}? Explain your choice. Answer:"
        # elif self.prompt_type == 9: return f"Goal: {doc['goal']}. Evaluate the two solutions: 1) {doc['sol1']}, 2) {doc['sol2']}. Which one aligns better with the goal? Answer:"
        # elif self.prompt_type == 10: return f"When trying to {doc['goal']}, which of these is the optimal solution and why? Option 1: {doc['sol1']}, Option 2: {doc['sol2']}. Answer:"
        #* changdae's
        #! Q/A prompt (style variation => excepted from analysis)
        #elif self.prompt_type == 1: return f"Given the goal \'{doc['goal']}\', you can"
        #! Q/C/A prompt
        elif self.prompt_type == 1: return f"Question: {doc['goal']}. \nAnswer candidates: 1) {doc['choices'][0]}, 2) {doc['choices'][1]}. \nAnswer:"
        #! P/Q/A prompt
        elif self.prompt_type == 2: return f"I am a physics professor with a strong understanding of physical commonsense. I will answer a given physics question from now. \nQuestion: {doc['goal']} \nAnswer:"
        #! P/Q/C/A prompt
        elif self.prompt_type == 3: return f"I am a physics professor with a strong understanding of physical commonsense. I will answer a given physics question from now. \nQuestion: {doc['goal']}. \nAnswer candidates: 1) {doc['choices'][0]}, 2) {doc['choices'][1]}. \nAnswer:"
        #! T/Q/A prompt
        elif self.prompt_type == 4: return f"For a given question requiring physical commonsense reasoning like \'{doc['goal']}\', you should answer like: "
        #! T/Q/C/A prompt
        elif self.prompt_type == 5: return f"For a given question requiring physical commonsense reasoning like \'Question: {doc['goal']}\', you should pick an answer across two candidates 1) {doc['choices'][0]} and 2) {doc['choices'][1]} like this, Answer:"
        else: raise ValueError(f'invalid prompt specification: ptype {self.prompt_type}')

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["goal"]
