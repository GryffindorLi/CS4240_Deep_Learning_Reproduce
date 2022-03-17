from typing import List

from pet.pvp import PVP, PVPS
from pet.utils import InputExample
from itertools import combinations

def _to_verbalizer(labels, n):
        ret = {}
        cnt = 1
        for i in range(1, n + 1):
            combs = combinations(labels, i)
            for comb in combs:
                ret[str(cnt)] = list(comb)
                cnt += 1
        return ret, cnt

class MFTC_PVP(PVP):
    """
    MFTC pattern-verbalizer pair (PVP).
    """

    # Set this to the name of the task
    TASK_NAME = "MFTC"

    # Set this to the verbalizer for the given task: a mapping from the task's labels (which can be obtained using
    # the corresponding DataProcessor's get_labels method) to tokens from the language model's vocabulary
    BASIC_LABEL = ["fairness", "non-moral", "purity", "degradation", "loyalty", "care", "cheating", \
        "betrayal", "subversion", "authority", "harm"]
    
    VERBALIZER, NUM_LABELS = _to_verbalizer(BASIC_LABEL, 4)
    

    def get_parts(self, example: InputExample):
        """
        This function defines the actual patterns: It takes as input an example and outputs the result of applying a
        pattern to it. To allow for multiple patterns, a pattern_id can be passed to the PVP's constructor. This
        method must implement the application of all patterns.
        """

        # We tell the tokenizer that both text_a and text_b can be truncated if the resulting sequence is longer than
        # our language model's max sequence length.
        text_a = self.shortenable(example.text_a)
        text_b = self.shortenable(example.text_b)

        # For each pattern_id, we define the corresponding pattern and return a pair of text a and text b (where text b
        # can also be empty).
        if self.pattern_id == 0:
            # this corresponds to the pattern [MASK]: a b
            return [text_a, "This is about", self.mask * 4], []
        elif self.pattern_id == 1:
            # this corresponds to the pattern [MASK] News: a || (b)
            return ["What is the next sentence about", text_a, self.mask * 4], []
        elif self.pattern_id == 2:
            return [text_a, "The previous is about", self.mask * 4], []
        elif self.pattern_id == 3:
            return ["What is this for", text_a, self.mask * 4], []
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))

    def verbalize(self, label) -> List[str]:
        return MFTC_PVP.VERBALIZER[label]


# register the PVP for this task with its name
PVPS[MFTC_PVP.TASK_NAME] = MFTC_PVP

if __name__ == "__main__":
    basic_label = ["fairness", "non-moral", "purity", "degradation", "loyalty", "care", "cheating", \
        "betrayal", "subversion", "authority", "harm"]
    v, n = _to_verbalizer(basic_label, 4)
    print(n)
    print(v)