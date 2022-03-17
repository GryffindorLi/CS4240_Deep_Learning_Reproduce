import csv
import os
from typing import List

from pet.task_helpers import MultiMaskTaskHelper
from pet.tasks import DataProcessor, PROCESSORS, TASK_HELPERS
from pet.utils import InputExample

class MFTC_Processor(DataProcessor):
    """
    MFTC data processor.
    """

    # Set this to the name of the task
    TASK_NAME = "MFTC"

    # Set this to the name of the file containing the train examples
    TRAIN_FILE_NAME = "labeled.csv"

    # Set this to the name of the file containing the dev examples
    DEV_FILE_NAME = "dev.csv"

    # Set this to the name of the file containing the test examples
    TEST_FILE_NAME = "test.csv"

    # Set this to the name of the file containing the unlabeled examples
    UNLABELED_FILE_NAME = "unlabeled.csv"

    # Set this to a list of all labels in the train + test data
    LABELS = ["fairness", "non-moral", "purity", "degradation", "loyalty", 
              "care", "cheating", "betrayal", "subversion", "authority", "harm"]

    # Set this to the column of the train/test csv files containing the input's text a
    TEXT_A_COLUMN = "text"

    # Set this to the column of the train/test csv files containing the input's text b or to -1 if there is no text b
    TEXT_B_COLUMN = -1

    # Set this to the column of the train/test csv files containing the input's gold label
    LABEL_COLUMN = ["fairness", "non-moral", "purity", "degradation", "loyalty", 
              "care", "cheating", "betrayal", "subversion", "authority", "harm"]

    def get_train_examples(self, data_dir: str) -> List[InputExample]:
        """
        This method loads train examples from a file with name `TRAIN_FILE_NAME` in the given directory.
        :param data_dir: the directory in which the training data can be found
        :return: a list of train examples
        """
        return self._create_examples(os.path.join(data_dir, MFTC_Processor.TRAIN_FILE_NAME), "train")

    def get_dev_examples(self, data_dir: str) -> List[InputExample]:
        """
        This method loads dev examples from a file with name `DEV_FILE_NAME` in the given directory.
        :param data_dir: the directory in which the dev data can be found
        :return: a list of dev examples
        """
        return self._create_examples(os.path.join(data_dir, MFTC_Processor.DEV_FILE_NAME), "dev")

    def get_test_examples(self, data_dir) -> List[InputExample]:
        """
        This method loads test examples from a file with name `TEST_FILE_NAME` in the given directory.
        :param data_dir: the directory in which the test data can be found
        :return: a list of test examples
        """
        return self._create_examples(os.path.join(data_dir, MFTC_Processor.TEST_FILE_NAME), "test")

    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        """
        This method loads unlabeled examples from a file with name `UNLABELED_FILE_NAME` in the given directory.
        :param data_dir: the directory in which the unlabeled data can be found
        :return: a list of unlabeled examples
        """
        return self._create_examples(os.path.join(data_dir, MFTC_Processor.UNLABELED_FILE_NAME), "unlabeled")

    def get_labels(self) -> List[str]:
        """This method returns all possible labels for the task."""
        return MFTC_Processor.LABELS

    def _create_examples(self, path, set_type, max_examples=-1, skip_first=0):
        """Creates examples for the training and dev sets."""
        examples = []

        with open(path) as f:
            reader = csv.DictReader(f, delimiter=',')
            for idx, row in enumerate(reader):
                guid = "%s-%s" % (set_type, idx)

                label = []
                for l in self.LABEL_COLUMN:
                    if row[l] == "1":
                        label.append(l)
                
                text_a = row[MFTC_Processor.TEXT_A_COLUMN]
                text_b = row[MFTC_Processor.TEXT_B_COLUMN] if MFTC_Processor.TEXT_B_COLUMN >= 0 else None

                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
                examples.append(example)

        return examples


# register the processor for this task with its name
PROCESSORS[MFTC_Processor.TASK_NAME] = MFTC_Processor

# optional: if you have to use verbalizers that correspond to multiple tokens, uncomment the following line
TASK_HELPERS[MFTC_Processor.TASK_NAME] = MultiMaskTaskHelper

if __name__ == "__main__":
    p = MFTC_Processor()
    train = p.get_train_examples("data")
    for i in range(10):
      print(train[i])