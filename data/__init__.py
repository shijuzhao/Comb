from .Squad import SquadDataset
from .NaturalInstructions import NIDataset
from .SuperNI import SuperNIDataset
from .Xsum import XsumDataset
from .LongBench import *

TRAIN_DATASETS = ["SQuAD", "Natural-Instructions", "XSum", "Super-Natural-Instructions"]
TEST_DATASETS = [HotpotQADataset, MultiNewsDataset, SAMSumDataset,
                MuSiQueDataset, WikiMQADataset]

DATASET_DICT = {
    "SQuAD": SquadDataset,
    "Natural-Instructions": NIDataset,
    "XSum": XsumDataset,
    "Super-Natural-Instructions": SuperNIDataset
}