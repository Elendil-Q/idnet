from omegaconf import OmegaConf
import hydra
from hydra import initialize, compose

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), 'loader'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'tests'))

os.environ['HYDRA_FULL_ERROR'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from utils.trainer import Trainer
from utils.validation import Validator


def test(trainer):
    test_cfg = compose(config_name="validation/dsec_test",
                       overrides=[]).validation
    Validator.get_test_type("dsec")(test_cfg).execute_test(
        trainer.model, save_all=False)


def test_co(trainer):
    test_cfg = compose(config_name="validation/dsec_co",
                       overrides=[]).validation
    Validator.get_test_type("dsec", "co")(
        test_cfg).execute_test(trainer.model, save_all=False)


@hydra.main(config_path="config", config_name="tid_eval")
# @hydra.main(config_path="config", config_name="id_eval")
def main(config):
    print(OmegaConf.to_yaml(config))

    trainer = Trainer(config)

    print("Number of parameters: ", sum(p.numel()
                                        for p in trainer.model.parameters() if p.requires_grad))

    if config.model.name == "RecIDE":
        test_co(trainer)
    elif config.model.name == "IDEDEQIDO":
        test(trainer)


if __name__ == '__main__':
    main()
