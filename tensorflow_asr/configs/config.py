# Copyright 2020 Huy Le Nguyen (@usimarit)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from . import load_yaml
from ..augmentations.augments import Augmentation
from ..utils.utils import preprocess_paths


class DatasetConfig:
    def __init__(self, config: dict = None):
        if not config: config = {}
        self.train_paths = config.get("train_paths", None)
        self.eval_paths = config.get("eval_paths", None)
        self.test_paths = config.get("test_paths", None)
        self.tfrecords_dir = config.get("tfrecords_dir", None)


class RunningConfig:
    def __init__(self, config: dict = None):
        if not config: config = {}
        self.batch_size = config.get("batch_size", 1)
        self.accumulation_steps = config.get("accumulation_steps", 1)
        self.num_epochs = config.get("num_epochs", 20)
        self.outdir = preprocess_paths(config.get("outdir", None))
        self.log_interval_steps = config.get("log_interval_steps", 500)
        self.save_interval_steps = config.get("save_interval_steps", 500)
        self.eval_interval_steps = config.get("eval_interval_steps", 1000)


class LearningConfig:
    def __init__(self, config: dict = None):
        if not config: config = {}
        self.augmentations = Augmentation(config.get("augmentations"))
        self.dataset_config = DatasetConfig(config.get("dataset_config"))
        self.optimizer_config = config.get("optimizer_config", {})
        self.running_config = RunningConfig(config.get("running_config"))


class Config:
    """ User config class for training, testing or infering """

    def __init__(self, path: str, learning: bool):
        config = load_yaml(preprocess_paths(path))
        self.speech_config = config.get("speech_config", {})
        self.decoder_config = config.get("decoder_config", {})
        self.model_config = config.get("model_config", {})
        if learning:
            self.learning_config = LearningConfig(config.get("learning_config"))
