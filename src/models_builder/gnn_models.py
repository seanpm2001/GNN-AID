import hashlib
import importlib.util
import json
import random
from math import ceil
from types import FunctionType
import numpy as np
import torch
import sklearn, sklearn.metrics
from torch.nn.utils import clip_grad_norm
from torch import tensor
import torch.nn.functional as F
from torch.cuda import is_available
from torch_geometric.data import DataLoader

from aux.configs import ModelManagerConfig, ModelModificationConfig, ModelConfig, CONFIG_CLASS_NAME, ConfigPattern, \
    CONFIG_OBJ
from aux.data_info import UserCodeInfo
from aux.utils import import_by_name, FRAMEWORK_PARAMETERS_PATH, model_managers_info_by_names_list, hash_data_sha256, \
    TECHNICAL_PARAMETER_KEY, IMPORT_INFO_KEY, OPTIMIZERS_PARAMETERS_PATH, FUNCTIONS_PARAMETERS_PATH
from aux.declaration import Declare
from explainers.explainer import ProgressBar



class Metric:
    available_metrics = {
        'Accuracy': sklearn.metrics.accuracy_score,
        'F1': sklearn.metrics.f1_score,
        'BalancedAccuracy': sklearn.metrics.balanced_accuracy_score,
        'Recall': sklearn.metrics.recall_score,
        'Precision': sklearn.metrics.precision_score,
        'Jaccard': sklearn.metrics.jaccard_score,
    }

    @staticmethod
    def add_custom(name, compute_function):
        """
        Register a custom metric.
        Example for accuracy:

        >>> Metric.add_custom('accuracy', lambda y_true, y_pred, normalize=False:
        >>>     int((y_true == y_pred).sum()) / (len(y_true) if normalize else 1))

        :param name: name to refer to this metric
        :param compute_function: function which computes metric result:
         f(y_true, y_pred, **kwargs) -> value
        """
        if name in Metric.available_metrics:
            raise NameError(f"Metric '{name}' already registered, use another name")
        Metric.available_metrics[name] = compute_function

    def __init__(self, name, mask, **kwargs):
        """
        :param name: name to refer to this metric
        :param mask: 'train', 'val', 'test', or a bool valued list
        :param kwargs: params used in compute function
        """
        self.name = name
        self.mask = mask
        self.kwargs = kwargs

    def compute(self, y_true, y_pred):
        if self.name in Metric.available_metrics:
            return Metric.available_metrics[self.name](y_true, y_pred, **self.kwargs)

        raise NotImplementedError()


class GNNModelManager:
    """ class of basic functions over models:
    training, one-step training, full training, evaluation, save and load principle
    """
    def __init__(self,
                 manager_config = None,
                 modification: ModelModificationConfig = None):
        """
        :param manager_config: socket to use for sending data to frontend
        :param modification: socket to use for sending data to frontend
        """
        if manager_config is None:
            # raise RuntimeError("model manager config must be specified")
            manager_config = ConfigPattern(
                        _config_class="ModelManagerConfig",
                        _config_kwargs={},
                    )
        elif isinstance(manager_config, ModelManagerConfig):
            manager_config = ConfigPattern(
                _config_class="ModelManagerConfig",
                _config_kwargs=manager_config.to_saveable_dict(),
            )
        # TODO Kirill, write raise Exception
        # else:
        #     raise Exception()

        # if modification is None:
        #     modification = ModelModificationConfig()

        if modification is None:
            # raise RuntimeError("model manager config must be specified")
            modification = ConfigPattern(
                        _config_class="ModelModificationConfig",
                        _config_kwargs={},
                    )
        elif isinstance(modification, ModelModificationConfig):
            modification = ConfigPattern(
                _config_class="ModelModificationConfig",
                _config_kwargs=modification.to_dict(),
            )
        # TODO Kirill, write raise Exception
        # else:
        #     raise Exception()

        self.manager_config = manager_config
        self.modification = modification

        # QUE Kirill do we need to store it? maybe pass when need to
        self.dataset_path = None

        self.gnn = None
        # We do not want store socket because it is not picklable for a subprocess
        self.socket = None
        self.stop_signal = False
        self.stats_data = None  # Stores some stats to be sent to frontend

    def train_model(self, **kwargs):
        pass

    def train_1_step(self, gen_dataset):
        """ Perform 1 step of model training. Can be called unlimited number of times.
        """
        # raise NotImplementedError()
        pass

    def train_full(self, gen_dataset, steps=None, **kwargs):
        """ Perform full cycle of model training. Can be called only once.
        """
        # raise NotImplementedError()
        pass

    def evaluate_model(self, **kwargs):
        pass

    def get_name(self):
        manager_name = self.manager_config.to_saveable_dict()
        # FIXME Kirill, make ModelManagerConfig and remove manager_name[CONFIG_CLASS_NAME]
        manager_name[CONFIG_CLASS_NAME] = self.__class__.__name__
        # for key, value in kwargs.items():
        #     manager_name[key] = value
        # manager_name = dict(sorted(manager_name.items()))
        json_str = json.dumps(manager_name, indent=2)
        return json_str

    def load_model(self, path=None, **kwargs):
        """
        Load model from torch save format
        """
        raise NotImplementedError()

    def save_model(self, path=None):
        """
        Save the model in torch format

        :param path: path to save the model. By default,
        the path is compiled based on the global class variables
        """
        raise NotImplementedError()

    def model_path_info(self):
        path, _ = Declare.models_path(self)
        return path

    def load_model_executor(self, path=None, **kwargs):
        """
        Load executor. Generates the download model path if no other path is specified.
        :param path: path to load the model. By default, the path is compiled based on the global
         class variables
        """

        if path is None:
            gnn_mm_name_hash = self.get_hash()
            model_dir_path, files_paths = Declare.declare_model_by_config(
                dataset_path=self.dataset_path,
                GNNModelManager_hash=gnn_mm_name_hash,
                model_ver_ind=kwargs.get('model_ver_ind') if 'model_ver_ind' in kwargs else
                self.modification.model_ver_ind,
                model_attack_type=self.modification.model_attack_type,
                epochs=self.modification.epochs,
                gnn_name=self.gnn.get_hash()
            )
            path = model_dir_path / 'model'
        else:
            model_dir_path = path

        # TODO Kirill, check default parameters in gnn
        self.load_model(path=path, **kwargs)
        self.gnn.eval()
        return model_dir_path

    def get_hash(self):
        """
        calculates the hash on behalf of the model manager required for storage. The sha256
        algorithm is used.
        """
        gnn_MM_name = self.get_name()
        json_object = json.dumps(gnn_MM_name)
        gnn_MM_name_hash = hash_data_sha256(json_object.encode('utf-8'))
        return gnn_MM_name_hash

    def save_model_executor(self, path=None, gnn_architecture_path=None):
        """
        Save executor, generates paths and prepares all information about the model
        and its parameters for saving

        :param gnn_architecture_path: path to save the architecture of the model,
        by default it forms the path itself.
        :param path: path to save the model. By default,
        the path is compiled based on the global class variables
        """
        if path is None:
            dir_path, files_paths = Declare.models_path(self)
            dir_path.mkdir(exist_ok=True, parents=True)
            path = dir_path / 'model'
            gnn_name_file = files_paths[0]
            gnn_mm_kwargs_file = files_paths[1]
        else:
            gnn_name_file = gnn_architecture_path / f"gnn={self.gnn.get_hash()}.json"
            gnn_mm_kwargs_file = gnn_architecture_path.parent / f"gnn_model_manager={self.get_hash()}.json"
        self.save_model(path)

        with open(gnn_name_file, "w") as f:
            f.write(self.gnn.get_name(obj_name_flag=True))
        with open(gnn_mm_kwargs_file, "w") as f:
            f.write(self.get_name())
        return path.parent

    @staticmethod
    def from_model_path(model_path, dataset_path, **kwargs):
        """
        Use information about model and model manager take gnn model,
        create gnn model manager object and load weights to the save model
        :param model_path: dict with information how create path to the model
        :param dataset_path: path to the dataset
        :param kwargs:
        :return: gnn model manager object and path to the model directory
        """

        model_dir_path, files_paths = Declare.declare_model_by_config(
            dataset_path=dataset_path,
            GNNModelManager_hash=str(model_path['gnn_model_manager']),
            epochs=int(model_path['epochs']) if model_path['epochs'] != 'None' else None,
            model_ver_ind=int(model_path['model_ver_ind']),
            model_attack_type=model_path['model_attack_type'],
            gnn_name=model_path['gnn'],
        )

        gnn_mm_file = files_paths[1]
        gnn_file = files_paths[0]

        gnn = GNNModelManager.take_gnn_obj(gnn_file=gnn_file)

        modification_config = ModelModificationConfig(
            epochs=int(model_path['epochs']) if model_path['epochs'] != 'None' else None,
            model_attack_type=model_path['model_attack_type'],
            model_ver_ind=int(model_path['model_ver_ind']),
        )

        with open(gnn_mm_file) as f:
            params = json.load(f)
            class_name = params.pop(CONFIG_CLASS_NAME)
            # class_name =
            # manager_config = ModelManagerConfig(**params)
            manager_config = ConfigPattern(**params)
        with open(FRAMEWORK_PARAMETERS_PATH, 'r') as f:
            framework_model_managers_info = json.load(f)
        if class_name in framework_model_managers_info.keys():
            klass = import_by_name(class_name, ["models_builder.gnn_models"])
            gnn_model_manager_obj = klass(
                gnn=gnn,
                manager_config=manager_config,
                modification=modification_config,
                dataset_path=dataset_path, **kwargs)
        else:
            mm_info = model_managers_info_by_names_list({class_name})
            klass = import_by_name(class_name, [mm_info[class_name][TECHNICAL_PARAMETER_KEY][IMPORT_INFO_KEY]])
            gnn_model_manager_obj = klass(
                gnn=gnn,
                manager_config=manager_config,
                dataset_path=dataset_path, **kwargs)

        gnn_model_manager_obj.load_model_executor()

        return gnn_model_manager_obj, model_dir_path

    def get_full_info(self):
        """
        Get available info about model for frontend
        """
        result = {}
        if hasattr(self, 'manager_config'):
            result["manager"] = self.manager_config.to_saveable_dict()
        if hasattr(self, 'modification'):
            result["modification"] = self.modification.to_saveable_dict()
        if hasattr(self, 'epochs'):
            result["epochs"] = f"Epochs={self.epochs}"
        return result

    def get_model_data(self):
        """
        :return: dict with the available functions of the model manager by the 'functions' key.
        """
        model_data = {}

        # Functions
        def get_own_functions(cls):
            return [x for x, y in cls.__dict__.items()
                    if isinstance(y, (FunctionType, classmethod, staticmethod))]

        model_data["functions"] = get_own_functions(type(self))
        return model_data

    @staticmethod
    def take_gnn_obj(gnn_file):
        with open(gnn_file) as f:
            params = json.load(f)
            class_name = params.pop(CONFIG_CLASS_NAME)
            obj_name = params.pop("obj_name")
            gnn_config = ModelConfig(**params)
        user_models_obj_dict_info = UserCodeInfo.user_models_list_ref()
        if class_name == 'FrameworkGNNConstructor':
            from models_builder.gnn_constructor import FrameworkGNNConstructor
            gnn = FrameworkGNNConstructor(gnn_config)
        else:
            if class_name not in user_models_obj_dict_info.keys():
                raise Exception(f"User class {class_name} does not defined")
            else:
                if obj_name is None:
                    try:
                        spec = importlib.util.spec_from_file_location(
                            class_name, user_models_obj_dict_info[class_name]['import_path'])
                        foo = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(foo)
                        gnn_class = getattr(foo, class_name)
                        gnn = gnn_class(gnn_config)
                    except:
                        raise Exception(f"Can't import user class {class_name}")
                else:
                    gnn = UserCodeInfo.take_user_model_obj(user_models_obj_dict_info[class_name]['import_path'],
                                                           obj_name)
        return gnn



