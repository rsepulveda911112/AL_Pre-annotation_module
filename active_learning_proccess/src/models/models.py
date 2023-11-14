import os

import pandas as pd
import numpy as np
import wandb
from sklearn.model_selection import train_test_split
import sklearn
from small_text.classifiers.factories import AbstractClassifierFactory
from transformers import AutoTokenizer, AutoModel
# from torch.nn.functional import normalize
import torch.nn.functional as F
#from sentence_transformers import SentenceTransformer
import torch
from tqdm.auto import tqdm

#New import
from active_learning_proccess.src.models.ClassificationModel import ClassificationModel


wandb.init(mode="disabled")


class TransformerBasedClassificationFactory(AbstractClassifierFactory):

    def __init__(self, model_type, model_name, use_cuda, labels_len, wandb_project, is_evaluate, best_result_config,
                 is_training, value_head, group, embedding, output_dir, model_args):

        """
        Parameters
        ----------
        transformer_model_args : TransformerModelArguments
            Name of the sentence transformer model.
        num_classes : int
            Number of classes.
        kwargs : dict
            Keyword arguments which will be passed to `TransformerBasedClassification`.
        """
        self.model_type = model_type
        self.model_name = model_name
        self.use_cuda = use_cuda
        self.labels_len = labels_len
        self.wandb_project = wandb_project
        self.is_evaluate = is_evaluate
        self.best_result_config = best_result_config
        self.is_training = is_training
        self.value_head = value_head
        self.group = group
        self.embedding = embedding
        self.output_dir = output_dir
        self.model_args = model_args

    def new(self):
        """Creates a new TransformerBasedClassification instance.

        Returns
        -------
        classifier : TransformerBasedClassification
            A new instance of TransformerBasedClassification which is initialized with the given keyword args `kwargs`.
        """
        return SequenceModel(self.model_type, self.model_name, self.use_cuda, self.labels_len, self.wandb_project,
                 self.is_evaluate, self.best_result_config, self.is_training, self.value_head, self.group,
                 self.embedding, self.output_dir, self.model_args)


class SequenceModel():
    def __init__(self, model_type, model_name, use_cuda, labels_len=None, wandb_project=None,
                 is_evaluate=False, best_result_config=None, is_training=False, value_head=0, group=None, embedding="PlanTL-GOB-ES/roberta-base-bne",
                 output_dir="", model_args=None):
        self.is_evaluate = is_evaluate
        self.init_again = True
        self.group = group
        self.use_cuda = use_cuda
        if embedding:
            self.tokenizer = AutoTokenizer.from_pretrained(embedding, use_fast=False)
            self.embedding_model = AutoModel.from_pretrained(embedding)

        ############ Hyperparameters #####################
        self.model_args = model_args
        self.model_args['evaluate_during_training'] = self.is_evaluate
        self.model_args['wandb_project'] = wandb_project
        self.model_args['output_dir'] = output_dir
        self.model_args['value_head'] = value_head
        weight = None
        if "weight" in model_args:
            weight = model_args["weight"]
        if is_training:
            self.wandb_project = wandb_project
            wandb_config = {}
            sweep_config = {}
            if wandb_project:
                wandb.init(config=wandb.config, project=wandb_project)
                wandb_config = wandb.config
                parse_wandb_param(wandb_config, self.model_args)
            if best_result_config:
                sweep_result = pd.read_csv(os.getcwd() + best_result_config)
                best_params = sweep_result.to_dict()
                print(best_params)
                parse_wandb_param(best_params, self.model_args)
            
            self.model = ClassificationModel(model_type, model_name, num_labels=labels_len, use_cuda=use_cuda,
                                             args=self.model_args, sweep_config=sweep_config, weight=weight)
        else:
            self.model = ClassificationModel(model_type, model_name, use_cuda=use_cuda, args=self.model_args, weight=weight)

    def fit(self, df_train, weights=None, df_eval=None):
        features = [[]] * len(df_train)
        df_train = pd.DataFrame(list(zip(df_train.x, df_train.y, features)), columns=['text', 'labels', 'features'])
        if self.is_evaluate and not isinstance(df_eval, pd.DataFrame):
            df_train, df_eval = train_test_split(df_train, test_size=0.2, train_size=0.8, random_state=1)

        if self.wandb_project and self.init_again:
            wandb.init(config=wandb.config, project=self.wandb_project, reinit=True, group=self.group, job_type="train")
            wandb_config = wandb.config
            parse_wandb_param(wandb_config, self.model_args)
        result = self.model.train_model(df_train, eval_df=df_eval, f1=sklearn.metrics.f1_score,
                                  acc=sklearn.metrics.accuracy_score)
        return result

    def prepare_predict(self, df_test):
        if not isinstance(df_test, pd.DataFrame):
            df_test = pd.DataFrame(list(zip(df_test.x, df_test.y)), columns=['text', 'labels'])
        labels_fic = len(df_test) * [0]
        labels_fic = pd.Series(labels_fic)
        # features = df_test['features']
        features = [[]] * len(df_test)
        features = pd.Series(features)
        if 'text_a' in df_test.columns:
            df_result = pd.concat([df_test['text_a'], df_test['text_b'], labels_fic, features], axis=1)
        else:
            df_result = pd.concat([df_test['text'], labels_fic, features], axis=1)
        value_in = df_result.values.tolist()
        y_predict, model_outputs_test = self.model.predict(value_in)
        return y_predict, model_outputs_test

    def predict(self, df_test, return_proba=False):
        y_predict, model_outputs_test = self.prepare_predict(df_test)
        y_predict = np.argmax(model_outputs_test, axis=1)
        return y_predict

    def predict_proba(self, df_test, return_proba=False):
        y_predict, model_outputs_test = self.prepare_predict(df_test)
        model_outputs_test = F.softmax(torch.from_numpy(model_outputs_test), dim=-1)
        return model_outputs_test.numpy()

    def embed(self, df_test, return_proba=True, pbar="tqdm", **embed_kwargs):
        proba = self.predict_proba(df_test, return_proba)
        print("Embedding tokenizer start")
        last_hidden_states = None
        batches = get_batches(df_test.x, self.tokenizer, batch_size=16)
        self.embedding_model.to('cuda')
        with torch.no_grad():
            batch_iterator = tqdm(
                batches,
                mininterval=0,
            )
            for i, batch_tuple in enumerate(batch_iterator):
                for t in batch_tuple:
                    input_ids = t.to('cuda')
                outputs = self.embedding_model(input_ids=input_ids)

                last_hidden_states_aux = outputs[0]
                last_hidden_states_aux = torch.mean(last_hidden_states_aux, dim=1)
                last_hidden_states_aux = last_hidden_states_aux.cpu()
                last_hidden_states_aux = last_hidden_states_aux.numpy()
                if last_hidden_states is None:
                    last_hidden_states = last_hidden_states_aux
                else:
                    last_hidden_states = np.concatenate([last_hidden_states, last_hidden_states_aux], axis=0)
        return last_hidden_states, proba


def get_batches(sentences, tokenizer, batch_size=2):
    input_ids = encode(sentences, tokenizer)
    tensor_dataset = torch.utils.data.TensorDataset(input_ids)
    tensor_dataloader = torch.utils.data.DataLoader(tensor_dataset, batch_size=batch_size)
    return tensor_dataloader


def encode(data, tokenizer):
    input_ids = []
    for text in data:
        tokenized_text = tokenizer.encode_plus(text,
                                               max_length=512,
                                               add_special_tokens=True,
                                               pad_to_max_length=True,
                                               #padding_side='right',
                                               truncation=True)
        input_ids.append(tokenized_text['input_ids'])
    return torch.tensor(input_ids, dtype=torch.long)


def parse_wandb_param(sweep_config, model_args):
    # Extracting the hyperparameter values
    cleaned_args = {}
    layer_params = []
    param_groups = []
    for key, value in sweep_config.items():
        if isinstance(value, dict):
            value = value[0]
        if key.startswith("layer_"):
            # These are layer parameters
            layer_keys = key.split("_")[-1]

            # Get the start and end layers
            start_layer = int(layer_keys.split("-")[0])
            end_layer = int(layer_keys.split("-")[-1])

            # Add each layer and its value to the list of layer parameters
            for layer_key in range(start_layer, end_layer):
                layer_params.append(
                    {"layer": layer_key, "lr": value,}
                )
        elif key.startswith("params_"):
            # These are parameter groups (classifier)
            params_key = key.split("_")[-1]
            param_groups.append(
                {
                    "params": [params_key],
                    "lr": value,
                    "weight_decay": model_args.weight_decay
                    if "bias" not in params_key
                    else 0.0,
                }
            )
        else:
            # Other hyperparameters (single value)
            cleaned_args[key] = value
    cleaned_args["custom_layer_parameters"] = layer_params
    cleaned_args["custom_parameter_groups"] = param_groups

    # Update the model_args with the extracted hyperparameter values
    model_args.update(cleaned_args)