import configparser
import os
import json
import argparse

import pandas as pd
import numpy as np
import wandb
from active_learning_proccess.src.common.load_data import load_data
from active_learning_proccess.src.common.score import scorePredict
from pathlib import Path
# from active_learning_proccess.src.common.util import compute_class_weight
from active_learning_proccess.src.common.api_inception import InceptionClient
from pycaprio.mappings import InceptionFormat
from active_learning_proccess.src.common.create_sofa import write_sentence_documents
import wsgi

from sklearn.utils.class_weight import compute_class_weight
from small_text import (
    EmptyPoolException,
    PoolBasedActiveLearner,
    PoolExhaustedException,
    LeastConfidence,
    BreakingTies,
    RandomSampling,
    ContrastiveActiveLearning,
    random_initialization_balanced,
    random_initialization
)
from small_text.data.datasets import TextDataset
from active_learning_proccess.src.models.models import TransformerBasedClassificationFactory

query = {"lc": LeastConfidence(),
         "bt": BreakingTies(),
         "rs": RandomSampling(),
         "cal": ContrastiveActiveLearning()
         }


def active_learning_process(config_path, config, df_model_args, is_unlabelled=False):
    ############# Load datasets ################
    df_train, label_encoder = load_data(config["train_file"], config['label'], config['filter_label'], config['filter_label_value'])
    df_test, _ = load_data(config["test_file"], config['label'], config['filter_label'], config['filter_label_value'])

    print(df_test['labels'].value_counts())
    print(df_train['labels'].value_counts())

    model_args = df_model_args.to_dict(orient='records')[0]
    ############### Calculate weights using sklearn ##################
    if "weight" in df_model_args:
        # weights = compute_class_weight(np.unique(df_train['labels'].values), list(df_train['labels'].values))
        weights = compute_class_weight(class_weight="balanced", classes=np.unique(df_train['labels'].values), y=df_train['labels'].values)
        model_args["weight"] = list(weights)

    os.environ["WANDB_API_KEY"] ="your_key"
    if not "id" in config:
        id = wandb.util.generate_id()
        config["group"] = config["group"] + id
        config["id"] = id
        if config["wandb_project"]:
            wandb.login()
            wandb.init(id=config["id"], config=wandb.config, project=config["wandb_project"], group=config["group"],
                       job_type="train")
    labels = ""
    clf_factory = TransformerBasedClassificationFactory(config["model_type"], config["model_name"], config["use_cuda"],
                                                        len(df_train['labels'].unique()), config["wandb_project"], config["is_evaluate"],
                                                        config["best_result_config"], config["is_training"],
                                                        len(df_train['features'][0]), config["group"], embedding=config["model_name"],
                                                        output_dir=os.getcwd() + '/models/' + config["output_model_dir"], model_args=model_args)
    query_strategy = query[config["query_strategy"]]
    df = df_train
    if is_unlabelled:
        df_unlabelled, _ = load_data(config["unlabelled_file"], None)
        df = pd.concat([df_train, df_unlabelled])
    train_dataset = TextDataset(df['text'].values, df['labels'].values)
    # Active learner
    active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, train_dataset)
    indices_initial = df_train.index.values
    y_initial = np.array([df_train['labels'].iloc[i] for i in indices_initial])
    active_learner.initialize_data(indices_initial, y_initial, retrain=False)
    if config["init"]:
        #indices_labeled = initialize_active_learner(active_learner, df_train, df_eval)
        pass
    elif config["query_strategy"] != "rs":
        indices_initial = df_train.index.values
        y_initial = np.array([df_train['labels'].iloc[i] for i in indices_initial])
        active_learner.initialize_data(indices_initial, y_initial)
        y_predict = active_learner._clf.predict(df_test)
        result, f1, accuracy_value = scorePredict(y_predict, df_test['labels'], list(df_test['labels'].unique()))
        print(result)
        #####################################   To add metric values in each training iteration ##############################
        if config["wandb_project"]:
            wandb.finish()
            wandb.login(key="your_key")
            wandb.init(id=config["id"], config=wandb.config, project=config["wandb_project"], resume="must")
            wandb.log({'example_counts': len(indices_initial)})
            wandb.log({'f_1_value_test': f1})
            wandb.log({'accuracy_value_test': accuracy_value})
            wandb.finish()

    ######################################## Only need is unlabelled pool exist ##############################
    if is_unlabelled:
        indices_queried = active_learner.query(num_samples=int(config["num_samples"]))
        df_queried = df.iloc[indices_queried]
        df_queried.to_csv(os.getcwd() + config["output_dir"] + '/unlabelled_to_annotator_{}.tsv'.format(config['num_iteration']), sep='\t', index=False)
        document_format = InceptionFormat.TEXT_SENTENCE_PER_LINE
        if not config["is_xmi"]:
            file_path = os.getcwd() + config["output_dir"] + '/unlabelled_to_annotator_{}.txt'.format(config['num_iteration'])
            with open(file_path, 'w') as f:
                for index, row in df_queried.iterrows():
                    f.write(str(row['id'])+' ' + row['text'] + '\n')
        else:
            document_format = InceptionFormat.UIMA_CAS_XMI_XML_1_1
            file_path = os.getcwd() + config["output_dir"] + '/unlabelled_to_annotator_{}.xmi'.format(
                config['num_iteration'])
            df_queried.reset_index(drop=True, inplace=True)
            y_predict = active_learner._clf.predict(df_queried)
            df_predict = pd.DataFrame(label_encoder.inverse_transform(y_predict), columns=["value"])
            labels_dict_list = []
            spans_list = []
            layers = []

            df_queried['id_str'] = df_queried['id'].map(str)
            df_queried['id_text'] = df_queried['id_str'] + ' ' + df_queried['text']
            df_queried.drop(axis=1, columns=['id_str'], inplace=True)
            sentences = df_queried['id_text'].values

            for element in config["dependent_layers"]:
                original_path = os.path.dirname(os.getcwd() + config_path)
                spans = None
                layers.append(element["layer"])
                for dependent in element["dependencies"]:
                    with open(original_path + '/' + dependent +'.json') as f:
                        dependent_config = json.load(f)
                    df_filter = df_queried
                    if "filter_label_value" in dependent_config:
                        df_filter = df_queried[df_predict["value"] == dependent_config['filter_label_value']]
                        index = df_filter.index
                        df_filter.reset_index(drop=True, inplace=True)
                    classfier = wsgi.server.get_classifier(dependent_config['service_name'])
                    if classfier.is_classification:
                        y_predict = classfier._predict(df_filter)
                        df_predict_aux = pd.DataFrame(y_predict, columns=[dependent.split('_')[1]], index=index)
                        df_predict_aux[dependent.split('_')[1]] = df_predict_aux[dependent.split('_')[1]].apply(lambda x: classfier.map_labels[x])
                        df_predict = pd.concat([df_predict, df_predict_aux], axis=1)
                        df_predict = df_predict.fillna('')
                    else:
                        spans = classfier._predict(sentences)
                        spans_list.append(spans)

                labels_dict_list.append(df_predict.to_dict('records'))


            # sentences, labels, 'Typesystem.xml', 'test_1.xmi', "webanno.custom.Violencia", labeled = True
            write_sentence_documents(sentences, labels_dict_list, spans_list, config["typesystem"], file_path, layers, True)

        client_inception = InceptionClient()
        client_inception.inset_document(config['inception_project'], file_path, document_format)
        index_to_drop_in_unlabelled = df_unlabelled[df_unlabelled['id'].isin(df_queried['id'])].index
        df_unlabelled.drop(axis=0, index=index_to_drop_in_unlabelled, inplace=True)
        df_unlabelled = df_unlabelled.sample(frac=1)
        df_unlabelled.to_csv(os.getcwd() + config["unlabelled_file"], sep='\t', index=False)
        
    config['num_iteration'] = int(config['num_iteration']) + 1
    with open(os.getcwd() + config_path, 'w') as f:
        f.write(json.dumps(config, indent=4))

def load_active_learning(config_path="/active_learning_proccess/config/comment.json",
                         model_arg="/active_learning_proccess/config/comment_model.json", df=None, is_unlabelled=False):
    with open(os.getcwd() + config_path) as f:
        config = json.load(f)
    df_model_args = pd.read_json(os.getcwd() + model_arg)
    ############# Concat with new training data ##################
    if isinstance(df, pd.DataFrame):
        df_train = pd.read_csv(os.getcwd() + config["train_file"], sep='\t')
        #################### Check if examples id there is in training dataset  ####################
        index_to_drop = df[df['id'].isin(df_train['id'])].index
        df.drop(axis=0, index=index_to_drop, inplace=True)
        if len(df) > 0:
            df.rename(columns={'label': config["label"]}, inplace=True)
            df_train = pd.concat([df_train, df], ignore_index=True)
            df.replace({"label": config["map_label"]}, inplace=True)
            df_train.to_csv(os.getcwd() + config["train_file"], sep='\t', index=False)
            ############ End Concat #############
            active_learning_process(config_path, config, df_model_args, is_unlabelled)
    else:
        active_learning_process(config_path, config, df_model_args, is_unlabelled)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## Required parameters

    parser.add_argument("--config_path",
                        default="/active_learning_proccess/config/vil.json",
                        type=str,
                        help="File path to configuration parameters.")

    parser.add_argument("--model_arg",
                        default="/active_learning_proccess/config/vil_model.json",
                        type=str,
                        help="File path to model configuration parameters."),

    parser.add_argument("--is_unlabelled",
                        default=False,
                        action='store_true',
                        help="This parameter should be True if you want to use unlabelled dataset.")


    args = parser.parse_args()

    config_path = args.config_path
    model_arg = args.model_arg
    is_unlabelled = args.is_unlabelled
    load_active_learning(config_path=config_path, model_arg=model_arg, is_unlabelled=is_unlabelled)
