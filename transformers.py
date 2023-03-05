###################################################
# Classification Tasks using transformers
###################################################

import numpy as np

# subset of categories that we will use
category_dict = {'comp.graphics': 'computer graphics',
                 'rec.sport.baseball': 'baseball',
                 'sci.electronics': 'science, electronics',
                 'talk.politics.guns': 'politics, guns'
                 }


def get_data(categories=None, portion=1.):
    """
    Get data for given categories and portion
    :param portion: portion of the data to use
    :return:
    """
    # get data
    from sklearn.datasets import fetch_20newsgroups
    data_train = fetch_20newsgroups(categories=categories, subset='train', remove=('headers', 'footers', 'quotes'),
                                    random_state=21)
    data_test = fetch_20newsgroups(categories=categories, subset='test', remove=('headers', 'footers', 'quotes'),
                                   random_state=21)

    # train
    train_len = int(portion*len(data_train.data))
    x_train = np.array(data_train.data[:train_len])
    y_train = data_train.target[:train_len]
    # remove empty entries
    non_empty = x_train != ""
    x_train, y_train = x_train[non_empty].tolist(), y_train[non_empty].tolist()

    # test
    x_test = np.array(data_test.data)
    y_test = data_test.target
    non_empty = np.array(x_test) != ""
    x_test, y_test = x_test[non_empty].tolist(), y_test[non_empty].tolist()
    return x_train, y_train, x_test, y_test


# Q1
def linear_classification(portion=1.):
    """
    Perform linear classification
    :param portion: portion of the data to use
    :return: classification accuracy
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    tf = TfidfVectorizer(stop_words='english', max_features=1000)
    x_train, y_train, x_test, y_test = get_data(categories=category_dict.keys(), portion=portion)
    x_train_tf = tf.fit_transform(x_train)
    x_test_tf = tf.transform(x_test)
    logistic_r = LogisticRegression()
    logistic_r.fit(x_train_tf, y_train)
    y_pred = logistic_r.predict(x_test_tf)
    accu = accuracy_score(y_test, y_pred)
    return accu


# Q2
def transformer_classification(portion=1.):
    """
    Transformer fine-tuning.
    :param portion: portion of the data to use
    :return: classification accuracy
    """
    import torch

    class Dataset(torch.utils.data.Dataset):
        """
        Dataset object
        """
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    from datasets import load_metric
    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    from transformers import Trainer, TrainingArguments
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('distilroberta-base', cache_dir=None)
    model = AutoModelForSequenceClassification.from_pretrained('distilroberta-base',
                                                               cache_dir=None,
                                                               num_labels=len(category_dict),
                                                               problem_type="single_label_classification")

    x_train, y_train, x_test, y_test = get_data(categories=category_dict.keys(), portion=portion)
    x = tokenizer.__call__(x_train, padding="longest", truncation=True)
    # y = tokenizer.__call__(y_train)
    x_t = tokenizer.__call__(x_test, padding="longest", truncation=True)
    # y_t = tokenizer.__call__(y_test)
    ds_train = Dataset(x, y_train)
    ds_test = Dataset(x_t, y_test)
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_test,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    return trainer.evaluate()


# Q3
def zeroshot_classification(portion=1.):
    """
    Perform zero-shot classification
    :param portion: portion of the data to use
    :return: classification accuracy
    """
    from transformers import pipeline
    from sklearn.metrics import accuracy_score
    import torch
    x_train, y_train, x_test, y_test = get_data(categories=category_dict.keys(), portion=portion)
    clf = pipeline("zero-shot-classification", model='cross-encoder/nli-MiniLM2-L6-H768',
                   device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    candidate_labels = list(category_dict.values())
    y_output = clf(x_test, candidate_labels=candidate_labels)
    y_pred = np.array(y_test)
    cat_dict = {'computer graphics': 0, 'baseball': 1,
                'science, electronics': 2, 'politics, guns': 3}
    i = 0
    for item in y_output:
        label = item['labels'][0]
        y_pred[i] = cat_dict[label]
        i += 1
    return accuracy_score(y_test, y_pred)


def graphs(portions):
    accu_1 = [0.7208222811671088, 0.8103448275862069, 0.8275862068965517]
    accu_2 = [0.8706896551724138, 0.8932360742705571, 0.9131299734748011]
    import plotly.graph_objects as go

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=portions, y=accu_1, name='train', mode='lines', line=dict(color="purple", width=2)))
    fig1.update_layout(title=f"Log-linear classifier Accuracy", xaxis=dict(title=r"$Portions$"),
                       yaxis=dict(title=r"Accuracy"),
                       width=900, height=600)
    fig1.write_image(f"./figures/log-linear.png")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=portions, y=accu_2, name='train', mode='lines', line=dict(color="purple", width=2)))
    fig2.update_layout(title=f"transformer classifier Accuracy", xaxis=dict(title=r"$Portions$"),
                       yaxis=dict(title=r"Accuracy"),
                       width=900, height=600)
    fig2.write_image(f"./figures/transformer.png")


if __name__ == "__main__":
    portions = [0.1, 0.5, 1.]
    # Q1
    print("Logistic regression results:")
    for p in portions:
        print(f"Portion: {p}")
        print(linear_classification(p))

    # Q2
    print("\nFinetuning results:")
    for p in portions:
        print(f"Portion: {p}")
        print(transformer_classification(portion=p))

    # Q3
    print("\nZero-shot result:")
    print(zeroshot_classification())

    graphs(portions)
