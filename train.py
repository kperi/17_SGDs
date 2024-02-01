import pandas as pd
import tqdm
import numpy as np
import random
import torch

import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score
import fire
import copy


idx2label = {
    0: "No Poverty",
    1: "Zero Hunger",
    2: "Good Health and Well-being",
    3: "Quality Education",
    4: "Gender Equality",
    5: "Clean Water and Sanitation",
    6: "Affordable and Clean Energy",
    7: "Decent Work and Economic Growth",
    8: "Industry, Innovation and Infrastructure",
    9: "Reducing Inequality",
    10: "Sustainable Cities and Communities",
    11: "Responsible Consumption and Production",
    12: "Climate Action",
    13: "Life Below Water",
    14: "Life On Land",
    15: "Peace, Justice, and Strong Institutions",
    16: "Partnerships for the Goals.",
}
label2idx = {label: idx for idx, label in idx2label.items()}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data(path):
    df = pd.read_csv(path, sep=";")
    df.sdg = df.sdg - 1

    # train only on accepted instances
    df_data = df[(df.volunteers == "accepted") & (df.label_osdg == "accepted")].copy()

    df_data["label"] = df_data.sdg.map(lambda v: idx2label[v])

    df_data = df_data.reset_index().drop("index", axis=1)
    return df_data


class SDGDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.df = df
        self.max_len = max_len

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        text = self.df.loc[idx].text.lower()
        label = self.df.loc[idx].sdg
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            truncation=True,
            padding="max_length",
        )
        ids = inputs["input_ids"]  # token ids
        mask = inputs[
            "attention_mask"
        ]  # token masks: 1 for text positions for the net to pay "attention"
        token_type_ids = inputs[
            "token_type_ids"
        ]  # token types for the task, here it would be always 0
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.int64),
            "text": text,
        }


class SDGNetwork(torch.nn.Module):
    def __init__(self, base_model, num_outputs):
        super(SDGNetwork, self).__init__()
        self.bert = AutoModel.from_pretrained(
            base_model
        )  # 786: bert representation size
        self.linear = torch.nn.Linear(768, num_outputs)

    def encode(self, ids, mask, token_type_ids):
        text_feats = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)[
            "pooler_output"
        ]
        return text_feats

    def decode(self, representation):
        return self.linear(representation)

    def forward(self, ids, mask, token_type_ids):
        representation = self.encode(ids, mask, token_type_ids)
        output = self.decode(representation)
        return output  # 17 outputs


def train(epoch, model, optimizer, criterion, training_loader, device):
    model.train()
    losses = []
    for _, data in tqdm.tqdm(enumerate(training_loader, 0), total=len(training_loader)):
        optimizer.zero_grad()

        ids = data["ids"].to(device, dtype=torch.long)
        mask = data["mask"].to(device, dtype=torch.long)
        token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
        targets = data["label"].to(device, dtype=torch.int64)

        outputs = model(ids, mask, token_type_ids)

        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    
    return np.mean(losses)


def validation(model, testing_loader, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data["ids"].to(device, dtype=torch.long)
            mask = data["mask"].to(device, dtype=torch.long)
            token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
            targets = data["label"].to(device, dtype=torch.int64)

            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(
                torch.argmax(outputs, dim=1).cpu().detach().numpy().tolist()
            )

    fin_outputs = np.array(fin_outputs)
    fin_targets = np.array(fin_targets)

    accuracy = accuracy_score(fin_targets, fin_outputs)
    f1_score_micro = f1_score(fin_targets, fin_outputs, average="micro")
    f1_score_macro = f1_score(fin_targets, fin_outputs, average="macro")

    return accuracy, f1_score_micro, f1_score_macro


def train_eval_loop(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion,
    train_loader,
    test_loader,
    device,
):
    best_score = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    EPOCHS = 10
    for epoch in range(EPOCHS):
        train(
            epoch,
            model,
            optimizer,
            criterion,
            training_loader=train_loader,
            device=device,
        )

        accuracy, f1_micro, f1_macro = validation(
            model=model, testing_loader=test_loader, device=device
        )

        # optimize for best f1_macro score
        if f1_macro > best_score:
            best_score = f1_macro

            # save the weights of the best model
            best_model_wts = copy.deepcopy(model.state_dict())
            print(
                f"Saving model:  Accuracy Score = {accuracy:0.3} || F1 Score (Macro) = {f1_macro:0.3}"
            )

    # reload the weights of the best f1score model
    model.load_state_dict(best_model_wts)


def main(
    device: str = "cuda",
    data_path="./data/Data_28.11.2023.csv",
    base_model="bert-base-uncased",
    lr=1e-05,
) -> None:
    set_seed(0)

  
    df_data = load_data(data_path)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    ds = SDGDataset(df_data, tokenizer=tokenizer)

    # train test split. We split the dataset in two parts, one 80% of the data for training and the rest 20% for testing
    # we also fix the random number generator seed to always have the same sequence of splits, eg make this reproducible
    test_size = int(len(ds) * 0.2)
    train_set, test_set = torch.utils.data.random_split(
        ds, [len(ds) - test_size, test_size]
    )

    num_outputs = df_data.sdg.unique().shape[0]

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=8, shuffle=False)

    model = SDGNetwork(base_model=base_model, num_outputs=num_outputs).to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    print("Training model...")
    train_eval_loop(model, optimizer, criterion, train_loader, test_loader, device)

    print("Saving model...")
    torch.save(model.state_dict(), "model.pt")


if __name__ == "__main__":
    fire.Fire(main)
