import lightning
import nltk.sentiment
import pandas as pd
import torch
import transformers
from sklearn import metrics, model_selection, utils


class MultiTask(lightning.LightningModule):
    def __init__(self, transformer_model, spam_weights=None, sentiment_weights=None, learning_rate=1e-4):
        """
        This is where the model's architecture is defined (layers, activations, etc.).
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.save_hyperparameters(ignore=["transformer_model"])

        # Transformer backbone
        self.transformer_model = transformer_model
        for _, param in self.transformer_model.named_parameters():  # Freeze all transformer layers
            param.requires_grad = False
        hidden_size = self.transformer_model.config.hidden_size

        # Task layers
        self.spam_head = torch.nn.Linear(hidden_size, 1)
        self.sentiment_head = torch.nn.Linear(hidden_size, 3)

        # Weights. Need to use register_buffer so that the tensors follow the device of the model
        self.register_buffer(
            "spam_weights",
            torch.tensor(spam_weights, dtype=torch.float) if spam_weights is not None else None, )
        self.register_buffer(
            "sentiment_weights",
            torch.tensor(sentiment_weights, dtype=torch.float) if sentiment_weights is not None else None, )

    def forward(self, inputs):
        """
        Here we define how we use the modules to operate on an input batch. First, we run the inputs through the
        transformer backbone, then we average the outputs to get a single vector for each input. Finally, we run the
        embeddings trough each task layer. For simplicity, only 1 Linear layer is used for each task.
        """
        t_outputs = self.transformer_model(**inputs)
        embeddings = self.average_pool(t_outputs.last_hidden_state, inputs["attention_mask"])

        spam_logits = self.spam_head(embeddings)
        sentiment_logits = self.sentiment_head(embeddings)

        return {
            "spam_logits": spam_logits,
            "sentiment_logits": sentiment_logits
        }

    def training_step(self, batch, batch_idx):
        """
        For each step, we run a batch through the forward function and compute the loss. For the spam task, we are only
        predicting 0 (for not spam) or 1 (for spam). For the sentiment task, we are predicting one of three classes:
        (0 for negative, 1 for neutral, 2 for positive).
        """
        outputs = self(batch["inputs"])

        loss_spam = torch.nn.BCEWithLogitsLoss(pos_weight=self.spam_weights)(
            outputs["spam_logits"].view(-1), batch["spam_label"])
        loss_sentiment = torch.nn.CrossEntropyLoss(weight=self.sentiment_weights)(
            outputs["sentiment_logits"], batch["sentiment_label"])
        loss = loss_spam + loss_sentiment

        self.log("train_loss_task1", loss_spam)
        self.log("train_loss_task2", loss_sentiment)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch["inputs"])

        loss_spam = torch.nn.BCEWithLogitsLoss()(
            outputs["spam_logits"].view(-1), batch["spam_label"])
        loss_sentiment = torch.nn.CrossEntropyLoss()(
            outputs["sentiment_logits"], batch["sentiment_label"])
        loss = loss_spam + loss_sentiment

        self.log("val_loss_task1", loss_spam, prog_bar=True)
        self.log("val_loss_task2", loss_sentiment, prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    @staticmethod
    def average_pool(last_hidden_states, attention_mask):
        # We don't want to include padded tokens, so use masked_fill to zero them out.
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class MultiTaskDataset(torch.utils.data.Dataset):
    def __init__(self, df_, tokenizer_):
        self.texts = df_["message"].tolist()
        self.spam_labels = df_["spam_label"].tolist()
        self.sentiment_labels = df_["sentiment_label"].tolist()
        self.tokenizer = tokenizer_

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        spam_label = self.spam_labels[idx]
        sentiment_label = self.sentiment_labels[idx]

        encoding = self.tokenizer(text, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}

        return {
            "inputs": encoding,
            "spam_label": torch.tensor(spam_label),
            "sentiment_label": torch.tensor(sentiment_label)
        }


def create_dataloader(df_, tokenizer_, batch_size=32, shuffle=False):
    dataset = MultiTaskDataset(df_, tokenizer_)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)


tokenizer = transformers.AutoTokenizer.from_pretrained("thenlper/gte-base")
model = transformers.AutoModel.from_pretrained("thenlper/gte-base")
df = pd.read_csv("hf://datasets/codesignal/sms-spam-collection/sms-spam-collection.csv")
df = df.rename(columns={"label": "spam_label"})
# Change spam label to a numerical value
df["spam_label"] = df["spam_label"].apply(lambda x: 1.0 if x == "spam" else 0.0)

# Download vader_lexicon (if not already present) for marking messages with sentiment value
nltk.download('vader_lexicon')
sia = nltk.sentiment.vader.SentimentIntensityAnalyzer()
df["sentiment_label"] = df["message"].apply(lambda x: sia.polarity_scores(x)["compound"])
df["sentiment_label"] = df["sentiment_label"].apply(lambda x: 2 if x >= 0.05 else 0 if x <= -0.05 else 1)

# Split dataset into 70% training, 15% validation, and 15% testing
train_df, temp_df = model_selection.train_test_split(df, test_size=0.3,
                                                     random_state=0)  # random_state for reproducibility
val_df, test_df = model_selection.train_test_split(temp_df, test_size=0.5, random_state=0)

# Create dataloaders
train_loader = create_dataloader(train_df, tokenizer, batch_size=32, shuffle=True)
val_loader = create_dataloader(val_df, tokenizer, batch_size=32, shuffle=False)
test_loader = create_dataloader(test_df, tokenizer, batch_size=32, shuffle=False)

# Optional step of class weights to deal with imbalance in dataset
spam_class_weights = utils.class_weight.compute_class_weight(
    class_weight="balanced", classes=train_df["spam_label"].unique(), y=train_df["spam_label"].tolist())[1:]
sentiment_class_weights = utils.class_weight.compute_class_weight(
    class_weight="balanced", classes=train_df["sentiment_label"].unique(), y=train_df["sentiment_label"].tolist())

# Instantiate a Trainer, and put it on GPU if available
trainer = lightning.Trainer(max_epochs=5, accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1)
model = MultiTask(transformer_model=model,
                  # spam_weights=spam_class_weights,
                  # sentiment_weights=sentiment_class_weights
                  )

# Train the model
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

print("Training finished. Evaluating on testing set...")
# Put the model into evaluation mode
model.eval()

# And run through the test dataset and print evaluation metrics
all_spam_preds = []
all_spam_labels = []
all_sentiment_preds = []
all_sentiment_labels = []

# Disable gradient updates for evaluation
with torch.no_grad():
    for b in test_loader:
        batch_inputs = b["inputs"]

        spam_labels = b["spam_label"]
        sentiment_labels = b["sentiment_label"]

        batch_outputs = model(batch_inputs)

        batch_spam_logits = batch_outputs["spam_logits"]
        batch_sentiment_logits = batch_outputs["sentiment_logits"]

        # Get predicted class indices
        # for spam_preds, If logit is >0.5, predict it as spam
        spam_preds = (torch.sigmoid(batch_spam_logits) > 0.5).cpu().numpy()
        # For sentiment_preds, predict the class wit the largest value
        sentiment_preds = torch.argmax(batch_sentiment_logits, dim=1).cpu().numpy()

        all_spam_preds.extend(spam_preds)
        all_spam_labels.extend(spam_labels.cpu().numpy())
        all_sentiment_preds.extend(sentiment_preds)
        all_sentiment_labels.extend(sentiment_labels.cpu().numpy())

# Compute metrics
print("Spam Classification Report:")
print(metrics.classification_report(all_spam_labels, all_spam_preds))
print("Accuracy:", metrics.accuracy_score(all_spam_labels, all_spam_preds))

print("Sentiment Classification Report:")
print(metrics.classification_report(all_sentiment_labels, all_sentiment_preds))
print("Accuracy:", metrics.accuracy_score(all_sentiment_labels, all_sentiment_preds))
