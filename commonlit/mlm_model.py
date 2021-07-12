import torch
import pytorch_lightning as pl
from transformers import AutoConfig, AutoModelForMaskedLM
from transformers.optimization import get_cosine_schedule_with_warmup


class BertMLMModel(pl.LightningModule):
    def __init__(
        self,
        max_steps=2500,
        learning_rate=1e-5,
        weight_decay=0.1,
        dropout=0.1,
        warmup_steps=0.06,  # percentage of steps to warmup for
        bert_model="roberta-base",
        freeze_layers=0,
        scheduler_rate=500,
        **kwargs,
    ):

        super(BertMLMModel, self).__init__()
        self.model_type = "BertClassifierModel"

        # Load Text Model
        config = AutoConfig.from_pretrained(f"../input/huggingfacemodels/{bert_model}/transformer")
        config.update({"layer_norm_eps": 1e-7, "hidden_dropout_prob": dropout}) 
        self.text_model = AutoModelForMaskedLM.from_pretrained(
            f"../input/huggingfacemodels/{bert_model}/transformer", config=config
        )

        # optimiser settings
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_steps = max_steps
        self.warmup = int(max_steps * warmup_steps)
        self.freeze_layers = freeze_layers
        self.scheduler_rate = scheduler_rate


    def forward(self, x):
        return self.text_model(**x)

    def training_step(self, batch, batch_idx):
        loss = self(batch).loss
        self.log('mlm_train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self(batch).loss
        self.log('mlm_val_loss', loss, prog_bar=True)


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        schedule = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup,
            num_training_steps=self.max_steps,
        )
        scheduler = {
            "scheduler": schedule,
            "interval": "step",  # runs per batch rather than per epoch
            "frequency": 1,
            "name": "learning_rate",  # uncomment if using LearningRateMonitor
        }
        return [optimizer], [scheduler]