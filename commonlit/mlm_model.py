import torch
import pytorch_lightning as pl
from transformers import AutoConfig, AutoModelForMaskedLM
from transformers.optimization import get_cosine_schedule_with_warmup
from deepspeed.ops.adam import FusedAdam
from deepspeed.ops.adam import DeepSpeedCPUAdam


class BertMLMModel(pl.LightningModule):
    def __init__(
        self,
        bert_model,
        max_steps=2500,
        learning_rate=1e-5,
        weight_decay=0.1,
        warmup_steps=0.06,  # percentage of steps to warmup for
        freeze_layers=0,
        scheduler_rate=500,
        **kwargs,
    ):

        super(BertMLMModel, self).__init__()
        self.model_type = "BertClassifierModel"

        # Load Text Model
        # config = AutoConfig.from_pretrained(f"../input/huggingfacemodels/{bert_model}/transformer")
        # config.update({"layer_norm_eps": 1e-7, "hidden_dropout_prob": dropout})
        self.text_model = AutoModelForMaskedLM.from_pretrained(
            f"../input/huggingfacemodels/{bert_model}/transformer"
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
        self.log("mlm_train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self(batch).loss
        self.log("mlm_val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        # disable weight decay on bias and layernorm
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.text_model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.text_model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=self.learning_rate
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
