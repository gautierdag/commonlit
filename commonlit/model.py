import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel, AutoConfig
from transformers.optimization import get_cosine_schedule_with_warmup


def get_bert_layerwise_lr_groups(bert_model, learning_rate=1e-5, layer_decay=0.9):
    """
    Gets parameter groups with decayed learning rate based on depth in network
    Layers closer to output will have higher learning rate

    Args:
        bert_model: A huggingface bert-like model (should have embedding layer and encoder)
        learning_rate: The learning rate at the output layer
        layer_decay: How much to decay the learning rate per depth (recommended 0.9-0.95)
    Returns:
        grouped_parameters (list): list of parameters with their decayed learning rates
    """

    n_layers = len(bert_model.encoder.layer) + 1  # + 1 (embedding)

    embedding_decayed_lr = learning_rate * (layer_decay ** (n_layers + 1))
    grouped_parameters = [
        {"params": bert_model.embeddings.parameters(), "lr": embedding_decayed_lr}
    ]
    for depth in range(1, n_layers):
        decayed_lr = learning_rate * (layer_decay ** (n_layers + 1 - depth))
        grouped_parameters.append(
            {
                "params": bert_model.encoder.layer[depth - 1].parameters(),
                "lr": decayed_lr,
            }
        )

    return grouped_parameters


class BertClassifierModel(pl.LightningModule):
    def __init__(
        self,
        bert_model,
        max_steps=2500,
        use_warmup=False,
        learning_rate=1e-5,
        weight_decay=0.1,
        dropout=0.1,
        warmup_steps=0.06,  # percentage of steps to warmup for
        dense_dim=None,
        custom_linear_init=True,
        freeze_layers=0,
        scheduler_rate=500,
        decay_lr=True,
        use_tanh_constraint=False,
        commonlit_loss_weight=1,  # loss multiplier for commonlit
        sqrt_mse_loss=False,  # whether to sqrt the loss during training
        **kwargs,
    ):

        super(BertClassifierModel, self).__init__()
        self.model_type = "BertClassifierModel"

        # Load Text Model
        config = AutoConfig.from_pretrained(
            f"../input/huggingfacemodels/{bert_model}/transformer"
        )
        config.update({"layer_norm_eps": 1e-7, "hidden_dropout_prob": 0.0})
        self.text_model = AutoModel.from_pretrained(
            f"../input/huggingfacemodels/{bert_model}/transformer", config=config
        )

        if dense_dim is None:  # use bert dimensionality
            dense_dim = self.text_model.config.hidden_size

        self.decay_lr = decay_lr
        self.use_tanh_constraint = use_tanh_constraint
        self.commonlit_loss_weight = commonlit_loss_weight
        self.sqrt_mse_loss = sqrt_mse_loss

        # self.dense = nn.Linear(self.text_model.config.hidden_size,
        #                        dense_dim)
        self.dense = nn.Sequential(
            nn.Linear(self.text_model.config.hidden_size, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1),
        )

        self.output_layers = nn.ModuleDict(
            [
                ["commonlit", nn.Linear(dense_dim, 1)],
                ["wiki", nn.Linear(dense_dim, 1)],
                ["onestop", nn.Linear(dense_dim, 1)],
                ["race", nn.Linear(dense_dim, 1)],
                ["ck_12", nn.Linear(dense_dim, 1)],
                ["weebit", nn.Linear(dense_dim, 1)],
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.criterion = nn.MSELoss(reduction="sum")
        self.eval_criterion = nn.MSELoss()

        # optimiser settings
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_steps = max_steps
        self.use_warmup = use_warmup
        self.warmup = int(max_steps * warmup_steps)
        self.freeze_layers = freeze_layers
        self.scheduler_rate = scheduler_rate

        if custom_linear_init:
            list(map(self.initialise, self.output_layers.values()))
            list(map(self.initialise, self.dense))

    def initialise(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0, std=self.text_model.config.initializer_range
            )
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, text_input, dataset_name="commonlit", **kwargs):
        outputs = self.text_model(**text_input)[0]
        # x = self.dropout(outputs[0]).mean(dim=1) # use CLS token
        # x = self.dense(x).tanh()

        weights = self.dense(outputs)
        x = torch.sum(weights * outputs, dim=1)

        x = self.dropout(x)
        predictions = self.output_layers[dataset_name](x)

        # whether to constrain output between the
        if dataset_name == "commonlit" and self.use_tanh_constraint:
            return ((predictions.tanh() * 2.9) - 1).squeeze(1)
        return predictions.squeeze(1)

    def training_step(self, batch, batch_nb):
        if "text_input1" in batch:
            predicted_targets_1 = self(
                text_input=batch["text_input1"], dataset_name=batch["dataset_name"]
            )
            predicted_targets_2 = self(
                text_input=batch["text_input2"], dataset_name=batch["dataset_name"]
            )
            target_loss = F.margin_ranking_loss(
                predicted_targets_1, predicted_targets_2, batch["target"]
            )
        else:
            predicted_targets = self(**batch)
            target_loss = self.criterion(predicted_targets, batch["target"])
            if self.sqrt_mse_loss:
                target_loss = torch.sqrt(target_loss)

        if batch["dataset_name"] == "commonlit":
            self.log(f"{batch['dataset_name']}_train_loss", target_loss, prog_bar=True)
            target_loss = target_loss * self.commonlit_loss_weight
        else:
            self.log(f"{batch['dataset_name']}_train_loss", target_loss)

        return target_loss

    def validation_step(self, val_batch, val_batch_idx, **kwargs):
        predicted_targets = self(**val_batch)
        target_loss = torch.sqrt(
            self.eval_criterion(predicted_targets, val_batch["target"])
        )
        self.log("val_loss", target_loss, prog_bar=True)
        return target_loss

    def configure_optimizers(self):
        # freeze bottom layers
        if self.freeze_layers > 0:
            modules = [
                self.text_model.embeddings,
                *self.text_model.encoder.layer[-self.freeze_layers :],
            ]  # freeze last X layers
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False

        if self.decay_lr:
            lr_groups = get_bert_layerwise_lr_groups(
                self.text_model, learning_rate=self.learning_rate
            )
            non_bert_layers = [
                self.dense,
                self.output_layers,
            ]  # important to update this
            non_bert_layers = [
                {"params": l.parameters(), "lr": self.learning_rate}
                for l in non_bert_layers
            ]
            lr_groups += non_bert_layers
        else:
            lr_groups = self.parameters()

        optimizer = torch.optim.AdamW(
            lr_groups, lr=self.learning_rate, weight_decay=self.weight_decay
        )

        if self.use_warmup:
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
        else:
            print(f"Scheduling with rate: {self.scheduler_rate}")
            scheduler_rate = self.scheduler_rate
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    patience=6,
                ),
                "monitor": "val_loss",
                "interval": "step",
                "reduce_on_plateau": True,
                "frequency": scheduler_rate,
            }
        return [optimizer], [scheduler]
