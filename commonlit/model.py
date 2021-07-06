
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel

class BertClassifierModel(pl.LightningModule):
    def __init__(self, 
                 max_steps=2500, 
                 use_warmup=False, 
                 learning_rate=1e-5,
                 weight_decay=0.1,
                 dropout=0.1,
                 warmup_steps=0.1, # percentage of steps to warmup for
                 dense_dim=None,
                 custom_linear_init=True,
                 bert_model="roberta-base",
                 freeze_layers=0,
                 scheduler_rate=500,
                 **kwargs
                ):

        super(BertClassifierModel, self).__init__()
        self.model_type = "BertClassifierModel"
         
        # Load Text Model
        self.text_model = AutoModel.from_pretrained(f"../input/huggingfacemodels/{bert_model}/transformer")

        if dense_dim is None: # use bert dimensionality
            dense_dim = self.text_model.config.hidden_size
        
        self.dense = nn.Linear(self.text_model.config.hidden_size, 
                               dense_dim)
        
        self.output_layers = nn.ModuleDict([
            ["commonlit", nn.Linear(dense_dim, 1)],
            ["wiki", nn.Linear(dense_dim, 1)],
            ["onestop", nn.Linear(dense_dim, 1)],
            ["race", nn.Linear(dense_dim, 1)],
            ["ck_12", nn.Linear(dense_dim, 1)],
            ["weebit", nn.Linear(dense_dim, 1)],
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.criterion = nn.MSELoss()
        
        # optimiser settings
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_steps = max_steps
        self.use_warmup = use_warmup
        self.warmup = int(max_steps*warmup_steps)
        self.freeze_layers = freeze_layers
        self.scheduler_rate = scheduler_rate
        
        if custom_linear_init:
            list(map(self.initialise, self.output_layers.values()))
            self.initialise(self.dense)

    def initialise(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.text_model.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, text_input, dataset_name="commonlit", **kwargs):
        outputs = self.text_model(**text_input)
        x = self.dropout(outputs[0]).mean(dim=1) # use CLS token
        x = self.dense(x).tanh()
        x = self.dropout(x)
        predictions = self.output_layers[dataset_name](x)
        if dataset_name == "commonlit":
            return ((predictions.tanh() * 2.9) - 1).squeeze(1)
        return predictions.squeeze(1)
        
    
    def training_step(self, batch, batch_nb):
        if "text_input1" in batch:
            predicted_targets_1 = self(text_input=batch["text_input1"], dataset_name=batch["dataset_name"])
            predicted_targets_2 = self(text_input=batch["text_input2"], dataset_name=batch["dataset_name"])
            target_loss = F.margin_ranking_loss(predicted_targets_1, predicted_targets_2,  batch["target"])
        else:
            predicted_targets = self(**batch)
            target_loss = torch.sqrt(self.criterion(predicted_targets, batch["target"]))

        if batch["dataset_name"] == "commonlit":
            self.log(f"{batch['dataset_name']}_train_loss", target_loss, prog_bar=True)
            target_loss = target_loss * 10
        else:
            self.log(f"{batch['dataset_name']}_train_loss", target_loss)

        return target_loss

    def validation_step(self, val_batch, val_batch_idx, **kwargs):   
        predicted_targets = self(**val_batch)
        target_loss = torch.sqrt(self.criterion(predicted_targets, val_batch["target"]))
        self.log("val_loss", target_loss, prog_bar=True)
        return target_loss
        
    def configure_optimizers(self):
        # freeze bottom layers
        if self.freeze_layers > 0:
            modules = [self.text_model.embeddings, *self.text_model.encoder.layer[-self.freeze_layers:]] # freeze last X layers
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False

        optimizer = torch.optim.AdamW(self.parameters(), 
                                      lr=self.learning_rate, 
                                      weight_decay=self.weight_decay)

        if self.use_warmup:
            def warm_decay(step):
                if step < self.warmup:
                    return  step / self.warmup
                return (self.max_steps-step)/(self.max_steps)
            scheduler = (
                {
                    "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, warm_decay),
                    "interval": "step", #runs per batch rather than per epoch
                    "frequency": 1,
                    "name" : "learning_rate" # uncomment if using LearningRateMonitor
                }
            )
        else:
            print(f"Scheduling with rate: {self.scheduler_rate}")
            scheduler_rate = self.scheduler_rate
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", patience=10,
                ),
                "monitor": "val_loss",
                "interval": "step",
                "reduce_on_plateau": True,
                "frequency": scheduler_rate
            }
        return [optimizer], [scheduler]