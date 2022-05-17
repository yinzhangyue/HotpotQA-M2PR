from fastNLP import Callback
import os
import wandb


class WandbCallback(Callback):
    def __init__(self, project: str, name: str, config: dict):
        r"""

        :param str name: project 项目名称
        :param str name: project 项目名称
        :param dict config: 模型超参
        :
        """
        super().__init__()
        self.project = project
        self.name = name
        self.config = config

    def on_train_begin(self):
        wandb.init(
            # Set entity to specify your username or team name
            # ex: entity="carey",
            # Set the project where this run will be logged
            project=self.project,
            name=self.name,
            # Track hyperparameters and run metadata
            config=self.config,
        )

    def on_train_end(self):
        wandb.finish()

    def on_valid_end(self, eval_result, metric_key, optimizer, is_better_eval):
        if "SpanSentenceMetric" in eval_result.keys():
            eval_result = eval_result["SpanSentenceMetric"]
            upload_result = {}
            for k in eval_result.keys():
                if "f1" in k or "em" in k:
                    upload_result[k] = eval_result[k]

        wandb.log(eval_result)

    def on_backward_begin(self, loss):
        wandb.log({"loss": loss})
