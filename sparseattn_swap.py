#!wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip
#!unzip wikitext-103-raw-v1.zip

import copy
import logging
import math
import os
from dataclasses import dataclass, field

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import (
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    RobertaForMaskedLM,
    RobertaTokenizerFast,
    TextDataset,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

torch.manual_seed(42)

torch.backends.cudnn.benchmark = True


@torch.jit.script
def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / 1.41421))


class MaskedLinear(nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_mask: Tensor = None,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(MaskedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

        self.weight_mask = weight_mask

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor, weight_mask: Tensor = None) -> Tensor:
        weight = self.weight
        if weight_mask is not None:
            weight = weight * weight_mask.type_as(input)

        if self.bias is not None:
            bias = self.bias
        else:
            bias = None

        output = F.linear(input.transpose(-1, -2), weight, bias).transpose(-1, -2)

        return output

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


class LongformerSelfAttention(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        max_pos = 4096
        self.pooler = MaskedLinear(max_pos * 3, max_pos)

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        # self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # if (
        #     self.position_embedding_type == "relative_key"
        #     or self.position_embedding_type == "relative_key_query"
        # ):
        #     self.max_position_embeddings = config.max_position_embeddings
        #     self.distance_embedding = nn.Embedding(
        #         2 * config.max_position_embeddings - 1, self.attention_head_size
        #     )

        self.layer_ids = layer_id

    def transpose_for_scores(self, x):
        bsz, seq_len = x.size()[:-1]
        new_x_shape = (
            bsz,
            self.num_attention_heads,
            seq_len,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def forward(
        self,
        hidden_states,
        weight_mask=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        input_shape = hidden_states.shape
        bsz, seq_len, hidden_dim = input_shape

        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        # assert query_layer.shape == (bsz, self.num_attention_heads, seq_len, self.attention_head_size)
        # assert key_layer.shape == (bsz, self.num_attention_heads, seq_len, self.attention_head_size)
        # assert value_layer.shape == (bsz, self.num_attention_heads, seq_len, self.attention_head_size)

        global_v = self.pooler(
            torch.cat([query_layer, key_layer, value_layer], dim=-2), weight_mask
        )  # (bsz, self.num_attention_heads, seq_len, self.attention_head_size)
        global_v = gelu(global_v)

        # (bsz, self.num_attention_heads, seq_len, 1)
        local_scores = torch.einsum(
            "bnij,bnji->bni", query_layer, key_layer.transpose(-1, -2)
        ).unsqueeze(-1)
        global_scores = torch.zeros_like(local_scores)
        attention_scores = torch.cat([local_scores, global_scores], dim=-1)
        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size
        )  # (bsz, self.num_attention_heads, seq_len, 2)
        ####dont need mask here
        # if attention_mask is not None:
        #     attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        v = torch.cat([value_layer.unsqueeze(-1), global_v.unsqueeze(-1)], dim=-1)
        v = v.transpose(
            -1, -2
        )  # (bsz, self.num_attention_heads, seq_len, 2, self.attention_head_size)

        attention_probs = attention_probs.unsqueeze(
            -2
        )  # (bsz, self.num_attention_heads, seq_len, 1, 2)

        context_layer = torch.matmul(
            attention_probs, v
        )  # (bsz, self.num_attention_heads, seq_len, 1, self.attention_head_size)

        context_layer = context_layer.squeeze(-2)
        context_layer = context_layer.permute(
            0, 2, 1, 3
        ).contiguous()  # (bsz, seq_len, self.num_attention_heads, self.attention_head_size)
        context_layer = context_layer.view(
            bsz, seq_len, -1
        )  # (bsz, seq_len, self.all_head_size)

        return (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )


class RobertaLongSelfAttention(LongformerSelfAttention):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        output_attentions=False,
    ):
        return super().forward(
            hidden_states,
            # attention_mask=attention_mask,
            # output_attentions=output_attentions,
        )


class RobertaLongForMaskedLM(RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        for i, layer in enumerate(self.roberta.encoder.layer):
            # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
            layer.attention.self = RobertaLongSelfAttention(config, layer_id=i)

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


def create_long_model(save_model_to, attention_window, max_pos):
    model = RobertaForMaskedLM.from_pretrained("roberta-base", hidden_dropout_prob=0.0)
    tokenizer = RobertaTokenizerFast.from_pretrained(
        "roberta-base", model_max_length=max_pos
    )
    config = model.config

    # extend position embeddings
    max_pos = 4096
    tokenizer.model_max_length = max_pos
    tokenizer.init_kwargs["model_max_length"] = max_pos
    (
        current_max_pos,
        embed_size,
    ) = model.roberta.embeddings.position_embeddings.weight.shape
    max_pos += 2  # NOTE: RoBERTa has positions 0,1 reserved, so embedding size is max position + 2
    config.max_position_embeddings = max_pos
    assert max_pos > current_max_pos
    # allocate a larger position embedding matrix
    new_pos_embed = model.roberta.embeddings.position_embeddings.weight.new_empty(
        max_pos, embed_size
    )
    # copy position embeddings over and over to initialize the new position embeddings
    k = 2
    step = current_max_pos - 2
    while k < max_pos - 1:
        new_pos_embed[
            k : (k + step)
        ] = model.roberta.embeddings.position_embeddings.weight[2:]
        k += step
    model.roberta.embeddings.position_embeddings.weight.data = new_pos_embed
    model.roberta.embeddings.position_ids.data = torch.tensor(
        [i for i in range(max_pos)]
    ).reshape(1, max_pos)

    # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
    config.attention_window = [attention_window] * config.num_hidden_layers
    for i, layer in enumerate(model.roberta.encoder.layer):
        longformer_self_attn = LongformerSelfAttention(config, layer_id=i)
        # longformer_self_attn.query = layer.attention.self.query
        # longformer_self_attn.key = layer.attention.self.key
        # longformer_self_attn.value = layer.attention.self.value

        # longformer_self_attn.query_global = copy.deepcopy(layer.attention.self.query)
        # longformer_self_attn.key_global = copy.deepcopy(layer.attention.self.key)
        # longformer_self_attn.value_global = copy.deepcopy(layer.attention.self.value)

        layer.attention.self = longformer_self_attn

        layer.attention.self.query = None
        layer.attention.self.key = None
        layer.attention.self.value = None

    logger.info(f"saving model to {save_model_to}")
    model.save_pretrained(save_model_to)
    tokenizer.save_pretrained(save_model_to)
    return model, tokenizer


def pretrain_and_evaluate(args, model_args, model, tokenizer, eval_only, model_path):
    val_dataset = TextDataset(
        tokenizer=tokenizer, file_path=args.val_datapath, block_size=model_args.max_pos
    )

    if eval_only:
        train_dataset = val_dataset
    else:
        logger.info(
            f"Loading and tokenizing training data is usually slow: {args.train_datapath}"
        )
        train_dataset = TextDataset(
            tokenizer=tokenizer,
            file_path=args.train_datapath,
            block_size=model_args.max_pos,
        )
        print("done loading")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    eval_loss = trainer.evaluate()
    eval_loss = eval_loss["eval_loss"]
    logger.info(f"Initial eval loss: {eval_loss}, eval bpc: {eval_loss/math.log(2)}")

    if not eval_only:
        trainer.train()
        trainer.save_model()

        eval_loss = trainer.evaluate()
        eval_loss = eval_loss["eval_loss"]
        logger.info(f"Eval bpc after pretraining: {eval_loss/math.log(2)}")


@dataclass
class ModelArgs:
    attention_window: int = field(
        default=256, metadata={"help": "Size of attention window"}
    )
    max_pos: int = field(default=4096, metadata={"help": "Maximum position"})


def main():
    parser = HfArgumentParser(
        (
            TrainingArguments,
            ModelArgs,
        )
    )
    training_args, model_args = parser.parse_args_into_dataclasses(
        look_for_args_file=False,
        args=[
            "--output_dir",
            "checkpoints",
            "--warmup_steps",
            "500",
            "--learning_rate",
            "0.0008",
            "--weight_decay",
            "0.01",
            "--adam_epsilon",
            "1e-6",
            "--adafactor",
            "True",
            "--max_steps",
            "10000",
            "--logging_steps",
            "10",
            "--save_steps",
            "5000",
            "--max_grad_norm",
            "5.0",
            "--per_device_eval_batch_size",
            "1",
            "--per_device_train_batch_size",
            "64",  # 2 for 32GB gpu with fp32
            "--gradient_accumulation_steps",
            "1",
            "--evaluation_strategy",
            "steps",
            "--eval_steps",
            "500",
            "--do_train",
            "--do_eval",
            "--prediction_loss_only",
            "--fp16",
        ],
    )
    training_args.val_datapath = "./wikitext-103-raw/wiki.valid.raw"
    training_args.train_datapath = "./wikitext-103-raw/wiki.train.raw"

    model_path = f"{training_args.output_dir}/roberta-base-{model_args.max_pos}"
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    ########################################################################################
    logger.info(f"Converting roberta-base into roberta-base-{model_args.max_pos}")
    model, tokenizer = create_long_model(
        save_model_to=model_path,
        attention_window=model_args.attention_window,
        max_pos=model_args.max_pos,
    )

    logger.info(f"Loading the model from {model_path}")
    tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
    model = RobertaLongForMaskedLM.from_pretrained(model_path)

    logger.info(f"Pretraining roberta-base-{model_args.max_pos} ... ")

    pretrain_and_evaluate(
        training_args,
        model_args,
        model,
        tokenizer,
        eval_only=False,
        model_path=training_args.output_dir,
    )


if __name__ == "__main__":
    main()
