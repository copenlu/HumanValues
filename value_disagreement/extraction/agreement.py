from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import AutoTokenizer, BertForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput


class OverriddenBertForSequenceClassification(BertForSequenceClassification):
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        with torch.no_grad():
            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # Get embedding CLS token from hidden state instead of pooled_output
        hids = outputs.last_hidden_state
        cls_hidden = hids[:, 0, :]
        cls_hidden = self.dropout(cls_hidden)
        logits = self.classifier(cls_hidden)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

def dynamically_construct_mlp(num_layers, emb_dim, starting_dim, num_labels):
    layers = []
    if num_layers == 0:
        layers.append(nn.Linear(starting_dim, num_labels))
    elif num_layers >= 1:
        layers.append(nn.Linear(emb_dim, starting_dim),)
        for i in range(num_layers - 1):
            layers.append(nn.GELU())
            layers.append(nn.Linear(int(starting_dim/ (2**i)), int(starting_dim / (2**(i+1)))))
        layers.append(nn.GELU())
        layers.append(nn.Linear(int(starting_dim / (2**(num_layers-1))), num_labels))
    return nn.Sequential(*layers)



class TransformerUserContextClassifier(BertForSequenceClassification):
    def __init__(self, *args, extras_dim=10, mlp_dim=100, mlp_layers=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.extras_dim = extras_dim
        concat_hidden_size = self.bert.config.hidden_size + (extras_dim * 2)
        self.mlp = dynamically_construct_mlp(mlp_layers, concat_hidden_size, mlp_dim, self.num_labels)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        user_context: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Get embedding CLS token from hidden state instead of pooled_output
        hids = outputs.last_hidden_state
        cls_hidden = hids[:, 0, :]
        cls_hidden = self.dropout(cls_hidden)
        if user_context is not None:
            cls_hidden = torch.cat((cls_hidden, user_context), dim=1)

        logits = self.mlp(cls_hidden)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        assert not torch.isnan(loss)
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ContextOnlyModel(nn.Module):
    def __init__(self, context_dims=100, num_layers=2, hidden_dims=200, problem_type=None, num_labels=2):
        super().__init__()
        self.num_labels = num_labels
        self.problem_type = problem_type
        self.mlp = dynamically_construct_mlp(num_layers, context_dims*2, hidden_dims, self.num_labels)

    def forward(self, user_context: torch.Tensor, attention_mask=None, input_ids=None, labels=None) -> torch.Tensor:
        logits = self.mlp(user_context)
        loss = None
        if labels is not None:
            if self.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        assert not torch.isnan(loss)
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )


class DebagreementTokenizer():
    def __init__(self, model_name, concat_mode):
        self.concat_mode = concat_mode
        if model_name == "override_bert":
            self.model_name = "bert-base-uncased"
        else:
            self.model_name: str = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def tokenize(self, examples):
        # Gather input text
        batch_size = len(examples['text_left'])
        if self.concat_mode == "mixin_token":
            batched_inputs = [examples['text_left'][i] + self.tokenizer.sep_token +
                              str(examples['sim_scores'][i]) + self.tokenizer.sep_token +
                              examples['text_right'][i] for i in range(batch_size)]
        else:
            batched_inputs = [examples['text_left'][i] + self.tokenizer.sep_token +
                              examples['text_right'][i] for i in range(batch_size)]

        # Tokenize~!
        samples = self.tokenizer(batched_inputs, truncation=True, padding=True)
        if "author_left_vectors" in examples and self.concat_mode == "concat_vector":
            profiles_left = np.array(examples['author_left_vectors'])
            profiles_right = np.array(examples['author_right_vectors'])
            samples['user_context'] = np.concatenate((profiles_left, profiles_right), axis=1)

        # Gather target labels
        samples['labels'] = examples['labels']
        return samples
