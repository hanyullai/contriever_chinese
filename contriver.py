from __future__ import annotations
import torch
from transformers import AutoTokenizer, AutoModel, BertModel
import json
import re

class Contriever(BertModel):
    def __init__(self, config, pooling="average", **kwargs):
        super().__init__(config, add_pooling_layer=False)
        if not hasattr(config, "pooling"):
            self.config.pooling = pooling

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        normalize=False,
    ):

        model_output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden = model_output["last_hidden_state"]
        last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)

        if self.config.pooling == "average":
            emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.config.pooling == "cls":
            emb = last_hidden[:, 0]

        if normalize:
            emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb

class QuestionReferenceDensity_forPredict(torch.nn.Module):
    def __init__(self, question_encoder_path, reference_encoder_path) -> None:
        super().__init__()
        self.question_encoder = Contriever.from_pretrained(question_encoder_path)
        self.reference_encoder = Contriever.from_pretrained(reference_encoder_path)

       
    def forward(self, question, sentences):
        temp = 0.05
        cls_q = self.question_encoder(**question)
        cls_r_sentences = self.reference_encoder(**sentences)
        cls_q /= temp
        results = torch.matmul(cls_q, torch.transpose(cls_r_sentences, 0, 1))
        return results
    
class QuestionReferenceDensityScorer:
    def __init__(self, question_encoder_path, reference_encoder_path, device=None) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(question_encoder_path)
        self.model = QuestionReferenceDensity_forPredict(question_encoder_path, reference_encoder_path)
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if not device else device
        self.model = self.model.to(self.device).eval()

    def get_embeddings(self, sentences: list[str]) -> torch.Tensor:
        # Tokenization and Inference
        torch.cuda.empty_cache()
        with torch.no_grad():
            question_inputs = self.tokenizer([sentences[0]], padding=True,
                                    truncation=True, return_tensors='pt')
            select_inputs = self.tokenizer(sentences[1:], padding=True,
                                    truncation=True, return_tensors='pt')
            for key in question_inputs:
                question_inputs[key] = question_inputs[key].to(self.device)
            for key in select_inputs:
                select_inputs[key] = select_inputs[key].to(self.device)
            
            outputs = self.model(question_inputs, select_inputs)
            sentence_embeddings = outputs

            return sentence_embeddings

    def score_documents_on_query(self, query: str, documents: list[str]) -> torch.Tensor:
        result = self.get_embeddings([query, *documents])
        return result[0]

    def select_topk(self, query: str, documents: list[str], k=1):
        """
        Returns:
            `ret`: `torch.return_types.topk`, use `ret.values` or `ret.indices` to get value or index tensor
        """
        scores = []
        max_batch = 500
        for i in range((len(documents) + max_batch - 1) // max_batch):
            scores.append(self.score_documents_on_query(query, documents[max_batch*i:max_batch*(i+1)]).to('cpu'))
        scores = torch.concat(scores)
        return scores.topk(min(k, len(scores)))

def test_contriever_scorer():
    sentences = open('retrieval_data.txt').read().split('\n')
    scorer = QuestionReferenceDensityScorer('ckpt/question_encoder', 'ckpt/reference_encoder')
    while True:
        query = input('Input your query >>>')
        print(scorer.score_documents_on_query(query, sentences))
        target_idx = scorer.select_topk(query, sentences, 3).indices
        result = [sentences[idx] for idx in target_idx]
        print(result)

if __name__ == "__main__":
    test_contriever_scorer()
