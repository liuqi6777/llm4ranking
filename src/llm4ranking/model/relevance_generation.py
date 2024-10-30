import torch

from llm4ranking.model.os_llm import OSLlm


class RelevanceGeneration(OSLlm):

    def create_messages(
        self,
        query: str,
        doc: str,
    ) -> str:
        input_context = f"Passage: {doc} Query:{query}\nDoes the passage answer the query? Answer:"
        messages = [{"role": "user", "content": input_context}]
        return messages

    @torch.no_grad()
    def __call__(self, query: str, doc: str) -> float:
        messages = self.create_messages(query, doc)
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            truncation=True,
        ).to(self.device)
        pos_token_id = self.tokenizer.convert_tokens_to_ids("Yes")
        logits = self.model(input_ids=input_ids)["logits"][0, -1, pos_token_id]
        return logits
