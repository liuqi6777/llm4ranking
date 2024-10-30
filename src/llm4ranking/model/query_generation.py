import torch

from llm4ranking.model.os_llm import OSLlm


class QueryGeneration(OSLlm):

    def create_messages(
        self,
        query: str,
        doc: str,
    ) -> str:
        input_context = f"Passage: {doc}\nPlease write a question based on this passage."
        messages = [{"role": "user", "content": input_context}, {"role": "assistant", "content": query}]
        return messages

    @torch.no_grad()
    def __call__(self, query: str, doc: str) -> float:
        messages = self.create_messages(query, doc)
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            return_tensors="pt",
            truncation=True,
        ).to(self.device)
        labels = self._mask_labels(messages, input_ids.clone())
        label_idx = (labels != -100).flatten()
        logits = self.model(input_ids=input_ids, labels=labels).logits[0, label_idx]
        labels = labels[0, label_idx]
        score = -torch.nn.functional.log_softmax(logits, dim=-1)[range(len(labels)), labels].sum().item()
        return score
