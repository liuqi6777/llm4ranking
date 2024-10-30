import torch

from llm4ranking.model.os_llm import OSLlm


class PairwiseComparison(OSLlm):

    def create_messages(
        self,
        query: str,
        doc1: str,
        doc2: str,
    ) -> str:
        input_context = f"Given a query: {query}, which of the following two passages is more relevant to the query?\n\nPassage A: {doc1}\n\nPassage B: {doc2}\n\nOutput Passage A or Passage B."
        messages = [{"role": "user", "content": input_context}]
        return messages

    @torch.no_grad()
    def __call__(self, query: str, doc1: str, doc2: str) -> float:
        prompt_1 = self.tokenizer.apply_chat_template(
            self.create_messages(query, doc1, doc2),
            add_generation_prompt=False,
            tokenize=False,
            truncation=True,
        ) + " Passage:"
        prompt_2 = self.tokenizer.apply_chat_template(
            self.create_messages(query, doc2, doc1),
            add_generation_prompt=False,
            tokenize=False,
            truncation=True,
        ) + " Passage:"
        input_ids = self.tokenizer([prompt_1, prompt_2], return_tensors="pt", truncation=True).input_ids.to(self.device)
        outputs = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=1,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=False,
            temperature=0.0,
            top_p=None,
        )
        output_1 = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        output_2 = self.tokenizer.decode(outputs[1][input_ids.shape[1]:], skip_special_tokens=True).strip()
        if output_1 == "A" and output_2 == "B":
            score = 1
        elif output_1 == "B" and output_2 == "A":
            score = -1
        else:
            score = 0
        return score
