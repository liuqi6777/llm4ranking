import re
import torch

from llm4ranking.model.os_llm import OSLlm


class ListwiseRanker(OSLlm):

    def _add_system_message(self) -> str:
        return f"You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query."

    def _add_prefix_prompt(self, query: str, num: int) -> str:
        return f"I will provide you with {num} passages, each indicated by a numerical identifier []. Rank the passages based on their relevance to the search query: {query}.\n"

    def _add_post_prompt(self, query: str, num: int) -> str:
        return f"Search Query: {query}.\nRank the {num} passages above based on their relevance to the search query. All the passages should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [] > ..., e.g., [4] > [2] > ..., Only respond with the ranking results, do not say anything else or explain."

    def create_messages(
        self,
        query: str,
        candidates: list[str],
    ) -> str:
        num = len(candidates)
        messages = []
        input_context = self._add_prefix_prompt(query, num)
        for i, content in enumerate(candidates):
            content = self._replace_number(content.strip())
            input_context += f"[{i + 1}] {content}\n"
        input_context += self._add_post_prompt(query, num)
        # messages.append({"role": "system", "content": self._add_system_message()})
        messages.append({"role": "user", "content": input_context})
        return messages

    def _replace_number(self, s: str) -> str:
        return re.sub(r"\[(\d+)\]", r"(\1)", s)

    def parse_output(self, output: str) -> list[int]:
        response = self._clean_response(output)
        response = [int(x) - 1 for x in response.split()]
        response = self._remove_duplicate(response)
        return response

    def _clean_response(self, response: str) -> str:
        new_response = ""
        for c in response:
            if not c.isdigit():
                new_response += " "
            else:
                new_response += c
        new_response = new_response.strip()
        return new_response

    def _remove_duplicate(self, response: list[int]) -> list[int]:
        new_response = []
        for c in response:
            if c not in new_response:
                new_response.append(c)
        return new_response

    @torch.no_grad()
    def __call__(self, query: str, candidates: list[str]) -> dict[str]:
        messages = self.create_messages(query, [p["content"] for p in candidates])
        outputs = self.get_response(messages)
        permutation = self.parse_output(outputs)
        original_rank = [tt for tt in range(len(candidates))]
        permutation = [ss for ss in permutation if ss in original_rank]
        permutation = permutation + [tt for tt in original_rank if tt not in permutation]
        return permutation
