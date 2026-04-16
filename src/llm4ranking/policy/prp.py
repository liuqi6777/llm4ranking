import re
from typing import Union

from llm4ranking.lm.base import BatchLMOutput, Capability, LMOutput
from llm4ranking.policy.base import PairwisePolicy, PolicyResult


DEFAULT_PROMPT_TEMPLATE = """Given a query: {{ query }}, which of the following two documents is more relevant to the query?

Document A: {{ doc1 }}

Document B: {{ doc2 }}

Only output "A" or "B", do not say anything else or explain.
"""


class PRP(PairwisePolicy):
    """PRP (Pairwise Ranking Prompting) that compares two documents at a time.
    
    This model takes pairs of documents and determines which one is more relevant
    to the query. It uses a simple A/B comparison approach.
    """

    DEFAULT_PROMPT_TEMPLATE = DEFAULT_PROMPT_TEMPLATE

    name = "PRP"
    supports_batch = True
    required_capabilities = {Capability.GENERATE, Capability.BATCH_GENERATE}

    def compare(
        self,
        query: str,
        doc1: str,
        doc2: str,
        return_lm_outputs: bool = False,
        **kwargs,
    ) -> Union[int, PolicyResult[int]]:
        """Compare two documents and determine which is more relevant.

        Args:
            query (str): The search query
            doc1 (str): First document to compare
            doc2 (str): Second document to compare
            return_lm_outputs (bool, optional): Whether to return LM outputs. Defaults to False.
            **kwargs: Additional arguments passed to the LM

        Returns:
            Union[int, tuple[int, LMOutput]]: 
                Returns 1 if doc1 is more relevant, -1 if doc2 is more relevant, 0 if tied.
                If return_lm_outputs is True, also returns the LM outputs.
        """
        result = self.compare_many(
            query,
            [(doc1, doc2)],
            return_lm_outputs=True,
            **kwargs,
        )
        if return_lm_outputs:
            lm_outputs = LMOutput(
                text=[
                    result.lm_outputs["forward"].text[0],
                    result.lm_outputs["reverse"].text[0],
                ]
            )
            return self.make_result(result.value[0], lm_outputs)

        return result.value[0]

    def create_batch_messages(
        self,
        query: str,
        doc_pairs: list[tuple[str, str]],
        reverse: bool = False,
    ) -> list[list[dict[str, str]]]:
        if reverse:
            return [self.create_messages(query, doc2, doc1) for doc1, doc2 in doc_pairs]
        return [self.create_messages(query, doc1, doc2) for doc1, doc2 in doc_pairs]

    def parse_batch_outputs(
        self,
        forward_outputs: BatchLMOutput,
        reverse_outputs: BatchLMOutput,
    ) -> list[int]:
        scores = []
        forward_texts = forward_outputs.text or []
        reverse_texts = reverse_outputs.text or []
        for forward_text, reverse_text in zip(forward_texts, reverse_texts):
            if self.parse_output(forward_text) == "a" and self.parse_output(reverse_text) == "b":
                scores.append(1)
            elif self.parse_output(forward_text) == "b" and self.parse_output(reverse_text) == "a":
                scores.append(-1)
            else:
                scores.append(0)
        return scores

    def compare_many(
        self,
        query: str,
        doc_pairs: list[tuple[str, str]],
        return_lm_outputs: bool = False,
        **kwargs,
    ) -> Union[list[int], PolicyResult[list[int]]]:
        forward_outputs = self.lm.generate_batch(
            self.create_batch_messages(query, doc_pairs, reverse=False),
            **kwargs,
        )
        reverse_outputs = self.lm.generate_batch(
            self.create_batch_messages(query, doc_pairs, reverse=True),
            **kwargs,
        )
        scores = self.parse_batch_outputs(forward_outputs, reverse_outputs)
        if return_lm_outputs:
            return self.make_result(scores, {"forward": forward_outputs, "reverse": reverse_outputs})
        return scores

    def create_messages(
        self,
        query: str,
        doc1: str,
        doc2: str,
    ) -> list[dict[str, str]]:
        """Create prompt messages for comparing two documents.

        Args:
            query (str): The search query
            doc1 (str): First document
            doc2 (str): Second document

        Returns:
            list[dict[str, str]]: Formatted prompt messages
        """
        messages = [
            {"role": "user", "content": self.prompt_template.render(query=query, doc1=doc1, doc2=doc2)}
        ]
        return messages

    def parse_output(self, output: str) -> str:
        """Parse the LM output into a document preference.

        Args:
            output (str): Raw output from the LM

        Returns:
            str: 'a' or 'b' indicating which document was preferred
        """
        if not isinstance(output, str):
            return ""

        match = re.search(r"\b([ab])\b", output.strip().lower())
        if match is None:
            return ""
        return match.group(1)
