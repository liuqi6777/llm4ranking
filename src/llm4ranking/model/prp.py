from typing import Union

from llm4ranking.lm.base import LMOutput
from llm4ranking.model.base import BaseRankingModel


DEFAULT_PROMPT_TEMPLATE = """Given a query: {{ query }}, which of the following two documents is more relevant to the query?

Document A: {{ doc1 }}

Document B: {{ doc2 }}

Only output "A" or "B", do not say anything else or explain.
"""


class PRP(BaseRankingModel):
    """PRP (Pairwise Ranking Prompting) that compares two documents at a time.
    
    This model takes pairs of documents and determines which one is more relevant
    to the query. It uses a simple A/B comparison approach.
    """

    DEFAULT_PROMPT_TEMPLATE = DEFAULT_PROMPT_TEMPLATE

    ranker = "pairwise"
    name = "PRP"

    def __call__(
        self,
        query: str,
        doc1: str,
        doc2: str,
        return_lm_outputs: bool = False,
        **kwargs,
    ) -> Union[int, tuple[int, LMOutput]]:
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
        lm_outputs = self.lm.generate(self.create_messages(query, doc1, doc2), return_num_tokens=True, **kwargs)
        lm_outputs_reverse = self.lm.generate(self.create_messages(query, doc2, doc1), return_num_tokens=True, **kwargs)

        outputs = [lm_outputs.text, lm_outputs_reverse.text]
        if self.parse_output(outputs[0]) == "a" and self.parse_output(outputs[1]) == "b":
            score = 1
        elif self.parse_output(outputs[0]) == "b" and self.parse_output(outputs[1]) == "a":
            score = -1
        else:
            score = 0

        if return_lm_outputs:
            lm_outputs.num_generated_tokens += lm_outputs_reverse.num_generated_tokens
            lm_outputs.num_processed_tokens += lm_outputs_reverse.num_processed_tokens
            lm_outputs.text = [lm_outputs.text, lm_outputs_reverse.text]
            return score, lm_outputs

        return score

    def create_messages(
        self,
        query: str,
        doc1: str,
        doc2: str,
    ) -> str:
        """Create prompt messages for comparing two documents.

        Args:
            query (str): The search query
            doc1 (str): First document
            doc2 (str): Second document

        Returns:
            str: Formatted prompt messages
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
        return output.strip().lower()[0]
