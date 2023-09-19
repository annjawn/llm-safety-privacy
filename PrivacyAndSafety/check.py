from typing import Any, Optional, List

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.prompts.base import StringPromptValue
from langchain.prompts.chat import ChatPromptValue
from langchain.schema import AIMessage, HumanMessage

from transformers import pipeline, AutoTokenizer
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

class SafetyPrivacyCheck:
    def __init__(
        self,        
        pii_labels: Optional[List[str]] = None,
        fail_on_pii: bool = False,
        pii_threshold: float = 0.5,
        toxicity_threshold: float = 0.8,
        pii_mask_character: str = "*",
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ):
        self.pii_labels = pii_labels
        self.fail_on_pii = fail_on_pii
        self.pii_threshold = pii_threshold
        self.toxicity_threshold = toxicity_threshold    
        self.mask_character = pii_mask_character    
        self.run_manager = run_manager

    def _print_verbose(self, message: str) -> None:
        if self.run_manager:
            self.run_manager.on_text(message)

    def _convert_prompt_to_text(self, prompt: Any) -> str:
        input_text = str()

        if isinstance(prompt, StringPromptValue):
            input_text = prompt.text
        elif isinstance(prompt, str):
            input_text = prompt
        elif isinstance(prompt, ChatPromptValue):
            message = prompt.messages[-1]
            self.chat_message_index = len(prompt.messages) - 1
            if isinstance(message, HumanMessage):
                input_text = message.content

            if isinstance(message, AIMessage):
                input_text = message.content
        else:
            raise ValueError(
                f"Invalid input type {type(input)}. "
                "Must be a PromptValue, str, or list of BaseMessages."
            )
        return input_text

    def _convert_text_to_prompt(self, prompt: Any, text: str) -> Any:
        if isinstance(prompt, StringPromptValue):
            return StringPromptValue(text=text)
        elif isinstance(prompt, str):
            return text
        elif isinstance(prompt, ChatPromptValue):
            messages = prompt.messages
            message = messages[self.chat_message_index]

            if isinstance(message, HumanMessage):
                messages[self.chat_message_index] = HumanMessage(
                    content=text,
                    example=message.example,
                    additional_kwargs=message.additional_kwargs,
                )
            if isinstance(message, AIMessage):
                messages[self.chat_message_index] = AIMessage(
                    content=text,
                    example=message.example,
                    additional_kwargs=message.additional_kwargs,
                )
            return ChatPromptValue(messages=messages)
        else:
            raise ValueError(
                f"Invalid input type {type(input)}. "
                "Must be a PromptValue, str, or list of BaseMessages."
            )
        
    def _perform_anonymization(self, prompt: str, results: Any) -> str:
        from presidio_anonymizer.entities import OperatorConfig
        operators = dict()
        for result in results:
            operators[result.entity_type] = OperatorConfig("mask", {"chars_to_mask": result.end-result.start,
                                                                    "masking_char": self.mask_character,
                                                                    "from_end": False})
        anonymizer = AnonymizerEngine()
        anonymized_results = anonymizer.anonymize(
            text=prompt, analyzer_results=results, operators=operators
        )
        return anonymized_results.text
    
    def _chunk_text(self,prompt: str) -> List[str]:
        tokenizer = AutoTokenizer.from_pretrained("tensor-trek/distilbert-toxicity-classifier")

        # Get token IDs for the input prompt
        token_ids = tokenizer.encode(prompt, add_special_tokens=True)

        # We account for special tokens that might be added during tokenization
        num_special_tokens = len(token_ids) - len(tokenizer.encode(prompt, add_special_tokens=False))
        max_chunk_length = 512 - num_special_tokens

        # Split token IDs into chunks of up to max_chunk_length IDs
        chunks = [tokenizer.decode(token_ids[i:i+max_chunk_length]) for i in range(0, len(token_ids), max_chunk_length)]

        return chunks

        
    def check(self, prompt: Any) -> str:  
        # convert prompt to text
        input_text = self._convert_prompt_to_text(prompt=prompt)
        
        # Chunk text if needed
        chunks = self._chunk_text(prompt=input_text)

        # initialize classifier
        classifier = pipeline("text-classification", model="tensor-trek/distilbert-toxicity-classifier")

        # perform classification
        self._print_verbose("Checking for Toxic content...\n")
        results = classifier(chunks)

        # check results
        for result in results:
            if result['label'] == 'TOXIC' and result['score'] > self.toxicity_threshold:
                self._print_verbose("Toxic content found in text. Stopping...\n")
                raise ValueError('Toxic content found in text. Stopping...\n')

        # perform PII detection
        analyzer = AnalyzerEngine()

        # Call analyzer to get results
        self._print_verbose("Checking for PII...\n")
        pii_results = analyzer.analyze(text=input_text,
                                       entities=self.pii_labels,
                                       language='en')
        
        if self.fail_on_pii:
            for result in pii_results:
                if result.score > self.pii_threshold:
                    self._print_verbose("PII found and fail_on_pii is True. Stopping...\n")
                    raise ValueError("PII found and fail_on_pii is True. Stopping...\n")
            
            # nothing was found return original text    
            return self._convert_text_to_prompt(prompt=prompt, text=input_text)
        else:
            output_text = self._perform_anonymization(prompt=input_text, results=pii_results)

            # return anonymized text
            return self._convert_text_to_prompt(prompt=prompt, text=output_text)