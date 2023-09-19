from typing import Any, Dict, List, Optional

from PrivacyAndSafety.check import SafetyPrivacyCheck
from langchain.callbacks.manager import (
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain

from langchain.pydantic_v1 import root_validator

class PrivacyAndSafetyChain(Chain):
    """A subclass of Chain, designed to apply moderation to LLMs."""

    output_key: str = "output"  #: :meta private:
    """Key used to fetch/store the output in data containers. Defaults to `output`"""

    input_key: str = "input"  #: :meta private:
    """Key used to fetch/store the input in data containers. Defaults to `input`"""

    pii_mask_character: str = "*"
    """Mask character for PII"""

    pii_labels: Optional[List[str]] = None
    """List of PII entities to check"""

    fail_on_pii: bool = False
    """Whether to fail when PII is detected"""

    pii_threshold: float = 0.5
    """PII confidence threshold"""

    toxicity_threshold: float = 0.8
    """Toxicity confidence threshold"""

    @root_validator(pre=True)
    def check_deps(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        try:
            import transformers     # noqa: F401
            import torch            # noqa: F401
        except ImportError:
            raise ModuleNotFoundError(
                "Could not import either transformers package. "
                "Please install transformers with `pip install transformers`."
                "Please install Pytorch with `pip install torch torchvision`"
            )
        try:
            import presidio_analyzer
            _ = presidio_analyzer.__name__
        except ImportError:
            raise ModuleNotFoundError(
                "Could not import either presidio-analyzer package. "
                "Please install transformers with `pip install presidio-analyzer`."
            )
        try:
            import presidio_anonymizer
            _ = presidio_anonymizer.__name__
        except ImportError:
            raise ModuleNotFoundError(
                "Could not import either presidio-anonymizer package. "
                "Please install transformers with `pip install presidio-anonymizer`."
            )
        try:
            import spacy
            from spacy.util import get_package_path

            # Check if 'en_core_web_lg' is installed
            package_path = get_package_path("en_core_web_lg")
            if not package_path:
                # If not installed, download it
                spacy.cli.download("en_core_web_lg")
                
        except ImportError:
            raise ModuleNotFoundError(
                "Could not import the spacy package. "
                "Please install it with `pip install spacy`."
            )
        return values

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        if run_manager:
            run_manager.on_text("Running PrivacyAndSafetyChain...\n")

        spc = SafetyPrivacyCheck(pii_labels=self.pii_labels,
                                 fail_on_pii=self.fail_on_pii,
                                 pii_threshold=self.pii_threshold,
                                 toxicity_threshold=self.toxicity_threshold,
                                 pii_mask_character=self.pii_mask_character,
                                 run_manager=run_manager)
        response = spc.check(prompt=inputs[self.input_keys[0]])
        return {self.output_key: response}


    @property
    def _chain_type(self) -> str:
        return "PrivacyAndSafetyChain"