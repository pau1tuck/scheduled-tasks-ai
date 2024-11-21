# apps/insights/services/openai/llm_integration.py
# TODO: Add logic to truncate long outputs from describe()
# TODO: Implement OpenAI error handling
# TODO: Detect and handle refusals
# ? Two-Phased Approach for Large Datasets?

from decouple import config
import instructor
from openai import OpenAI
from pydantic import BaseModel
from typing import List

# Load the OpenAI API key from environment variables
openai_api_key = config("OPENAI_API_KEY")

# Initialize OpenAI client with Instructor
client = instructor.from_openai(OpenAI(api_key=openai_api_key))


# Define structured output models
class KeyMetric(BaseModel):
    metric_name: str
    value: float


class SummaryOutput(BaseModel):
    plain_english_summary: str
    key_metrics: List[KeyMetric]


# Generate a structured summary using OpenAI API
def generate_summary(statistical_summary: str) -> SummaryOutput:
    prompt = f"""
    The following is a statistical summary of a dataset:

    {statistical_summary}

    Please:
    1. Write a plain English summary of the dataset.
    2. Highlight key metrics.
    """
    try:
        # Make the API call
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",  # Ensure the correct model is used
            messages=[{"role": "user", "content": prompt}],
            response_model=SummaryOutput,
        )
        return response

    except Exception as e:
        raise ValueError(f"Error generating summary with OpenAI: {e}")
