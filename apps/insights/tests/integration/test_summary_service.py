# apps/insights/tests/test_summary_service.py

import os
import pytest
from apps.insights.services.summary_service import process_week
from apps.insights.services.openai.schemas import SummaryOutput
from django.conf import settings

print(f"SECRET_KEY in Test: {settings.SECRET_KEY}")


def test_process_week():
    """
    Test the summary service with the actual CSV file and a fixed start date.
    Prints the output for manual verification.
    """
    file_path = os.path.join(os.path.dirname(__file__), "../data/ga4_data.csv")
    start_date = "2024-01-08"
    week_number = 1  # Testing for Week 1

    try:
        # Process the week
        result = process_week(file_path, start_date, week_number)

        # Verify the output type
        assert isinstance(
            result, SummaryOutput
        ), "Result is not a SummaryOutput object."

        # Print dataset summary and key metrics for manual verification
        print("Dataset Summary:")
        print(result.dataset_summary)
        print("\nKey Metrics:")
        for metric in result.key_metrics:
            print(f"{metric.name}: {metric.value}")

        print("Test completed successfully.")

    except Exception as e:
        pytest.fail(f"Summary service test failed: {e}")
