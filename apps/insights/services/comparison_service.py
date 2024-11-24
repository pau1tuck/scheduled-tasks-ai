# apps/insights/services/comparison_service.py
"""
Comparison Service for Dataset Summaries
Handles LLM comparison generation and logging for two dataset summaries.
"""

import json
import logging
from django.db import transaction
from apps.insights.models.comparison import Comparison, KeyMetricComparison
from apps.insights.models.summary import Summary
from apps.insights.services.openai.comparison_generator import generate_comparison
from apps.insights.services.openai.schemas import ComparisonOutput

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def prepare_summary(data_summary: dict) -> str:
    """
    Combines dataset_summary and key_metrics from a structured dataset summary into a single string for LLM input.

    Args:
        data_summary (dict): A dictionary containing 'dataset_summary' (str) and 'key_metrics' (list of dicts).

    Returns:
        str: A combined string representation of the dataset summary and its key metrics.
    """
    try:
        if not data_summary.get("dataset_summary"):
            raise ValueError("Missing 'dataset_summary' in data_summary.")

        if not data_summary.get("key_metrics"):
            raise ValueError("Missing 'key_metrics' in data_summary.")

        key_metrics_str = "\n".join(
            f"- {metric['name']}: {metric['value']}"
            for metric in data_summary["key_metrics"]
            if "name" in metric and "value" in metric
        )

        if not key_metrics_str:
            logging.warning("Key metrics are empty or malformed.")

        return f"{data_summary['dataset_summary']}\n\nKey Metrics:\n{key_metrics_str}"
    except Exception as e:
        logging.error(f"Failed to prepare summary: {e}")
        raise


def process_comparison(summary1_id: int, summary2_id: int) -> ComparisonOutput:
    """
    Processes two dataset summaries, merges them into strings, and generates a structured comparison.

    Args:
        summary1_id (int): The database ID of the first summary.
        summary2_id (int): The database ID of the second summary.

    Returns:
        ComparisonOutput: A structured comparison containing a summary and key metrics comparison.
    """
    try:
        logging.info("Starting comparison of dataset summaries...")

        # Fetch the summaries from the database
        summary1_obj = Summary.objects.get(id=summary1_id)
        summary2_obj = Summary.objects.get(id=summary2_id)

        # Prepare data_summary dictionaries
        data_summary1 = {
            "dataset_summary": summary1_obj.dataset_summary,
            "key_metrics": list(summary1_obj.keymetric_set.values("name", "value")),
        }
        data_summary2 = {
            "dataset_summary": summary2_obj.dataset_summary,
            "key_metrics": list(summary2_obj.keymetric_set.values("name", "value")),
        }

        # Validate and prepare text strings for the LLM
        summary1 = prepare_summary(data_summary1)
        summary2 = prepare_summary(data_summary2)

        logging.info("Generated summaries for comparison.")
        logging.debug(f"Summary 1: {summary1}")
        logging.debug(f"Summary 2: {summary2}")

        # Generate comparison using LLM
        comparison_result = generate_comparison(summary1, summary2)

        # Save comparison to database
        save_comparison_to_database(summary1_id, summary2_id, comparison_result)

        # Save comparison to file
        save_comparison_to_file(comparison_result, summary1_id, summary2_id)

        # Log detailed results
        logging.info("Comparison completed successfully.")
        logging.debug(f"Raw comparison result: {comparison_result}")

        return comparison_result

    except Summary.DoesNotExist as e:
        logging.error(f"Summary not found: {e}")
        raise ValueError(f"Summary not found: {e}")

    except Exception as e:
        logging.error(f"Unexpected error during comparison: {e}")
        raise


def save_comparison_to_database(
    summary1_id: int, summary2_id: int, comparison_result: ComparisonOutput
):
    """
    Save the LLM comparison result into the database.

    Args:
        summary1_id (int): ID of the first summary (Week 1).
        summary2_id (int): ID of the second summary (Week 2).
        comparison_result (ComparisonOutput): The structured comparison result from LLM.
    """
    try:
        with transaction.atomic():
            # Fetch the summaries
            summary1 = Summary.objects.get(id=summary1_id)
            summary2 = Summary.objects.get(id=summary2_id)

            logging.info(
                f"Saving comparison for summaries {summary1_id} and {summary2_id}..."
            )

            # Create the Comparison object
            comparison = Comparison.objects.create(
                summary1=summary1,
                summary2=summary2,
                comparison_summary=comparison_result.comparison_summary,
            )

            # Create KeyMetricComparison objects
            for metric in comparison_result.key_metrics_comparison:
                KeyMetricComparison.objects.create(
                    comparison=comparison,
                    name=metric.name,
                    value1=metric.value1,
                    value2=metric.value2,
                    description=metric.description,
                )

            logging.info(
                f"Comparison saved successfully for summaries {summary1_id} and {summary2_id}."
            )

    except Summary.DoesNotExist as e:
        logging.error(f"Summary not found: {e}")
        raise ValueError(f"Summary not found: {e}")
    except Exception as e:
        logging.error(f"Failed to save comparison to the database: {e}")
        raise


def save_comparison_to_file(
    comparison_result: ComparisonOutput, summary1_id: int, summary2_id: int
):
    """
    Saves the structured comparison result to a JSON file resembling the database entry.

    Args:
        comparison_result (ComparisonOutput): The structured comparison result.
        summary1_id (int): The database ID of the first summary.
        summary2_id (int): The database ID of the second summary.
    """
    try:
        file_path = "comparison_output.json"
        logging.info(f"Saving comparison result to {file_path}...")

        # Construct the data dictionary to match database structure
        data = {
            "summary1": summary1_id,
            "summary2": summary2_id,
            "comparison_summary": comparison_result.comparison_summary,
            "key_metrics_comparison": [
                {
                    "name": metric.name,
                    "value1": metric.value1,
                    "value2": metric.value2,
                    "description": metric.description,
                    "percentage_difference": (
                        ((metric.value2 - metric.value1) / metric.value1) * 100
                        if metric.value1 != 0
                        else None
                    ),
                }
                for metric in comparison_result.key_metrics_comparison
            ],
        }

        # Write to the JSON file
        with open(file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

        logging.info("Comparison result saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save comparison result to file: {e}")
        raise
