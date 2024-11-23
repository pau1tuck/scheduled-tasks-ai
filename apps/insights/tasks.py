# apps/insights/tasks.py
"""
Task definitions for the Insights app.
These tasks integrate with Django-Q to run asynchronously.
"""

import logging
from datetime import timedelta
from django.utils import timezone
from django_q.models import Schedule
from django_q.tasks import async_task, result_group

from apps.insights.services.summary_service import process_week

logger = logging.getLogger(__name__)


def process_week_task(file_path: str, start_date: str, week: int):
    """
    Processes a single week's data and generates an LLM summary.

    Args:
        file_path (str): Path to the CSV file.
        start_date (str): Start date for the week (YYYY-MM-DD).
        week (int): Week number (1 or 2).
    """
    try:
        logger.info(f"Processing Week {week} starting from {start_date}...")
        result = process_week(file_path, start_date, week)
        logger.info(f"Week {week} summary generated successfully.")
        return {
            "dataset_summary": result.dataset_summary,
            "key_metrics": result.key_metrics,
        }
    except Exception as e:
        logger.error(f"Failed to process Week {week}: {e}")
        raise


def schedule_two_summaries(file_path: str, start_date: str):
    """
    Schedules tasks to process Week 1 after a 1-minute delay,
    followed by processing Week 2 in sequence using group logic.
    """
    try:
        logger.info("Scheduling Week 1 task to run in 1 minute...")

        # Create a group for dependent tasks
        group_name = "process_summaries"

        # Schedule Week 1
        Schedule.objects.create(
            func="apps.insights.tasks.process_week_task",
            args=(file_path, start_date, 1),  # Pass arguments as a tuple
            schedule_type=Schedule.ONCE,
            next_run=timezone.now() + timedelta(minutes=1),
            group=group_name,  # Assign task to a group
        )
        logger.info("Week 1 task scheduled successfully.")

        # Calculate Week 2 start date
        week2_start_date = pd.to_datetime(start_date) + pd.Timedelta(days=7)

        # Add Week 2 to the group (ensures it runs after Week 1)
        async_task(
            "apps.insights.tasks.process_week_task",
            file_path,
            week2_start_date.strftime("%Y-%m-%d"),
            2,
            group=group_name,  # Ensure it runs as part of the same group
        )
        logger.info("Week 2 task added to the group successfully.")

    except Exception as e:
        logger.error(f"Failed to schedule two summaries: {e}")
        raise


def fetch_group_results(group_name: str):
    """
    Fetches the results of tasks in the specified group.

    Args:
        group_name (str): Name of the group to fetch results from.

    Returns:
        list: Results of all tasks in the group.
    """
    try:
        logger.info(f"Fetching results for group: {group_name}")
        results = result_group(group_name)
        if not results:
            logger.warning(f"No results found for group: {group_name}")
        return results
    except Exception as e:
        logger.error(f"Failed to fetch results for group {group_name}: {e}")
        raise


# Example trigger for manual testing:
# schedule_two_summaries("/path/to/ga4_data.csv", "2024-01-01")
