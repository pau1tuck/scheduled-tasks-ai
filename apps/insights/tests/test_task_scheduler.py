# apps/insights/tests/test_task_scheduler.py
import os
import pytest
from django_q.tasks import async_task, result
from apps.insights.tasks import process_week_task
import time


@pytest.mark.django_db
def test_process_week_task():
    """
    Test the process_week_task function via async_task and print the result.
    """
    # Correctly resolve the file path
    file_path = os.path.join(os.path.dirname(__file__), "../data/ga4_data.csv")
    start_date = "2024-01-01"  # Example start date
    week_number = 1  # Testing for Week 1

    try:
        # Trigger the task via Django-Q
        task_id = async_task(
            "apps.insights.tasks.process_week_task",
            file_path,
            start_date,
            week_number,
        )

        # Wait for the result with retries
        retries = 10
        result_data = None
        while retries > 0:
            result_data = result(task_id)
            if result_data is not None:
                break
            retries -= 1
            time.sleep(1)  # Wait 1 second before retrying

        # Validate and print the result
        assert result_data is not None, "Task did not return a result."
        print("Task Result:")
        print(result_data)

    except Exception as e:
        pytest.fail(f"Task test failed: {e}")
