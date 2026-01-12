
import os
from tqdm import tqdm
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

class ParallelExecutor:
    """
    Generic parallel executor for running functions with items in parallel.
    """

    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize the parallel executor.

        Args:
            max_workers: Maximum number of worker threads. If None, uses CPU count.
        """
        self.max_workers = max_workers

    def run_in_parallel(self, function, item_list: List,
                       progress_desc: str = "Processing",
                       max_workers: Optional[int] = None, **kwargs) -> List:
        """
        Execute a function in parallel for each item.

        Args:
            function: Function to execute. Should accept (item, **kwargs) as arguments.
            item_list: List of items to process.
            progress_desc: Description for the progress bar.
            max_workers: Override the default max_workers for this execution.
            **kwargs: Additional keyword arguments to pass to the function.

        Returns:
            List of results from the function calls (in order of completion).
        """
        if not item_list:
            return []

        # Determine number of workers
        workers = max_workers or self.max_workers
        if workers is None:
            workers = min(len(item_list), os.cpu_count() or 1)

        print(f"    {progress_desc}: {len(item_list)} items using {workers} workers...")

        results = []

        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(function, item, **kwargs): item
                for item in item_list
            }

            # Process completed tasks with progress bar
            with tqdm(total=len(item_list), desc=progress_desc, unit="item") as pbar:
                for future in as_completed(future_to_item):
                    item = future_to_item[future]
                    try:
                        result = future.result()
                        results.append(result)
                        pbar.update(1)
                    except Exception as exc:
                        print(f'Processing item {item} generated an exception: {exc}')
                        results.append(None)  # Add None for failed items
                        pbar.update(1)

        return results

    def run_in_parallel_no_return(self, function, item_list: List,
                                 progress_desc: str = "Processing",
                                 max_workers: Optional[int] = None, **kwargs) -> None:
        """
        Execute a function in parallel for each item without collecting results.
        More memory efficient when you don't need the return values.

        Args:
            function: Function to execute. Should accept (item, **kwargs) as arguments.
            item_list: List of items to process.
            progress_desc: Description for the progress bar.
            max_workers: Override the default max_workers for this execution.
            **kwargs: Additional keyword arguments to pass to the function.
        """
        if not item_list:
            return

        # Determine number of workers
        workers = max_workers or self.max_workers
        if workers is None:
            workers = min(len(item_list), os.cpu_count() or 1)

        print(f"    {progress_desc}: {len(item_list)} items using {workers} workers...")

        import time
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(function, item, **kwargs): item
                for item in item_list
            }

            print(f"    All {len(item_list)} tasks submitted, waiting for completion...")

            # Process completed tasks with progress bar
            with tqdm(total=len(item_list), desc=progress_desc, unit="item",
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
                for future in as_completed(future_to_item):
                    item = future_to_item[future]
                    try:
                        future.result()  # Don't store the result
                        pbar.update(1)
                    except Exception as exc:
                        print(f'Processing item {item} generated an exception: {exc}')
                        pbar.update(1)

        elapsed = time.time() - start_time
        print(f"    Completed {progress_desc} in {elapsed:.2f} seconds ({elapsed/len(item_list):.2f}s per item)")
