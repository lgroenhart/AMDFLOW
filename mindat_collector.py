import openmindat
import os
import csv
import json
import time
import sys
from requests.exceptions import Timeout, ConnectionError
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set API key as user input
API_KEY = input("Enter your Mindat API key: ").strip()
os.environ["MINDAT_API_KEY"] = API_KEY

from openmindat import GeomaterialRetriever, LocalitiesRetriever

def fetch_with_retry(retriever_call, max_retries=3, delay=2):
    """Helper function to retry failed requests"""
    for attempt in range(max_retries):
        try:
            result = retriever_call()
            return result
        except (Timeout, ConnectionError) as e:
            if attempt < max_retries - 1:
                wait_time = delay * (2 ** attempt)  # Exponential backoff
                logger.warning(f"Request timed out (attempt {attempt + 1}/{max_retries}). "
                             f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed after {max_retries} attempts: {e}")
                raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
    return None

def main():
    material_id = str(3314)  # Pyrite
    country = "Portugal"
    
    # Step 1: Get pyrite data with retry logic
    logger.info(f"Fetching pyrite (ID: {material_id}) data...")
    try:
        gr = GeomaterialRetriever()
        gr.expand("locality").id_in(material_id)
        
        # Use retry logic for the save operation
        fetch_with_retry(lambda: gr.saveto("../data/mindat_data"))
        logger.info("Pyrite data saved successfully.")
    except Exception as e:
        logger.error(f"Failed to fetch pyrite data: {e}")
        return
    
    # Step 2: Load pyrite data
    try:
        with open("../data/mindat_data/v1_geomaterials.json", "r") as file:
            pyrite_dict = json.load(file)
        
        pyrite_locs = [str(r) for r in pyrite_dict["results"][0]["locality"]]
        logger.info(f"Found {len(pyrite_locs)} pyrite localities.")
    except Exception as e:
        logger.error(f"Failed to load pyrite data: {e}")
        return
    
    # Step 3: Fetch localities in batches with periodic saving
    batch_size = 100
    save_every_n_batches = 5  # Save data every 5 batches
    total_data = []
    failed_batches = []

    # Create output directory
    os.makedirs("../data/mindat_data", exist_ok=True)

    # Define checkpoint files
    progress_file = "../data/mindat_data/progress.json"
    partial_data_file = "../data/mindat_data/partial_data.json"

    # Load existing progress if available
    start_idx = 0
    if os.path.exists(progress_file) and os.path.exists(partial_data_file):
        try:
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
                start_idx = progress_data.get('last_processed_idx', 0)
                loaded_failed_batches = progress_data.get('failed_batches', [])
                loaded_total_count = progress_data.get('total_retrieved', 0)
            
            with open(partial_data_file, 'r') as f:
                total_data = json.load(f)
            
            logger.info(f"Resuming from batch starting at index {start_idx}")
            logger.info(f"Already have {len(total_data)} localities loaded from previous run")
            logger.info(f"Previously failed batches: {len(loaded_failed_batches)}")
            
            # Continue with existing failed batches
            failed_batches = loaded_failed_batches
            
        except Exception as e:
            logger.warning(f"Failed to load progress files: {e}. Starting from scratch.")
            start_idx = 0
    else:
        logger.info("No previous progress found. Starting from scratch.")

    # Track successful batches for periodic saving
    successful_batch_count = 0

    for i in range(start_idx, len(pyrite_locs), batch_size):
        batch_start = i
        batch_end = min(i + batch_size, len(pyrite_locs))
        batch_ids = pyrite_locs[batch_start:batch_end]
        
        batch_num = i//batch_size + 1
        total_batches = (len(pyrite_locs) + batch_size - 1)//batch_size
        
        logger.info(f"Processing batch {batch_num}/{total_batches} "
                f"(indices {batch_start}-{batch_end-1})")
        
        try:
            # Create a new retriever for each batch
            lr = LocalitiesRetriever()
            id_str = ",".join(batch_ids)
            lr.id_in(id_str).country(country).fields("longitude,latitude,guid,txt")
            
            # Fetch with retry logic
            batch_result = fetch_with_retry(
                lambda: lr.get_dict(),
                max_retries=3,
                delay=5
            )
            
            if batch_result and "results" in batch_result:
                batch_data = batch_result["results"]
                total_data.extend(batch_data)
                successful_batch_count += 1
                logger.info(f"  → Successfully retrieved {len(batch_data)} localities")
                logger.info(f"  → Total collected so far: {len(total_data)} localities")
            else:
                logger.warning(f"  → No results returned for batch {batch_num}")
                failed_batches.append((batch_start, batch_end))
            
            # Save progress after each batch
            progress_data = {
                'last_processed_idx': batch_end,
                'failed_batches': failed_batches,
                'total_batches': total_batches,
                'current_batch': batch_num,
                'total_retrieved': len(total_data),
            }
            
            with open(progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
            
            # Save partial data periodically or after each batch
            if successful_batch_count % save_every_n_batches == 0 or batch_num == total_batches:
                with open(partial_data_file, 'w') as f:
                    json.dump(total_data, f, indent=2)
                logger.info(f"  → Saved partial data ({len(total_data)} localities)")
            
            # Add a small delay between batches
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"  → Failed to process batch {batch_num}: {e}")
            failed_batches.append((batch_start, batch_end))
            
            # Save progress even if batch fails
            progress_data = {
                'last_processed_idx': batch_end,
                'failed_batches': failed_batches,
                'total_batches': total_batches,
                'current_batch': batch_num,
                'total_retrieved': len(total_data),
                'last_error': str(e),
            }
            
            with open(progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
            
            time.sleep(5)  # Longer delay after failure

    # Step 4: Final output with merged data
    from datetime import datetime

    # Generate final filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"../data/mindat_data/v1_localities_{country}_{timestamp}.json"

    # Prepare final output
    output_data = {
        "count": len(total_data),
        "next": None,
        "previous": None,
        "results": total_data,
        "metadata": {
            "material_id": material_id,
            "country": country,
            "total_localities_processed": len(pyrite_locs),
            "successfully_retrieved": len(total_data),
            "failed_batches": len(failed_batches),
            "total_batches": total_batches,
            "completed_timestamp": datetime.now().isoformat()
        }
    }

    # Save final output
    with open(output_filename, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Final data saved to {output_filename}")
    logger.info(f"Successfully retrieved {len(total_data)} localities out of {len(pyrite_locs)}")

    # Save a summary report
    summary_file = f"../data/mindat_data/summary_{timestamp}.json"
    summary = {
        "total_requested": len(pyrite_locs),
        "retrieved": len(total_data),
        "failed": len(failed_batches) * batch_size,
        "success_rate": f"{(len(total_data)/len(pyrite_locs)*100):.1f}%",
        "failed_batches": failed_batches,
        "output_file": output_filename,
        "timestamp": datetime.now().isoformat()
    }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Clean up temporary files if run completed successfully
    if len(failed_batches) == 0:
        if os.path.exists(progress_file):
            os.remove(progress_file)
            logger.info("Progress file cleaned up")
        if os.path.exists(partial_data_file):
            os.remove(partial_data_file)
            logger.info("Partial data file cleaned up")
    else:
        logger.warning(f"{len(failed_batches)} batches failed. Temporary files preserved.")
        
        # Save failed batches for manual inspection
        failed_file = f"../data/mindat_data/failed_batches_{timestamp}.json"
        failed_details = {
            "failed_batches": failed_batches,
            "batch_size": batch_size,
            "total_failed_items": len(failed_batches) * batch_size,
            "failed_indices": [],
            "timestamp": datetime.now().isoformat()
        }
        
        # Add specific failed indices
        for start, end in failed_batches:
            failed_details["failed_indices"].extend(list(range(start, end)))
        
        with open(failed_file, "w") as f:
            json.dump(failed_details, f, indent=2)
        
        logger.info(f"Failed batches details saved to {failed_file}")
main()