import pandas as pd
import os
import hashlib
from collections import Counter
from datetime import datetime, timedelta

def get_content_hash(content):
    """Generate a hash from the content."""
    return hashlib.md5(content.encode('utf-8')).hexdigest()[:10]

def process_hits_csv(csv_path, team_id_filter=None, chunk_size=1000, timestamp_threshold_days=None):
    """
    Process the hits.csv file in chunks and create individual text files.
    
    Args:
        csv_path: Path to the hits.csv file
        team_id_filter: Optional team ID to filter by. If None, process all teams.
        chunk_size: Number of rows to process at once
        timestamp_threshold_days: Optional number of days to filter by. If None, process all timestamps.
    """
    # Create output directories for each status
    status_dirs = {
        'RELEVANT': 'relevant_posts',
        'REPLIED': 'relevant_posts',  # Same as RELEVANT
        'IGNORED': 'irrelevant_posts',  # Map to irrelevant_posts
        'NEW': 'new_posts',
        'CONTENT_REMOVED': 'content_removed_posts'
    }
    
    # Create all directories
    for dir_path in set(status_dirs.values()):
        os.makedirs(dir_path, exist_ok=True)
    
    # Initialize counters
    total_processed = 0
    status_counts = {status: 0 for status in status_dirs.keys()}
    f5bot_filtered_count = 0
    team_filtered_count = 0
    timestamp_filtered_count = 0
    
    # Calculate timestamp threshold if specified
    timestamp_threshold = None
    if timestamp_threshold_days is not None:
        timestamp_threshold = datetime.now() - timedelta(days=timestamp_threshold_days)
        print(f"Filtering out items older than {timestamp_threshold.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Track unique content hashes
    seen_hashes = set()
    duplicate_hashes = set()
    
    # Process the CSV file in chunks
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        for _, row in chunk.iterrows():
            # Skip rows with empty content
            if pd.isna(row.get('content')) or not row.get('content'):
                continue
            
            # Filter by timestamp if specified
            if timestamp_threshold is not None:
                timestamp_str = row.get('timestamp', '')
                if timestamp_str:
                    try:
                        if '+' in timestamp_str:
                            base_timestamp, tz = timestamp_str.split('+')
                            item_timestamp = datetime.strptime(base_timestamp, '%Y-%m-%d %H:%M:%S')
                        else:
                            item_timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                        
                        if item_timestamp < timestamp_threshold:
                            timestamp_filtered_count += 1
                            continue
                    except (ValueError, TypeError) as e:
                        print(f"Error parsing timestamp: {timestamp_str}")
                        print(f"Error type: {type(timestamp_str)}")
                        print(f"Error message: {e}")
                        pass
            
            # Filter by team ID if specified
            team_id = row.get('team_id', 'unknown')
            if team_id_filter is not None and team_id != team_id_filter:
                team_filtered_count += 1
                continue
            
            # Get content and URL
            content = str(row.get('content', ''))
            url = str(row.get('url', ''))
            
            # Filter out posts containing "F5Bot"
            if "F5Bot" in content:
                f5bot_filtered_count += 1
                continue
                
            # Get the status
            status = row.get('status', '')
            if status not in status_dirs:
                continue  # Skip unknown statuses
                
            # Create a filename based on the content hash
            content_hash = get_content_hash(content)
            
            # Check for duplicates
            if content_hash in seen_hashes:
                duplicate_hashes.add(content_hash)
            else:
                seen_hashes.add(content_hash)
                
                # Ensure filename has .txt suffix
                final_filename = f"{content_hash}.txt"
                
                # Determine output directory
                output_dir = status_dirs[status]
                
                # Write content to file with URL prepended
                output_path = os.path.join(output_dir, final_filename)
                with open(output_path, 'w', encoding='utf-8') as f:
                    if url:
                        f.write(f"{url}\n\n")
                    f.write(content)
                
                # Update status counter
                status_counts[status] += 1
            
            total_processed += 1
        
        # Print progress
        print(f"Processed {total_processed} entries so far...")
    
    print(f"Processing complete!")
    print(f"Total entries processed: {total_processed}")
    print(f"Unique entries: {len(seen_hashes)}")
    print(f"Duplicate entries: {len(duplicate_hashes)}")
    print(f"F5Bot filtered entries: {f5bot_filtered_count}")
    if team_id_filter is not None:
        print(f"Team ID filtered entries (not team {team_id_filter}): {team_filtered_count}")
    if timestamp_threshold is not None:
        print(f"Timestamp filtered entries (older than {timestamp_threshold_days} days): {timestamp_filtered_count}")
    
    print("\nStatus breakdown:")
    # Group RELEVANT and REPLIED
    relevant_count = status_counts['RELEVANT'] + status_counts['REPLIED']
    print(f"RELEVANT/REPLIED: {relevant_count}")
    
    # Print other statuses
    for status, count in status_counts.items():
        if status not in ['RELEVANT', 'REPLIED']:
            print(f"{status}: {count}")
    
    print(f"\nFiles saved to the following directories:")
    for status, dir_path in status_dirs.items():
        print(f"- {dir_path} ({status})")

if __name__ == "__main__":
    # Path to CSV file
    csv_path = "Supabase Snippet Filter Tracking Hits for Team 1 Sept.csv"
    
    # Team ID to filter by (set to None to process all teams)
    team_id_filter = 1
    
    # Timestamp threshold in days (set to None to process all timestamps)
    timestamp_threshold_days = 90  # Filter out items older than 90 days
    
    # Process the CSV file
    process_hits_csv(csv_path, team_id_filter=team_id_filter, timestamp_threshold_days=timestamp_threshold_days) 