import pandas as pd
import os
import hashlib
from collections import Counter

def get_content_hash(content):
    """Generate a hash from the content."""
    return hashlib.md5(content.encode('utf-8')).hexdigest()[:10]

def process_hits_csv(csv_path, team_id_filter=None, chunk_size=1000):
    """
    Process the hits.csv file in chunks and create individual text files.
    
    Args:
        csv_path: Path to the hits.csv file
        team_id_filter: Optional team ID to filter by. If None, process all teams.
        chunk_size: Number of rows to process at once
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
    
    # Track unique content hashes
    seen_hashes = set()
    duplicate_hashes = set()
    
    # Process the CSV file in chunks
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        for _, row in chunk.iterrows():
            # Skip rows with empty content
            if pd.isna(row.get('content')) or not row.get('content'):
                continue
            
            # Filter by team ID if specified
            team_id = row.get('team_id', 'unknown')
            if team_id_filter is not None and team_id != team_id_filter:
                team_filtered_count += 1
                continue
            
            # Filter out posts containing "F5Bot"
            content = str(row.get('content', ''))
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
                
                # Write content to file
                output_path = os.path.join(output_dir, final_filename)
                with open(output_path, 'w', encoding='utf-8') as f:
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
    # csv_path = "hits.csv"
    csv_path = "tracking_hits_rows.csv"
    
    # Team ID to filter by (set to None to process all teams)
    team_id_filter = 1
    
    # Process the CSV file
    process_hits_csv(csv_path, team_id_filter=team_id_filter) 