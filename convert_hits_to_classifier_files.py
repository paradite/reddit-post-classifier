import pandas as pd
import os
import hashlib

def get_content_hash(content):
    """Generate a hash from the content."""
    return hashlib.md5(content.encode('utf-8')).hexdigest()[:10]

def process_hits_csv(csv_path, chunk_size=1000):
    """
    Process the hits.csv file in chunks and create individual text files.
    
    Args:
        csv_path: Path to the hits.csv file
        chunk_size: Number of rows to process at once
    """
    # Create output directories at the same level
    relevant_dir = "relevant_posts"
    irrelevant_dir = "irrelevant_posts"
    os.makedirs(relevant_dir, exist_ok=True)
    os.makedirs(irrelevant_dir, exist_ok=True)
    
    # Initialize counters
    total_processed = 0
    relevant_count = 0
    irrelevant_count = 0
    f5bot_filtered_count = 0
    
    # Process the CSV file in chunks
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        for _, row in chunk.iterrows():
            # Skip rows with empty content
            if pd.isna(row.get('content')) or not row.get('content'):
                continue
            
            # Filter out posts containing "F5Bot"
            content = str(row.get('content', ''))
            if "F5Bot" in content:
                f5bot_filtered_count += 1
                continue
                
            # Determine if the content is relevant (both REPLIED and RELEVANT are considered relevant)
            status = row.get('status', '')
            is_relevant = status in ['RELEVANT', 'REPLIED']
            
            # Create a filename based on the content hash
            content_hash = get_content_hash(content)
            
            # Ensure filename has .txt suffix
            final_filename = f"{content_hash}.txt"
            
            # Determine output directory
            output_dir = relevant_dir if is_relevant else irrelevant_dir
            
            # Write content to file
            output_path = os.path.join(output_dir, final_filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Update counters
            total_processed += 1
            if is_relevant:
                relevant_count += 1
            else:
                irrelevant_count += 1
        
        # Print progress
        print(f"Processed {total_processed} entries so far...")
    
    print(f"Processing complete!")
    print(f"Total entries processed: {total_processed}")
    print(f"Relevant entries: {relevant_count}")
    print(f"Irrelevant entries: {irrelevant_count}")
    print(f"F5Bot filtered entries: {f5bot_filtered_count}")
    print(f"Files saved to '{relevant_dir}' and '{irrelevant_dir}' directories with .txt suffix")

if __name__ == "__main__":
    # Path to CSV file
    # csv_path = "hits.csv"
    csv_path = "tracking_hits_rows.csv"
    
    # Process the CSV file
    process_hits_csv(csv_path) 