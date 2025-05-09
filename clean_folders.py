import os
import glob

def clean_folders():
    """
    Delete all .txt files from the classifier folders to prepare for new data.
    """
    # List of folders to clean
    folders = [
        'relevant_posts',
        'irrelevant_posts',
        'new_posts',
        'content_removed_posts'
    ]
    
    total_deleted = 0
    
    for folder in folders:
        if not os.path.exists(folder):
            print(f"Folder {folder} does not exist, skipping...")
            continue
            
        # Get all .txt files in the folder
        txt_files = glob.glob(os.path.join(folder, '*.txt'))
        
        # Delete each file
        for file_path in txt_files:
            try:
                os.remove(file_path)
                total_deleted += 1
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
        
        print(f"Cleaned {len(txt_files)} files from {folder}")
    
    print(f"\nTotal files deleted: {total_deleted}")

if __name__ == "__main__":
    clean_folders() 