import os
import csv

# Create output directory
OUTPUT_DIR = os.path.abspath("entailment_output")
print(f"Creating output directory at: {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Test file creation
def create_test_file():
    filepath = os.path.join(OUTPUT_DIR, "test_file.csv")
    print(f"Attempting to create test file at: {filepath}")
    
    try:
        # Create a simple CSV file
        with open(filepath, mode='w', newline='', encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Column1", "Column2", "Column3"])
            writer.writerow(["Value1", "Value2", "Value3"])
            writer.writerow(["Value4", "Value5", "Value6"])
        
        # Verify file was created
        if os.path.exists(filepath):
            print(f"SUCCESS: Test file created at {filepath}")
            print(f"File size: {os.path.getsize(filepath)} bytes")
        else:
            print(f"ERROR: File does not exist after creation attempt: {filepath}")
    except Exception as e:
        print(f"ERROR creating test file: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print(f"Current working directory: {os.getcwd()}")
    
    # Check if output directory exists and is writable
    if os.path.exists(OUTPUT_DIR):
        print(f"Output directory exists: {OUTPUT_DIR}")
        if os.access(OUTPUT_DIR, os.W_OK):
            print("Output directory is writable")
        else:
            print("WARNING: Output directory is not writable!")
    else:
        print(f"WARNING: Output directory does not exist despite creation attempt: {OUTPUT_DIR}")
    
    # Try to create a test file
    create_test_file()
    
    # List all files in the output directory
    print("\nFiles in output directory:")
    if os.path.exists(OUTPUT_DIR):
        files = os.listdir(OUTPUT_DIR)
        if files:
            for filename in files:
                file_path = os.path.join(OUTPUT_DIR, filename)
                file_size = os.path.getsize(file_path)
                print(f"  - {filename} ({file_size} bytes)")
        else:
            print("  No files found in the output directory.")
    else:
        print("  Output directory does not exist.")