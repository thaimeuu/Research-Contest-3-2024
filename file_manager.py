import os

def delete_ricegr_files(folder_path):
    try:
        # List all files in the specified folder
        files = os.listdir(folder_path)

        # Iterate through each file in the folder
        for file in files:
            file_path = os.path.join(folder_path, file)

            # Check if the file is a .txt file
            if file.endswith(".ricegr"):
                print(f"Deleting {file}...")
                os.remove(file_path)
                
            # Toggle break on/off
            break

        print("Deletion completed.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    delete_ricegr_files("dataset_2_xml/.Asian-African panel_CIAT/Asian-African panel_New")
