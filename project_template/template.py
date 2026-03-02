# generate necessary boilerplate code for project initialization
import os

def initialize_project_structure():
    project_name = "customer-churning"
    os.makedirs(project_name, exist_ok=True)
    
    init_file_path = os.path.join(project_name, "__init__.py")
    with open(init_file_path, "w") as f:
        f.write("# initialization file for customer-churning project\n")

def create_files_and_directories():
    directories = [
        'src',
        "data",
        "notebooks",
        'models',
        'api',
        'config',
        'tests'
    ]
    files ={
        'data': [],
        'notebooks': [],
        'models': [],
        'api': [],
        'config': [],
        'tests': [],
        'src': ['__init__.py', 'preprocess.py',
                'train.py', 'serve.py', 'ui.py',
                'utils.py'] 
    }
    for directory in directories:
        os.makedirs( directory, exist_ok=True)

    for parent_dir, file_list in files.items():
        for file_name in file_list:
            file_path = os.path.join( parent_dir, file_name)
            with open(file_path, "w") as f:
                f.write("")  # create an empty file

def main():
    #initialize_project_structure()
    print("Creating directories and files...")
    create_files_and_directories()

if __name__ == "__main__":
    main()