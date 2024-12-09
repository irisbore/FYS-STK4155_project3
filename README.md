# FYS-STK4155_project3    
    ├── data                            
    │   └── dataset                 
    ├── docs/chatgpt                    # Includes screenshots of conversations with chatgpt
    ├── results                         # Images of the plotted results
    ├── src                             # Source files for running our experiments
    │   ├── utils                       # Scripts containing useful functions 
    │   ├── model                       # Classes for the different models
    │   ├── testing                     # Folder for testing our code
    │   └── ...
    ├── undeliverables                  # Not in the project delivery. If we tested something we wanted to archive
    └──  requirements.txt               # Python libraries. Install with ´pip -r requirements.txt´


# Running experiments

- All experiments in `.py´ files can be ran by python3 filename.py
- This will use the default config, with the same name as the script file, with .yaml extension
- To provide a config: `python3 filename.py --path_to_config´

