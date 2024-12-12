# FYS-STK4155_project3    
    ├── data                            # Placeholder for the MNIST data set that will be downloaded on the local machine when running the source scripts          
    ├── docs                            # Screenshots of conversations with ChatGPT
    ├── results                         # Yaml-files with the results of the experiments and/or images of the plotted results
    ├── src                             # Source files for running our experiments 
    │   ├── utils                       # Scripts containing all functions needed for running our experiments
    │   ├── models                      # Classes for the two different models
    │   ├── grid_search
            ├── CNN                     # Contains one python script for running all the grid searches for CNN and one separate yaml for each experiment. See instructions below
            ├── LogReg                  # Run all experiments by running the python file. Used the provided yaml file as default
    │   └── ...
    └──  requirements.txt               # Python libraries needed for running the code. Install with ´pip -r requirements.txt´


# Running experiments

- All experiments in `.py´ files can be ran by python3 filename.py
- This will use the default config, with the same name as the script file, with .yaml extension
- To provide a config: `python3 filename.py --path_to_config CONFIG´

