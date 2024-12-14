# FYS-STK4155_project3    
    ├── data                            # Placeholder for the data set (will be downloaded on the local machine when running the source scripts)          
    ├── docs                            # Screenshots of conversations with ChatGPT (and latex code for report)
    ├── results                         # Files with the results of the experiments and/or images of the plotted results. Also a placeholder for a model that is finished training, on local machine
    ├── src                             # Code to run our experiments (description in the scripts)
    │   ├── utils                       # Python files with functions needed for running our experiments
    │   ├── models                      # One python file for each model class
    │   ├── grid_search
    │   │   ├── CNN                     # Contains one python script for running the grid search for CNN, and one separate yaml file for each of the experiments. See instructions below
    │   │   └── LogReg                  # Run all experiments by running the python file. Used the provided yaml file as default
    │   └── evaluation                  # One folder for each model with a script for evaluation and a separate yaml configuration
    │       ├── CNN                     
    │       └── LogReg
    └──  requirements.txt               # Python libraries needed for running the code. Install with ´pip -r requirements.txt´


# Running experiments

- The experiments in the folder ´grid_search/CNN´ can be ran by `python3 run_cnn_gs.py --path_to_config CONFIG´, where CONFIG is the yaml file of the experiment you want to run
- Except for the experiments in the folder ´grid_search/CNN´, all ´.py´ files can be ran by python3 filename.py
- This will use the default config, with the same name as the script file, with .yaml extension
- To provide a config: `python3 filename.py --path_to_config CONFIG´. 


