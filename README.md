# Team Name: 

# Group Members: 
Rohit Vakkalagadda (rv324@cornell.edu)
Kaitlyn Lu (kl995@cornell.edu)
Lucas He (th689@cornell.edu)

## Dataset: ATP Tennis Matches (2024)
Downloading the Dataset: https://github.com/JeffSackmann/tennis_atp (scroll down to find the 2024 one)

## Checking that you downloaded the dataset: 
Run 'python -m unittest test_dataset.py' in the main folder. You should see the following result: 

----------------------------------------------------------------------
Ran 2 tests in 0.000s

OK

----------------------------------------------------------------------

This ensures you have correctly downloaded the dataset.

## Frontend using Streamlit
We are using Streamlit to host our project. To see our website, run 'streamlit run frontend/app.py'

Make sure you are using the newest version of streamlit. You can verify this by running 'pip install --upgrade streamlit'

Once you have launched the website, navigate to the visualization section to see our plots

### Project Description: 

We wish to predict the winner of the 2025 Australian Open using player data to create power rankings from 2024


### Using the Repository

You can use download this repository to visualize the data and implement machine learning algorithms

Files: 
datavis.ipynb: Jupyter notebook including everything from loading the dataset, cleaning the dataset, and visualizing the data with plots.
datavis_tests.ipynb: Jupyter notebook testing all data manipulation functions from datavis.ipynb
Makefile: file that runs useful tests.
plots/ : contains useful plots visualizing the data
milestones/ : documenting the progress of this project
frontend/ : includes the code for the frontend of this project.
README.md: this file, has useful info about the project
