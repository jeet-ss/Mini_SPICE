# Mini_SPICE
Simplified Implementation of SPICE

Contributors: Jeet Sen Sarma, Jibak Sarkar

Special Thanks to: Prof. Dr. Meinard Müller, Simon Schwär


## Implementation
1. Datasets are loaded using "data_files/dataloader.py"
2. Constant Q Transform is generated for the datasets in "generate_data.py" and stored in directory "CQT_data" 
3. The Entire model along with two Decoders can be found in "utils/model.py"
4. Training Methodology is present is "utils/training_script.py"
5. We train the model with our data in "train.py" and store the loss values in a numpy dictionary
6. The preliminary evaluation and comparision with original model can be found at "spice.ipynb" notebook


## Evaluation
1. For evaluation of Mini-SPICE refer to the notebook evaluate_model.ipynb
2. For evaluation of SPICE refer to the notebook mini_SPICE.ipynb
3. For evaluation of CREPE refer to [Pietro's code](https://github.com/pf-mpa/FAU-Music-Processing-Internship/blob/pietro/crepe_eval.py)
