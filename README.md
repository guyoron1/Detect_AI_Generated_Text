﻿# Detect_AI_Generated_Text


  
## Datamix Phase

### Step 1: Fetching LLM-Generated Data

- Build methods to pull datasets from their respective places under fetch_data.py
- Datasets to be used:
  - Daigt v2 (Guy) - https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset - Guy
  - OUTFOX (Imry) - https://github.com/ryuryukke/OUTFOX
  - Ghostbuster (Imry) - https://github.com/vivek3141/ghostbuster-data
  - T5 Persuade Synthetic (Guy) - https://www.kaggle.com/datasets/conjuring92/fpe-processed-dataset?select=mlm_essays_processed.csv
  - GPT2 Output https://github.com/openai/gpt-2-output-dataset

TODO: 8.12

### Step 2: Fetching Student-Generated Data

- Datasets to be used:
  - Persuade corpus - https://www.kaggle.com/datasets/nbroad/persaude-corpus-2/data
  - Other corpora that are not specifically essays such as Ellipse, NarrativeQA, Wikipedia, IMDB Movie Reviews
- Add mechanism (or think of mechanism) for generating prompts for promptless human writing


### Step 3: Build Pipeline for Generating Essays Ourselves (GPT, Claude, Mistral, Llama)

- Which models to use?
- Which generation configs and prompting strategies to use? 
- Consider prompting for different tasks: Prompting to rewrite existing essays, Prompting to fill-in-the-blank...


### Step 4: Build adversary examples (augmented essays)
- Generate examples via mechanisms for bypassing LLM content detection, to train the models on them


### Step 5: Preprocessing and Cleaning



## Model Architecture Phase

###

## Finetuning Phase 


"Our best performing models were trained on 160k samples (without pre-processing), out of which 40k were human written."


gpt2 data is just llm generated text without prompts.

Formatting outside data to competition format (for train data):
- features are prompt_name, prompt_text, essay, is_prompt_llm_generated,generated



1. Finish standardization of existing LLM generated and Student generated datasets in format.py (Imry)
2. Build mechanism for generating prompt based on llm\student generated item (Guy)
3. Write e-mail to Tomer regarding our datamix plan and what is best practice for optimizing on data (and not on model parameters) (imry)
For afterwards
4. Write mechanism for generating new essays based on existing prompts
5. Look into fine-tuning state of the art models on persuade - what is instruction tuning?
6. How do we do datamix optimization?
