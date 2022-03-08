# Explainability with XBertScore
This repository contains code for the following paper: https://aclanthology.org/2021.eval4nlp-1.16.pdf


When you use it please cite:

```
@inproceedings{leiter-2021-reference,
    title = "Reference-Free Word- and Sentence-Level Translation Evaluation with Token-Matching Metrics",
    author = "Leiter, Christoph Wolfgang",
    booktitle = "Proceedings of the 2nd Workshop on Evaluation and Comparison of NLP Systems",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.eval4nlp-1.16",
    doi = "10.18653/v1/2021.eval4nlp-1.16",
    pages = "157--164",
   }
```

To reproduce the scores run `xai/SystemPaperScores.py`. The code is documented with comments in the respective files.


### Installation
This project was run on Windows, therefore there might be some compatibility issues with linux. To install the packages
use the `requirements.txt` file that was build with `pipreqs`. 
The root directory of this project can be referred to by importing `from project_root import ROOT_DIR`.