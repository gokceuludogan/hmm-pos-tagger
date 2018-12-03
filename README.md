# hmm-pos-tagger

HMM for POS Tagging


## Installation

* The required packages for the program can be downloaded by the below command:

      pip3 install −r requirements.txt
    
## Usage

      python hmm.py −−data <path> −−split <percentage> −−unknown to singleton <0 or 1> −−print sequences <0 or 1>
  
 * **data:** corresponds to the path of the file containing data.
 * **split:** refers to the split percentage of the data for dividing data into train and test data. By default, it is equal to 90 which means 90% of the data is used in training and 10% of the data is used in testing.
 * **unknown to singleton:** specifies prediction type of unknown words. If it’s 1, then the unknown words are assumed to act like singleton words. By default it is 0 which means probable tags of unknown words are predicted by morphological patterns and the mean probabilities of these tags are used.
* **print sequences:** Prints obtained and expected sequences to console, if 1 is chosen. Default value is 0 which means the sequences are not printed.

*NOTE: Data should be the same format with [METU-Sabancı Turkish Dependency Treebank](https://web.itu.edu.tr/gulsenc/treebanks.html)*
