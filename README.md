# VFTModels

Verbal Fluency Task Analysis Tool using traditional rule-based, list-based and a novel **semantic relatedness method**.

This GitHub repository contains the code referred in the publication **XXXX** (will be updated after acceptance of the paper). Please refer to this publication for further details.

When using this software, please cite the work as: **XXXX** (will be updated after acceptance of the paper)

The code of this repository should allow you to:
1. Train a semantic relatedness model using a Wikipedia corpus
2. Analyze the words of a verbal fluency task (VFT) using a semantic relatedness model
3. Analyze the words of a semantic verbal fluency task using traditional list-based clustering
4. Analyze the words of a phonematic verbal fluency task using traditional rule-based clustering

# Requirements

The scripts were tested using the following software:

* Python 3.10
* pandas 1.3.5
* matplotlib 3.5.1
* scipy 1.7.3
* gensim 4.2.0

# How to use

## 1. Semantic Relatedness Model Training using a Wikipedia Corpus

Pre-trained semantic relatedness models can be obtained from:  

**Hähnel, Tom. (2022). Multilanguage Semantic Relatedness Models for Verbal Fluency Tasks. Zenodo. https://doi.org/10.5281/zenodo.7429321**

Please note that each semantic relatedness model is composed of two files (one .kv and one .kv.vectors.npy file). Both files need to have the same base name (except the different file name extensions) and need to be stored within the same directory.

If a semantic relatedness model is required for a different language, it can be also trained individually following this procedure:

First, download the correct Wikipedia corpus. Thus, visit https://dumps.wikimedia.org/backup-index.html and click the link for your desired language (e.g. enwiki for English Wikipedia). Then locate the correct Wikipedia corpus (named ...-pages-articles.xml.bz2, e.g. enwiki-20221020-pages-articles.xml.bz2) and download it.

To train the model, the algorithm first preprocesses the Wikipedia corpus and saves the preprocessed corpus as a .txt file. 

When running the modeltraining_word2vec.py script you have to specify the following filenames by parameters:
* The name/path of the Wikipedia corpus (e.g. enwiki-20221020-pages-articles.xml.bz2)
* The name/path where the preprocessed file should be saved (e.g. enwiki-20221020-pages-articles.txt). If the file does already exist, it will be used and not recreated.
* The name/path of the output file, i.e. the trained model (e.g. enwiki-20221020-pages-articles.kv). The second .kv.vectors.npy file will be created within the same directory.

Now, to train a new semantic relatedness model, execute the modeltraining_word2vec.py using python and replace the file names accordingly:

    python modeltraining_word2vec.py -i dewiki-20221020-pages-articles.xml.bz2 -p dewiki-20221020-pages-articles.txt -o dewiki-20221020-pages-articles.kv
    
The script will need some hours of computation time and print the progress after creating the preprocessed file.

The following parameters can be used with the script:

    -h, --help            show this help message and exit  
    -i input_file, --Input input_file: wikipedia corpus used as input file (.xml.bz2-file)
    -p preprocessed_file, --Preprocessed preprocessed_file: file of preprocessed wikipedia corpus (.txt). If this file already exists, it will be used for model training. Else it will be created from the input file  
    -o output_file, --Output output_file: output file for trained model (.kv-file). The second .kv.vectors.npy file will be created within the same directory.  
    -w number_of_workers, --Workers number_of_workers: number of workers (default=number of cpu cores - 1)  
    -d number_of_dimensions, --Dimensions number_of_dimensions: number of dimensions of the word2vec model (default=500)  
    -s window_size, --Size window_size: size of window used for word2vec model training (default=10)  
    -m minimum_word_count, --Min minimum_word_count: minimum word count used for word2vec model training (default=5)  
    -a algorithm_type, --Algorithm algorithm_type: algorithm used for word2vec model training (default=1; 1 for skip-gram; 0 for CBOW)  

## 2. Analyzing the Words of a Verbal Fluency Task (VFT) using a Semantic Relatedness Model

After downloading or self-training a semantic relatedness model (see chapter above), this model can be used to identify clusters in the words produced by a VFT.

A short tutorial can be found in the Jupyter Notebook file example/Example.ipynb.

After importing the Word2VecModel module, the semantic relatedness model can be initialized:

    from clustering_word2vec import Word2VecModel
    model = Word2VecModel()
    model.load_wv("enwiki-20221020-pages-articles.kv")

The file name of the semantic relatedness model *enwiki-20221020-pages-articles.kv* needs to be updated to the downloaded or self-trained model file name/location. Also, the second model with the extension .kv.vectors.npy (e.g. *enwiki-20221020-pages-articles.kv.vectors.npy*) needs to be stored in the same directory.

After initializing the model, the clusters can be identified using:

    model.calculate_clusterids(words, sim_threshold=threshold)

The parameter *threshold* defines the pairwise semantic relatedness threshold which defines whether two words belong to a cluster or not. The value 0.3 should be used for phonematic VFTs and 0.4 for semantic VFTs. The parameter *words* needs to be a pd.DataFrame with a column *words* which contains all words produced in one VFT. A pd.DataFrame will be returned which contains another column indicating the clusters. All words sharing the same ID in this column are considered as part of the same cluster.  

## 3. Analyzing the Words of a Semantic Verbal Fluency Task (VFT) using Traditional List-based Clustering

To compare the clustering results of the new Semantic Relatedness Model with traditional clustering methods, we implemented also traditional rule-based and list-based clustering methods. The rules are based on the publication of Troyer et al.:

**Troyer, A. K., Moscovitch, M. & Winocur, G. Clustering and switching as two components of verbal fluency: Evidence from younger and older healthy adults. Neuropsychology 11, 138–146 (1997). DOI: 10.1037/0894-4105.11.1.138**

The traditional list-based clustering is based on thematic lists of animals. These lists need to be provided as a .csv file. A German example can be found in the file "database/de/animal_categories.csv". The file requires a column with the header *category* and a column with the header *word*. 

A short tutorial how to perform list-based clustering can be found in the Jupyter Notebook file example/Example.ipynb.

First, the clustering model needs to be initialized and populated using the animal-lists-file:

    model = TraditionalClustering()
    model.initialize_semantic_list(filename="../database/de/animal_categories.csv")

After initializing the model, the clusters can be identified using:

    model.calculate_clusterids_semantic(words)

The parameter *words* needs to be a pd.DataFrame with a column *words* which contains all words which were produced in one VFT. A pd.DataFrame will be returned which contains another column indicating the clusters. All words sharing the same ID in this column are considered as part of the same cluster.  


## 4. Analyzing the Words of a Phonematic Verbal Fluency Task (VFT) using Traditional Rule-based Clustering

The traditional rule-based clustering is based on phonematic rules shared by sequential words. These rules need to be stored in a database. A German example can be found in the file "database/de/phonematic_pairs.csv". The file requires a column *word1* and *word2* where each word pair produced by a patient needs to be stored. Each word pair needs to be stored only once. For each word pair, the word occurring first in the alphabet should be word1 and the other word should be treated as word2. Additionally, the file requires 4 more rows: *first_two*, *rhyme*, *vowel_diff_only*, *homonyms* which indicate the four rules used for identifying clusters. The value *1* indicates that the rule is fulfilled and *0* indicates that it is not.  

A short tutorial how to perform list-based clustering can be found in the Jupyter Notebook file example/Example.ipynb.

First, the clustering model needs to be initialized and populated using the phonematic-rules-file:

    model = TraditionalClustering()
    model.initialize_phonematic_list(filename="../database/de/phonematic_pairs.csv")

After initializing the model, the clusters can be identified using:

    model.calculate_clusterids_phonematic(words)

The parameter *words* needs to be a pd.DataFrame with a column *words* which contains all words which were produced in one VFT. A pd.DataFrame will be returned which contains another column indicating the clusters. All words sharing the same ID in this column are considered as part of the same cluster.  
