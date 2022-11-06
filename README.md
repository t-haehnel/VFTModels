# VFTModels

Verbal Fluency Task Analysis Tool using traditional rule-based, list-based and a novel semantic relatedness method.

This Github repository contains the code for the publication **XXXX**. Please refer to this publication for further details.

When using this software, please cite this work as: **XXXX**

This code allows:
1. training a Semantic Relatedness Model using a Wikipedia Corpus
2. Analyzing the words of a Verbal Fluency Task (VFT) using a Semantic Relatedness Model
3. Analyzing the words of a Semantic Verbal Fluency Task (VFT) using traditional list-based clustering
4. Analyzing the words of a Phonematic Verbal Fluency Task (VFT) using traditional rule-based clustering

# Requirements

The scripts were tested using the following software:

* Python 3.10
* pandas 1.3.5
* matplotlib 3.5.1
* scipy 1.7.3
* gensim 4.2.0

# How to use

## 1. Semantic Relatedness Model training using a Wikipedia Corpus

Pretrained Semantic Relatedness Models can be downloaded from https://mega.nz/folder/RD0jlK5Q#Pd_aNR5rrpyXrzD-bv1YUg
Please not, that a Semantic Relatedness Model is composed of two files (one .kv and one .kv.vectors.npy file). Both files need to have the same base name and only different file name extensions.

If a Semantic Relatedness Model is required for a different language, it can be also trained individually following this procedure:

First, download a Wikipedia Corpus. Thus, go to from https://dumps.wikimedia.org/backup-index.html and click the link for your language (e.g. enwiki for English Wikipedia). Then locate the correct Wikipedia Corpus (named ...-pages-articles.xml.bz2, e.g. enwiki-20221020-pages-articles.xml.bz2) and download it to your computer.

To train the model, the algorithm first preprocesses the Wikipedia Corpus and saves the preprocessed corpus as a .txt file. 

When running the modeltraining_word2vec.py script you have to specify the following filenames:
* The name/path of the Wikipedia corpus (e.g. enwiki-20221020-pages-articles.xml.bz2)
* The name/path where the preprocessed file should be saved (e.g. enwiki-20221020-pages-articles.txt). If the file does already exists, it will be used and not recreated.
* The name/path of the output file, i.e. the trained model (e.g. enwiki-20221020-pages-articles.kv)

Now, to train a new Semantic Relatedness Model, execute the modeltraining_word2vec.py with python: (replace the file names accordingly)

    python modeltraining_word2vec.py -i dewiki-20221020-pages-articles.xml.bz2 -p dewiki-20221020-pages-articles.txt -o dewiki-20221020-pages-articles.kv
    
The script will need a few hours of computation time and print some progress after creating the preprocessed file.

The following parameters can be used with the script:

    -h, --help            show this help message and exit  
    -i input_file, --Input input_file: wikipedia corpus used as input file (.xml.bz2-file)
    -p preprocessed_file, --Preprocessed preprocessed_file: file of preprocessed wikipedia corpus (.txt). If this file already exists, it will be used for model training. Else it will be created from the input file  
    -o output_file, --Output output_file: output file for trained model (.kv-file)  
    -w number_of_workers, --Workers number_of_workers: number of workers (default=number of cpu cores - 1)  
    -d number_of_dimensions, --Dimensions number_of_dimensions: number of dimenions of the word2vec model (default=500)  
    -s window_size, --Size window_size: size of window used for word2vec model training (default=10)  
    -m minimum_word_count, --Min minimum_word_count: minimum word count used for word2vec model training (default=5)  
    -a algorithm_type, --Algorithm algorithm_type: algorithm used for word2vec model training (default=1; 1 for skip-gram; 0 for CBOW)  

## 2. Analyzing the words of a Verbal Fluency Task (VFT) using a Semantic Relatedness Model

After downloading or self-training a Semantic Reladetdness Model (see chapter above), this model can be used to identify clusters in the words produced by a Verbal Fluency Tasks.

A short tutorial can be found in the Jupyter Notebook file example/Example.ipynb.

After importing the Word2VecModel module, the Semantic Relatedness Model can be initialized:

    from clustering_word2vec import Word2VecModel
    model = Word2VecModel()
    model.load_wv("enwiki-20221020-pages-articles.kv")

The file name of the Semantic Relatedness Model *enwiki-20221020-pages-articles.kv* needs to be updated to the downloaded or self-trained model file. Also, the second model with the extension .kv.vectors.npy (e.g. *enwiki-20221020-pages-articles.kv.vectors.npy*) needs to be stored in the same directory.

After initializing the model, the clusters can be identified using:

    model.calculate_clusterids(words, sim_threshold=threshold)

The parameter *threshold* defines the pairwise semantic relatedness threshold which defines whether two words belong to a cluster or not. The value 0.3 should be used for phonematic VFTs and 0.4 for semantic VFTs. The parameter *words* needs to be a pd.DataFrame with a column *words* which contains all words which were produced in one VFT. A pd.DataFrame will be returned which contains another column indicating the clusters. All words sharing the same ID in this column are part of the same cluster.  

## 3. Analyzing the words of a Semantic Verbal Fluency Task (VFT) using traditional list-based clustering

To compare the clustering results of the new Semantic Relatedness Model with traditional clustering methods, we implemented also traditional rule-based and list-based clustering methods. The algorithms used are based on the publication of Troyer et al.:

**1. Troyer, A. K., Moscovitch, M. & Winocur, G. Clustering and switching as two components of verbal fluency: Evidence from younger and older healthy adults. Neuropsychology 11, 138–146 (1997). DOI: 10.1037/0894-4105.11.1.138**

The traditional list-based clustering is based on thematic lists of animals. These need to be provided as a .csv file. An German example can be found in the file "database/de/animal_categories.csv". The file requires a column with the header "category" and a column with the header "word". 

A short tutorial how to perform list-based clustering can be found in the Jupyter Notebook file example/Example.ipynb.

First, the clustering model needs to be initialized and populated using the animal-lists-file:

    model = TraditionalClustering()
    model.initialize_semantic_list(filename="../database/de/animal_categories.csv")

After initializing the model, the clusters can be identified using:

    model.calculate_clusterids_semantic(words, sim_threshold=threshold)

The parameter *words* needs to be a pd.DataFrame with a column *words* which contains all words which were produced in one VFT. A pd.DataFrame will be returned which contains another column indicating the clusters. All words sharing the same ID in this column are part of the same cluster.  


## 4. Analyzing the words of a Phonematic Verbal Fluency Task (VFT) using traditional rule-based clusterin

The traditional rule-based clustering is based on phonematic rules shared by sequential words. These rules need to be stored in a database. An German example can be found in the file "database/de/phonematic_pairs.csv". The file requires a column "word1" and "word2" where each word pair produced by a patient needs to be stored. Each word pair needs to be stored only once. For each word pair, the word occuring first in the alphabet should be word1 and the other word should be treated as word2. Additionaly, the file requires 4 more rows: first_two, rhyme, vowel_diff_only, homonyms which indicate the four rules used for identifying clusters. The value *1* indicates that the rule is fulfilled and *0* indicates that it is not.  

A short tutorial how to perform list-based clustering can be found in the Jupyter Notebook file example/Example.ipynb.

First, the clustering model needs to be initialized and populated using the animal-lists-file:

    model = TraditionalClustering()
    model.initialize_phonematic_list(filename="../database/de/phonematic_pairs.csv")

After initializing the model, the clusters can be identified using:

    model.calculate_clusterids_phonematic(words, sim_threshold=threshold)

The parameter *words* needs to be a pd.DataFrame with a column *words* which contains all words which were produced in one VFT. A pd.DataFrame will be returned which contains another column indicating the clusters. All words sharing the same ID in this column are part of the same cluster.  
