import numpy as np
import pandas as pd


class TraditionalClustering:
    """A class for all semantic clustering algorithms using category lists and phonematic algorithms using rules (
    Troyer et al.) """
    semantic_categories = pd.DataFrame()
    phonematic_pairs = pd.DataFrame()

    def initialize_phonematic_list(self, filename: str):
        """
        Loads a .csv file which includes a list of all occuring phonemic-word-pairs. The .csv file needs to have
        the columns word1, word2, first_two, rhyme, vowel_diff_only, homonyms
        :param filename: path of the .csv file
        """
        self.phonematic_pairs = pd.read_csv(filename)

        # check columns
        cols = ["word1", "word2", "first_two", "rhyme", "vowel_diff_only", "homonyms"]
        cols_paircheck = ["first_two", "rhyme", "vowel_diff_only", "homonyms"]

        if len(self.phonematic_pairs.columns) != len(cols):
            print("WARNING: word list has wrong number of columns: ")
            print(self.phonematic_pairs.columns)

        for col in cols:
            if col not in self.phonematic_pairs.columns:
                print("WARNING: word list has no column '" + col + "': ")
                print(self.phonematic_pairs.columns)

        # check lowercase start
        non_lowercase_start = self.phonematic_pairs["word1"] != self.phonematic_pairs["word1"].map(
            lambda s: str.lower(s))
        if np.any(non_lowercase_start):
            print("WARNING: non-lower-case starting word1 in list " + filename + ": ")
            print(self.phonematic_pairs[non_lowercase_start])

        non_lowercase_start = self.phonematic_pairs["word2"] != self.phonematic_pairs["word2"].map(
            lambda s: str.lower(s))
        if np.any(non_lowercase_start):
            print("WARNING: non-lower-case starting word2 in list " + filename + ": ")
            print(self.phonematic_pairs[non_lowercase_start])

        # check spaces at begin/end
        trimmed = self.phonematic_pairs["word1"].map(lambda s: s.strip()) != self.phonematic_pairs["word1"]
        if np.any(trimmed):
            print("WARNING: non-trimmed word1 in list " + filename + ": ")
            print(self.phonematic_pairs[trimmed])
        trimmed = self.phonematic_pairs["word2"].map(lambda s: s.strip()) != self.phonematic_pairs["word2"]
        if np.any(trimmed):
            print("WARNING: non-trimmed word2 in list " + filename + ": ")
            print(self.phonematic_pairs[trimmed])

        # sort word1/word2 alphabetically (word1 should be before word2 within the alphabet)
        not_sorted_bool = self.phonematic_pairs["word1"] > self.phonematic_pairs["word2"]
        not_sorted_word1 = self.phonematic_pairs.loc[not_sorted_bool, "word1"].copy()
        self.phonematic_pairs.loc[not_sorted_bool, "word1"] = self.phonematic_pairs.loc[not_sorted_bool, "word2"]
        self.phonematic_pairs.loc[not_sorted_bool, "word2"] = not_sorted_word1

        # find duplicates:
        concated_words = self.phonematic_pairs["word1"] + "-" + self.phonematic_pairs["word2"]
        duplicates = self.phonematic_pairs[concated_words.duplicated()]
        if len(duplicates) > 0:
            print("WARNING: duplicates found in list " + filename + ": ")
            print(duplicates)

        # calculate is_pair-column
        self.phonematic_pairs["is_pair"] = 0
        for col in cols_paircheck:
            self.phonematic_pairs["is_pair"] += self.phonematic_pairs[col]
        self.phonematic_pairs["is_pair"] = self.phonematic_pairs["is_pair"].map(lambda i: 1 if i >= 1 else 0)

    def initialize_semantic_list(self, filename: str):
        """
        Loads a .csv file which includes a list of all semantical-category-associations. The .csv file needs to have
        2 columns (category, word)
        :param filename: path of the .csv file
        """
        # load csv file
        self.semantic_categories = pd.read_csv(filename)

        # check columns
        if len(self.semantic_categories.columns) != 2:
            print("WARNING: word list has wrong number of columns: ")
            print(self.semantic_categories.columns)

        if "category" not in self.semantic_categories.columns:
            print("WARNING: word list has no column 'category': ")
            print(self.semantic_categories.columns)

        if "word" not in self.semantic_categories.columns:
            print("WARNING: word list has no column 'word': ")
            print(self.semantic_categories.columns)

        # check double animal entries
        duplicates = self.semantic_categories[self.semantic_categories.duplicated()]
        if duplicates.shape[0] > 0:
            print("WARNING: duplicate word found in list " + filename + ": ")
            print(duplicates)

        # check lowercase start
        non_uppercase_start = self.semantic_categories["word"].str[0] != self.semantic_categories["word"].map(
            lambda s: str.upper(s[0]))
        if np.any(non_uppercase_start):
            print("WARNING: lower-case starting word in list " + filename + ": ")
            print(self.semantic_categories[non_uppercase_start])

        # check spaces at begin/end
        trimmed = self.semantic_categories["word"].map(lambda s: s.strip()) != self.semantic_categories["word"]
        if np.any(trimmed):
            print("WARNING: non-trimmed word in list " + filename + ": ")
            print(self.semantic_categories[trimmed])

    def get_words_from_category(self, category: str) -> np.array:
        """
        Returns all words of a semantic category given by the wordlist
        :param category: the semantic category
        :return: a np.array of all words which are associated with this semantic category
        """
        return self.semantic_categories[self.semantic_categories["category"] == category]["word"].values

    def get_categories_from_word(self, word: str, printwarning: bool = True) -> np.array:
        """
        Returns all semantic categories which are associated with a given word
        :param word: the word
        :param printwarning: bool, whether to print a warning if one word is not found in the semantic category list
        :return: a np.array of all categories associated with this word
        """
        if printwarning and word not in self.semantic_categories["word"].values:
            print("WARNING: " + word + " does not exist in semantic category list")
        return self.semantic_categories[self.semantic_categories["word"] == word]["category"].values

    def get_categories_from_wordpair(self, word1: str, word2: str, printwarning: bool = True) -> np.array:
        """
        Returns all semantic categories, which are associated with both words (word1 AND word2)
        :param word1: the first word
        :param word2: the second word
        :param printwarning: bool, whether to print a warning if one word is not found in the semantic category list
        :return: np.array of all categories associated with this word
        """
        if printwarning and (word1 not in self.semantic_categories["word"].values):
            print("WARNING: " + word1 + " does not exist in semantic category list")
        if printwarning and (word2 not in self.semantic_categories["word"].values):
            print("WARNING: " + word2 + " does not exist in semantic category list")
        categories1 = self.semantic_categories[self.semantic_categories["word"] == word1]["category"].values
        categories2 = self.semantic_categories[self.semantic_categories["word"] == word2]["category"].values
        return np.intersect1d(categories1, categories2)

    def get_commonrules_from_wordpair(self, word1: str, word2: str, printwarning: bool = True) -> set:
        """
        Returns all fullfilled phonematic rules of a word pair
        :param word1: the first word
        :param word2: the second word
        :param printwarning: bool, whether to print a warning if one word is not found in the phonematic pair list
        :return: set of all fullfilled rules
        """

        word1 = word1.lower()
        word2 = word2.lower()

        # sort words
        if word1 > word2:
            tmp = word1
            word1 = word2
            word2 = tmp

        # check if words do exist
        if printwarning:
            notfound = self.check_wordlist_phonematic(pd.Series([word1, word2]))
            if notfound.shape[0] > 0:
                print("WARNING: words not found in phonemic pair list: ")
                print(notfound)

        # check if words are a pair
        search_res = self.phonematic_pairs.loc[(self.phonematic_pairs["word1"] == word1) &
                                               (self.phonematic_pairs["word2"] == word2)]
        # pair not found
        if search_res.shape[0] != 1:
            if printwarning:
                print("WARNING: word pair is not in phonemic pair list: ")
            return {}

        # else check which rules are fullfilled
        shared_rules = []
        for rule in ["first_two", "rhyme", "vowel_diff_only", "homonyms"]:
            if search_res[rule].values[0] == 1:
                shared_rules.append(rule)

        return set(shared_rules)

    def check_same_category(self, word1: str, word2: str, printwarning: bool = True) -> bool:
        """
        Checks if both words are part of any same semantic category
        :param printwarning: bool, whether to print a warning if one word is not found in the semantic category list
        :param word1: first word
        :param word2: second word
        :return: bool, whether both words are part of any same semantic category
        """
        if printwarning and (word1 not in self.semantic_categories["word"].values):
            print("WARNING: " + word1 + " does not exist in semantic category list")
        if printwarning and (word2 not in self.semantic_categories["word"].values):
            print("WARNING: " + word2 + " does not exist in semantic category list")
        return len(self.get_categories_from_wordpair(word1, word2, printwarning)) > 0

    def check_phonemic_pair(self, word1: str, word2: str, printwarning: bool = True):
        """
        Checks if both words form a phonemic pair by checking the phonemic word list
        :param word1: one word
        :param word2: the other word (order of the words doesn't matter)
        :param printwarning: bool, whether to print a warning if the pair is not found in the phonemic word list
        :return:
        """
        # make word lower characters
        word1 = word1.lower()
        word2 = word2.lower()

        # sort words
        if word1 > word2:
            tmp = word1
            word1 = word2
            word2 = tmp

        # check if words do exist
        if printwarning:
            notfound = self.check_wordlist_phonematic(pd.Series([word1, word2]))
            if notfound.shape[0] > 0:
                print("WARNING: words not found in phonemic pair list: ")
                print(notfound)

        # check if words are a pair
        search_res = self.phonematic_pairs.loc[(self.phonematic_pairs["word1"] == word1) &
                                               (self.phonematic_pairs["word2"] == word2)]
        # pair not found
        if search_res.shape[0] != 1:
            return False

        return search_res["is_pair"].values[0] == 1

    def check_wordlist_semantic(self, words_to_check: np.array) -> list:
        """
        Checks if all words are contained by the loaded wordlist
        :param words_to_check: a np.array of all words which should be checked
        :return: a np.array containing all words not found in the loaded wordlist
        """
        words_to_check = words_to_check[words_to_check != ""]
        return words_to_check[list(map(lambda s: s not in self.semantic_categories["word"].values, words_to_check))]

    def check_wordlist_phonematic(self, words_to_check: np.array) -> pd.DataFrame:
        """
        Checks if all word-pairs (sequential words in the list) are contained by the loaded phonemic list
        :param words_to_check: a np.array of all words which should be checked
        :return: a pd.DataFrame containing all word pairs which were not found in the loaded phonemic word list
        """
        words_to_check = words_to_check[words_to_check != ""]
        if isinstance(words_to_check, pd.DataFrame) or isinstance(words_to_check, pd.Series):
            words_to_check = words_to_check.reset_index(drop=True)
        non_existing_pairs = pd.DataFrame(columns=["word1", "word2"])
        for i in range(0, len(words_to_check) - 1):
            word1 = words_to_check[i]
            word2 = words_to_check[i + 1]
            # lower characters
            word1 = word1.lower()
            word2 = word2.lower()
            # sort word pair
            if word1 > word2:
                temp = word1
                word1 = word2
                word2 = temp
            # search occurences
            search_result = self.phonematic_pairs[(self.phonematic_pairs["word1"] == word1) &
                                                  (self.phonematic_pairs["word2"] == word2)]
            if search_result.shape[0] != 1:
                non_existing_pairs = pd.concat([non_existing_pairs, pd.DataFrame({"word1": [word1], "word2": [word2]})])

        return non_existing_pairs

    def calculate_clusterids_semantic(self, intervals: pd.DataFrame, printwarning: bool = True) -> np.array:
        """
        Calculates a np.array with IDs for each found cluster. If a word does not belong to a cluster, the value will
        be set to NAN. Clusters are counted from 1 to cluster_max. Clusters are defined as a chain of words where
        all concurrent neighbors share at least one semantic category
        :param printwarning: bool, whether to print a warning if one word is not found in the semantic category list
        :param intervals: pd.DataFrame with column 'words' (additional columns may be passed)
        :return: the given pd.DataFrame with 2 additional columns: cluster (indicating the cluster ID) +
        category_before (indicating the categories shared with the row before)
        """
        intervals.reset_index(drop=True, inplace=True)
        intervals["cluster"] = np.NAN
        intervals["category_before"] = ""

        cluster_id = 0
        curr_cluster_lists = {}

        # if printwartning=True -> check if all words exist
        if printwarning:
            for i in range(0, intervals.shape[0]):
                word = intervals.loc[i, "word"]
                if printwarning and word not in self.semantic_categories["word"].values and word != "":
                    print("WARNING: " + word + " does not exist in semantic category list")

        for i in range(0, intervals.shape[0]):
            curr_lists = self.get_categories_from_word(intervals.loc[i, "word"])

            if len(set(curr_lists).intersection(curr_cluster_lists)) > 0:
                curr_cluster_lists = set(curr_lists).intersection(curr_cluster_lists)
                intervals.loc[i, "category_before"] = ", ".join(curr_cluster_lists)
            else:
                curr_cluster_lists = curr_lists
                cluster_id += 1

            intervals.loc[i, "cluster"] = cluster_id

        return intervals

    def calculate_clusterids_phonematic(self, intervals: pd.DataFrame, printwarning: bool = True) -> np.array:
        """
        Calculates a np.array with IDs for each found cluster. If a word does not belong to a cluster, the value will
        be set to zero. Clusters are counted from 1 to cluster_max. Clusters are defined as a chain of words where
        all neighbors share at least one phonematic criterium.
        :param printwarning: bool, whether to print a warning if one word is not found in the phonematic pair list
        :param intervals: pd.DataFrame with column 'words'
        :return: the given pd.DataFrame with 1 additional column: cluster (indicating the cluster ID)
        """

        intervals.reset_index(drop=True, inplace=True)
        intervals["cluster"] = np.NAN
        intervals["rule_before"] = ""

        # if printwartning=True -> check if all words exist
        if printwarning:
            not_found = self.check_wordlist_phonematic(intervals[intervals["word"] != ""]["word"])
            if not_found.shape[0] > 0:
                print("WARNING: words do not exist in phonemic pair list: ")
                print(not_found)

        cluster_id = 0
        curr_cluster_lists = {}
        intervals.loc[0, "cluster"] = cluster_id

        for i in range(1, intervals.shape[0]):
            curr_lists = self.get_commonrules_from_wordpair(intervals.loc[i - 1, "word"], intervals.loc[i, "word"])

            if len(curr_lists) == 0:
                cluster_id +=1
                curr_cluster_lists = {}
            else:
                if len(curr_cluster_lists) > 0:
                    if len(curr_lists.intersection(curr_cluster_lists)) > 0:
                        curr_cluster_lists = curr_lists.intersection(curr_cluster_lists)
                    else:
                        cluster_id += 1
                        curr_cluster_lists = {}
                else:
                    curr_cluster_lists = curr_lists

            intervals.loc[i, "rule_before"] = ", ".join(curr_cluster_lists)
            intervals.loc[i, "cluster"] = cluster_id

        return intervals
