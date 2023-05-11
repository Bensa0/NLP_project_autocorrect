import random
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
import itertools
from gensim.models import FastText
from spellchecker import SpellChecker

def generate_combinations(word, max_output):
    # Get all possible permutations of the word's letters
    if(len(word)>10):
        #print(word + " is greater than 10")
        return [word]
    permutations = list(itertools.permutations(word))
    # Create a set to store unique combinations
    combinations = set()
    # For each permutation, create a new combination by joining the letters in a random order
    for permutation in permutations:
        combination = ''.join(random.sample(permutation, len(permutation)))
        # Add the combination to the set, if it's not the original word
        if combination != word:
            combinations.add(combination)
            # Stop generating combinations if the maximum number is reached
            if len(combinations) == max_output:
                break
    # Convert the set to a list and return it
    return list(combinations)

def generate_word_lists(words, max_combinations):
    words_list = []
    for word in tqdm(words, desc='Generating word combinations'):
        misspellings = generate_combinations(word, max_combinations)
        words_list.append([word] + misspellings)
    return words_list

def generate_word_pairs(words_list):
    word_pairs = []
    for word_group in tqdm(words_list, desc='Generating word pairs'):
        correct_word = word_group[0]
        for incorrect_word in word_group[1:]:
            word_pairs.append([correct_word, incorrect_word])
    return word_pairs

def find_correct_word(input_word, word_list, model,topn = 5):
    # Get most similar words for the given input
    similar_words = model.wv.most_similar(input_word, topn=topn)
    #print(f"Most similar words for '{input_word}':", similar_words)

    # Find the correct word for the input_word
    correct_word = None
    for word, similarity in similar_words:
        for word_group in word_list:
            if word in word_group:
                correct_word = word_group[0]
                break
        if correct_word:
            break

    if correct_word:
        print(f"The correct word for '{input_word}' is '{correct_word}'.")
        return correct_word ,similar_words
    else:
        print("Could not find a similar word in words_list.")
        return None, similar_words
    
    from spellchecker import SpellChecker

def correct_sentence(sentence, word_list , model):
    # Initialize spellchecker
    spell = SpellChecker()

    # Tokenize sentence into words
    words = sentence.split()

    # Find misspelled words
    misspelled = spell.unknown(words)

    # Create a list to store corrected words
    corrected = []

    # Loop through misspelled words and get correct spelling using find_correct_word
    for word in words:
        if word in misspelled:
            correct_word, _ = find_correct_word(word, word_list, model)
            if correct_word:
                corrected.append(correct_word)
            else:
                corrected.append(word)
        else:
            corrected.append(word)

    # Join corrected words into a sentence
    corrected_sentence = ' '.join(corrected)

    return corrected_sentence