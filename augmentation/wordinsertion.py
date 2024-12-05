import random
import string
from augmentation.augmentation import Augmentation, ASCII_SAMPLE_SPACE
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

PROMPT_STRING_JOIN_SEPARATOR = ""

def get_antonyms(word):
    antonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.antonyms():
                antonyms.extend([ant.name() for ant in lemma.antonyms()])
    return antonyms

def antonyms(words, percentage):
    word_index = []
    for i,word in enumerate(words):
        antonyms = get_antonyms(word)
        if antonyms:
            word_index.append(i)
    word_indices = random.choices(
        word_index,
        k=min(len(word_index), int(percentage * len(words))),
    )
    new_words = []
    for index in word_indices:
        antonyms = get_antonyms(words[index])
        word = random.choice(antonyms)
        new_words.append(word)
    insert_indices = random.sample(
        range(len(words)),
        k=min(len(word_index), int(percentage * len(words))),
    )
    return new_words, insert_indices

def synonyms(words, percentage):
    word_index = []
    for i,word in enumerate(words):
        synonyms = wordnet.synsets(words[i])
        if synonyms:
            word_index.append(i)
    word_indices = random.sample(
        word_index,
        k = int(percentage * len(words)),
    )
    new_words = []
    for index in word_indices:
        synonyms = wordnet.synsets(words[index])
        word = random.choice(synonyms).lemmas()[0].name()
        new_words.append(word)
    insert_indices = random.sample(
        range(len(words)),
        k = min(len(word_index), int(percentage * len(words))),
    )
    return new_words, insert_indices

def stop_words(words, percentage):
    stop_words = stopwords.words('english')
    new_words = random.choices(
        stop_words, 
        k = int(percentage * len(words)),
    )
    insert_indices = random.sample(
        range(len(words)),
        k = int(percentage * len(words)),
    )
    return new_words, insert_indices


punctuation_list = ['.',',',';',':','?','!']
def punctuation(words,percentage):
    new_words = random.choices(
        punctuation_list, 
        k = int(percentage * len(words)),
    )
    insert_indices = random.sample(
        range(len(words)),
        k = int(percentage * len(words)),
    )
    return new_words, insert_indices


class RandomWordInsertion(Augmentation):
    """A random edit augmentation."""

    def __init__(
        self,
        percentage: float,
    ):
        """Initializes the augmentation.

        Args:
            percentage: The percentage of synonym words to insert.
        """

        self.percentage = percentage
        self._sample_space = ASCII_SAMPLE_SPACE
    
    def __call__(
        self,
        prompt: str,
    ) -> str:
        """First pick words and get the synonyms. Then find the position to insert."""
        words = word_tokenize(prompt)
        # get the synonyms first.
        # new_words, insert_indices = synonyms(words, self.percentage)

        # get the antonyms first.
        # new_words, insert_indices = antonyms(words, self.percentage)

        # get the stop words first
        new_words, insert_indices = stop_words(words, self.percentage)

        # get the punctuation first
        # new_words, insert_indices = punctuation(words, self.percentage)

        # get the output prompt
        output = []
        new_index = 0
        for index in range(len(words)):
            if index in insert_indices:
                output.append(new_words[new_index])
                new_index += 1
            output.append(words[index])
        result = []
        for i, token in enumerate(output):
            if i > 0 and token not in string.punctuation:
                result.append(' ')
            result.append(token)
        return ''.join(result)