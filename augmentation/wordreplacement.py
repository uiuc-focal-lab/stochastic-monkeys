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
    replace_indices = random.sample(
        word_index,
        k=min(len(word_index), int(percentage * len(words))),
    )
    for index in replace_indices:
        antonyms = get_antonyms(words[index])
        word = random.choice(antonyms)
        words[index] = word
    return words

def synonyms(words, percentage):
    word_index = []
    for i,word in enumerate(words):
        synonyms = wordnet.synsets(words[i])
        if synonyms:
            word_index.append(i)
    replace_indices = random.sample(
        word_index,
        k=int(percentage * len(words)),
    )
    for index in replace_indices:
        synonyms = wordnet.synsets(words[index])
        word = random.choice(synonyms).lemmas()[0].name()
        words[index] = word
    return words

# def stop_words(words, percentage):
#     stop_words = stopwords.words('english')



class RandomWordRepalcement(Augmentation):
    """A random edit augmentation."""

    def __init__(
        self,
        percentage: float,
    ):
        """Initializes the augmentation.

        Args:
            percentage: The percentage of words to replace.
        """

        self.percentage = percentage
        self._sample_space = ASCII_SAMPLE_SPACE
    
    def __call__(
        self,
        prompt: str,
    ) -> str:
        """Randomly replaces a fixed percentage of words in the prompt."""
        words = word_tokenize(prompt)
        # get the index of words that have synonym first.

        
        # get the antonym first.
        words = antonyms(words, self.percentage)

        # get the output prompt
        result = []
        for i, token in enumerate(words):
            if i > 0 and token not in string.punctuation:
                result.append(' ')
            result.append(token)
        return ''.join(result)
