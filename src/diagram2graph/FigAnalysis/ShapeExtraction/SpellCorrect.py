from __future__ import print_function
from spellchecker import SpellChecker


class SpellCorrect:
    
    def __init__(self):
        self.text_file_path = "./DLWords.txt"
    
    
    def correctDLWord(self, misspelled):
        spellDL = SpellChecker(language = None) 
        spellDL.word_frequency.load_text_file(self.text_file_path)
        #spellDL.export("DL_dictionary.gz", gzipped = True)    
        misspelled_lower = misspelled.lower()
        
        if misspelled_lower in spellDL:
            return (misspelled, 1.0)
        else: 
            correct = spellDL.correction(misspelled_lower)
            prob = spellDL.word_probability(misspelled_lower, total_words = 1)
            return (correct, prob)
        
    def correctRegularWord(self, misspelled):
        spell = SpellChecker()  # load default dictionary
        misspelled_lower = misspelled.lower()
        
        if misspelled_lower in spell:
            return misspelled
        else: 
            correct = spell.correction(misspelled_lower)
            return correct

    def correctListWord(self, misspelled, fig_text):
        if fig_text != []:
            spellList = SpellChecker(language = None)  # load default dictionary
            spellList.word_frequency.load_words(fig_text)
            misspelled_lower = misspelled.lower()
            
            if misspelled_lower in spellList:
                return (misspelled, 1.0)
            else: 
                correct = spellList.correction(misspelled_lower)
                prob = spellList.word_probability(misspelled_lower, total_words = 1)
                return (correct, prob)
        else:
            return (None, 0.0)

            
    def correctWord(self, misspelled, fig_text):
        
        (correctListWord, probListWord) = self.correctListWord(misspelled, fig_text)

        if correctListWord != None and correctListWord == misspelled and probListWord == 1.0: # found word in fig_list
           return correctListWord
        elif correctListWord != None  and (correctListWord != misspelled and probListWord == 0.0): # corrected word from fig_list
            return correctListWord
        elif (correctListWord == None) or (correctListWord == misspelled and probListWord == 0.0):# not found word in fig_list
           
            (correctDL, probDL) = self.correctDLWord(misspelled)

            if correctDL == misspelled and probDL == 1.0: # found word in dictionary
                return correctDL
            elif correctDL != misspelled and probDL == 0.0: # corrected word from DL dictionary
                return correctDL               
            else: #elif correctDL == misspelled.lower() and probDL == 0.0: #Not corrected by DL dictionary, returned original
                correct = self.correctRegularWord(misspelled)
                return correct
            
