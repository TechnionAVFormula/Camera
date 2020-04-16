import PyPDF2 
import textract
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

class Text():
    def __init__(self, filePath):
        self = filePath
        #open("example2.txt", "r+")

    def readFile(self, path):
        txtFile1 = open(path, "r+")
        txtFile2 = open(path, "r+")

        # convert given txt to a list of str
        lines1 = txtFile1.readlines()
        lines2 = txtFile2.readlines()

        #creating 3 lists
        a = []
        b = []
        c = []

        #finding first 3 groups of names
        for i_line in range(len(lines1)):
            line1 = lines1[i_line]
            j_line = i_line + 1

            for j_line in range(len(lines1)):
                line2 = lines1[j_line]

                if(fuzz.token_set_ratio(line1.lower(), line2.lower()) > 70):
                    if(len(a) == 0):
                        a.append(line1)
                        break
                    else:
                        if(fuzz.token_set_ratio(a[0].lower(), line2.lower()) == 100):
                            break
                        else:
                            if(len(b) == 0):
                                b.append(line1)
                                break
                            else:
                                if(fuzz.token_set_ratio(b[0].lower(), line2.lower()) == 100):
                                    break
                                else:
                                    if(len(c) == 0):
                                        c.append(line1)
                                        break
                                pass
                            pass
                        pass
                    pass

        #overlooping all the names and adding the rest to the lists
        for i_line in range(len(lines1)):
            line2 = lines1[i_line]
            strOptions = [a[0], b[0], c[0]]
            highest = process.extractOne(line2, strOptions)
            if(highest[1] > 60):
                if(highest[0] == a[0] and fuzz.partial_ratio(line2.lower(),highest[0].lower()) > 70):
                    a.append(line2)
                    lines2.remove(line2)
                else:
                    if(highest[0] == b[0] and fuzz.partial_ratio(line2.lower(),highest[0].lower()) > 60):
                        b.append(line2)
                        lines2.remove(line2)
                    else:
                        if(highest[0] == c[0] and fuzz.partial_ratio(line2.lower(),highest[0].lower()) > 70):
                            c.append(line2)
                            lines2.remove(line2)
                    pass
                pass
            
        #another check of the rest of the names
        for i_line in range(len(lines2)):
            line2 = lines2[i_line]
            check_a = fuzz.ratio(a[0].lower(),line2.lower())
            check_b = fuzz.ratio(b[0].lower(),line2.lower())
            check_c = fuzz.ratio(c[0].lower(),line2.lower())
            if(check_a > check_b and check_a > check_c and check_a > 80):
                a.append(line2)
            else:
                if(check_b > check_c and check_b > 40):
                    b.append(line2)
                else:
                    if(check_c > 80):
                        c.append(line2)
                pass
            pass

        #deleting the first 2 that are the same
        a.pop(0)
        b.pop(0)
        c.pop(0) 


        #printing
        print("#1")
        print(a)
        print("#2")
        print(b)
        print("#3")
        print(c)



        


print('Begin: ')
path = "example1.txt"
text_1 = Text(path)
text_1.readFile(path)
print('Done')
