#!/usr/bin/python3

# mainCodeParse.py
#
# A small utility to parse a code file and pull out methods and classes
#


import os
from time import strftime


#File and directory names and types
whereAREwe = os.getcwd()

fName = '/home/womo1998/Documents/Uni/masterarbeit/code/StabilDiagram.py' #grab the name of our input file

try:
    codeF = open(fName, 'r') # open file

except:
    print("Can't open that file!")

else:
    #if the file was processed successfully
    clssList = [] # here i'll hold all my class names
    AllLines = []
    #first gather up all the classes
    print("Gathering classes")
    
    for tLine in codeF:
        # look for classes so we can find calls to them
        # we could do the same for method calls,
        # but there would be lots of those.
        leftPsh = tLine.lstrip()
        if "class" in tLine and "(" in tLine \
        and ")" in tLine and leftPsh[0] != "#":
            #strip the other text and keep name of the class
            tLine = tLine.replace('class', '')
            startvars = tLine.find("(")
            # endvars = tLine.find(")") + 1
            #just out of curiosity
            #remThis = tLine[startvars:endvars]
        
            #don't need replace just truncate
            tLine = tLine[0:startvars]
            tLine=tLine.lstrip() #take whitespace off
        
            #put that class into a list
            clssList.append(tLine)
            AllLines.append(tLine)
            
        if " def " in tLine and leftPsh[0] != "#":
            # not a comment only line
            AllLines.append(tLine.partition('(')[0])
        if leftPsh.startswith('self.') and '=' in tLine:
            AllLines.append(tLine.partition('=')[0])
    #reset the pointer to make another pass
    codeF.seek(0)

AllLines_str=''
for line in AllLines:
    AllLines_str+=line+'\n'
print(AllLines_str)
