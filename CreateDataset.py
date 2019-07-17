"""
Michael Lombardo
Siamese Workshop
Dataset generator for Omniglot, creates csv of selected languages.
"""

import csv
import os
import glob
import random

def readFolder(data, path, lang, charz):
    data_path = os.path.join('data/' + path + '/','*g')
    files = glob.glob(data_path)
    for file in files:
        data.append([lang, charz, file])
    return data


"""
Reads all files in the data/Language, captures name of files
and their classification, writes to csv for easy access for dataloader
"""
def main():
    """
    Korean is a similar looking language that can stress a Model
    Ojibwe_(Canadian_Aboriginal_Syllabics) is a small and different language
    Hebrew
    Braille
    """
    languages = ['Greek']
    counts = [24]
    dataset = []

    for i in range(len(languages)):
        for j in range(1,counts[i]+1):
            if j >= 10:
                charPath = languages[i] + '/character' + str(j)
            else:
                charPath = languages[i] + '/character0' + str(j)

            dataset = readFolder(dataset, charPath, i, j)
    print("~~ Writing Model Names to parsedData.csv ~~")
    with open('parsedData24.csv', "w+") as csv_file:
        csv_file.truncate()
        writer = csv.writer(csv_file)
        for elem in dataset:
            writer.writerow(elem)
    print("~~ Done ~~")




if __name__ == "__main__":
    main()
