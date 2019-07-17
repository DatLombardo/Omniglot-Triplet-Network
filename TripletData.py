def readModelCSV(filename):
      '''
      1-by-N(videos)
      '''
      dataContainer = []
      with open(filename) as infile:
          for line in infile:
              line = line.replace('"', '')
              line = line.replace('\n','')
              data = line.split(",")
              dataContainer.append(data)
      return dataContainer

def splitData(data):
    #Set at 0 as no character can be char0
    charCount = data[0][1]
    newData = []
    dimension = []
    for point in data:
        num = point[1]
        if num == charCount:
            dimension.append(point)
        else:
            newData.append(dimension)
            dimension = []
            charCount = num
            dimension.append(point)
    newData.append(dimension)
    return newData

def main(dataFile, trainSplit, batchSize, w, h):
    import torch
    import numpy as np
    import cv2
    from torch.utils.data import Dataset, DataLoader
    import random
    import itertools

    class TripletData(Dataset):
        def __init__(self, data, width, height, transforms=None):
            self.data = data
            self.width = width
            self.height = height

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            #Images are [0][2] & [1][2]
            img1 = np.divide(cv2.imread(self.data[idx][0][2]).astype(np.float32), 255)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            #img1 = img1[...,[2,1,0]]
            img1 = torch.tensor(cv2.resize(img1, (self.width, self.height)))
            img2 = np.divide(cv2.imread(self.data[idx][1][2]).astype(np.float32), 255)
            #img2 = img2[...,[2,1,0]]
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            img2 = torch.tensor(cv2.resize(img2, (self.width, self.height)))
            img3 = np.divide(cv2.imread(self.data[idx][2][2]).astype(np.float32), 255)
            #img2 = img2[...,[2,1,0]]
            img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
            img3 = torch.tensor(cv2.resize(img3, (self.width, self.height)))

            #Classes are [0][1] & [1][1]
            classList = [int(self.data[idx][0][1]),int(self.data[idx][1][1]), int(self.data[idx][2][1])]
            charClass = torch.tensor(classList, dtype=torch.float32)
            #nameList = [int(self.data[idx][0][2]), int(self.data[idx][1][2])]
            #fileNums = torch.tensor(nameList, dtype=torch.float32)

            return {'A' : img1, 'P' : img2,  'N' : img3, 'character': charClass}

    print("\n~~~| ModelDataloader.py Execution |~~~")
    dataFile = 'parsedData24.csv'
    loadedData = readModelCSV(dataFile)
    dataSplit = splitData(loadedData)
    del loadedData
    tripletCombs = []
    for charSet in dataSplit:
        charSplit = []
        for anch in charSet:
            pos = charSet[random.randint(0,len(charSet)-1)]
            negNum = random.randint(0,len(dataSplit)-1)
            neg = dataSplit[negNum][random.randint(0,len(dataSplit[negNum])-1)]
            if (pos[2][23:] == anch[2][23:]):
                pos = charSet[random.randint(0,len(anch)-1)]
            if (neg[2][23:] == anch[2][23:] or neg[1] == anch[1]):
                negNum = random.randint(0,len(dataSplit)-1)
                neg = dataSplit[negNum][random.randint(0,len(dataSplit[negNum])-1)]
            charSplit.append([anch, pos, neg])
        tripletCombs.append(charSplit)
    """
    for i in tripletCombs:
        for j in i:
            print(j[0][1], j[1][1], j[2][1])
    """
    trainSplit = 0.8
    trainSize = int(trainSplit * len(tripletCombs[0]))
    testSize = len(tripletCombs[0]) - trainSize
    train, test = torch.utils.data.random_split(tripletCombs[0], [trainSize, testSize])
    trainSet = np.array(train)
    testSet = np.array(test)
    for i in range(1,len(tripletCombs)):
        trainSize = int(trainSplit * len(tripletCombs[i]))
        testSize = len(tripletCombs[i]) - trainSize
        train, test = torch.utils.data.random_split(tripletCombs[i], [trainSize, testSize])
        trainSet = np.concatenate((trainSet, train), axis=0)
        testSet = np.concatenate((testSet, test), axis=0)

    del tripletCombs, dataSplit

    print("Loaded dataset")
    trainLoader = DataLoader(TripletData(trainSet, w, h), batch_size=batchSize, shuffle=True)
    testLoader = DataLoader(TripletData(testSet, w, h), batch_size=batchSize, shuffle=True)

    print("~~~| ModelDataloader.py Complete |~~~\n")
    return trainLoader, testLoader

if __name__ == "__main__":
    main()
