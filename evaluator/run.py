import os

import numpy as np

from evaluator.fileReader import readImage
from evaluator.fileReader import readTextFile
import re

path = os.path.dirname(os.path.abspath(__file__))

optimalFolder = path + "/optimal"  # you may have to specify the complete path
studentFolder = path + "/student"  # you may have to specify the complete path
colorValueSlackRange = 40
blackValueThreshold = 100  # colors below 100 is black
pixelRangeCheck = 4
checkEightSurroundingPixels = True


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r"(\d+)", text)]


def readFilesFromFolder(directory):
    allFiles = []
    for filename in sorted(os.listdir(directory), key=natural_keys):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            filename = os.path.join(directory, filename)
            allFiles.append(readImage(filename))
        elif filename.endswith(".txt"):
            filename = os.path.join(directory, filename)
            allFiles.append(readTextFile(filename))
        elif filename.endswith(".mat"):
            filename = os.path.join(directory, filename)
            allFiles.append(readHSIImage(filename))
    return allFiles


def comparePics(studentPic, optimalSegmentPic):
    # for each pixel in studentPic, compare to corresponding pixel in optimalSegmentPic
    global colorValueSlackRange
    global checkEightSurroundingPixels
    global pixelRangeCheck

    height, width = studentPic.shape

    counter = 0  # counts the number of similar pics
    numberOfBlackPixels = 0
    for w in range(width):
        for h in range(height):
            # if any pixel nearby or at the same position is within the range, it counts as correct
            color1 = studentPic[h][w]
            color2 = optimalSegmentPic[h][w]
            if color1 < blackValueThreshold:
                # black color
                numberOfBlackPixels += 1
                if int(color1) == int(color2):
                    counter += 1
                    continue
                elif checkEightSurroundingPixels:
                    # check surroundings
                    correctFound = False
                    for w2 in range(w - pixelRangeCheck, w + pixelRangeCheck + 1):
                        if correctFound:
                            break
                        for h2 in range(h - pixelRangeCheck, h + pixelRangeCheck + 1):
                            if w2 >= 0 and h2 >= 0 and w2 < width and h2 < height:

                                color2 = optimalSegmentPic[h2][w2]
                                if (
                                    color1 - colorValueSlackRange < color2
                                    and color2 < colorValueSlackRange + color1
                                ):
                                    correctFound = True
                                    counter += 1
                                    break

    return counter / max(numberOfBlackPixels, 1)


def eval_files(optimal=optimalFolder, student=studentFolder):
    optimalFiles = readFilesFromFolder(optimal)
    studentFiles = readFilesFromFolder(student)
    results = np.zeros(len(studentFiles))
    totalScore = 0
    for i, student in enumerate(studentFiles):
        highestScore = 0
        for opt in optimalFiles:
            result1 = comparePics(opt, student)
            result2 = comparePics(student, opt)
            # 			print("PRI 1: %.2f" % result1)
            # 			print("PRI 2: %.2f" % result2)
            result = min(result1, result2)
            highestScore = max(highestScore, result)
        totalScore += highestScore
        a = highestScore * 100
        results[i] = a
        print(f"{i} Score: %.2f" % a + "%")
    a = totalScore / len(studentFiles) * 100
    print("Total Average Score: %.2f" % a + "%")
    return results


if __name__ == "__main__":
    eval_files()
