import serial
import sys
import os
import time
import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
This program only is tested for Radar IWR6843
"""

# Function to configure the serial ports and send the data from the configuration file to the radar


def serialConfig(configFileName):

    # Open the serial ports for the configuration and the data ports
    # Linux
    CLIport = serial.Serial('/dev/ttyACM0', 115200)
    Dataport = serial.Serial('/dev/ttyACM1', 921600)

    # Windows
    # CLIport = serial.Serial('COM9', 115200)
    # Dataport = serial.Serial('COM10', 921600)

    # Read the configuration file and send it to the board
    config = [line.rstrip('\r\n') for line in open(configFileName)]
    for i in config:
        CLIport.write((i+'\n').encode())
        print(i)
        time.sleep(0.01)

    return CLIport, Dataport

# ------------------------------------------------------------------

# Function to parse the data inside the configuration file


def parseConfigFile(configFileName):
    # Initialize an empty dictionary to store the configuration parameters
    configParameters = {}

    # Read the configuration file and send it to the board
    config = [line.rstrip('\r\n') for line in open(configFileName)]
    for i in config:

        # Split the line
        splitWords = i.split(" ")

        # Hard code the number of antennas, change if other configuration is used
        numRxAnt = 4
        numTxAnt = 3

        # Get the information about the profile configuration
        if "profileCfg" in splitWords[0]:
            startFreq = int(float(splitWords[2]))
            idleTime = int(splitWords[3])
            rampEndTime = float(splitWords[5])
            freqSlopeConst = float(splitWords[8])
            numAdcSamples = int(splitWords[10])
            numAdcSamplesRoundTo2 = 1

            while numAdcSamples > numAdcSamplesRoundTo2:
                numAdcSamplesRoundTo2 = numAdcSamplesRoundTo2 * 2

            digOutSampleRate = int(splitWords[11])

        # Get the information about the frame configuration
        elif "frameCfg" in splitWords[0]:

            chirpStartIdx = int(splitWords[1])
            chirpEndIdx = int(splitWords[2])
            numLoops = int(splitWords[3])
            numFrames = int(splitWords[4])
            framePeriodicity = float(splitWords[5])

    # Combine the read data to obtain the configuration parameters
    numChirpsPerFrame = (chirpEndIdx - chirpStartIdx + 1) * numLoops
    configParameters["numDopplerBins"] = numChirpsPerFrame / numTxAnt
    configParameters["numRangeBins"] = numAdcSamplesRoundTo2
    configParameters["rangeResolutionMeters"] = (
        3e8 * digOutSampleRate * 1e3) / (2 * freqSlopeConst * 1e12 * numAdcSamples)
    configParameters["rangeIdxToMeters"] = (3e8 * digOutSampleRate * 1e3) / (
        2 * freqSlopeConst * 1e12 * configParameters["numRangeBins"])
    configParameters["dopplerResolutionMps"] = 3e8 / (2 * startFreq * 1e9 * (
        idleTime + rampEndTime) * 1e-6 * configParameters["numDopplerBins"] * numTxAnt)
    configParameters["maxRange"] = (
        300 * 0.9 * digOutSampleRate)/(2 * freqSlopeConst * 1e3)
    configParameters["maxVelocity"] = 3e8 / \
        (4 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * numTxAnt)

    return configParameters

# ------------------------------------------------------------------

# Function to draw the plot


def draw(detObj):
    x, y, z, v = [], [], [], []

    if len(detObj["x"]) > 0:

        fig.clf()
        ax = fig.add_subplot(111, projection="3d")

        ax.set_zlim(bottom=-5, top=5)
        ax.set_ylim(bottom=0, top=10)
        ax.set_xlim(left=-4, right=4)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        x = -detObj["x"]
        y = detObj["y"]
        z = detObj["z"]
        v = detObj["velocity"]

        ax.scatter(x, y, z, c='r', marker='o', s=10)
        plt.pause(0.01)  # show


class readradar():
    def __init__(self, configFileName='./cfg/indoor.cfg', folderName="./data", num=600):
        self.byteBuffer = np.zeros(2**15, dtype='uint8')
        self.byteBufferLength = 0
        self.folderName = folderName
        self.num = num
        self.configFileName = configFileName


    def run(self, pipe):
        # Get the configuration parameters from the configuration file
        configParameters = parseConfigFile(self.configFileName)

        # Set the plot
        # fig = plt.figure()
        # plt.ion()
        # ax = Axes3D(fig)

        # Configurate the serial port.
        # The `Dataport` will start to read data immediately after the `serialConfig` function
        time.sleep(2)
        CLIport, Dataport = {}, {}
        CLIport, Dataport = serialConfig(self.configFileName)

        detObj, frameData = {}, []
        currentIndex, dataOk = 0, 0

        pipe.send("Radar is ready")
        print("Radar -> Camera: radar is ready to start")
        pipe.recv()
        print("Both sensors are ready to start")

        while True:
            # check if there is data
            dataOk, frameNumber, detObj, timestamp = self.readAndParseData68xx(
                Dataport, configParameters)

            if dataOk:
                # Store the current frame into frameData
                frameData.append(
                    dict(Data=detObj, Time=timestamp, Frame_ID=currentIndex))
                print("Radar count: " + str(currentIndex))
                # draw(detObj)    # very time consuming: > 0.1s
                currentIndex += 1

            if currentIndex == self. num:
                CLIport.write(('sensorStop\n').encode())
                CLIport.close()
                Dataport.close()

                # Saved as pickle file
                print("Radar Done!")
                self.outputFile = self.folderName + '/pointcloud' + '.pkl'
                f = open(self.outputFile, 'wb')
                pickle.dump(frameData, f)
                f.close()
                break

    # Funtion to read and parse the incoming data
    def readAndParseData68xx(self, Dataport, configParameters):

        # Constants
        OBJ_STRUCT_SIZE_BYTES = 12
        BYTE_VEC_ACC_MAX_SIZE = 2**15
        MMWDEMO_UART_MSG_DETECTED_POINTS = 1
        MMWDEMO_UART_MSG_RANGE_PROFILE = 2
        maxBufferSize = 2**15
        tlvHeaderLengthInBytes = 8
        pointLengthInBytes = 16
        magicWord = [2, 1, 4, 3, 6, 5, 8, 7]

        # Initialize variables
        magicOK = 0  # Checks if magic number has been read
        dataOK = 0  # Checks if the data has been read correctly
        frameNumber = 0
        detObj = {}

        readBuffer = Dataport.read(Dataport.in_waiting)
        byteVec = np.frombuffer(readBuffer, dtype='uint8')
        byteCount = len(byteVec)

        # Check that the buffer is not full, and then add the data to the buffer
        if (self.byteBufferLength + byteCount) < maxBufferSize:
            self.byteBuffer[self.byteBufferLength:self.byteBufferLength +
                            byteCount] = byteVec[:byteCount]
            self.byteBufferLength = self.byteBufferLength + byteCount

        # Check that the buffer has some data
        if self.byteBufferLength > 16:

            # Check for all possible locations of the magic word
            possibleLocs = np.where(self.byteBuffer == magicWord[0])[0]

            # Confirm that is the beginning of the magic word and store the index in startIdx
            startIdx = []
            for loc in possibleLocs:
                check = self.byteBuffer[loc:loc+8]
                if np.all(check == magicWord):
                    startIdx.append(loc)

            # Check that startIdx is not empty
            if startIdx:

                # Remove the data before the first start index
                if startIdx[0] > 0 and startIdx[0] < self.byteBufferLength:
                    self.byteBuffer[:self.byteBufferLength-startIdx[0]
                                    ] = self.byteBuffer[startIdx[0]:self.byteBufferLength]
                    self.byteBuffer[self.byteBufferLength-startIdx[0]:] = np.zeros(
                        len(self.byteBuffer[self.byteBufferLength-startIdx[0]:]), dtype='uint8')
                    self.byteBufferLength = self.byteBufferLength - startIdx[0]

                # Check that there have no errors with the byte buffer length
                if self.byteBufferLength < 0:
                    self.byteBufferLength = 0

                # word array to convert 4 bytes to a 32 bit number
                word = [1, 2**8, 2**16, 2**24]

                # Read the total packet length
                totalPacketLen = np.matmul(self.byteBuffer[12:12+4], word)

                # Check that all the packet has been read
                if (self.byteBufferLength >= totalPacketLen) and (self.byteBufferLength != 0):
                    magicOK = 1

        # If magicOK is equal to 1 then process the message
        if magicOK:
            # word array to convert 4 bytes to a 32 bit number
            word = [1, 2**8, 2**16, 2**24]

            # Initialize the pointer index
            idX = 0

            # Read the header
            magicNumber = self.byteBuffer[idX:idX+8]
            idX += 8
            version = format(np.matmul(self.byteBuffer[idX:idX+4], word), 'x')
            idX += 4
            totalPacketLen = np.matmul(self.byteBuffer[idX:idX+4], word)
            idX += 4
            platform = format(np.matmul(self.byteBuffer[idX:idX+4], word), 'x')
            idX += 4
            frameNumber = np.matmul(self.byteBuffer[idX:idX+4], word)
            idX += 4
            timeCpuCycles = np.matmul(self.byteBuffer[idX:idX+4], word)
            idX += 4
            numDetectedObj = np.matmul(self.byteBuffer[idX:idX+4], word)
            idX += 4
            numTLVs = np.matmul(self.byteBuffer[idX:idX+4], word)
            idX += 4
            subFrameNumber = np.matmul(self.byteBuffer[idX:idX+4], word)
            idX += 4

            # Read the TLV messages
            for tlvIdx in range(numTLVs):

                # word array to convert 4 bytes to a 32 bit number
                word = [1, 2**8, 2**16, 2**24]

                # Check the header of the TLV message
                tlv_type = np.matmul(self.byteBuffer[idX:idX+4], word)
                idX += 4
                tlv_length = np.matmul(self.byteBuffer[idX:idX+4], word)
                idX += 4

                # Read the data depending on the TLV message
                if tlv_type == MMWDEMO_UART_MSG_DETECTED_POINTS:

                    # Initialize the arrays
                    x = np.zeros(numDetectedObj, dtype=np.float32)
                    y = np.zeros(numDetectedObj, dtype=np.float32)
                    z = np.zeros(numDetectedObj, dtype=np.float32)
                    velocity = np.zeros(numDetectedObj, dtype=np.float32)

                    for objectNum in range(numDetectedObj):

                        # Read the data for each object
                        x[objectNum] = self.byteBuffer[idX:idX +
                                                       4].view(dtype=np.float32)
                        idX += 4
                        y[objectNum] = self.byteBuffer[idX:idX +
                                                       4].view(dtype=np.float32)
                        idX += 4
                        z[objectNum] = self.byteBuffer[idX:idX +
                                                       4].view(dtype=np.float32)
                        idX += 4
                        velocity[objectNum] = self.byteBuffer[idX:idX +
                                                              4].view(dtype=np.float32)
                        idX += 4

                    # Store the data in the detObj dictionary
                    detObj = {"numObj": numDetectedObj, "x": x,
                              "y": y, "z": z, "velocity": velocity}
                    dataOK = 1

            # Remove already processed data
            if idX > 0 and self.byteBufferLength > idX:
                shiftSize = totalPacketLen

                self.byteBuffer[:self.byteBufferLength -
                                shiftSize] = self.byteBuffer[shiftSize:self.byteBufferLength]
                self.byteBuffer[self.byteBufferLength - shiftSize:] = np.zeros(
                    len(self.byteBuffer[self.byteBufferLength - shiftSize:]), dtype='uint8')
                self.byteBufferLength = self.byteBufferLength - shiftSize

                # Check that there are no errors with the buffer length
                if self.byteBufferLength < 0:
                    self.byteBufferLength = 0

        return dataOK, frameNumber, detObj, time.time()
