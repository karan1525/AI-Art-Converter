# Author - Karan Bhargava
# Poems from -> http://www.familyfriendpoems.com/ (passed in through a scanner one at a line)

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from PIL import Image
from PIL import ImageFilter
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
from pybrain.supervised.trainers import BackpropTrainer
import random
import subprocess
import time

# Defined all the colors that the AI is trained on
happy_color1 = '#FF9A55'
happy_color2 = '#FFEA6C'
happy_color3 = '#54FFFB'
happy_color4 = '#E7B2FF'
happy_color5 = '#89FFCC'
happy_color6 = '#20FBEA'
happy_color_7 = '#DCFF26'
color_green = '#00FF00'
color_yellow = '#FFFF00'
color_orange = '#FFA500'

angry_color1 = '#637A8A'
angry_color2 = '#162340'
angry_color3 = '#05383B'
angry_color4 = '#00282B'
angry_color5 = '#00171A'
color_brown = '#A52A2A'
color_red = '#FF0000'
color_magenta = '#FF00FF'

neutral_color1 = '#727B84'
neutral_color2 = '#DF9496'
neutral_color3 = '#F6F4DA'
neutral_color4 = '#F4F3EE'
neutral_color5 = '#D9E2E1'
neutral_color6 = '#C4BDAC'
neutral_color7 = '#EFEEEE'
color_blue = '#0000FF'
color_beige = '#F5F5DC'
color_gray = '#808080'

def main():
    # Get the input of the sentence to give the AI
    # sentence = input('Enter a sentence please: ')

    with open('Poem.txt') as f:
        text = f.read()

    # Analyze the sentence
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(text)

    # Create a data set of emotions and colors defined with it
    ds = SupervisedDataSet(1, 3)

    ds.addSample((1,), (2, 3, 4))  # Happy -> 1, Green is 2, Yellow is 3, Orange is 4
    ds.addSample((-1,), (5, 6, 7))  # Mad -> -1, Brown is 5, Red is 6, Black is 7
    ds.addSample((0,), (8, 9, 10))  # Meh -> 0, Blue is 8, Gray is 9, Beige is 10

    # Build a neural net with the above provided data set
    net = buildNetwork(1, 3, bias=True, hiddenclass=TanhLayer)

    # train the data set using a hidden TanhLayer
    trainer = BackpropTrainer(net, ds)
    print(trainer.train())

    # Testing for the trained neural network
    # for inp, tar in ds:
    #     print([net.activate(inp), tar])

    # print(ds['target'][0]) #input or target

    # get the sentiment value from Vader (which returns in a cryptic format)
    sentimentValue = value_comparator(vs)

    # The AI takes about 10 seconds to think how it feels about the picture
    print("Let me think about that for a moment..")
    say("Let me think about that for a moment.")

    #Let the method sleep for 10 seconds
    time.sleep(10)

    # if sentiment is happy, the trained AI chooses a happy color and draws a picture
    if sentimentValue is 1:
        print("Happy!")
        say("That line made me happy!")

        testImage = Image.new("RGB", (10, 10), (255, 255, 255))
        pixel = testImage.load()

        for x in range(10):
            for y in range(10):

                colorRandNumber = random.randrange(1,6)
                pixel[x, y] = (hex_to_rgb(getRandomHappyColor(colorRandNumber)))

        testImage.filter(ImageFilter.DETAIL)
        size_tuple = 800, 800
        testImage = testImage.resize(size_tuple)
        testImage.show()

    # if sentiment is angry/sad, the trained AI chooses a sad color and draws a picture
    elif sentimentValue is -1:
        print("Angry!")
        say("That line made me angry!")
        testImage = Image.new("RGB", (10, 10), (255, 255, 255))
        pixel = testImage.load()

        for x in range(10):
            for y in range(10):
                colorRandNumber = random.randrange(1, 6)
                pixel[x, y] = (hex_to_rgb(getRandomSadColor(colorRandNumber)))

        testImage.filter(ImageFilter.BLUR)
        size_tuple = 800,800
        testImage = testImage.resize(size_tuple)
        testImage.show()

    # Else the picture is neutral and the trained AI draws a meh picture
    else:
        print("Meh")
        say("That line was meh")
        testImage = Image.new("RGB", (10, 10), (255, 255, 255))
        pixel = testImage.load()

        for x in range(10):
            for y in range(10):
                colorRandNumber = random.randrange(1, 6)
                pixel[x, y] = (hex_to_rgb(getRandomNeutralColor(colorRandNumber)))

        size_tuple = 800, 800
        testImage = testImage.resize(size_tuple)
        testImage.show()


# A helper method to understand the cryptic score provided by the sentiment analyzer
def value_comparator(polarityScores):
    negativeScore = polarityScores['neg']
    neutralScore = polarityScores['neu']
    positiveScore = polarityScores['pos']

    if positiveScore > negativeScore:
        return 1  # the sentence was positive
    elif negativeScore > positiveScore:
        return -1  # the sentence was negative
    else:
        return 0  # the sentence was neutral


# A helper method for the AI to say what it is thinking
def say(text):
    subprocess.call('say ' + text, shell= True)


# A helper method to get a random happy color
def getRandomHappyColor(number):
    if number is 1:
        return happy_color1 or color_orange
    elif number is 2:
        return happy_color2 or color_green
    elif number is 3:
        return happy_color3 or color_yellow
    elif number is 4:
        return happy_color4 or happy_color6
    else:
        return happy_color5 or happy_color_7


# A helper method to get a random sad color
def getRandomSadColor(number):
    if number is 1:
        return angry_color1 or color_brown
    elif number is 2:
        return angry_color2 or color_red
    elif number is 3:
        return angry_color3 or color_magenta
    elif number is 4:
        return angry_color4
    else:
        return angry_color5


# A helper method to get a random neutral color
def getRandomNeutralColor(number):
    if number is 1:
        return neutral_color1 or color_blue
    elif number is 2:
        return neutral_color2 or color_beige
    elif number is 3:
        return neutral_color3 or color_gray
    elif number is 4:
        return neutral_color4 or neutral_color6
    else:
        return neutral_color5 or neutral_color7


# A helper method to convert HEX values to RGB values
def hex_to_rgb(value):
    """Return (red, green, blue) for the color given as #rrggbb."""
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


# To make the program executable
if __name__ == '__main__':
    main()
