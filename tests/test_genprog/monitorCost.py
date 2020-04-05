import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

style.use('fivethirtyeight')

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)


def animate(i):
    graph_data = open('./outputs/generationsCost.csv', 'r').read()
    lines = graph_data.split('\n')
    #print ("animate(): lines = {}".format(lines))
    generations = []
    trainingChampionCostList = []
    validationChampionCostList = []
    trainingMedianCostList = []
    validationMedianCostList = []

    headers = lines[0].split(',')
    for lineNdx in range (1, len(lines)):
        line = lines[lineNdx]
        if len(line) > 1 :
            generation, trainingChampionCost, validationChampionCost, trainingMedianCost, validationMedianCost = \
             line.split(',')
            try:
                generations.append(int(generation))
            except:
                generations.append(None)
            try:
                trainingChampionCostList.append(float(trainingChampionCost))
            except ValueError:
                trainingChampionCostList.append(None)
            try:
                validationChampionCostList.append(float(validationChampionCost))
            except ValueError:
                validationChampionCostList.append(None)
            try:
                trainingMedianCostList.append(float(trainingMedianCost))
            except ValueError:
                trainingMedianCostList.append(None)
            try:
                validationMedianCostList.append(float(validationMedianCost))
            except ValueError:
                validationMedianCostList.append(None)


    ax1.clear()
    plt.ylim((0, 1.0))
    ax1.plot(generations, trainingChampionCostList, label=headers[1])
    ax1.plot(generations, validationChampionCostList, label=headers[2])
    ax1.plot(generations, trainingMedianCostList, label=headers[3])
    ax1.plot(generations, validationMedianCostList, label=headers[4], c='fuchsia')

    ax1.legend(shadow=True, fancybox=True, loc='upper left')

ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()