import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

style.use('fivethirtyeight')

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)


def animate(i):
    graph_data = open('./outputs/comparison.csv', 'r').read()
    lines = graph_data.split('\n')
    #print ("animate(): lines = {}".format(lines))
    xs = []
    targetList = []
    predictionList = []


    headers = lines[0].split(',')
    for lineNdx in range (1, len(lines)):
        line = lines[lineNdx]
        if len(line) > 1 :
            x, target, prediction = \
             line.split(',')
            try:
                xs.append(float(x))
            except:
                xs.append(None)
            try:
                targetList.append(float(target))
            except ValueError:
                taregtList.append(None)
            try:
                predictionList.append(float(prediction))
            except ValueError:
                predictionList.append(None)



    ax1.clear()
    plt.ylim((-1.5, 1.5))
    plt.scatter(xs, targetList, label='target')
    plt.scatter(xs, predictionList, label='prediction')

    ax1.legend(shadow=True, fancybox=True, loc='upper left')

ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()