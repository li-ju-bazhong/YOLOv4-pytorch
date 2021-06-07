import matplotlib.pyplot as plt
import numpy as np

def draw_plot(rec, prec, mrec, mprec, class_name):
    plt.plot(rec, prec, "-o")
    # add a new penultimate point to the list (mrec[-2], 0.0)
    # since the last line segment (and respective area) do not affect the AP value
    area_under_curve_x = mrec[:-1]  # + [mrec[-2]] + [mrec[-1]]
    area_under_curve_y = mprec[:-1]  # + [0.0] + [mprec[-1]]

    plt.fill_between(
        area_under_curve_x,
        0,
        area_under_curve_y,
        alpha=0.2,
        edgecolor="r",
    )
    # set window title
    fig = plt.gcf()  # gcf - get current figure
    fig.canvas.set_window_title("AP " + class_name)
    # set plot title
    plt.title(
        class_name, fontsize=44
    )  ####################################################################
    # plt.title('class: ' + text, fontsize=24)########################################################################
    # plt.suptitle('This is a somewhat long figure title', fontsize=16)
    # set axis titles
    plt.xlabel("Recall", fontsize=32)
    plt.ylabel("Precision", fontsize=32)
    # optional - set axes
    axes = plt.gca()  # gca - get current axes
    axes.set_xlim([0.0, 1.0])
    axes.set_ylim([0.0, 1.05])  # .05 to give some extra space
    # Alternative option -> wait for button to be pressed
    # while not plt.waitforbuttonpress(): pass # wait for key display
    # Alternative option -> normal display
    #plt.show()
    # save the plot
    fig.savefig("file/" + class_name + ".png")
    # plt.cla()  # clear axes for next plot
def file_chiose(class_name):
    files = open("file/%s.txt" % class_name, "r", encoding="utf-8")

    files = files.readlines()

    mrec = eval(files[1])
    mprec = eval(files[3])
    rec = eval(files[5])+[eval(files[5])[-2]]+[eval(files[5])[-1]]
    prec = eval(files[7])+[eval(files[7])[-2]]+[eval(files[7])[-1]]
    return rec, prec, mrec, mprec

def draw():
    class_names = ["aeroplane", "bicycle"]

    rec_0, pre_0, mrec_0, mprec_0 = file_chiose(class_names[0])
    rec_1, pre_1, mrec_1, mprec_1 = file_chiose(class_names[1])

    plt.plot(rec_0, pre_0, "g^")
    plt.plot(rec_1, pre_1, "bs")
    # add a new penultimate point to the list (mrec[-2], 0.0)
    # since the last line segment (and respective area) do not affect the AP value
    area_under_curve_x = mrec_0[:-1]  # + [mrec[-2]] + [mrec[-1]]
    area_under_curve_y = mprec_0[:-1]  # + [0.0] + [mprec[-1]]
    #plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor="r")
    area_under_curve_x_1 = mrec_1[:-1]  # + [mrec[-2]] + [mrec[-1]]
    area_under_curve_y_1 = mprec_1[:-1]  # + [0.0] + [mprec[-1]]
    #plt.fill_between(area_under_curve_x_1, 0, area_under_curve_y_1, alpha=0.2, edgecolor="r")

    # set window title
    fig = plt.gcf()  # gcf - get current figure
    fig.canvas.set_window_title("AP " + "class_name")
    # set plot title
    plt.title(
        "class_name", fontsize=44
    )  ####################################################################
    # plt.title('class: ' + text, fontsize=24)########################################################################
    # plt.suptitle('This is a somewhat long figure title', fontsize=16)
    # set axis titles
    plt.xlabel("Recall", fontsize=32)
    plt.ylabel("Precision", fontsize=32)
    # optional - set axes
    axes = plt.gca()  # gca - get current axes
    axes.set_xlim([0.0, 1.0])
    axes.set_ylim([0.0, 1.05])  # .05 to give some extra space
    # Alternative option -> wait for button to be pressed
    # while not plt.waitforbuttonpress(): pass # wait for key display
    # Alternative option -> normal display
    plt.show()
    # save the plot
    fig.savefig("file/" + "class_name" + ".png")

def np_size(file):
    return np.array(file).shape

def down_sample(data, points=7130):
    size = np_size(data)[0]
    k = size//points
    new_data = []
    for i in range(points):
        site = k*i
        new_data.append(data[site])
    return new_data

def file():
    file = open("file/class_shape.txt", 'a', encoding="utf-8")
    for name in class_names:
        rec, _, mrec, _ = file_chiose(name)
        line = "class: {}, rec shape: {}, mrec shape: {}\n".format(name, np_size(rec), np_size(mrec))
        print(line)
        file.write(line)

def dra_multi_downsample(class_name):
    rec, prec, mrec, mprec = file_chiose(class_name)

    rec_new = down_sample(rec)
    prec_new = down_sample(prec)
    mrec_new = down_sample(mrec)
    mprec_new = down_sample(mprec)
    draw_plot(rec, prec, mrec, mprec, class_name)
    draw_plot(rec_new, prec_new, mrec_new, mprec_new, class_name+"_new")
class_names = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
        "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
        "sheep", "sofa", "train", "tvmonitor"]
for i in range(len(class_names)):
    dra_multi_downsample(class_names[i])





