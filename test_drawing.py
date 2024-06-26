# import matplotlib.pyplot as plt

# # 数据
# x = [1, 2, 3, 4, 5]
# y = [10, 8, 5, 3, 6]
# error = [1, 0.5, 1.2, 0.8, 0.3]

# # 创建画布和子图
# fig, ax = plt.subplots()

# # 绘制折线图
# ax.plot(x, y, marker='o', linestyle='-')

# # 绘制误差线
# ax.errorbar(x, y, yerr=error, linestyle='None', marker='None', capsize=5)

# # 设置图表标题和轴标签
# ax.set_title('Line Chart with Error Bars')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# plt.savefig('test.png')
# # 显示图表
# plt.show()

# ## with arrow nice
# import seaborn as sns
# import matplotlib.pyplot as plt

# # 数据
# x = [1, 2, 3, 4, 5]
# y1 = [10, 8, 5, 3, 6]
# y2 = [7, 5, 3, 2, 4]

# # 创建画布和子图
# fig, ax = plt.subplots()

# # 绘制折线图
# sns.lineplot(x=x, y=y1, marker='o', linestyle='-', label='Line 1', ax=ax)
# sns.lineplot(x=x, y=y2, marker='o', linestyle='-', label='Line 2', ax=ax)

# # 添加注释
# ax.annotate('Annotation 1', xy=(2, 8), xytext=(3, 8),
#             arrowprops=dict(arrowstyle='->'))
# ax.annotate('Annotation 2', xy=(4, 2), xytext=(3, 3),
#             arrowprops=dict(arrowstyle='->'))

# # 设置图表标题和轴标签
# ax.set_title('Multiple Lines with Annotations')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# plt.savefig('test.png')
# # 显示图表
# plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# from matplotlib import rcParams

# rcParams['text.usetex'] = True
# rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{mathptmx}'

# import matplotlib as mpl
# mpl.rcParams.update(mpl.rcParamsDefault)
# 数据
def line_accurary_rankK():
    x = np.array([1, 5, 10, 50])
    TIRG = np.array([14.73,42.42,58.75,86.15])
    MAAF= np.array([12.15,36.39,49.87,79.90])
    ARTEMIS = np.array([19.43,48.99,64.03,89.74])
    CIRPLANT = np.array([19.82,52.53,68.77,92.63])
    CLIP4CIR = np.array([33.94,67.08,80.33,95.67])
    errorTIRG = np.array([7.49,22.75,32.4,57.87])
    errorMAAF = np.array([6.12,17.91,25.81,48.68])
    errorARTEMIS = np.array([9.5,27.48,38.24,64.45])
    errorCIRPLANT = np.array([11.39,31.65,43.33,69.61])
    errorCLIP4CIR = np.array([18.81,41.48,52.25,74.42])

    default_colors = sns.color_palette()

    # 创建画布和子图
    fig, ax = plt.subplots()

    # 绘制折线图
    sns.lineplot(x=x, y=TIRG, marker='o', linestyle='-',color=default_colors[0], label='TIRG', ax=ax)
    sns.lineplot(x=x, y=MAAF, marker='o', linestyle='-',color=default_colors[1], label='MAAF', ax=ax)
    sns.lineplot(x=x, y=ARTEMIS, marker='o', linestyle='-',color=default_colors[2], label='ARTEMIS', ax=ax)
    sns.lineplot(x=x, y=CIRPLANT, marker='o', linestyle='-',color=default_colors[3], label='CIRPLANT', ax=ax)
    sns.lineplot(x=x, y=CLIP4CIR, marker='o', linestyle='-',color=default_colors[4], label='CLIP4CIR', ax=ax)

    # ax.errorbar(x, TIRG, yerr=errorTIRG, fmt='none', ecolor='black', capsize=4)

    # sns.lineplot(x=x,y=errorTIRG,marker='o',linestyle='dashed',color=default_colors[0],ax=ax)
    # sns.lineplot(x=x,y=errorMAAF,marker='o',linestyle='dashed',color=default_colors[1],ax=ax)
    # sns.lineplot(x=x,y=errorARTEMIS,marker='o',linestyle='dashed',color=default_colors[2],ax=ax)
    # sns.lineplot(x=x,y=errorCIRPLANT,marker='o',linestyle='dashed',color=default_colors[3],ax=ax)
    # sns.lineplot(x=x,y=errorCLIP4CIR,marker='o',linestyle='dashed',color=default_colors[4],ax=ax)


    ax.fill_between(x, errorTIRG, TIRG, alpha=0.3)
    ax.fill_between(x, errorMAAF, MAAF, alpha=0.3)
    ax.fill_between(x, errorARTEMIS, ARTEMIS, alpha=0.3)
    ax.fill_between(x, errorCIRPLANT, CIRPLANT, alpha=0.3)
    ax.fill_between(x, errorCLIP4CIR, CLIP4CIR, alpha=0.3)

    # 绘制误差线
    # ax.errorbar(x, y1, yerr=error1, linestyle='None', marker='None', capsize=5)
    # ax.errorbar(x, y2, yerr=error2, linestyle='None', marker='None', capsize=5)

    # 设置图表标题和轴标签
    # ax.set_title('Multiple Models Accuracy with Average visual Corruption')
    ax.set_xlabel('Rank K')
    ax.set_ylabel('Accuracy')
    plt.savefig('./paper_images/cirr_corruption_accurary.png')

    # 显示图表
    plt.show()

# tips = sns.load_dataset("tips")
# sns.histplot(data=tips, x="day", hue="sex", multiple="dodge", shrink=.8)
# plt.savefig('test.png')
# histgram bar
import seaborn as sns
import matplotlib.pyplot as plt

# 数据
def histgram():
    categories = ['CIRR_image', 'CIRR_image_sub']
    # TIRG = [CIRR_rank1_waverageimagecorruption, sub_rank1_w_average_cprrupti]
    # for image corruption
    TIRG = [0.51,0.73]
    MAAF= [0.5,0.77]
    ARTEMIS = [0.49,0.76]
    CIRPLANT = [0.57,0.78]
    CLIP4CIR = [0.55,0.74]

    # for text corruption
    TIRG_text = [0.5,0.85]
    MAAF_text = [0.97,0.97]
    ARTEMIS_text = [0.78,0.89]
    CIRPLANT_text = [0.94,0.96]
    CLIP4CIR_text = [0.89,0.94]

    TIRG = TIRG+TIRG_text
    MAAF = MAAF+MAAF_text
    ARTEMIS = ARTEMIS+ARTEMIS_text
    CIRPLANT = CIRPLANT+CIRPLANT_text
    CLIP4CIR = CLIP4CIR+CLIP4CIR_text
    categories = categories + ['CIRR_text', 'CIRR_sub_text']

    # 组合数据
    data = {
        'Categories': categories * 5,
        'Values': TIRG + MAAF + ARTEMIS + CIRPLANT + CLIP4CIR,
        'Group':['TIRG'] * len(categories) + ['MAAF'] * len(categories) + ['ARTEMIS'] * len(categories) + ['CIRPLANT'] * len(categories) + ['CLIP4CIR'] * len(categories)
    }

    # 创建画布和子图
    fig, ax = plt.subplots()

    # 使用 Seaborn 绘制柱状图
    sns.barplot(x='Categories', y='Values', hue='Group', data=data, ax=ax)
    plt.ylim(0.4, 1)

    # 设置图例
    ax.legend()

    # 设置图表标题和轴标签
    # ax.set_title('Average relative robustness for image corruptions on CIRR and CIRR_sub')
    # ax.set_xlabel('Categories')
    ax.set_ylabel('Relative Robustness')    

    # 调整图表布局
    plt.tight_layout()

    plt.savefig('./paper_images/histgram_test.png',dpi=300)

    # 显示图表
    plt.show()

def line_relativeRobustness_rankK():
    x = np.array([1, 5, 10, 50])
    TIRG = np.array([0.51,0.54,0.55,0.67])
    MAAF= np.array([0.5,0.49,0.52,0.61])
    ARTEMIS = np.array([0.49,0.56,0.60,0.72])
    CIRPLANT = np.array([0.57,0.60,0.63,0.75])
    CLIP4CIR = np.array([0.55,0.62,0.65,0.78])

    default_colors = sns.color_palette()

    # 创建画布和子图
    fig, ax = plt.subplots()

    # 绘制折线图
    sns.lineplot(x=x, y=TIRG, marker='o', linestyle='-',color=default_colors[0], label='TIRG', ax=ax)
    sns.lineplot(x=x, y=MAAF, marker='o', linestyle='-',color=default_colors[1], label='MAAF', ax=ax)
    sns.lineplot(x=x, y=ARTEMIS, marker='o', linestyle='-',color=default_colors[2], label='ARTEMIS', ax=ax)
    sns.lineplot(x=x, y=CIRPLANT, marker='o', linestyle='-',color=default_colors[3], label='CIRPLANT', ax=ax)
    sns.lineplot(x=x, y=CLIP4CIR, marker='o', linestyle='-',color=default_colors[4], label='CLIP4CIR', ax=ax)

    # ax.errorbar(x, TIRG, yerr=errorTIRG, fmt='none', ecolor='black', capsize=4)   
    ax.set_xlabel('Rank K')
    ax.set_ylabel('Relative Robustness')
    plt.savefig('./paper_images/relative_robustness.png')

    # 显示图表
    plt.show()

def number_distribution_img():
    import seaborn as sns
    import matplotlib.pyplot as plt

    # 示例数据
    categories = ['One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten']
    
    # test  = [46.1333333,	39.3777778,	16.2666667,	5.9555556,	2.4,	1.1555556,	0.3555556,	0.5333333,	0.3555556,	0.9777778]
    # test = [37.9779099,	36.4485981,	16.1777778,	5.8666667	,2.4,	1.1555556,	0.3555556	,0.2666667,	0.2666667,	0.2666667]
    test = [37.94567063,	36.41765705,	15.44991511,	5.602716469,	2.292020374,	1.103565365,	0.339558574,	0.25466893,	0.339558574,	0.25466893]

    train = [40.976621,	36.9227226,	12.7490499,	5.2286076,	1.9117816,	1.2553265,	0.3685362,	0.3109524,	0.0691005,	0.2073016]

    test = [round(item,1) for item in test]
    train = [round(item,1) for item in train]

    data = {
        'Categories': categories * 2,
        'Values':  train + test,
        'Group':['Test'] * len(categories) + ['Train'] * len(categories)
    }
    # values2 = [7, 9, 6, 10]

    # 设置图形大小
    fig,ax= plt.subplots(figsize=(8, 6))

    # 绘制柱状图
    # default_colors = sns.color_palette()

    sns.barplot(x='Categories', y='Values',hue = 'Group', data=data, ax=ax)
    # for i, v in enumerate(test):
    #     ax.annotate(str(v), xy=(i, v), ha='center', va='bottom')
    # plt.ylim(0.4,1)

    for p in ax.patches:
    # 获取柱子的高度
        height = p.get_height()
    # 在柱子的中心位置添加数值
        ax.annotate(height, (p.get_x() + p.get_width() / 2, height), ha='center', va='bottom')

    ax.legend()

    # show number on top of bar
    # for i in range(len(categories)):
    #     plt.text(i, values1[i], str(values1[i]), ha='center', va='bottom')

    # 调整柱状图间的间距
    # plt.subplots_adjust(wspace=0.2)

    # 添加标题和标签
    # plt.title("Comparison of Values")
    # plt.xlabel("Categories")
    plt.ylabel("Percentage")
    plt.savefig('./paper_images/number_distribution_train_test.png')

    # 显示图形
    plt.show()

def attribute_distribution_img():
    import seaborn as sns
    import matplotlib.pyplot as plt

    # 示例数据
    categories = ['Color',  'Size','Shape',]
    values1 = [0.821649976	,0.234620887,0.017167382	]

    # values2 = [7, 9, 6, 10]

    # 设置图形大小
    plt.figure(figsize=(8, 6))

    # 绘制柱状图
    default_colors = sns.color_palette()

    sns.barplot(x=categories, y=values1, alpha=0.4)
    # sns.barplot(x=categories, y=values2, color='red', alpha=0.7)

    # show number on top of bar
    # for i in range(len(categories)):
    #     plt.text(i, values1[i], str(values1[i]), ha='center', va='bottom')

    # 调整柱状图间的间距
    plt.subplots_adjust(wspace=0.2)

    # 添加标题和标签
    # plt.title("Comparison of Values")
    # plt.xlabel("Categories")
    plt.ylabel("Percentage")
    plt.savefig('./paper_images/attributes_distribution.png')

    # 显示图形
    plt.show()

def test():
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    # Sample data
    categories = ['Category 1', 'Category 2', 'Category 3']
    values1 = [10, 15, 8]
    values2 = [20, 25, 12]

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Set the width of each bar
    bar_width = 0.35

    # Generate an array of indices for the bars
    bar_indices = np.arange(len(categories))

    # Plot the barplot
    sns.barplot(x=bar_indices, y=values1, color='skyblue', label='Values 1', alpha=0.7)
    sns.barplot(x=bar_indices + bar_width, y=values2, color='orange', label='Values 2', alpha=0.7)

    # Add value annotations
    for i, (v1, v2) in enumerate(zip(values1, values2)):
        ax.text(i, v1 + 1, str(v1), ha='center', va='bottom')
        ax.text(i + bar_width, v2 + 1, str(v2), ha='center', va='bottom')

    # Set x-axis tick labels
    ax.set_xticks(bar_indices + bar_width / 2)
    ax.set_xticklabels(categories)

    # Set labels and title
    ax.set_xlabel('Categories')
    ax.set_ylabel('Values')
    ax.set_title('Bar Plot with Multiple Bars per Category')

    # Add a legend
    ax.legend()
    plt.savefig('./paper_images/recall_rank.png')
    # Show the plot
    plt.show()


def recall_wrank():
    # CLEAN PERFGORMANCE 
    # x = np.array([1, 5, 10, 50])
    # # TIRG = np.array([14.73,42.42,58.75,86.15])
    # MAAF= np.array([12.15,36.39,49.87,79.90])
    # # ARTEMIS = np.array([19.43,48.99,64.03,89.74])
    # CIRPLANT = np.array([19.82,52.53,68.77,92.63])
    # CLIP4CIR = np.array([33.94,67.08,80.33,95.67])
    # # errorTIRG = np.array([7.49,22.75,32.4,57.87])
    # errorMAAF = np.array([6.12,17.91,25.81,48.68])
    # # errorARTEMIS = np.array([9.5,27.48,38.24,64.45])
    # errorCIRPLANT = np.array([11.39,31.65,43.33,69.61])
    # errorCLIP4CIR = np.array([18.81,41.48,52.25,74.42])


    # AVERAGE PERFORMANCE AND STD
    x = np.array([1, 5, 10, 50])
    TIRG=np.array([5.64,    16.9,  24.4,47.08])
    MAAF=np.array([6.12,   17.91, 25.81,48.68])
    ARTEMIS=np.array([6.68, 19.23,27.47,51.00])
    CIRPLANT=np.array([11.39, 31.65,43.33,69.61])
    CLIP4CIR=np.array([18.81, 41.48,52.25,74.42])
    BLIP2 = np.array([14.46,38.45,51.86,79.42])
    BLIP2_CIR = np.array([21.41,45.93,57.03,78.35])
    instructBLIP = np.array([6.3,18.89,27.31,50.26])

    IMAGE_CLIP=np.array([5.17,14.98,21.59,41.0])
    TEXT_CLIP=np.array([11.55,27.68,37.03,59.63])
    IMAGE_RESNET50=np.array([6.31,19.57,28.45,52.26])

    TIRG_std=np.array([2.91,8.32,11.34,16.43])
    MAAF_std=np.array([2.59,7.68,10.57,15.44])
    ARTEMIS_std=np.array([3.44,9.0,11.97,16.18])
    CIRPLANT_std=np.array([4.43,11.78,15.09,17.37])
    CLIP4CIR_std=np.array([8.36,16.11,18.41,18.17])
    BLIP2_std=np.array([2.58,5.3,6.77,7.26])
    IMAGE_CLIP_std=np.array([1.55,5.09,7.21,11.4])
    TEXT_CLIP_std=np.array([4.65,10.19,12.89,15.88])

    BLIP2_CIR_std = np.array([4.48,8.47,9.27,8.73])
    instructBLIP_std = np.array([1.39,4.17,5.82,8.9])


    IMAGE_RESNET50_std=np.array([1.87,6.84,10.31,15.4])
    # IMAGE_RESNET50_var =np.array(np.var(i) for i in IMAGE_RESNET50_std)


    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    
    # 创建主图对象
    fig, ax = plt.subplots()

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # import pdb
    # pdb.set_trace()

    # 绘制原图
    ax.plot(x, TIRG, marker='o', label='TIRG',color=colors[0])
    ax.plot(x, MAAF, marker='o', label='MAAF',color=colors[1])
    ax.plot(x, ARTEMIS, marker='o', label='ARTEMIS',color=colors[2])
    ax.plot(x, CIRPLANT, marker='o', label='CIRPLANT', color=colors[3])
    ax.plot(x, CLIP4CIR, marker='o', label='CLIP4CIR', color=colors[4])
    ax.plot(x, BLIP2, marker='o', label='BLIP2', color=colors[5])
    ax.plot(x, BLIP2_CIR, marker='o', label='BLIP2-CIR', color=colors[6])
    ax.plot(x, instructBLIP, marker='o', label='instructBLIP', color=colors[7])

    ax.plot(x, IMAGE_CLIP, marker='o', label='IMAGE-CLIP', color=colors[8])
    ax.plot(x, TEXT_CLIP, marker='o', label='TEXT-CLIP',   color=colors[9])
    ax.plot(x, IMAGE_RESNET50, marker='o', label='IMAGE-RESNET50', color=np.array([0.19215686, 0.50980392, 0.74117647, 1.        ]))


    ax.errorbar(x, TIRG, yerr=TIRG_std, fmt='o', capsize=4,color=colors[0],ecolor=colors[0])
    ax.errorbar(x, MAAF, yerr=MAAF_std, fmt='o', capsize=4,color=colors[1],ecolor=colors[1])
    ax.errorbar(x, ARTEMIS, yerr=ARTEMIS_std, fmt='o', capsize=4,color=colors[2],ecolor=colors[2])
    ax.errorbar(x, CIRPLANT, yerr=CIRPLANT_std, fmt='o', capsize=4,color=colors[3],ecolor=colors[3])
    ax.errorbar(x, CLIP4CIR, yerr=CLIP4CIR_std, fmt='o', capsize=4, color=colors[4], ecolor=colors[4])
    ax.errorbar(x, BLIP2, yerr=BLIP2_std, fmt='o', capsize=4, color=colors[5], ecolor=colors[5])
    ax.errorbar(x, BLIP2_CIR, yerr=BLIP2_CIR_std, fmt='o', capsize=4, color=colors[6], ecolor=colors[6])
    ax.errorbar(x, instructBLIP, yerr=instructBLIP_std, fmt='o', capsize=4, color=colors[7], ecolor=colors[7])


    ax.errorbar(x, IMAGE_CLIP, yerr=IMAGE_CLIP_std, fmt='o', capsize=4, color=colors[8], ecolor=colors[8])
    ax.errorbar(x, TEXT_CLIP, yerr=TEXT_CLIP_std, fmt='o', capsize=4, color=colors[9], ecolor=colors[9])
    ax.errorbar(x, IMAGE_RESNET50, yerr=IMAGE_RESNET50_std, fmt='o', capsize=4, color=np.array([0.19215686, 0.50980392, 0.74117647, 1.]), ecolor=np.array([0.19215686, 0.50980392, 0.74117647, 1.]))
    

    # 创建插图对象
    # ax_zoom = ax.inset_axes((0.5, 0.1, 0.45, 0.3))  # tune position of the sub figure


    # 设置局部放大的区域
    # x_zoom = [9, 10, 11]
    # y_zoom = [21, 22, 23]
    # error_zoom = [0.4, 0.3, 0.2]  # 局部放大区域的误差值

    # # 绘制局部放大的折线图
    # ax_zoom.plot(x_zoom, y_zoom, marker='o', label='Zoomed Line')
    # ax_zoom.errorbar(x_zoom, y_zoom, yerr=error_zoom, fmt='o', capsize=4, label='Zoomed Error')

    # # 设置局部放大区域的坐标范围
    # ax_zoom.set_xlim(9, 11)
    # ax_zoom.set_ylim(20, 25)

    # # 设置插图的边框样式
    # ax_zoom.spines['top'].set_visible(False)
    # ax_zoom.spines['right'].set_visible(False)
    # ax_zoom.spines['bottom'].set_visible(False)
    # ax_zoom.spines['left'].set_visible(False)

    # # 添加网格线
    # ax_zoom.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.set_xlabel('Rank K',fontsize=16)
    ax.set_ylabel('Recall@K',fontsize=16)
    # ax.set_title('Line Plot with Zoomed Area')
    ax.legend(fontsize=6)
    plt.savefig('./paper_images/recall_rank_blips.png',dpi=300)

    plt.show()


def relativerobustness_wrank():
    # AVERAGE PERFORMANCE AND STD
    x = np.array([1, 5, 10, 50])
    TIRG=np.array([0.38,0.41,0.44,0.56])
    MAAF=np.array([0.5,0.49,0.52,0.61])
    ARTEMIS=np.array([0.4,0.43,0.47,0.59])
    CIRPLANT=np.array([0.57,0.6,0.63,0.75])
    CLIP4CIR=np.array([0.55,0.62,0.65,0.78])
    BLIP2 = np.array([1.03,0.97,0.96,0.95])
    BLIP2_CIR = np.array([0.88,0.9,0.9,0.94])
    instructBLIP = np.array([0.82,0.82,0.84,0.9])

    IMAGE_CLIP=np.array([0.6,0.6,0.6,0.65])
    TEXT_CLIP=np.array([0.7,0.71,0.72,0.79])
    IMAGE_RESNET50=np.array([0.6,0.56,0.56,0.64])

    TIRG_std=np.array([0.2,0.2,0.21,0.19])
    MAAF_std=np.array([0.21,0.22,0.21,0.19])
    ARTEMIS_std=np.array([0.2,0.2,0.2,0.19])
    CIRPLANT_std=np.array([0.22,0.22,0.22,0.19])
    CLIP4CIR_std=np.array([0.25,0.25,0.23,0.19])
    BLIP2_std = np.array([0.18,0.13,0.13,0.09])
    BLIP2_CIR_std = np.array([0.18,0.17,0.15,0.1])
    instructBLIP_std = np.array([0.18,0.18,0.18,0.16])

    IMAGE_CLIP_std=np.array([0.18,0.2,0.2,0.18])
    TEXT_CLIP_std=np.array([0.28,0.26,0.25,0.21])

    IMAGE_RESNET50_std=np.array([0.18,0.2,0.2,0.19])

    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # 创建主图对象
    fig, ax = plt.subplots()

    # 绘制原图
    ax.plot(x, TIRG, marker='o', label='TIRG',color=colors[0])
    ax.plot(x, MAAF, marker='o', label='MAAF',color=colors[1])
    ax.plot(x, ARTEMIS, marker='o', label='ARTEMIS',color=colors[2])
    ax.plot(x, CIRPLANT, marker='o', label='CIRPLANT', color=colors[3])
    ax.plot(x, CLIP4CIR, marker='o', label='CLIP4CIR', color=colors[4])
    ax.plot(x, BLIP2, marker='o', label='BLIP2', color=colors[5])
    ax.plot(x, BLIP2_CIR, marker='o', label='BLIP2-CIR', color=colors[6])
    ax.plot(x, instructBLIP, marker='o', label='instructBLIP', color=colors[7])

    ax.plot(x, IMAGE_CLIP, marker='o', label='IMAGE-CLIP', color=colors[8])
    ax.plot(x, TEXT_CLIP, marker='o', label='TEXT-CLIP',   color=colors[9])
    ax.plot(x, IMAGE_RESNET50, marker='o', label='IMAGE-RESNET50',  color=np.array([0.19215686, 0.50980392, 0.74117647, 1.        ]))


    ax.errorbar(x, TIRG, yerr=TIRG_std, fmt='o', capsize=4,color=colors[0],ecolor=colors[0])
    ax.errorbar(x, MAAF, yerr=MAAF_std, fmt='o', capsize=4,color=colors[1],ecolor=colors[1])
    ax.errorbar(x, ARTEMIS, yerr=ARTEMIS_std, fmt='o', capsize=4,color=colors[2],ecolor=colors[2])
    ax.errorbar(x, CIRPLANT, yerr=CIRPLANT_std, fmt='o', capsize=4,color=colors[3],ecolor=colors[3])
    ax.errorbar(x, CLIP4CIR, yerr=CLIP4CIR_std, fmt='o', capsize=4, color=colors[4], ecolor=colors[4])
    ax.errorbar(x, BLIP2, yerr=BLIP2_std, fmt='o', capsize=4, color=colors[5], ecolor=colors[5])
    ax.errorbar(x, BLIP2_CIR, yerr=BLIP2_CIR_std, fmt='o', capsize=4, color=colors[6], ecolor=colors[6])
    ax.errorbar(x, instructBLIP, yerr=instructBLIP_std, fmt='o', capsize=4, color=colors[7], ecolor=colors[7])

    ax.errorbar(x, IMAGE_CLIP, yerr=IMAGE_CLIP_std, fmt='o', capsize=4, color=colors[8], ecolor=colors[8])
    ax.errorbar(x, TEXT_CLIP, yerr=TEXT_CLIP_std, fmt='o', capsize=4, color=colors[9], ecolor=colors[9])
    ax.errorbar(x, IMAGE_RESNET50, yerr=IMAGE_RESNET50_std, fmt='o', capsize=4, color=np.array([0.19215686, 0.50980392, 0.74117647, 1.        ]), ecolor=np.array([0.19215686, 0.50980392, 0.74117647, 1.        ]))
    

    # 创建插图对象
    # ax_zoom = ax.inset_axes((0.5, 0.1, 0.45, 0.3))  # tune position of the sub figure


    # # 设置局部放大的区域
    # x_zoom = [3, 4, 5]
    # y_zoom = [3, 4, 5]
    # error_zoom = [0.4, 0.3, 0.2]  # 局部放大区域的误差值

    # # 绘制局部放大的折线图
    # ax_zoom.plot(x_zoom, y_zoom, marker='o', label='Zoomed Line')
    # ax_zoom.errorbar(x_zoom, y_zoom, yerr=error_zoom, fmt='o', capsize=4, label='Zoomed Error')

    # # 设置局部放大区域的坐标范围
    # ax_zoom.set_xlim(3, 5)
    # ax_zoom.set_ylim(2, 6)

    # # 设置插图的边框样式
    # ax_zoom.spines['top'].set_visible(False)
    # ax_zoom.spines['right'].set_visible(False)
    # ax_zoom.spines['bottom'].set_visible(False)
    # ax_zoom.spines['left'].set_visible(False)

    # # 添加网格线
    # ax_zoom.grid(True, linestyle='--', alpha=0.5)

    ax.xaxis.label.set_fontsize(16)
    ax.yaxis.label.set_fontsize(16)
    ax.tick_params(axis='both', labelsize=16)

    # ax.set_ylabel('Recall@5',fontsize=14)    
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    ax.set_xlabel('Rank K',fontsize=16)
    ax.set_ylabel('Relative robustness@K',fontsize=16)
    # ax.set_title('Line Plot with Zoomed Area')
    ax.legend()
    plt.savefig('./paper_images/relativerobustness_rank_blips.png',dpi=300)

    plt.show()

def model_size():
    import seaborn as sns
    import matplotlib.pyplot as plt

    # 创建示例数据
    x = [24.4, 25.81, 27.47, 43.33, 52.25,51.86,57.03,27.31 ] #,28.45,37.03]   # R10 in cirr average corrupted recall 
    x_clean = [55.05,49.87,59.02,68.77,80.33 ]# ,50.44,51.22] # R10 in cirr clean recall
    y = [0.44, 0.52, 0.47, 0.63, 0.65,0.96,0.90,0.84]# 0.56,0.72]  # relative robustness
    y_clean = [1,1,1,1,1]#,1,1]

    model_size = [30.7,34.6,29.9,155.5,237.3,4917.3,1227.7,7913.2]# 25.6, 146.2] 4917279934  30721947 tirg;1227671008 BLIP2-CIR; 7913155712 instructblip  
    model_size = [item for item in model_size]
    categories = ['TIRG', 'MAAF', 'ARTEMIS', 'CIRPLANT', 'CLIP4CIR','BLIP2','BLIP2-CIR','instructBLIP']#,'IMAGE-RESNET50','TEXT-ONLY']  # 不同点的类别

    # 绘制散点图
    fig, ax = plt.subplots()
    colors = np.random.rand(8)  # 指定每个点的颜色
    print(colors)
    scatter = ax.scatter(x, y, s=model_size,c=colors, cmap='viridis')
    for i, label in enumerate(categories):
        ax.annotate(label, (x[i], y[i]), textcoords="offset points", xytext=(0, 10), ha='center')
    
    for idx,text in enumerate(ax.texts):
        if idx in [0,1,2,3,4]:
            text.set_y(text.get_position()[1] +0)
        elif idx in [6]:
            text.set_y(text.get_position()[1] +10)
        elif idx in [7]:
            text.set_y(text.get_position()[1] +35)
        else:
            text.set_y(text.get_position()[1] +27)

    ax.set_xticks(np.arange(20, 65, 10),fontsize=16)
    ax.set_yticks(np.arange(0.4, 1.12, 0.1),fontsize=16)

    ax.set_xlabel('Average Recall@10',fontsize=16)
    ax.set_ylabel('Averave relative robustness',fontsize=16)

    # ax.set_ylabel('Recall@5',fontsize=14)    
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)

    
    # sns.scatterplot(x=x, y=y, hue=categories, palette='tab10', size="size")
    
    # sns.scatterplot(x=x_clean, y=y_clean, hue=categories, palette='tab10')

    # 显示图形
    import matplotlib.pyplot as plt
    plt.savefig('./paper_images/model_size_blips.png',dpi=300)
    plt.show()

def demooo():
    import seaborn as sns
    import matplotlib.pyplot as plt

    # 创建示例数据
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    sizes = [20, 40, 60, 80, 100]  # 点的大小数组

    # 创建绘图对象
    fig, ax = plt.subplots()

    # 使用Matplotlib的scatter函数绘制散点图
    scatter = ax.scatter(x, y, s=sizes)

    # 使用Seaborn设置绘图风格
    sns.set()

    # 添加图例
    legend1 = ax.legend(*scatter.legend_elements(), title="Sizes")
    ax.add_artist(legend1)
    plt.savefig('./paper_images/demooo.png')
    # 显示图形
    plt.show()


def histgram_number():
    categories = ['CIRR','Number1-10', 'Number1-3', 'Number4-10']
    # TIRG = [CIRR_rank1_waverageimagecorruption, sub_rank1_w_average_cprrupti]
    # for image corruption
    TIRG = [36.35, 39.64,	38.85	,36.13]
    MAAF= [32.19,  32.53,	31.57	,27.73]
    ARTEMIS = [40.05, 39.56,	37.81	,41.18]
    CIRPLANT = [48.82,  45.07,	43.95,	42.86]
    CLIP4CIR = [62.94, 64.18	,63.14,	63.87]

    # for text corruption
    # TIRG_text = [0.5,0.85]
    # MAAF_text = [0.97,0.97]
    # ARTEMIS_text = [0.78,0.89]
    # CIRPLANT_text = [0.94,0.96]
    # CLIP4CIR_text = [0.89,0.94]

    # TIRG = TIRG+TIRG_text
    # MAAF = MAAF+MAAF_text
    # ARTEMIS = ARTEMIS+ARTEMIS_text
    # CIRPLANT = CIRPLANT+CIRPLANT_text
    # CLIP4CIR = CLIP4CIR+CLIP4CIR_text
    # categories = categories + ['CIRR_text', 'CIRR_sub_text']

    # 组合数据
    data = {
        'Categories': categories * 5,
        'Values': TIRG + MAAF + ARTEMIS + CIRPLANT + CLIP4CIR,
        'Group':['TIRG'] * len(categories) + ['MAAF'] * len(categories) + ['ARTEMIS'] * len(categories) + ['CIRPLANT'] * len(categories) + ['CLIP4CIR'] * len(categories)
    }

    # 创建画布和子图
    fig, ax = plt.subplots()

    # 使用 Seaborn 绘制柱状图
    sns.barplot(x='Categories', y='Values', hue='Group', data=data, ax=ax)
    # plt.ylim(0.4, 1)
    for p in ax.patches:
    # 获取柱子的高度
        height = p.get_height()
    # 在柱子的中心位置添加数值

        ax.annotate(round(height,1), (p.get_x() + p.get_width() / 2, height), ha='center', va='bottom',fontsize=12) # keep only the integer
    # 设置图例
    ax.legend()

    # 设置图表标题和轴标签
    # ax.set_title('Average relative robustness for image corruptions on CIRR and CIRR_sub')
    # ax.set_xlabel('Categories')
    ax.set_ylabel('Recall@5',fontsize=14)    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # 调整图表布局
    plt.tight_layout()

    plt.savefig('./paper_images/histgram_number.png',dpi=300)

    # 显示图表
    plt.show()


def relativerobustness_wrank_subCIRR():
    # AVERAGE PERFORMANCE AND STD
    x = np.array([1, 2, 3])
    TIRG=np.array([0.65,0.62,0.59])
    MAAF=np.array([0.77,0.71,0.66])
    ARTEMIS=np.array([0.68,0.64,0.62])
    CIRPLANT=np.array([0.78,0.79,0.78])
    CLIP4CIR=np.array([0.74,0.76,0.78])
    IMAGE_CLIP=np.array([0.8,0.76,0.72])
    IMAGE_RESNET50=np.array([0.87,0.79,0.73])
    TEXT_CLIP=np.array([0.77,0.79,0.79])


    TIRG_std=np.array([0.17,0.18,0.19])
    MAAF_std=np.array([0.16,0.17,0.19])
    ARTEMIS_std=np.array([0.17,0.18,0.18])
    CIRPLANT_std=np.array([0.16,0.17,0.18])
    CLIP4CIR_std=np.array([0.18,0.18,0.18])
    IMAGE_CLIP_std=np.array([0.11,0.15,0.17])
    IMAGE_RESNET50_std=np.array([0.1,0.14,0.17])
    TEXT_CLIP_std=np.array([0.21,0.21,0.21])


    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # 创建主图对象
    fig, ax = plt.subplots()

    # 绘制原图
    ax.plot(x, TIRG, marker='o', label='TIRG',color=colors[0])
    ax.plot(x, MAAF, marker='o', label='MAAF',color=colors[1])
    ax.plot(x, ARTEMIS, marker='o', label='ARTEMIS',color=colors[2])
    ax.plot(x, CIRPLANT, marker='o', label='CIRPLANT', color=colors[3])
    ax.plot(x, CLIP4CIR, marker='o', label='CLIP4CIR', color=colors[4])
    ax.plot(x, IMAGE_CLIP, marker='o', label='IMAGE-CLIP', color=colors[5])
    ax.plot(x, TEXT_CLIP, marker='o', label='TEXT-CLIP',   color=colors[6])
    ax.plot(x, IMAGE_RESNET50, marker='o', label='IMAGE-RESNET50', color=colors[7])


    ax.errorbar(x, TIRG, yerr=TIRG_std, fmt='o', capsize=4,color=colors[0],ecolor=colors[0])
    ax.errorbar(x, MAAF, yerr=MAAF_std, fmt='o', capsize=4,color=colors[1],ecolor=colors[1])
    ax.errorbar(x, ARTEMIS, yerr=ARTEMIS_std, fmt='o', capsize=4,color=colors[2],ecolor=colors[2])
    ax.errorbar(x, CIRPLANT, yerr=CIRPLANT_std, fmt='o', capsize=4,color=colors[3],ecolor=colors[3])
    ax.errorbar(x, CLIP4CIR, yerr=CLIP4CIR_std, fmt='o', capsize=4, color=colors[4], ecolor=colors[4])
    ax.errorbar(x, IMAGE_CLIP, yerr=IMAGE_CLIP_std, fmt='o', capsize=4, color=colors[5], ecolor=colors[5])
    ax.errorbar(x, TEXT_CLIP, yerr=TEXT_CLIP_std, fmt='o', capsize=4, color=colors[6], ecolor=colors[6])
    ax.errorbar(x, IMAGE_RESNET50, yerr=IMAGE_RESNET50_std, fmt='o', capsize=4, color=colors[7], ecolor=colors[7])
    

    # 创建插图对象
    # ax_zoom = ax.inset_axes((0.5, 0.1, 0.45, 0.3))  # tune position of the sub figure


    # # 设置局部放大的区域
    # x_zoom = [3, 4, 5]
    # y_zoom = [3, 4, 5]
    # error_zoom = [0.4, 0.3, 0.2]  # 局部放大区域的误差值

    # # 绘制局部放大的折线图
    # ax_zoom.plot(x_zoom, y_zoom, marker='o', label='Zoomed Line')
    # ax_zoom.errorbar(x_zoom, y_zoom, yerr=error_zoom, fmt='o', capsize=4, label='Zoomed Error')

    # # 设置局部放大区域的坐标范围
    # ax_zoom.set_xlim(3, 5)
    # ax_zoom.set_ylim(2, 6)

    # # 设置插图的边框样式
    # ax_zoom.spines['top'].set_visible(False)
    # ax_zoom.spines['right'].set_visible(False)
    # ax_zoom.spines['bottom'].set_visible(False)
    # ax_zoom.spines['left'].set_visible(False)

    # # 添加网格线
    # ax_zoom.grid(True, linestyle='--', alpha=0.5)

    ax.xaxis.label.set_fontsize(14)
    ax.yaxis.label.set_fontsize(14)
    ax.tick_params(axis='both', labelsize=14)

    # ax.set_ylabel('Recall@5',fontsize=14)    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    ax.set_xlabel('Rank K',fontsize=14)
    ax.set_ylabel('Relative robustness@K',fontsize=14)
    # ax.set_title('Line Plot with Zoomed Area')
    ax.legend(fontsize="small")
    plt.savefig('./paper_images/relativerobustness_rank_cirrsub.png',dpi=300)

    plt.show()



def recall_wrank_subCIRR():
    # AVERAGE PERFORMANCE AND STD
    x = np.array([1, 2, 3])
    TIRG=np.array([23.86,35.89,42.3])
    MAAF=np.array([22.36,34.99,42.49])
    ARTEMIS=np.array([28,40.42,46.7])
    CIRPLANT=np.array([30.04,48.22,59.52])
    CLIP4CIR=np.array([46.83,62.25,69.27])
    IMAGE_CLIP=np.array([16.39,27.51,34.61])
    IMAGE_RESNET50=np.array([17.7,31.39,41.46])
    TEXT_CLIP=np.array([39.84,51.46,56.22])


    TIRG_std=np.array([6.11,10.34,13.6])
    MAAF_std=np.array([4.66,8.63,11.91])
    ARTEMIS_std=np.array([6.88,11.1,13.77])
    CIRPLANT_std=np.array([6.05,10.34,13.67])
    CLIP4CIR_std=np.array([11.57,14.86,16.49])
    IMAGE_CLIP_std=np.array([2.24,5.31,8.08])
    IMAGE_RESNET50_std=np.array([2.13,5.59,9.47])
    TEXT_CLIP_std=np.array([10.65,13.37,14.72])


    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # 创建主图对象
    fig, ax = plt.subplots()

    # 绘制原图
    ax.plot(x, TIRG, marker='o', label='TIRG',color=colors[0])
    ax.plot(x, MAAF, marker='o', label='MAAF',color=colors[1])
    ax.plot(x, ARTEMIS, marker='o', label='ARTEMIS',color=colors[2])
    ax.plot(x, CIRPLANT, marker='o', label='CIRPLANT', color=colors[3])
    ax.plot(x, CLIP4CIR, marker='o', label='CLIP4CIR', color=colors[4])
    ax.plot(x, IMAGE_CLIP, marker='o', label='IMAGE-CLIP', color=colors[5])
    ax.plot(x, TEXT_CLIP, marker='o', label='TEXT-CLIP',   color=colors[6])
    ax.plot(x, IMAGE_RESNET50, marker='o', label='IMAGE-RESNET50', color=colors[7])


    ax.errorbar(x, TIRG, yerr=TIRG_std, fmt='o', capsize=4,color=colors[0],ecolor=colors[0])
    ax.errorbar(x, MAAF, yerr=MAAF_std, fmt='o', capsize=4,color=colors[1],ecolor=colors[1])
    ax.errorbar(x, ARTEMIS, yerr=ARTEMIS_std, fmt='o', capsize=4,color=colors[2],ecolor=colors[2])
    ax.errorbar(x, CIRPLANT, yerr=CIRPLANT_std, fmt='o', capsize=4,color=colors[3],ecolor=colors[3])
    ax.errorbar(x, CLIP4CIR, yerr=CLIP4CIR_std, fmt='o', capsize=4, color=colors[4], ecolor=colors[4])
    ax.errorbar(x, IMAGE_CLIP, yerr=IMAGE_CLIP_std, fmt='o', capsize=4, color=colors[5], ecolor=colors[5])
    ax.errorbar(x, TEXT_CLIP, yerr=TEXT_CLIP_std, fmt='o', capsize=4, color=colors[6], ecolor=colors[6])
    ax.errorbar(x, IMAGE_RESNET50, yerr=IMAGE_RESNET50_std, fmt='o', capsize=4, color=colors[7], ecolor=colors[7])
    

    # 创建插图对象
    # ax_zoom = ax.inset_axes((0.5, 0.1, 0.45, 0.3))  # tune position of the sub figure


    # # 设置局部放大的区域
    # x_zoom = [3, 4, 5]
    # y_zoom = [3, 4, 5]
    # error_zoom = [0.4, 0.3, 0.2]  # 局部放大区域的误差值

    # # 绘制局部放大的折线图
    # ax_zoom.plot(x_zoom, y_zoom, marker='o', label='Zoomed Line')
    # ax_zoom.errorbar(x_zoom, y_zoom, yerr=error_zoom, fmt='o', capsize=4, label='Zoomed Error')

    # # 设置局部放大区域的坐标范围
    # ax_zoom.set_xlim(3, 5)
    # ax_zoom.set_ylim(2, 6)

    # # 设置插图的边框样式
    # ax_zoom.spines['top'].set_visible(False)
    # ax_zoom.spines['right'].set_visible(False)
    # ax_zoom.spines['bottom'].set_visible(False)
    # ax_zoom.spines['left'].set_visible(False)

    # # 添加网格线
    # ax_zoom.grid(True, linestyle='--', alpha=0.5)

    ax.xaxis.label.set_fontsize(14)
    ax.yaxis.label.set_fontsize(14)
    ax.tick_params(axis='both', labelsize=14)

    # ax.set_ylabel('Recall@5',fontsize=14)    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    ax.set_xlabel('Rank K',fontsize=14)
    ax.set_ylabel('Recall@K',fontsize=14)
    # ax.set_title('Line Plot with Zoomed Area')
    ax.legend()
    plt.savefig('./paper_images/recall_rank_cirrsub.png',dpi=300)

    plt.show()



if __name__ == '__main__':
    # recall_wrank()
    # histgram_number()
    # recall_wrank_subCIRR()
    # recall_wrank()
    
    # demooo()
    # relativerobustness_wrank()
    # line_accurary_rankK()
    model_size()
    # recall_wrank()
