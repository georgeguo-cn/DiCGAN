import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def line_chart(filename, out_name, out_form):
    cfgan = []
    with open('CFGAN_log_' + filename + '.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            try:
                datas = line[:-1].split(',')
                if datas[1].strip() == 'top10':
                    cfgan.append(float(datas[2].split(':')[1]))
            except:
                pass
    dicgan = []
    with open('DiCGAN_log_' + filename + '.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            try:
                datas = line[:-1].split(',')
                if datas[1].strip() == 'top10':
                    dicgan.append(float(datas[2].split(':')[1]))
            except:
                pass

    lens = max(len(cfgan), len(dicgan))
    names = [str(x) for x in list(range(lens))]

    x = range(0, len(names))
    print(x)
    print(names)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(range(len(cfgan)), cfgan, label='CFGAN')
    plt.plot(range(len(dicgan)), dicgan, label='DiCGAN')
    font1 = {'weight': 'normal', 'size': 16}
    plt.legend(loc=4, prop=font1)  # 让图例生效
    ax.grid(ls='--', axis='y')

    plt.xticks(x, names)
    plt.yticks([0.00, 0.05, 0.10, 0.15, 0.20, 0.25])
    ax.tick_params(labelsize=16)

    # plt.margins(0, 0.1)
    plt.subplots_adjust(bottom=0.10)
    font2 = {'weight': 'bold', 'size': 18}
    plt.xlabel('epoch', fontdict=font2)  # X轴标签
    plt.ylabel("Precision@5", fontdict=font2)  # Y轴标签

    for label in ax.get_xticklabels():
        label.set_visible(False)
    for label in ax.get_xticklabels()[::50]:
        label.set_visible(True)

    ax.spines['bottom'].set_linewidth(2)  # 设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2)  # 设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2)  # 设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)  # 设置顶部坐标轴的粗细
    if out_form == 'pdf':
        out_pdf = PdfPages(out_name + '.pdf')
        out_pdf.savefig(bbox_inches='tight')
        out_pdf.close()
    else:
        plt.savefig(out_name + '.png', dpi=900)
    plt.show()

if __name__ == '__main__':
    filename = 'ABaby'
    line_chart(filename, filename + '-trend', 'pdf') # Ciao,Ababy