import pandas as pd
import csv
import os
import numpy as np
import random
import time


'''
解析IP，将IP转换成整数，返回列表
'''
def ProcessIP(ipstr):
    res = []
    ips = ipstr.split(',')
    for ip in ips:
        if ip[0] == " ":
            ip = ip[2:-1]
        else:
            ip = ip[1:-1]
        value = 0
        try:
            for j, i in enumerate(ip.split('.')[::-1]):
                value += 256 ** j * int(i)
            res.append(value)
        except ValueError:
            continue
    return res

def FileStandard(ROOT_PATH, feature_file, standard_file):
    if os.path.isfile(ROOT_PATH + standard_file):
        print("reuslt file ", ROOT_PATH+standard_file, " has been exist, exit")
        return
    data = pd.read_csv(ROOT_PATH+feature_file, sep=",", engine='python')  
    columns = ["domain","label","rescount","interval","ipcount","ipvar"] 
    for i in range(2, 6):
        maxv = data.iloc[:, i].max() 
        minv = data.iloc[:, i].min()
        if maxv == minv:
            data.iloc[:, i] = 0
        else:
            data.iloc[:, i] = (data.iloc[:, i] - minv)/(maxv - minv)
    
    # print(data.iloc[:, 3])
    data.to_csv(ROOT_PATH+standard_file, mode='a', header=columns, index=False)

'''
生成Flux域名列表
返回值：List
'''
def GenerateLabel(ROOT_PATH, label_file):
    data2 = pd.read_csv(ROOT_PATH+label_file, sep=",", engine='python', iterator=True)
    chunk_size = 1000
    labels = []
    loop = True
    while loop:
        try:
            label = data2.get_chunk(chunk_size)
            for i in range(label.shape[0]):
                if label.iloc[i, 1] == 1:
                    labels.append(label.iloc[i, 0])
        except StopIteration:
            loop = False
            print("Stop Iteration, GenerateLabel")
    # print(labels)
    return labels
'''
生成标准化后的，可输入训练模型中问数据文件
训练集和测试集公用
主要工作：数据标准化
'''
def OriginObjFile(ROOT_PATH, feature_file, result_file):
    if os.path.isfile(ROOT_PATH + result_file):
        print("reuslt file ", ROOT_PATH+result_file, " has been exist, exit")
        return

    data = pd.read_csv(ROOT_PATH+feature_file, sep=",", engine='python', iterator=True)
    chunk_size = 1000
    loop = True
    last_domain = ""
    last_label = 0
    count = 0
    sums = [0] * 4
    with_header = True
    columns = ["domain","label","rescount","interval","ipcount","ipvar"]
    while loop:
        try:
            # ["domain","label","rescount","interval","ipcount","ipvar"]
            feature = data.get_chunk(chunk_size)
            feature_sums = []
            for i in range(feature.shape[0]):
                if last_domain != "" and last_domain != feature.iloc[i, 0]:
                    current = [last_domain, last_label, 0, 0, 0, 0]
                    current[2:] = [x/count for x in sums]
                    feature_sums.append(current)
                    # print(current)
                    count = 0
                    sums = [0] * 4
                count += 1
                sums[0] += feature.iloc[i, 2]
                sums[1] += feature.iloc[i, 3]
                sums[2] += feature.iloc[i, 4]
                sums[3] += feature.iloc[i, 5]

                last_domain = feature.iloc[i, 0]
                last_label = feature.iloc[i, 1]

            res = pd.DataFrame(data=feature_sums, index=None)
            if res.shape[0] == 0:
                continue

            if with_header:
                with_header = False
                res.to_csv(ROOT_PATH+result_file, mode='a', header=columns, index=False)
            else:
                res.to_csv(ROOT_PATH+result_file, mode='a', header=False, index=False)
        except StopIteration:
            loop = False
            final = [last_domain, last_label, 0, 0, 0, 0]
            final[2:] = [x/count for x in sums]
            res = pd.DataFrame(data=[final], index=None)
            res.to_csv(ROOT_PATH+result_file, mode='a', header=False, index=False)
            print("Stop Iteration, OriginObjFile")

'''
链接文件，主要将训练集中的数据打上标签
'''
def ConcatFile(ROOT_PATH, data_file, label_file, result_file):
    if os.path.isfile(ROOT_PATH + result_file):
        print("reuslt file ", ROOT_PATH+result_file, " has been exist, exit")
        return
    # 获取标签
    labels = GenerateLabel(ROOT_PATH, label_file)

    # 连接文件
    data1 = pd.read_csv(ROOT_PATH+data_file, sep=",", engine='python', iterator=True)
    chunk_size = 1000
    loop = True
    with_header = True
    columns = ["count","time_first","time_last","rrname","rrtype","bailiwick","rdata", "label"]
    while loop:
        try:
            # count,time_first,time_last,rrname,rrtype,bailiwick,rdata
            pdns = data1.get_chunk(chunk_size)
            insert_labels = []
            for i in range(pdns.shape[0]):
                if pdns.iloc[i, 3] in labels:
                    insert_labels.append(1)
                else:
                    insert_labels.append(0)
            pdns['label'] = insert_labels
            if with_header:
                with_header = False
                pdns.to_csv(ROOT_PATH+result_file, mode='a', header=columns, index=False)
            else:
                pdns.to_csv(ROOT_PATH+result_file, mode='a', header=False, index=False)
        except StopIteration:
            loop = False
            print("Stop Iteration, ConcatFile")

'''
提取数据集中的有效数据
例如ip个数，时间之类的，不知道域名字符串有没有作用
确定的特征：
域名domain、标签label、解析次数rescount、间隔时间interval、解析的IP个数ipcount、解析出的IP网段变化幅度ipvar
'''
def ExtractData(ROOT_PATH, origin_label_file, ip_info_file, result_file, is_test):
    if os.path.isfile(ROOT_PATH + result_file):
        print("reuslt file ", ROOT_PATH+result_file, " has been exist, exit")
        return

    data = pd.read_csv(ROOT_PATH+origin_label_file, sep=",", engine='python', iterator=True)
    chunk_size = 1000
    loop = True
    columns = ["domain","label","rescount","interval","ipcount","ipvar"]
    with_header = True
    while loop:
        try:
            data_tmp = data.get_chunk(chunk_size)
            domains = []
            for i in range(data_tmp.shape[0]):
                current = []
                #domian
                current.append(data_tmp.iloc[i, 3]) 
                # label
                if is_test:
                    current.append(0)
                else:
                    current.append(data_tmp.iloc[i, 7]) 
                # rescount
                current.append(data_tmp.iloc[i, 0]) 

                # interval
                time_first = data_tmp.iloc[i, 1]
                time_last = data_tmp.iloc[i, 2]
                current.append(time_last - time_first) 

                # ipcount
                ipstr = data_tmp.iloc[i, 6]
                ips = ProcessIP(ipstr[1:-1])
                current.append(len(ips)) 

                # ipvar
                if len(ips) == 0:
                    current.append(0)
                else:
                    current.append(np.std(ips)) 

                domains.append(current) 

            res = pd.DataFrame(data=domains, index=None)
            if with_header:
                with_header = False
                res.to_csv(ROOT_PATH+result_file, mode='a', header=columns, index=False)
            else:
                res.to_csv(ROOT_PATH+result_file, mode='a', header=False, index=False)
        except StopIteration:
            loop = False
            print("Stop Iteration, ExtractData")

'''
数据采样，进行平衡
返回数据训练集与测试集
'''
def SampleBalance(ROOT_PATH=r"data\fastflux_dataset", 
                feature_file = r"\feature_standard.csv"):
    random.seed(time.time())
    data = pd.read_csv(ROOT_PATH+feature_file, sep=",", engine='python')
    key = data.keys()[0:]
    print(key)
    result = []
    count = [0, 0]
    for i in range(data.shape[0]):
        if data.label[i] == 1:
            result.append(data.iloc[i, :])
            count[0] += 1
            continue
        choose = random.uniform(0, 1)
        if choose < 0.35:
            count[1] += 1
            result.append(data.iloc[i, :])
    print("Finish sample, data count [flux: %d, no flux %d]" % (count[0], count[1]))
    return result

'''
产生带标签的数据
'''
def OriginLableFile(ROOT_PATH, test_file, label_file, final_file):
    labels = GenerateLabel(ROOT_PATH, label_file)

    data1 = pd.read_csv(ROOT_PATH+test_file, sep=",", engine='python', iterator=True)
    chunk_size = 1000
    loop = True
    last_domain = ""
    while loop:
        try:
            # count,time_first,time_last,rrname,rrtype,bailiwick,rdata
            pdns = data1.get_chunk(chunk_size)
            domains = []
            for i in range(pdns.shape[0]):
                if last_domain != "" and last_domain == pdns.iloc[i, 3]:
                    continue
                else:
                    domains.append([pdns.iloc[i, 3], 0])
                    if pdns.iloc[i, 3] in labels:
                        domains[-1][1] = 1
                    last_domain = domains[-1][0]
            res = pd.DataFrame(data=domains, index=None)
            res.to_csv(ROOT_PATH+final_file, mode='a', header=False, index=False)
        except StopIteration:
            loop = False
            print("Stop Iteration, OriginLableFile")


if __name__ == '__main__':
    ROOT_PATH = r"data\fastflux_dataset"

    train_file1 = r"\train\pdns.csv"
    train_file2 = r"\train\fastflux_tag.csv"
    result_file = r"\origin_label.csv"
    test_pdns = r"\test\pdns.csv"
    final_file = r"\final.csv"
    feature_file = r"\feature.csv"
    # feature_file = r"\demo.csv"
    feature_summary = r"\feature_summary.csv"
    # feature_summary = r"\demo_summary.csv"
    standard_file = r"\feature_standard.csv"
    # train_file2 = r"data\fastflux_dataset\demo.csv"
    ConcatFile(ROOT_PATH, train_file1, train_file2, result_file)
    # OriginLableFile(ROOT_PATH, test_pdns, train_file2, final_file)
    ExtractData(ROOT_PATH, result_file, None, feature_file, 0)

    OriginObjFile(ROOT_PATH, feature_file, feature_summary)
    FileStandard(ROOT_PATH, feature_summary, standard_file)
    SampleBalance(ROOT_PATH, standard_file)
