import csv
import sys
import re
import pickle
from Dict import Dict

# for line in diff['a']:
#     # if len(diff['a'][line]['code'].split()) > 3:
#     remove_code = diff['a'][line]['code'].strip()
#     remove_code = ' '.join(split_sentence(remove_code).split())
#     remove_code = ' '.join(remove_code.split(' '))
#     removed_code.append(remove_code)
#     for word in remove_code.split():
#         code_dict.add(word)
#     # remove_code = 'removed _ code'
#     file_codes.append((line, remove_code))
#     if len(removed_code) > 10: break
                
                


csv.field_size_limit(1000000000)
raw_diff_path = "C:\\Zhijie\\fyp\\defects4j\\result\\raw_diff\\raw_diff_exception_type_exception_only.csv"
dic_path = "C:\\Zhijie\\fyp\\fyp\\transformer\\word_dic_exception_type.pkl"
label_dic_path = "C:\\Zhijie\\fyp\\fyp\\transformer\\label_dic_exception_type.pkl"

def split_sentence(sentence):
    sentence = sentence.replace('.', ' . ').replace('_', ' ').replace('@', ' @ ')\
        .replace('-', ' - ').replace('~', ' ~ ').replace('%', ' % ').replace('^', ' ^ ')\
        .replace('&', ' & ').replace('*', ' * ').replace('(', ' ( ').replace(')', ' ) ')\
        .replace('+', ' + ').replace('=', ' = ').replace('{', ' { ').replace('}', ' } ')\
        .replace('|', ' | ').replace('\\', ' \ ').replace('[', ' [ ').replace(']', ' ] ')\
        .replace(':', ' : ').replace(';', ' ; ').replace(',', ' , ').replace('?', ' ? ').replace('/', ' / ')
    sentence = ' '.join(sentence.split())
    return sentence

def parse_diff(diff, word_dic):
    str_result = ""
    split_hunk_format = r'@@ -[0-9]*,[0-9]* +[0-9]*\+[0-9]*,[0-9]* @@'
    split_diff = re.split(split_hunk_format, diff)
    modified_file = split_diff[0].split('\n')[0]
    is_modified_file_added = False
    for i in range(1, len(split_diff)):
        if (not is_modified_file_added):
            is_modified_file_added = True
            str_result += " <modifiedfile> "
        hunk = split_diff[i]
        index = hunk.find("diff --git")
        if (index != -1): 
            is_modified_file_added = False
            modified_file = hunk[index:].split('\n')[0]
            hunk = hunk[:index]
        diff_lines = hunk.splitlines()
        for line in diff_lines:
            if (line.startswith('-')):
                code = line[1:].strip()
                code = ' '.join(split_sentence(code).split())
                for word in code.split():
                    word_dic.add(word)
                str_result += " <del> " + code
            elif (line.startswith('+')):
                code = line[1:].strip()
                code = ' '.join(split_sentence(code).split())
                for word in code.split():
                    word_dic.add(word)
                str_result += " <add> " + code
            else:
                code = line.strip()
                code = ' '.join(split_sentence(code).split())
                for word in code.split():
                    word_dic.add(word)
                # unchanged
                str_result += " <unc> " + code
    return str_result, word_dic

def generate_input_diff(raw_diff_path, dic_path, label_dic_path):
    word_dic = Dict(lower=True)
    label_dic = {}
    label_count = 0
    output_result = list()
    input_diff_path = "C:\\Zhijie\\fyp\\fyp\\transformer\\input_diff_exception_type.csv"
    with open(raw_diff_path, mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            diff = row[0]
            label = row[1]
            if (label not in label_dic):
                label_dic[label] = label_count
                label_count += 1
            parsed_diff, word_dic = parse_diff(diff, word_dic)
            print(parsed_diff)
            output_result.append([parsed_diff, label])
            
    with open(input_diff_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for row in output_result:
            writer.writerow(row)
            
    with open(dic_path, 'wb') as dic_file:
        #total 33194, select top 30000
        word_dic = word_dic.prune(29987)
        pickle.dump(word_dic.get_dict(), dic_file)
    
    with open(label_dic_path, 'wb') as dic_file:
        pickle.dump(label_dic, dic_file)
        
        
        
        
        
        
        
        


def convert_words_to_nums(dic, diff, max_len=512):
    diff_list = diff.split()
    diff_num_list = list()
    for word in diff_list:
        if(word not in dic):
            diff_num_list.append(dic["<NULL>"])
        else:
            diff_num_list.append(dic[word])
    diff_num_list = diff_num_list[:max_len]
    if(len(diff_num_list) < max_len):
        for i in range(max_len - len(diff_num_list)):
            diff_num_list.append(dic["<NULL>"])
    return diff_num_list

def generate_limit_len_input_diff(dic, label_dic, input_path):
    output = "C:\\Zhijie\\fyp\\fyp\\transformer\\input_diff_limit_len_512_exception_type_limit_to_4_new.pkl"
    result = list()
    count_Exception = 0
    with open(input_path, mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            diff = row[0]
            label = row[1]
            diff = convert_words_to_nums(dic, diff, 512)
            if (label == 'IllegalArgumentException' or label == 'NullPointerException' or label == 'IllegalArgumentException' or label == 'IllegalStateException'):
                label = label_dic[label]
            else:
                label = label_dic['OtherException']
                count_Exception+=1
            result.append([diff, label])
    pickle.dump(result, open(output, 'wb'))
    data = pickle.load(open(output,'rb'))
    print(data)
    # with open(output, mode='w', newline='', encoding='utf-8') as file:
    #     writer = csv.writer(file)
    #     for row in result:
    #         writer.writerow(row)
  
def counting():
    dic = {}
    with open("C:\\Zhijie\\fyp\\fyp\\transformer\\input_diff_exception_type.csv",'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            if (row[1] not in dic):
                dic[row[1]] = 1
            else:
                dic[row[1]] = dic[row[1]] + 1
    print(dic)


if __name__=="__main__":
    # counting()
    # generate_input_diff(raw_diff_path, dic_path, label_dic_path)
    dic = pickle.load(open(dic_path, 'rb'))
    # label_dic = pickle.load(open(label_dic_path, 'rb'))
    label_dic = {'OtherException':0, 'NullPointerException':1, 'IllegalArgumentException':2, "IllegalStateException":3}
    # print(data)
    generate_limit_len_input_diff(dic, label_dic, "C:\\Zhijie\\fyp\\fyp\\transformer\\input_diff_exception_type.csv")
    
    