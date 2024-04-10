import os
import pickle
import csv
import subprocess
import re
from transformer.Dict import Dict
import numpy as np



#info_header = ['buggy_commit', 'fix_commit', 'repo_id', 'report_url', 'bug_status', 'is_exception_inducing', 'exception_list']
def get_commits_from_defects4j():
    dic = {}
    dic["Chart_result.pkl"] = "jfree_jfreechart"
    dic["Cli_result.pkl"] = "apache_commons-cli"
    dic["Closure_result.pkl"] = "google_closure-compiler"
    dic["Codec_result.pkl"] = "apache_commons-codec"
    dic["Collections_result.pkl"] = "apache_commons-collections"
    dic["Compress_result.pkl"] = "apache_commons-compress"
    dic["Csv_result.pkl"] = "apache_commons-csv"
    dic["Gson_result.pkl"] = "google_gson"
    dic["JacksonCore_result.pkl"] = "FasterXML_jackson-core"
    dic["JacksonDatabind_result.pkl"] = "FasterXML_jackson-databind"
    dic["JacksonXml_result.pkl"] = "FasterXML_jackson-dataformat-xml"
    dic["Jsoup_result.pkl"] = "jhy_jsoup"
    dic["JxPath_result.pkl"] = "apache_commons-jxpath"
    dic["Lang_result.pkl"] = "apache_commons-lang"
    dic["Math_result.pkl"] = "apache_commons-math"
    dic["Mockito_result.pkl"] = "mockito_mockito"
    dic["Time_result.pkl"] = "JodaOrg_joda-time"
    info_path = "C:\\Zhijie\\fyp\\defects4j\\result"
    csv_result = list()
    for file in os.listdir(info_path):
        
        if (not os.path.isdir(f"{info_path}\\{file}") and not file == 'Chart_result.pkl' and not file == 'combined_result.csv' and not file=='transformer_input.pkl'):
            # if(dic[file] in analyzed_repo): continue
            info = pickle.load(open(f'{info_path}\\{file}', 'rb'))
            for row in info.values():
                is_exception_inducing = 0
                if (row[5]): is_exception_inducing = 1
                csv_result.append([dic[file], row[0], row[1], is_exception_inducing])
    with open('C:\\Zhijie\\fyp\\fyp\\baseline\\defects4j_commit_id.csv', mode='w', encoding='utf-8', newline="") as csv_output:
        csv_writer = csv.writer(csv_output)
        for row in csv_result:
            csv_writer.writerow(row)    
    
def get_commits_from_regs4j():
    csv_output_path = "regs4j_commit_id.csv"
    csv_output = list()
    with open("C:\\Zhijie\\fyp\\defects4j\\result\\commit_diff_with_context\\combined_input_1060bugfree_1032buggy.pkl", "rb") as f:
        data = pickle.load(f)
        count = 0
        for elem in data:
            try:
                if(elem['label'] == 1): continue
                print(count)
                count+=1
                repo = elem['repo']
                owner = elem['owner']
                owner_repo = owner + "_" + repo
                commit_id = elem['commit_id']
                csv_output.append([owner_repo, commit_id, 0])
            except Exception as e:
                print(e)
    with open(csv_output_path, mode='w', encoding='utf-8', newline="") as csv_output_file:
        csv_writer = csv.writer(csv_output_file)
        for row in csv_output:
            csv_writer.writerow(row)
  
def parse_diff_transformer(raw_diff, is_for_tansformer, is_raw):
    if(is_raw):
        str_result = ""
        split_hunk_format = r'diff --git'
        split_diff = re.split(split_hunk_format,raw_diff)
        result = ""
        for i in range(1, len(split_diff)):
            result +=  "diff --git" + split_diff[i]
        return result
    if (is_for_tansformer):
        str_result = ""
        split_hunk_format = r'@@ -[0-9]*,[0-9]* +[0-9]*\+[0-9]*,[0-9]* @@'
        split_diff = re.split(split_hunk_format,raw_diff)
        for i in range(1, len(split_diff)):
            hunk = split_diff[i]
            index = hunk.find("diff --git")
            if (index != -1): 
                hunk = hunk[:index]
            diff_lines = hunk.splitlines()
            
            for line in diff_lines:
                if (line.startswith('-')):
                    l = line[1:].strip()
                    str_result += " <del> " + l
                elif (line.startswith('+')):
                    l = line[1:].strip()
                    str_result += " <add> " + l 
                else:
                    l = line[1:].strip()
                    # unchanged
                    str_result += " <unc> " + l 
        return str_result
    
def split_sentence(sentence):
    sentence = sentence.replace('.', ' . ').replace('_', ' ').replace('@', ' @ ')\
        .replace('-', ' - ').replace('~', ' ~ ').replace('%', ' % ').replace('^', ' ^ ')\
        .replace('&', ' & ').replace('*', ' * ').replace('(', ' ( ').replace(')', ' ) ')\
        .replace('+', ' + ').replace('=', ' = ').replace('{', ' { ').replace('}', ' } ')\
        .replace('|', ' | ').replace('\\', ' \ ').replace('[', ' [ ').replace(']', ' ] ')\
        .replace(':', ' : ').replace(';', ' ; ').replace(',', ' , ').replace('<', ' < ')\
        .replace('>', ' > ').replace('?', ' ? ').replace('/', ' / ')
    sentence = ' '.join(sentence.split())
    return sentence

def parse_diff_deepjit_cc2vec(diff, code_dict):
    added_lines = list()
    removed_lines = list()
    file_codes = list()
    split_hunk_format = r'@@ -[0-9]*,[0-9]* +[0-9]*\+[0-9]*,[0-9]* @@'
    split_diff = re.split(split_hunk_format, diff)
    if (len(split_diff) <= 1): return added_lines, removed_lines, file_codes
    for i in range(1, len(split_diff)):
        hunk = split_diff[i]
        index = hunk.find("diff --git")
        if (index != -1): 
            hunk = hunk[:index]
        diff_lines = hunk.splitlines()
        
        for line in diff_lines:
            if (line.startswith('-')):
                if len(removed_lines) > 10: continue
                l = line[1:].strip()
                l = ' '.join(split_sentence(l).split())
                l = ' '.join(l.split(' '))
                removed_lines.append(l)
                for word in l.split():
                    code_dict.add(word)
                file_codes.append((line, l))
            elif (line.startswith('+')):
                if len(added_lines) > 10: continue
                l = line[1:].strip()
                l = ' '.join(split_sentence(l).split())
                l = ' '.join(l.split(' '))
                added_lines.append(l)
                for word in l.split():
                    code_dict.add(word)
                file_codes.append((line, l))
    return added_lines, removed_lines, file_codes, code_dict

def obtain_commit_info(buggy_commit_id, msg_dict, code_dict, working_commit_id=""):
    format_code = list()
    files_code = list()
    raw_code = list()
    commit_msg = ""
    try:
        get_modified_file_cmd = f'git show --pretty="format:" --name-only {working_commit_id} {buggy_commit_id}'
        output = subprocess.check_output(get_modified_file_cmd, shell=True).decode('utf-8', errors='ignore')
        result = output.split('\n')
        result = result[:-1]
        get_commit_msg_cmd = f'git show {buggy_commit_id}'
        commit_msg = subprocess.check_output(get_commit_msg_cmd, shell=True).decode('utf-8', errors='ignore')
        date_format = r'Date:.+\n'
        commit_msg = re.split(date_format, commit_msg)[1]
        diff_format = r'diff --git'
        commit_msg = re.split(diff_format, commit_msg)[0].strip()
        commit_msg = split_sentence(commit_msg)
        commit_msg = ' '.join(commit_msg.split(' ')).lower()
        for word in commit_msg.split():
            msg_dict.add(word)
        
        for modified_file in result:
            if(len(modified_file) == 0): continue
            get_commit_diff_cmd = f'git diff {working_commit_id} {buggy_commit_id} -- {modified_file}'
            commit_diff = subprocess.check_output(get_commit_diff_cmd, shell=True).decode('utf-8', errors='ignore')
            if (len(commit_diff.strip()) == 0):
                continue
            print(commit_diff)
            added_lines, removed_lines, file_codes, code_dict = parse_diff_deepjit_cc2vec(commit_diff, code_dict)
            file_codes.sort(key = lambda x: x[0])
            raw_code.extend([code[1] for code in file_codes])
            raw_code = raw_code[:10]
            format_code.append("added _ code removed _ code")
            files_code.append({'added_code': added_lines, 'removed_code': removed_lines})
    except Exception as e:
        print(e)
    return commit_msg, format_code, files_code, raw_code, msg_dict, code_dict

def split_data(args, data):
    idx1 = args.size + args.test
    idx2 = args.test
    idx = int(len(data)*0.8)

    if args.size:
        return data[-idx1:-idx2], data[-idx2:]

    return data[:idx], data[idx:]

def generate_deepjit_cc2vec_input():
    msg_dict = Dict(lower=True)
    code_dict = Dict(lower=True)
    defects4j_commit_id_path = "C:\\Zhijie\\fyp\\fyp\\baseline\\defects4j_commit_id.csv"
    regs4j_commit_id_path = "C:\\Zhijie\\fyp\\fyp\\baseline\\regs4j_commit_id.csv"
    ids, labels, msgs, codes, deepjit_codes, deepjit_raw_codes = [], [], [], [], [], []
    with open(defects4j_commit_id_path, 'r', encoding='utf-8', newline='') as file:
        csv_reader = csv.reader(file)    
        for row in csv_reader:
            owner_repo = row[0]
            buggy_commit = row[1]
            working_commit = row[2]
            label = row[3]
            os.chdir(f'C:\\Zhijie\\fyp\\defects4j\\buggy_version_repos\\{owner_repo}')
            msg, format_code, files_code, raw_code, msg_dict, code_dict = obtain_commit_info(buggy_commit, msg_dict, code_dict, working_commit)
            if(len(format_code) == 0): continue
            ids.append(buggy_commit)
            labels.append(label)
            msgs.append(msg)
            deepjit_codes.append(format_code)
            deepjit_raw_codes.append(raw_code)
            codes.append(files_code)
    with open(regs4j_commit_id_path, 'r', encoding='utf-8', newline='') as file:
        csv_reader = csv.reader(file)    
        for row in csv_reader:
            owner_repo = row[0]
            buggy_commit = row[1]
            label = row[2]
            os.chdir(f'C:\\Zhijie\\fyp\\dataset\\repos\\{owner_repo}')
            msg, format_code, files_code, raw_code, msg_dict, code_dict = obtain_commit_info(buggy_commit, msg_dict, code_dict)
            if(len(format_code) == 0): continue
            ids.append(buggy_commit)
            labels.append(label)
            msgs.append(msg)
            deepjit_codes.append(format_code)
            deepjit_raw_codes.append(raw_code)
            codes.append(files_code)
    
    indices = np.random.permutation(len(ids))
    split = int(0.8 * len(ids))
    max_codes_len = 0
    for elem in codes:
        max_codes_len = max(max_codes_len, len(elem))
    for elem in codes:
        if (len(elem) < max_codes_len):
            for i in range(max_codes_len - len(elem)):
                elem.append({})
    
    max_codes_len = 0
    for elem in deepjit_codes:
        max_codes_len = max(max_codes_len, len(elem))
    for elem in deepjit_codes:
        if (len(elem) < max_codes_len):
            for i in range(max_codes_len - len(elem)):
                elem.append("")
    
    max_codes_len = 0
    for elem in deepjit_raw_codes:
        max_codes_len = max(max_codes_len, len(elem))
    for elem in deepjit_raw_codes:
        if (len(elem) < max_codes_len):
            for i in range(max_codes_len - len(elem)):
                elem.append("")
    train_ids = np.take(ids, indices[:split], axis=0)
    train_labels = np.take(labels, indices[:split], axis=0)
    train_msgs = np.take(msgs, indices[:split], axis=0)
    deepjit_train_codes = np.take(deepjit_codes, indices[:split], axis=0)
    deepjit_train_raw_codes = np.take(deepjit_raw_codes, indices[:split], axis=0)
    train_codes = np.take(codes, indices[:split], axis=0)
    
    test_ids = np.take(ids, indices[split:], axis=0)
    test_labels = np.take(labels, indices[split:], axis=0)
    test_msgs = np.take(msgs, indices[split:], axis=0)
    deepjit_test_codes = np.take(deepjit_codes, indices[split:], axis=0)
    deepjit_test_raw_codes = np.take(deepjit_raw_codes, indices[split:], axis=0)
    test_codes = np.take(codes, indices[split:], axis=0)
    
    deepjit_train_data = [train_ids, train_labels,
                          train_msgs, deepjit_train_codes]
    deepjit_train_raw_data = [train_ids, train_labels,
                          train_msgs, deepjit_train_raw_codes]
    deepjit_test_data = [test_ids, test_labels, test_msgs, deepjit_test_codes]
    deepjit_test_raw_data = [test_ids, test_labels, test_msgs, deepjit_test_raw_codes]

    deepjit_all_data = [ids, labels, msgs, deepjit_codes]
    deepjit_all_raw_data = [ids, labels, msgs, deepjit_raw_codes]


    cc2vec_train_data = [train_ids, train_labels, train_msgs, train_codes]
    cc2vec_test_data = [test_ids, test_labels, test_msgs, test_codes]

    cc2vec_all_data = [ids, labels, msgs, codes]

    dextend_train_data = [train_ids, train_labels,
                          train_msgs, deepjit_train_codes]
    dextend_test_data = [test_ids, test_labels, test_msgs, deepjit_test_codes]

    dextend_all_data = [ids, labels, msgs, deepjit_codes]

    raw_dextend_train_data = [train_ids, train_labels,
                          train_msgs, deepjit_train_raw_codes]
    raw_dextend_test_data = [test_ids, test_labels, test_msgs, deepjit_test_raw_codes]

    raw_dextend_all_data = [ids, labels, msgs, deepjit_raw_codes]
    
    os.chdir("C:\\Zhijie\\fyp\\fyp\\baseline")
    
    with open('./deepjit/train.pkl', 'wb') as f:
        pickle.dump(deepjit_train_data, f)
    with open('./deepjit/test.pkl', 'wb') as f:
        pickle.dump(deepjit_test_data, f)
    with open('./deepjit/all.pkl', 'wb') as f:
        pickle.dump(deepjit_all_data, f)

    with open('./deepjit/train_raw.pkl', 'wb') as f:
        pickle.dump(deepjit_train_raw_data, f)
    with open('./deepjit/test_raw.pkl', 'wb') as f:
        pickle.dump(deepjit_test_raw_data, f)
    with open('./deepjit/all_raw.pkl', 'wb') as f:
        pickle.dump(deepjit_all_raw_data, f)

    with open('./cc2vec/train.pkl', 'wb') as f:
        pickle.dump(cc2vec_train_data, f)
    with open('./cc2vec/test.pkl', 'wb') as f:
        pickle.dump(cc2vec_test_data, f)
    with open('./cc2vec/all.pkl', 'wb') as f:
        pickle.dump(cc2vec_all_data, f)

    with open('./cc2vec/train_dextend.pkl', 'wb') as f:
        pickle.dump(dextend_train_data, f)
    with open('./cc2vec/test_dextend.pkl', 'wb') as f:
        pickle.dump(dextend_test_data, f)
    with open('./cc2vec/all_dextend.pkl', 'wb') as f:
        pickle.dump(dextend_all_data, f)

    with open('./cc2vec/train_dextend_raw.pkl', 'wb') as f:
        pickle.dump(raw_dextend_train_data, f)
    with open('./cc2vec/test_dextend_raw.pkl', 'wb') as f:
        pickle.dump(raw_dextend_test_data, f)
    with open('./cc2vec/all_dextend_raw.pkl', 'wb') as f:
        pickle.dump(raw_dextend_all_data, f)
        
    msg_dict = msg_dict.prune(100000)
    code_dict = code_dict.prune(100000)
    project_dict = [msg_dict.get_dict(), code_dict.get_dict()]
    pickle.dump(project_dict, open("./dict.pkl", 'wb'))
    pickle.dump(project_dict, open("./deepjit/dict.pkl", 'wb'))
    pickle.dump(project_dict, open("./cc2vec/dict.pkl", 'wb'))
    pickle.dump(project_dict, open("./feature/dict.pkl", 'wb'))

    print('Train data size: {}, Bug size: {}'.format(
        len(train_labels), sum(train_labels)))
    print('Test data size: {}, Bug size: {}'.format(
        len(test_labels), sum(test_labels)))
    
    
def preprocess_data():
    data = pickle.load(open('C:\\Zhijie\\fyp\\fyp\\baseline\\deepjit\\train.pkl', 'rb'))
    msgs = list()
    labels = list()
    codes = list()
    ids = list()
    for id in data[0]:
        ids.append(id)
    for label in data[1]:
        labels.append(label)
    for msg in data[2]:    
        msgs.append(msg)
    for code in data[3]:
        codes.append(code)   
    train_data = [ids, labels, msgs, codes]     
    with open('C:\\Zhijie\\fyp\\fyp\\baseline\\deepjit\\train_list.pkl', 'wb') as f:
        pickle.dump(train_data, f)
        
    data = pickle.load(open('C:\\Zhijie\\fyp\\fyp\\baseline\\deepjit\\test.pkl', 'rb'))
    msgs = list()
    labels = list()
    codes = list()
    ids = list()
    for id in data[0]:
        ids.append(id)
    for label in data[1]:
        labels.append(int(label))
    for msg in data[2]:    
        msgs.append(msg)
    for code in data[3]:
        codes.append(code) 
    test_data =  [ids, labels, msgs, codes]     
    with open('C:\\Zhijie\\fyp\\fyp\\baseline\\deepjit\\test_list.pkl', 'wb') as f:
        pickle.dump(test_data, f)
        
    data = pickle.load(open('C:\\Zhijie\\fyp\\fyp\\baseline\\cc2vec\\train.pkl', 'rb'))
    msgs = list()
    labels = list()
    codes = list()
    ids = list()
    for id in data[0]:
        ids.append(id)
    for label in data[1]:
        labels.append(int(label))
    for msg in data[2]:    
        msgs.append(msg)
    for code in data[3]:
        shrink = list()
        for ele in code:
            if(len(ele) > 0):
                shrink.append(ele)
        codes.append(shrink) 
    train_data =  [ids, labels, msgs, codes]     
    with open('C:\\Zhijie\\fyp\\fyp\\baseline\\cc2vec\\train_list.pkl', 'wb+') as f:
        pickle.dump(train_data, f)
        
    data = pickle.load(open('C:\\Zhijie\\fyp\\fyp\\baseline\\cc2vec\\test.pkl', 'rb'))
    msgs = list()
    labels = list()
    codes = list()
    ids = list()
    for id in data[0]:
        ids.append(id)
    for label in data[1]:
        labels.append(int(label))
    for msg in data[2]:    
        msgs.append(msg)
    for code in data[3]:
        shrink = list()
        for ele in code:
            if(len(ele) > 0):
                shrink.append(ele)
        codes.append(shrink) 
    test_data =  [ids, labels, msgs, codes]     
    with open('C:\\Zhijie\\fyp\\fyp\\baseline\\cc2vec\\test_list.pkl', 'wb+') as f:
        pickle.dump(test_data, f)
      
      
    data = pickle.load(open('C:\\Zhijie\\fyp\\fyp\\baseline\\cc2vec\\train_dextend.pkl', 'rb'))
    msgs = list()
    labels = list()
    codes = list()
    ids = list()
    for id in data[0]:
        ids.append(id)
    for label in data[1]:
        labels.append(int(label))
    for msg in data[2]:    
        msgs.append(msg)
    for code in data[3]:
        shrink = list()
        for ele in code:
            if(len(ele) > 0):
                shrink.append(ele)
        codes.append(shrink) 
    train_data =  [ids, labels, msgs, codes]     
    with open('C:\\Zhijie\\fyp\\fyp\\baseline\\cc2vec\\train_dextend_list.pkl', 'wb+') as f:
        pickle.dump(train_data, f)  
        
        
    data = pickle.load(open('C:\\Zhijie\\fyp\\fyp\\baseline\\cc2vec\\test_dextend.pkl', 'rb'))
    msgs = list()
    labels = list()
    codes = list()
    ids = list()
    for id in data[0]:
        ids.append(id)
    for label in data[1]:
        labels.append(int(label))
    for msg in data[2]:    
        msgs.append(msg)
    for code in data[3]:
        shrink = list()
        for ele in code:
            if(len(ele) > 0):
                shrink.append(ele)
        codes.append(shrink) 
    test_data =  [ids, labels, msgs, codes]     
    with open('C:\\Zhijie\\fyp\\fyp\\baseline\\cc2vec\\test_dextend_list.pkl', 'wb+') as f:
        pickle.dump(test_data, f)  
    
    
if __name__=="__main__":
    # generate_deepjit_cc2vec_input()
    # data = pickle.load(open('C:\\Zhijie\\fyp\\fyp\\baseline\\deepjit\\train.pkl', 'rb'))
    # for elem in data:
    #     print(elem)
    # print(data)
    preprocess_data()