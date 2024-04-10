from re import T
from openai import OpenAI
import os
import requests
import csv
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc
import re
import sys


csv.field_size_limit(100000000)

diff_path = "C:\\Zhijie\\fyp\\semantic-preserving-detector\\diff_files\\"


not_recorded_commit_list = ["alibaba_canal_b893aafae764aada0b8122b9cb7f7f40472c9487_b131e8cfa4c212497b075d5175aa3425aa6fd8c3"]


def extract_changes(file_path):
    changes = list()
    change = ""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if (line.strip() =="---------------- METHOD BREAK ---------------") :
                changes.append(change)
                change = ""
            else:
                change += line + "\n"
    return changes


def extract_pure_refactoring_commit_contains():
    client = OpenAI(api_key="")
    for file in os.listdir(diff_path):
        try:
            print(f"start judging file {file}")
            is_refactoring = False
            is_time_out = False
            changes = extract_changes(diff_path + file)
            for change in changes:
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        response_format={ "type": "text" },
                        messages=[
                            {"role": "user", "content": f"is this a pure refactoring? answer in yes or no: \"{change}\""}
                        ],
                        timeout = 10
                    )
                except requests.exceptions.Timeout:
                    print("The request timed out.")
                    is_time_out = True
                    break
                res = response.choices[0].message.content
                print(res)
                if ("Yes" in res or "yes" in res or "YES" in res):
                    is_refactoring = True
                    print("contains refactoring")
                    with open("pure_refactoring_code.txt", 'a', encoding="utf-8") as f:
                        f.write(file + ":1\n")
                        f.close()
                    break
            if (not is_refactoring and not is_time_out): 
                with open("pure_refactoring_code.txt", 'a', encoding="utf-8") as f:
                    f.write(file + ":0\n")
                    f.close()
        except Exception as e:
            print(e)
            
def extract_pure_refactoring_commit_complete():
    client = OpenAI(api_key="")
    for file in os.listdir(diff_path):
        try:
            print(f"start judging file {file}")
            is_refactoring = True
            is_time_out = False
            changes = extract_changes(diff_path + file)
            for change in changes:
                try:
                    response = client.chat.completions.create(
                        model="gpt-4",
                        response_format={ "type": "text" },
                        messages=[
                            {"role": "user", "content": f"is this a pure refactoring? answer in yes or no: \"{change}\""}
                        ],
                        timeout = 10
                    )
                except requests.exceptions.Timeout:
                    print("The request timed out.")
                    is_time_out = True
                    break
                res = response.choices[0].message.content
                print(res)
                if ("No" in res or "NO" in res or "no" in res):
                    is_refactoring = False
                    print("not refactoring")
                    with open("pure_refactoring_code.txt", 'a', encoding="utf-8") as f:
                        f.write(file + ":0\n")
                        f.close()
                    break
            if (is_refactoring and not is_time_out): 
                with open("pure_refactoring_code.txt", 'a', encoding="utf-8") as f:
                    f.write(file + ":1\n")
                    f.close()
        except Exception as e:
            print(e)

def count_pure_refactoring():
    count_1 = 0
    count_0 = 0
    with open('pure_refactoring_code.txt', 'r') as f:
        for line in f:
            parts = line.split(':')
            if len(parts) >= 2:
                if (parts[1][0] == '1'):
                    count_1 += 1
                else:
                    count_0 += 1
    print(f'count_1:{count_1}')
    print(f'count_0:{count_0}')
    
def check_unrecorded_commit():
    recorded_commit = list()
    with open('C:\\Zhijie\\fyp\\refactoring_bugs\\working_ric\\pure_refactoring_working_ric_complete.txt', 'r') as f:
        for line in f:
            recorded_commit.append(line[:-3])
    for file in os.listdir(diff_path):
        if (file not in recorded_commit):
            print(file)
            
def try_gpt_4():
    client = OpenAI(api_key="")
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            response_format={ "type": "text" },
            messages=[
                {"role": "user", "content": f"Am I interacting with gpt4?"}
            ],
            timeout = 10
        )
        print(response)
    except requests.exceptions.Timeout:
        print("The request timed out.")
        is_time_out = True

def check_exception_inducing_diff(recorded_path):
    client = OpenAI(api_key="")
    last_position = 0
    with open(recorded_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            last_position += 1
    print(last_position)
    with open('C:\\Zhijie\\fyp\\refactoring_bugs\\gpt_input_512_len\\gpt_input_diff_limit_len.csv', 'r', newline="", encoding='utf-8') as file:
        reader = csv.reader(file)
        y_test = list()
        y_pred = list()
        count = 0
        for row in reader:
            if (count < last_position):
                count+=1
                continue
            count+=1
            is_time_out = False
            diff = row[0]
            # diff_list = re.split(r'[0-9]+|[()|\"\'&$_%#!<>{}=\+*@ .,\\/\n;-]', diff)
            # new_list = list()
            # for elem in diff_list:
            #     if (len(elem) != 0):
            #         new_list.append(elem)
            # if(len(new_list) > 512):
            #     new_list = new_list[:512]
            # diff = ' '.join(new_list)
            label = int(row[1])
            y_test.append(label)
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    response_format={ "type": "text" },
                    messages=[
                        {"role": "user", "content": f"\
                         You are an expert in software engineering.\
                         I'll give you a commit diff.\
                         In this commit diff, each modified file content starts with diff --git,\
                         each added line started with <add>, each removed line started with <del>, each unchanged line started with <unc>.\
                         You must tell me if this commit code change will lead to exception. answer in lower case yes or no: \"{diff}\""}
                    ],
                    timeout = 50
                )
            except requests.exceptions.Timeout:
                print("The request timed out.")
                is_time_out = True
                break
            predict = 1
            res = response.choices[0].message.content
            if (is_time_out): print('timeout')
            if ("No" in res or "NO" in res or "no" in res):
                predict = 0
            elif ("Yes" in res or "YES" in res or "yes" in res):
                predict = 1
            else:
                print('error!!!!!!!!!!!!!!!!!')
            y_pred.append(predict)
            print(f'count: {count}, actual: {label}, predict: {predict}')
            recorded_result = [label, predict]
            with open(recorded_path, mode='a', newline='') as output:
                writer = csv.writer(output)
                writer.writerow(recorded_result)

    
    
def check_gpt_result():
    y_test = list()
    y_pred = list()
    with open('C:\\Zhijie\\fyp\\fyp\\baseline\\gpt\\gpt3.5_prediction_result.csv', 'r', newline="", encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            y_test.append(int(row[0]))
            y_pred.append(int(row[1]))
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Calculate precision
    precision = precision_score(y_test, y_pred)

    # Calculate recall
    recall = recall_score(y_test, y_pred)

    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("AUC:", roc_auc)

def count():
    with open("C:\\Zhijie\\fyp\\fyp\\baseline\\gpt\\gpt_input_diff_limit_len.csv", 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        bug_count = 0
        bug_free_count = 0
        for row in csv_reader:
            if (row[1] == '0'):
                bug_free_count += 1
            else:
                bug_count += 1
        print(bug_free_count)
        print(bug_count)

if __name__ == "__main__":
    
    # try_gpt_4()
    count()
    # check_gpt_result()
    # check_exception_inducing_diff("C:\\Zhijie\\fyp\\fyp\\baseline\\gpt\\gpt3.5_prediction_result.csv")
    # check_exception_inducing_diff()
    # extract_pure_refactoring_commit_complete()
    # count_pure_refactoring()
    # check_unrecorded_commit()
    # with open(f'{diff_path}\\alibaba_canal_b893aafae764aada0b8122b9cb7f7f40472c9487_b131e8cfa4c212497b075d5175aa3425aa6fd8c3', 'r', encoding='utf-8') as f:
    #     print(f)
    #     for line in f:
    #         print(line)
    
    