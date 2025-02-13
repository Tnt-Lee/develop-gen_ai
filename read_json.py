import json
import csv
import os

# 假设你的JSON文件路径是 'c:\\Users\\youzi\\Desktop\\AI\\myGitHub\\deepseek\\qianchengwuyou\\java_20250213_009.json'
file_path = r'qianchengwuyou\java_20250213_009.json'

# 打开并读取JSON文件
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 获取 jobName 和 jobAreaString
jobs = data['resultbody']['job']['items']

# 打开CSV文件以写入模式
with open('out_qc/output_dl_java.csv', 'w+', newline='', encoding='utf-8') as csv_file:
    
    # 创建DictWriter对象
    writer = csv.DictWriter(csv_file, fieldnames=['jobName','jobAreaString','provideSalaryString','fullCompanyName','companyTypeString','jobDescribe','termStr'])
    

    # 打印CSV头部
    print("Job Name,Job Area String,Job Salary,Company Name,Company Type,Job Describe,Term Str")
    # 写入CSV文件的表头
    writer.writeheader()


    for jb in jobs:
        job_name = jb.get('jobName')
        job_area = jb.get('jobAreaString')
        job_salary = jb.get('provideSalaryString')
        company_name = jb.get('fullCompanyName')
        company_type = jb.get('companyTypeString')
        job_describe = jb.get('jobDescribe')
        term_str = jb.get('termStr')
        # 写入JSON数据      
        writer.writerow({
            'jobName': job_name,
            'jobAreaString': job_area,
            'provideSalaryString': job_salary,
            'fullCompanyName': company_name,
            'companyTypeString': company_type,
            'jobDescribe': job_describe,
            'termStr': term_str
        })

        # 打印CSV数据行
        print(f"{job_name},{job_area},{job_salary},{company_name},{company_type},{job_describe},{term_str}")
