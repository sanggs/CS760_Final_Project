import pandas as pd
import csv
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='Specify filename to process')
parser.add_argument('--csvfile', type=str, default='data.csv')
parser.add_argument('--outputfile', type=str, default='processed.csv')

args = parser.parse_args()

def get_data_from_csv(filename):
	row_count = 0
	survey_data = None
	data = []
	with open(filename) as csvfile:
		survey_data = csv.reader(csvfile)
		for row in survey_data:
			row_data = []
			for entry in row:
				row_data.append(entry)
			row_count += 1
			data.append(row_data)
	print("Read {} rows from {}".format(row_count, filename))
	return data

def generate_feature_index_dictionary(feature_row):
	d = {}
	index = 0
	for feature_name in feature_row:
		d[feature_name] = index
		index += 1
	return d

csvfile = args.csvfile
data = get_data_from_csv(csvfile)
feature_dict = generate_feature_index_dictionary(data[0])

fg01 = ['SOC1', 'SOC2A', 'SOC3A']
fg02 = ['SOC2B', 'SOC3B']
fg03 = ['PHYS8']
fg04 = ['PHYS1A' ,'PHYS1B' ,'PHYS1C' ,'PHYS1D' ,'PHYS1E' ,'PHYS1F' ,'PHYS1G' ,'PHYS1H' ,'PHYS1I' ,'PHYS1J' ,'PHYS1K' ,'PHYS1L' ,'PHYS1M' ,'PHYS1N' ,'PHYS1O' ,'PHYS1P' ,'PHYS1Q', 'PHYS7_1', 
'PHYS7_2', 'PHYS7_3']
fg05 = ['SOC5A', 'SOC5B', 'SOC5C', 'SOC5D', 'SOC5E']
fg06 = ['PHYS2_1', 'PHYS2_2', 'PHYS2_3', 'PHYS2_4', 'PHYS2_5', 'PHYS2_6', 'PHYS2_7', 'PHYS2_8', 'PHYS2_9', 'PHYS2_10', 'PHYS2_11', 'PHYS2_12', 'PHYS2_13', 'PHYS2_14', 'PHYS2_15', 'PHYS2_16', 'PHYS2_17', 'PHYS2_18', 'PHYS2_19', 'PHYS10A', 'PHYS10B', 'PHYS10C', 'PHYS10D', 'PHYS10E']
fg07 = ['ECON8A', 'ECON8B', 'ECON8C', 'ECON8D', 'ECON8E', 'ECON8F', 'ECON8G', 'ECON8H', 'ECON8I', 'ECON8J', 'ECON8K', 'ECON8L', 'ECON8M', 'ECON8N', 'ECON8O', 'ECON8P', 'ECON8Q', 'ECON8R', 'ECON8S']
fg08 = ['ECON7_1', 'ECON7_2', 'ECON7_3', 'ECON7_4', 'ECON7_5', 'ECON7_6', 'ECON7_7', 'ECON7_8']
fg09 = ['ECON1', 'ECON4A', 'ECON4B']
fg10 = ['ECON6A', 'ECON6B', 'ECON6C', 'ECON6D', 'ECON6E', 'ECON6F', 'ECON6G', 'ECON6H', 'ECON6I', 'ECON6J', 'ECON6K', 'ECON6L']
fg11 = ['ECON5A_A', 'ECON5A_B']
fg12 = ['PHYS9A', 'PHYS9B', 'PHYS9C', 'PHYS9D', 'PHYS9E', 'PHYS9F', 'PHYS9G', 'PHYS9H']
fg13 = ['PHYS3A']
fg14 = ['PHYS3B']
fg15 = ['PHYS3C']
fg16 = ['PHYS3D']
fg17 = ['PHYS3E']
fg18 = ['PHYS3F']
fg19 = ['PHYS3G']
fg20 = ['PHYS3H']
fg21 = ['PHYS3I']
fg22 = ['PHYS3J']
fg23 = ['PHYS3K']
fg24 = ['PHYS3L']
fg25 = ['PHYS3M']
fg26 = ['PHYS4']
fg27 = ['PHYS5']
fg28 = ['PHYS6']
fg29 = ['AGE7']
fg30 = ['GENDER']
fg31 = ['RACE_R2']
fg33 = ['EDUCATION', 'EDUC4']
fg35 = ['HH01S']
fg36 = ['HH25S']
fg37 = ['HH612S']
fg38 = ['HH1317S']
fg39 = ['HH18OVS']
fg40 = ['P_DENSE']

feature_grouping_list = [fg01, fg02, fg03, fg04, fg05, fg06, fg07, fg08, fg09, fg10, fg11, fg12, fg13, fg14, fg15, fg16, fg17, fg18, fg19, fg20, fg21, fg22, fg23, fg24, fg25, fg26, fg27, fg28, fg29, fg30, fg31, fg33, fg35, fg36, fg37, fg38, fg39, fg40]

values_to_ignore = ['', 'NA', 'N/A', '77', '88', '98', '99', '777', '888', '998', '999']

skip_count = {}
for feature_group in feature_grouping_list:
	for feature_name in feature_group:
		index = feature_dict[feature_name]
		for row_id in range(1, len(data)): #Starting indexing from 1 since row 0 is the header
			if data[row_id][index] in values_to_ignore:
				if row_id in skip_count.keys():
					skip_count[row_id] += 1
				else:
					skip_count[row_id] = 1

print("Number of samples skipped: ", len(skip_count.keys()))


'''Grouping features'''
grouped_features = []

for row_id in range(1, len(data)):
	if row_id not in skip_count.keys():
		feature = []
		for feature_group in feature_grouping_list:
			feature_val = 0
			for feature_name in feature_group:
				index = feature_dict[feature_name]
				feature_val += int(data[row_id][index])
			feature.append(feature_val)
		grouped_features.append(feature)

grouped_feature_headers = ['Social Interaction pre COVID', 'Social Interaction post COVID', 'Self assessment of Health', 'Physical symptoms', 'Mental Health Symptoms', 'Level of reponse to covid', 'Impact of COVID19 restrictions on Financial Status', 'Ability to pay off expenses', 'Employment Status', 'Financial Assistance', 'Ability to buy food', 'Medical Coverage level', 'Diabetes', 'Hypertension', 'Heart Disease', 'Asthma', 'COPD', 'Bronchitis', 'Allergies', 'Presence of Mental Health Condition', 'Cystic Fibrosis', 'Liver disease', 'Cancer', 'Compromised Immune System', 'Obesity', 'COVID test result', 'COVID test result for family', 'Death from COVID', 'Age', 'Gender', 'Race', 'Education', 'Number of HH members age 0-1', 'Number of HH members age 2-5', 'Number of HH members age 6-12', 'Number of HH members age 13-17', 'Number of HH members age 18+', 'Population Density']

grouped_features.insert(0, grouped_feature_headers)
print("Number of samples in new data: ", len(grouped_features)-1)

dirpath = os.path.dirname(csvfile)
output_file = str(os.path.splitext(os.path.basename(csvfile))[0]) + '_grouped.csv'
output_file = os.path.join(dirpath, output_file)

with open(output_file, "w") as f:
	writer = csv.writer(f)
	writer.writerows(grouped_features)
