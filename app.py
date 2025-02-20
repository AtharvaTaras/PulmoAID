import joblib
import random
import torch
from datetime import datetime
import pandas as pd
import streamlit as st
from streamlit import session_state as state
from PIL import Image
from gemini_llm import LLM, DataManager, LungCancerVGG16Fusion
import os
import shap
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter('ignore')

# INIT PATHS
# train_csv = r'data\all_features_combined_new.csv'
# val_csv = r'data\all_validation_features_combined.csv'
# logo_img = r'images\logo.png'
# arch_img = r'images\architecture.png'

train_csv = os.path.join('data', 'all_features_combined_new.csv')
val_csv = os.path.join('data', 'all_validation_features_combined.csv')
logo_img = os.path.join('images', 'logo.png')
arch_img = os.path.join('images', 'architecture.png')

# modelpaths =  {
# 	"Logistic Regression":r'classifiers\FusionModel LR_97.41.pkl',
# 	"KNN":r'classifiers\FusionModel KNN_73.28.pkl',
# 	"Naive Bayes":r'classifiers\FusionModel NB_78.45.pkl',
# 	"Random Forest":r'classifiers\FusionModel RFC_91.38.pkl',
# 	"XGBoost":r'classifiers\FusionModel XGB_92.24.pkl'
# }

modelpaths = {
    "Logistic Regression": os.path.join('classifiers', 'FusionModel LR_97.41.pkl'),
    "KNN": os.path.join('classifiers', 'FusionModel KNN_73.28.pkl'),
    "Naive Bayes": os.path.join('classifiers', 'FusionModel NB_78.45.pkl'),
    "Random Forest": os.path.join('classifiers', 'FusionModel RFC_91.38.pkl'),
    "XGBoost": os.path.join('classifiers', 'FusionModel XGB_92.24.pkl')
}


# INIT SLICES
feature_cols = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8']
demographic_cols = ['age', 'ethnic', 'gender', 'height', 'race', 'weight']
smoking_hist = ['age_quit', 'cigar', 'cigsmok', 'pipe', 'pkyr', 'smokeage', 'smokeday', 'smokelive', 'smokework', 'smokeyr']
llm_sent = ['llm_sentiment']
clinical = ['biop0', 'bioplc', 'proclc', 'can_scr', 'canc_free_days']
treatments = ['procedures', 'treament_categories', 'treatment_types', 'treatment_days']


if 'layout' not in state: state.layout = 'centered'
if 'login' not in state: state.login = False
if 'scans' not in state: state.scans = []
if 'pil_images' not in state: state.pil_images = []
if 'subject' not in state: state.subject = 'N/A'

st.set_page_config(page_title='PulmoAID', 
				   layout=state.layout,
				   page_icon='🫁')


@st.cache_resource
def utilloader(utility:str):
	if utility == 'llm':
		return LLM(st.secrets["keys"]["api"])
	
	if utility == 'manager':
		torch.manual_seed(0)
		device = torch.device('cpu')
		VGG_16 = LungCancerVGG16Fusion().to(device)
		modelpath = os.path.join("models", "best_vgg16.pth")
		VGG_16.load_state_dict(torch.load(modelpath, weights_only=True, map_location=device))
		VGG_16.eval()

		return DataManager(VGG_16)
	
	if utility == 'subject_list':
		data = pd.read_csv(train_csv)
		return list(data['Subject'])

	if utility == 'classifier_csv':
		return pd.read_csv(os.path.join("data", "all_combined.csv")) 
	
	if utility == 'llm_csv':
		return pd.read_csv(os.path.join("data", "all_combined_descriptive.csv"))


@st.cache_resource
def load_classifier(name:str):
	return joblib.load(modelpaths[name])


@st.cache_resource
def generate_outcome(features=[], subject='', classifier='', full_row=None) -> str:
	global csvdata, feature_cols

	row = csvdata[csvdata['Subject'] == int(subject)]
	# newrow = features + row[demographic_cols + smoking_hist + llm_sent].values.flatten().tolist()
	newrow = row[feature_cols + demographic_cols + smoking_hist + llm_sent].values.flatten().tolist()
	model = load_classifier(classifier)

	if full_row is not None:
		try:
			outcome = model.predict_proba(full_row)
			probability_negative = outcome[0][0] * 100
			probability_positive = outcome[0][1] * 100

			if probability_negative > probability_positive:
				result = f"""
				✅ **Subject `{subject}` has tested _Negative_.**  
				- **Confidence:** `{probability_negative:.2f}%`
				"""
			else:
				result = f"""
				⚠️ **Subject `{subject}` has tested _Positive_.**  
				- **Confidence:** `{probability_positive:.2f}%`
				"""

		except AttributeError:
			outcome = model.predict(full_row)
			result = f"""
			🧪 **Subject `{subject}` has tested:**  
			**{"🟢 Negative" if int(outcome[0]) == 0 else "🔴 Positive"}**
			"""

		return result

	try:
		outcome = model.predict_proba([newrow])
		probability_negative = outcome[0][0] * 100
		probability_positive = outcome[0][1] * 100

		if probability_negative > probability_positive:
			result = f"""
			✅ **Subject `{subject}` has tested _Negative_.**  
			- **Confidence:** `{probability_negative:.2f}%`
			"""
		else:
			result = f"""
			⚠️ **Subject `{subject}` has tested _Positive_.**  
			- **Confidence:** `{probability_positive:.2f}%`
			"""

	except AttributeError:
		outcome = model.predict([newrow])
		result = f"""
		🧪 **Subject `{subject}` has tested:**  
		**{"🟢 Negative" if int(outcome[0]) == 0 else "🔴 Positive"}**
		"""

	return result


# @st.cache_resource
def generate_shap_plot(base: pd.DataFrame, subject: str):
	np.random.seed(0)
	model = load_classifier('XGBoost')

	features = ['n1', 'n2', 'n3', 'n4',
				'age', 'ethnic', 'gender', 'height', 'race', 'weight',
				'age_quit', 'cigar', 'cigsmok', 'pipe', 'pkyr', 'smokeage', 'smokeday',
				'smokelive', 'smokework', 'smokeyr']
	X = base[features]
	y = base['lung_cancer']

	# Fit model before SHAP calculation
	model.fit(X, y)
	
	subject_index = base[base['Subject'] == int(subject)].index
	explainer = shap.TreeExplainer(model)
	shap_values = explainer.shap_values(X)
	subject_shap_values = shap_values[subject_index]

	plt.figure(figsize=(12, 8))
	shap.summary_plot(shap_values, X, max_display=20, show=False)

	# Get feature importance order
	mean_abs_shap = np.abs(shap_values).mean(axis=0)
	feature_importance_order = np.argsort(-mean_abs_shap)[:20]
	
	# Plot subject points
	for i, idx in enumerate(feature_importance_order):
		plt.scatter(subject_shap_values[0][idx], i, color='black', edgecolor='white', s=50, zorder=3)

	plt.title("SHAP Summary Plot with Subject Highlighted")
	plt.tight_layout()
	
	return plt


# @st.cache_resource
# def load_image(subject:str):
# 	path = r"A:\Software Projects\NLST-Dataset\images_all"
# 	imagepaths = []

# 	for root, _, files in os.walk(path):
# 		for file in files:
# 			if subject in files:
# 				imagepaths.append(os.path.join(root, file))

# 	return imagepaths[8:8+16]


def doctor_page():
	global csvdata, llmdata
	
	with st.sidebar:
		st.header(body='Doctor Portal')
		st.divider()
		st.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
		st.write(f'Welcome Dr.Tushar')

		logout = st.button(label='Logout', use_container_width=True)
		if logout:
			state.login = False
			state.layout = 'centered'
			st.rerun()

		state.subject_selection = st.selectbox(label='Patient ID', options=state.subject_list)
		state.model_selection = st.selectbox(label='Classifier', options=list(modelpaths.keys()))

		clinical_data = st.toggle(label='Clinical Data')
		demographic_data = st.toggle(label='Demographic Data')
		smoking_history = st.toggle(label='Smoking History')

		patient_obs = st.button(label='Patient\'s Observations', use_container_width=True)
		doctor_notes = st.text_area(label='Doctor\'s Notes')

	st.image(image=logo_img, use_column_width=True)

	information, images_clinical, diagnostics, ai = st.tabs(['Information', 'Images and Clinical', 'Diagnostics', 'Talk To AI'])

	with information:
		st.image(image=arch_img)
		# st.markdown("![Alt Text](https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExeTZybHlnNjl2aGFtaTZlNjN2ejk4ZHl3bGxmdDRvbWcyOTVjY2F1MSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/WtUK5I9TbWiRcGrVZh/giphy.gif)")
		# st.markdown("![Alt Text](https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExbGZmc2Z2d2pwM25pODlodXpzem0weXBzeXVxMjdiOG9yYnRmbWRmNyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/h8j1vnUMElormqm8E8/giphy.gif)")

		st.write(""" 
		PulmoAID -  Enabling AI-based diagnostics for lung cancer using advanced multimodal feature fusion approach.
		""".strip())

		st.code(body=""" 
DATA SUMMARY STATISTICS 

Training and Testing Dataset
	1A (Positive Patients) - 310
	1G (NEgative Patients) - 269
	
Validation (Field Testing) Dataset - 
	2A (Positive Patients) - 312
	2G (NEgative Patients) - 184
		""".strip(), language=None)

		st.write(""" 
Model Summary - 

This model integrates multimodal feature fusion to detect lung cancer from CT scan images. 
It employs a pretrained VGG-16 network for feature extraction, capturing deep spatial representations from the input images. 
These extracted features are then processed through a fully connected neural network (FCNN), 
serving as a fusion layer to integrate and refine the learned representations. 
Finally, the fused features are passed to a logistic regression classifier, which performs binary classification to 
predict the likelihood of lung cancer. This architecture effectively combines deep learning-based feature
extraction with traditional classification techniques to enhance diagnostic accuracy.
		""".strip())


	with images_clinical:
		uploaded_files = st.file_uploader(label='Upload Scans', accept_multiple_files=True, type=["jpg", "jpeg", "png"])
		submit = st.toggle(
			label='Generate SHAP Plot (Please upload CT Scans First)' if uploaded_files == [] \
					else "Generate SHAP Plot", 
			disabled=True if uploaded_files == [] else False)

		if uploaded_files != []: st.image(uploaded_files[0], use_column_width=True, caption=f'Image01_Current')
			
		if uploaded_files != [] and submit:
			shap_plot = generate_shap_plot(base=csvdata, subject=state.selected_subject)
			st.pyplot(shap_plot, use_container_width=True)


	with diagnostics:
		st.write(""" 
		Comparison of current analysis with the last diagnostics in terms of
		probability key factors that are different.
		""".strip())

		submit = st.toggle(
			label='Generate Fusion Model Prediction (Please upload CT Scans First)' if uploaded_files == [] \
					else "Generate Fusion Model Prediction", 
			disabled=True if uploaded_files == [] else False)

		if uploaded_files != [] and submit:
			nameset = set()

			for file in uploaded_files:
				name = file.name
				nameset.add(name.split('_')[0])
				try:
					image = Image.open(file).convert("RGB")
					state.pil_images.append(image)

				except Exception as e:
					st.error(f"Error processing '{file.name}': {e}")
				
			if len(nameset) > 1:
				st.error('Input files are of different subjects, please give images for one subject only.')

			else:
				current_subject = nameset.pop()
				with st.spinner(text='Running Model...'):
					features = Manager.extract_features(imagelist=state.pil_images)
					outcome = generate_outcome(features, current_subject, state.model_selection)
					st.markdown(outcome)
					# st.image(uploaded_files[0], use_column_width=True, caption=f'Image01_{current_subject}')

		else:
			state.selected_subject = state.subject_selection
		
		edited_data = {}
		original_columns = csvdata.columns.tolist()
		c1, c2, c3 = st.columns(3)
	
		original = csvdata[csvdata['Subject'] == int(state.selected_subject)]

		def process_data(section_name, columns):
			"""Handles editing and storing modified data while preserving structure."""
			slice_df = csvdata[['Subject'] + columns]
			data_df = slice_df[slice_df['Subject'] == int(state.subject_selection)].T
			data_df.columns = ['Value']

			# Editable dataframe
			edited_df = st.data_editor(data_df, use_container_width=True)

			# Store edited values while avoiding duplicate 'Subject' columns
			edited_df = edited_df.T
			edited_df = edited_df.drop(columns=['Subject'], errors='ignore')
			edited_df.insert(0, 'Subject', state.subject_selection)  # Ensure 'Subject' is the first column
			
			edited_data[section_name] = edited_df

		with c1:
			if clinical_data:
				st.write('Clinical Data')
				process_data('Clinical', clinical)

		with c2:
			if demographic_data:
				st.write('Demographic Data')
				process_data('Demographic', demographic_cols)

		with c3:
			if smoking_history:
				st.write('Smoking History')
				process_data('Smoking History', smoking_hist)

		if edited_data:
			final_edited_df = pd.concat(edited_data.values(), axis=1)

			# Remove duplicate columns (keeping the first occurrence)
			final_edited_df = final_edited_df.loc[:, ~final_edited_df.columns.duplicated()]
			final_edited_df = final_edited_df.reindex(columns=original_columns, fill_value=None)
			final_edited_df = final_edited_df.fillna(original.set_index('Subject').loc[state.selected_subject])

			# st.write("Edited Data (Preserved Column Order, No Missing Values):")
			# st.dataframe(final_edited_df)

			new_pred = st.toggle('Generate New Prediction')
			if new_pred:
				new_X = final_edited_df[feature_cols + demographic_cols + smoking_hist + llm_sent]
				new_results = generate_outcome(subject=state.selected_subject, classifier=state.model_selection, full_row=new_X)
				st.markdown(new_results)



	with ai:
		state.llm.set_prompt(fr'''
You are an intelligent AI mdeical assistant.
Refer to the patient data given below (patient is referred to as "Subject"). It is related to a lung cancer study.
					   
{llmdata[llmdata['Subject'] == int(state.subject_selection)][demographic_cols + smoking_hist + clinical + llm_sent + ['lung_cancer']].to_dict(orient='records')}

Some fields that do have a clear description are described below - 
bioplc - Had a biopsy related to lung cancer?
biop0 - Had a biopsy related to positive screen?
proclc - Had any procedure related to lung cancer?
can_scr - Result of screen associated with the first confirmed lung cancer diagnosis Indicates whether the cancer followed a positive negative, or missed screen, or whether it occurred after the screening years.
0="No Cancer", 1="Positive Screen", 2="Negative Screen", 3="Missed Screen", 4="Post Screening"
canc_free_days - Days until the date the participant was last known to be free of lung cancer. 
llm_sentiment - AI generated sentiment variable for cancer likeliness from 0 - 10.
lung_cancer - Actual clinical test outcome for lung cancer (0 = negative, 1 = positive)

Based on this data, a doctor will be interacting with you and ask you some questions. Answer these questions. 
Answer them as per your knowledge and understanding.
If any question is unrelated to lung cancer or the medical field in general, respectfully decilne to answer that question.
''')

		if "chat_history" not in state: state.chat_history = []

		sample_text = random.choice([
					"Summarize this patient for me...",
					"Tell me more about this subject...",
					"Explain this patient's smoking history in detail..."
					])
		
		user_input = st.chat_input(placeholder=sample_text)

		if user_input:
			state.chat_history.append({"role":"User", "content":user_input})
			response = state.llm.ask(user_input)
			state.chat_history.append({"role":"AI", "content":response})
			# st.rerun()

		for message in state.chat_history:
			with st.chat_message(message["role"]):
				st.markdown(message["content"])

		# Display all existing messages
		
		# for message in state.chat_history:
		# 	with st.chat_message(message["role"]):
		# 		st.markdown(message["content"])

		# # Set up placeholder for new responses
		# response_placeholder = st.empty()

		# # Sample placeholder texts
		# sample_text = random.choice([
		# 	"Summarize this patient for me...",
		# 	"Tell me more about this subject...",
		# 	"Explain this patient's smoking history in detail..."
		# ])

		# user_input = st.chat_input(placeholder=sample_text)

		# # Process new input
		# if user_input:
		# 	# Add and display user message immediately
		# 	state.chat_history.append({"role": "User", "content": user_input})
		# 	with st.chat_message("User"):
		# 		st.markdown(user_input)
			
		# 	# Generate and display AI response
		# 	with st.chat_message("AI"):
		# 		with st.spinner("Thinking..."):
		# 			response = state.llm.ask(user_input)
		# 			st.markdown(response)
			
		# 	# Add AI response to history
		# 	state.chat_history.append({"role": "AI", "content": response})


def patient_page(patient_id:str):
	global csvdata, llmdata

	state.subject = patient_id
	csv_row = csvdata[csvdata['Subject'] == int(patient_id)]
	llm_row = llmdata[llmdata['Subject'] == int(patient_id)]

	with st.sidebar:
		st.header(body='Patient Portal')
		st.divider()
		st.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
		st.write(f'Welcome {state.subject}')

		# state.show_scans = st.button(label='Load Scans')
		# state.llm_temp = st.slider(label='LLM Temperature', min_value=0.00, max_value=1.00)
		# state.sys_prompt = st.text_area(label='System Prompt')

		show_hist = st.toggle('Show History')
		show_reports = st.toggle('View Results')

		notes = st.text_area('Doctor\'s Notes')
		observations = st.text_area('My Observations')

		logout = st.button(label='Logout', use_container_width=True)

		if logout:
			state.chat_history = []
			state.login = False
			st.rerun()

	st.title('Patient Dashboard')
	st.divider()

	diagnostics, history, ai = st.tabs(['Diagnostics', 'My History', 'Talk To VDoctor'])

	with diagnostics:
		if show_reports:
			shap = generate_shap_plot(base=csvdata, subject=patient_id)
			st.pyplot(fig=shap, use_container_width=True)

	with history:
		
		if show_hist:
			col1, col2 = st.columns(2)

			with col1:
				st.write('Demographic History')
				row_dm = csvdata[csvdata['Subject'] == int(state.subject)]
				slice_dm = row_dm[demographic_cols].T  
				slice_dm.columns = ['Data'] 
				st.dataframe(data=slice_dm, use_container_width=True)

			with col2:
				st.write('Smoking History')
				row_sm = csvdata[csvdata['Subject'] == int(state.subject)]
				slice_sm = row_sm[smoking_hist].T  
				slice_sm.columns = ['Data'] 
				st.dataframe(data=slice_sm, use_container_width=True)



	with ai:
		state.llm.set_prompt(f'''
You are a helpful AI doctor.
Your task is to respond to the patient's queries to the best of your knowledge.
Refer to patient info given below.
{llmdata[llmdata['Subject'] == int(patient_id)][demographic_cols + smoking_hist + clinical + llm_sent + treatments + ['lung_cancer']].to_dict(orient='records')}

					  '''.strip())

		if "chat_history" not in state: state.chat_history = []

		for message in state.chat_history:
			with st.chat_message(message["role"]):
				st.markdown(message["content"])

		user_input = st.chat_input(placeholder="Summarize this patient for me...")

		if user_input:
			state.chat_history.append({"role":"User", "content":user_input})
			response = state.llm.ask(user_input)
			state.chat_history.append({"role":"AI", "content":response})
			# st.rerun()


def main():
	
	if 'login' not in state:
		state.login = False
	if 'user' not in state:
		state.user = None
	
	if not state.login:
		st.image(image=logo_img, use_column_width=True)
		st.title('Login')
		
		username = st.text_input(label='Username/Patient ID')
		password = st.text_input(label='Password', type='password')
		
		col1, col2 = st.columns(2)
		
		with col1:
			if st.button("Doctor", use_container_width=True):
				state.user = "Doctor"
		
		with col2:
			if st.button("Patient", use_container_width=True):
				state.user = "Patient"
		
		if username and password and state.user:
			if state.user == "Doctor" and username == st.secrets["keys"]["username"] and password == st.secrets["keys"]["password"]:
				state.login = True
				state.user = "Doctor"
				# st.rerun()

			elif state.user == "Patient" and username.strip().isnumeric():
				tmp = int(username.strip())
				
				if tmp in state.subject_list and password == st.secrets["password"]:
					state.login = True
					state.subject = username.strip()
					# st.rerun()

				else:
					st.error("Invalid Patient ID or password")
			else:
				st.error("Invalid credentials or user type selection!")
	
	elif state.login and state.user == "Doctor":
		if 'chat_history' not in state: state.chat_history = []
		doctor_page()
	
	elif state.login and state.user == "Patient":
		if 'chat_history' not in state: state.chat_history = []
		# state.scans = load_image(str(state.subject))
		patient_page(state.subject)


if __name__ == "__main__":
	csvdata = utilloader('classifier_csv')
	llmdata = utilloader('llm_csv')
	state.llm = utilloader('llm')
	Manager = utilloader('manager')

	state.subject_list = list(csvdata['Subject'])
	# state.login = True
	# state.user = 'Doctor'
	# patient_page('100158')
	
	main()