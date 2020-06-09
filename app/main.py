import plac
import random
import spacy
import sys
from spacy.util import minibatch, compounding
from pathlib import Path
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/health")
def hello():
	return jsonify({'text':'server is up !'})

@app.route('/extractNetAssetValueFromText', methods = ['POST'])
def upload_text():
	print(request.data,file = sys.stderr)
	return jsonify(nameEntityExtraction(request.data))

#Data for training Model
TRAIN_DATA = [

	(
	 'Canyon Value Realization MAC 18, Ltd.Performance net of management fees and applicable incentive allocation as of 01/31/2019 Net Performance january MTD 2019 YTD AUM +5.25% $165 million',
	 {"entities":[(173,185,"NAV")]}
	),
	 
	 (
	 "Assets Under Management (approx. as of 03/01/19) Estimated Master Fund Net Assets: $125 million",
	 {"entities":[(83,95,"NAV")]}
	 ),
	 (
	 "jan 0.27 -2.22 0.26 feb -0.10 are exceptions",
	 {"entities":[(4,8,"NA"),(9,14,"NA"),(15,19,"NA"),(24,29,"NA")]}
	 ),
	 (
	 "$125 million Assets Under Management (approx. as of 03/01/19) Estimated Master Fund Net Assets for AIR UTOPIA LTD",
	 {"entities":[(0,12,"NAV")]}
	 ),
	 (
	 "$565 million Assets Under Management (approx. as of 03/02/19) Estimted Master Fund Net Assets for FSO GMBH& CO.KG",
	 {"entities":[(0,12,"NAV")]}
	 ),
	 (
	 "If $A or $A: then dont take these values",
	 {"entities":[(3,5,"INCNAV"),(9,12,"INCNAV")]}
	 ),
	 (
	 "Toal Aum is $2000.1 m and fund Aum is $4.2/$24.3 m",
	 {"entities":[(12,21,"NAV"),(38,50,"NAVTYPE")]}
	 ),
	 (
	 "2018 is an year type and is not nav value",
	 {"entities":[(0,4,"YEAR")]}
	 ),
	 (
	 "18% or 19.1% is a type of percentage value and is not a nav value",
	 {"entities":[(0,3,"PERCENT"),(7,12,"PERCENT")]}
	 ),
]
	 

@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)

def nameEntityExtraction(text, model=None, output_dir=None, n_iter=100):
	"""Load the model, setup the pipeline and train the entity recognizer"""
	if model is not None:
		nlp = spacy.load(model) # load existing spaCy model
		print("Model Loaded '%s'" % model)
	else:
		nlp = spacy.blank("en") #create blank language class
		print("Created blank Model")
		
		
	# create the built-in pipeline components and add them to the pipeline
	# npm.create_pipe works for built-ins that are registered with spaCy
	if "ner" not in nlp.pipe_names:
		ner= nlp.create_pipe("ner")
		nlp.add_pipe(ner,last=True)
		print("Create Pipe")
	# otherwise, get it so we can add labels
	else:
		ner = nlp.get_pipe("ner")
		print("get pipe")
		
	# add labels
	
	for _, annotations in TRAIN_DATA:
		for ent in annotations.get("entities"):
			ner.add_label(ent[2])
	
	# get names of other pipes to disable them during training
	other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
	with nlp.disable_pipes(*other_pipes): #only tarin NER
		# reset and initialize the weights randomly - but only if we'read
		# training a new model
		if model is None:
			nlp.begin_training()
			print("Begin Training")
		for itn in range(n_iter):
			print("Iteration -> ",itn)
			random.shuffle(TRAIN_DATA)
			losses ={}
			#batch up the examples using spaCy's minbatch
			batches = minibatch(TRAIN_DATA, size=compounding(8.0,64.0,1.001))
			for batch in batches:
				texts, annotations= zip(*batch)
				nlp.update(
					texts, # batch of texts
					annotations, # batch of annotations
					drop = 0.5, #droupout - make it harder to memorise data
					losses = losses,
				)
		my_objects=[]
		TEST_DATA = text.decode("utf-8")
		print(TEST_DATA)
				
		nlp.to_disk("./models")
		print("Model saved and/or Updated")
		 
		# test the saved model
		print("Loading Model")
		nlp2 = spacy.load("./models")
		 
		doc = nlp2(TEST_DATA)
		for ent in doc.ents:
			print(ent.text + " " + ent.label_)
			my_objects.append({'Text':ent.text,'Type':ent.label_})
	return my_objects
			
			
			
# program to find position of word in given string
#word = '18% or 19.1% is a type of percentage value and is not a nav value'
#search = '19.1%'
#result = word.find(search) 
#print ("start", result )
#print ("end",result + len(search)) 
 
