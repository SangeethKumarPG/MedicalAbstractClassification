import streamlit as st
import tensorflow as tf 
import tensorflow_hub as hub 
from tensorflow.keras import mixed_precision


#creating a function to split document into sentences
def split_sentences(text):
    return text.split('. ')

#creating a function to return the position of the sentence
def position(sentence_list):
    index_list = []
    for index, sentence in enumerate(sentence_list):
        index_list.append(index)
    return index_list

# creating a function to covert the sentences to character tokens
def convert_char_tokens(sentence_list):
    char_list = []
    for token in sentence_list:
        char_list.append(" ".join(list(token)))

    return char_list

#creating a function to get the total lines
def get_total_lines(text):
    total_abstract_lines = []
    for sentence in text.split('. '):
        total_abstract_lines.append(len(text.split('. ')))
        
    return total_abstract_lines

# Combining the functions to generate a single tensor with position, total_lines, sentences, characters
def generate_tensor_from_text(text):
    sentence_list = split_sentences(text)
    position_list = position(sentence_list)
    characters = convert_char_tokens(sentence_list)
    total_lines = get_total_lines(text)
    
    position_encoded = tf.one_hot(position_list, depth=15)
    
    total_lines_encoded = tf.one_hot(total_lines, depth=20)
    return (position_encoded, total_lines_encoded, tf.constant(sentence_list), tf.constant(characters))

def generate_results_from_probs(probs, text):
    target_classes = ['BACKGROUND', 'CONCLUSIONS', 'METHODS', 'OBJECTIVE', 'RESULTS']
    preds = tf.argmax(probs, axis=1)
    pred_classes = [target_classes[i] for i in preds]
    #creating a dictionary to store predictions
    prediction_dict = dict()
    sentence_list = split_sentences(text)
    #visualizing the prediction
    for i, line in enumerate(sentence_list):
        print(f"{pred_classes[i]}: {line}\n")
        if pred_classes[i] in prediction_dict:
            prediction_dict[pred_classes[i]] = prediction_dict[pred_classes[i]] +". "+line
        else:
            prediction_dict[pred_classes[i]] = line
    return prediction_dict


st.header("Medical Abstract Classification")
st.write("Convert a bulk medical abstract into sections of'BACKGROUND', 'CONCLUSIONS', 'METHODS', 'OBJECTIVE', 'RESULTS', making it easier to read")

# sample_text = "Nutritional support of surgical and critically ill patients has undergone significant advances since 1936 when Studley demonstrated a direct relationship between pre-operative weight loss and operative mortality. The advent of total parenteral nutrition followed by the extraordinary progress in parenteral and enteral feedings, in addition to the increased knowledge of cellular biology and biochemistry, have allowed clinicians to treat malnutrition and improve surgical patient's outcomes. We reviewed the literature for the current status of perioperative nutrition comparing parenteral nutrition with enteral nutrition. In a surgical patient with established malnutrition, nutritional support should begin at least 7-10 days prior to surgery. Those patients in whom eating is not anticipated beyond the first five days following surgery should receive the benefits of early enteral or parenteral feeding depending on whether the gut can be used. Compared to parenteral nutrition, enteral nutrition is associated with fewer complications, a decrease in the length of hospital stay, and a favorable cost-benefit analysis. In addition, many patients may benefit from newer enteral formulations such as Immunonutrition as well as disease-specific formulations."

with st.form("text_input_form"):
    sample_text = st.text_area(label="Paste the medical abstract here")
    submitted = st.form_submit_button("Submit")

if submitted:
    test_tensor = generate_tensor_from_text(sample_text)
    mixed_precision.set_global_policy('mixed_bfloat16')
    model = tf.keras.models.load_model("loaded_model_format")
    probs = model.predict(test_tensor)
    pred_dict = generate_results_from_probs(probs, sample_text)
    print(pred_dict)
    for (key, value) in pred_dict.items():
        st.write(f"{key}:\n{value}\n")