import pandas as pd
from transformers import pipeline

# Load CSV data
data = pd.read_csv('NutritionalFacts_Fruit_Vegetables_Seafood.csv', encoding="ISO-8859-1")
print(data.head())

# Load pre-trained models
intent_classifier = pipeline('text-classification', model='distilbert-base-uncased')
ner_model = pipeline('ner', model='dbmdz/bert-large-cased-finetuned-conll03-english')
sentiment_analyzer = pipeline('sentiment-analysis')

def classify_intent(text):
    return intent_classifier(text)[0]

def extract_entities(text):
    return ner_model(text)

def analyze_sentiment(text):
    return sentiment_analyzer(text)[0]

text_example = "Tell me about the nutritional content of an apple."
print("Intent:", classify_intent(text_example))
print("Entities:", extract_entities(text_example))
print("Sentiment:", analyze_sentiment(text_example))

class Context:
    def __init__(self):
        self.context_data = {}

    def update_context(self, user_id, key, value):
        if user_id not in self.context_data:
            self.context_data[user_id] = {}
        self.context_data[user_id][key] = value

    def get_context(self, user_id, key, default=None):
        return self.context_data.get(user_id, {}).get(key, default)

user_context = Context()

def generate_response(user_id, text):
    intent = classify_intent(text)['label']
    entities = extract_entities(text)
    sentiment = analyze_sentiment(text)['label']

    if intent == 'nutritional_content':
        food_item = next((ent['word'] for ent in entities if ent['entity'] == 'B-FOOD'), None)
        if food_item:
            nutrition_info = data[data['food_name'].str.contains(food_item, case=False)]
            if not nutrition_info.empty:
                response = f"Nutritional content of {food_item}: {nutrition_info.to_dict(orient='records')[0]}"
            else:
                response = f"Sorry, I don't have information on {food_item}."
        else:
            response = "Please specify a food item."
    else:
        response = "I'm not sure how to respond to that."

    user_context.update_context(user_id, 'last_intent', intent)
    return response
