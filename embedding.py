import pandas as pd # dataframe manipulation
import numpy as np # linear algebra
from sentence_transformers import SentenceTransformer


model_data = pd.read_csv("model_data.csv", sep = ",")


# -------------------- First Step --------------------
def compile_text(x):

    text =  f"""Crime description: {x['Crime description']},
                Victim's age: {x["Victim's age"]},
                Victim's sex: {x["Victim's sex"]},
                Victim's race: {x["Victim's race"]}
            """

    return text

sentences = model_data.apply(lambda x: compile_text(x), axis=1).tolist()

# -------------------- Second Step --------------------

model = SentenceTransformer(r"sentence-transformers/paraphrase-MiniLM-L6-v2")
output = model.encode(sentences=sentences,
         show_progress_bar=True,
         normalize_embeddings=True)

model_data_embedding = pd.DataFrame(output)
model_data_embedding

model_data_embedding.to_csv('embedded_model_data.csv', index=False)
