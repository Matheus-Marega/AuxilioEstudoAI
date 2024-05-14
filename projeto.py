
#Importacao de Bibliotecas que serao utilizadas
import streamlit as st
import numpy as np
import pandas as pd
import google.generativeai as genai

GOOGLE_API_KEY = "YOUR API KEY"  #Coloque sua Chave Api da Google
genai.configure(api_key=GOOGLE_API_KEY)

df = pd.read_csv("SEU ARQUIVO CSV")
df.columns = ["Topico", "Conteudo"]


model = "models/embedding-001"#Modelo de embedding que sera utilizado



def embed_fn(title, text): #Funcao que calculara os embeddings
  return genai.embed_content(model= model,
                                 content=text,
                                 title=title,
                                 task_type= "RETRIEVAL_DOCUMENT")["embedding"]

df["Embeddings"] = df.apply(lambda row: embed_fn(row["Topico"], row["Conteudo"]), axis=1)

def gerar_buscar_consulta(consulta, base, model): #Funcao que ira consultar o Dataframe e retornara Informacoes
  embedding_consulta = genai.embed_content(model= model,
                                 content=consulta,
                                 task_type= "RETRIEVAL_QUERY")["embedding"]

  produtos_escalares = np.dot(np.stack(df["Embeddings"]), embedding_consulta)

  indice = np.argmax(produtos_escalares)
  return df.iloc[indice]["Conteudo"]



generation_config = { #Parametros como o modelo se comportara
    "temperature": 0.5,
    "candidate_count": 1
}


def main(): 
    st.set_page_config(
        page_title="Teste AI", page_icon=";bird")
    st.header("Consultor de respostas")
    consulta = st.text_area("Faca sua pergunta") #Front-end da aplicacao

    if consulta:
        st.write("Gerando uma resposta baseado nos arquivos...")
        trecho = gerar_buscar_consulta(consulta, df, model)

        prompt = f"""Reescreva o texto de forma que nao adicione informacoes que nao estao no texto. 
Pergunta: {consulta}, Texto para ser reescrito: {trecho}""" #Engenharia de Prompt para o modelo reescrever a pergunta
        
        model_2 = genai.GenerativeModel("gemini-1.0-pro", generation_config=generation_config) #Modelo do Gemini
        response = model_2.generate_content(prompt) #Geracao de resposta com base no prompt

        print("Resposta: ",response.text)

        st.info(response.text)

if __name__ == "__main__":
    main()