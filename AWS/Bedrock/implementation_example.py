# %% [markdown]
# ## Demonstração com o Amazon Bedrock

# %% [markdown]
# ### 1 - Bedrock Setup
#
# Este notebook pode ser executado no Amazon SageMaker Studio, utilizando o Kernel Data Science 3.0.

# %%
# %pip install --no-build-isolation --force-reinstall \
#     "boto3>=1.28.57" \
#     "awscli>=1.29.57" \
#     "botocore>=1.31.57"

# %%
# In case when it's necessary to restart kernel
#import IPython

#IPython.Application.instance().kernel.do_shutdown(True)

# # %%
# %pip install langchain==0.0.309

# # %%
# %pip install pypdf

# # %%
# %pip install "faiss-cpu>=1.7,<2" sqlalchemy --quiet

# %%
#%pip freeze | grep boto
#%pip freeze | grep aws
#%pip freeze | grep langchain

# %% [markdown]
# O ambiente onde o notebook será executado precisa ter permissões na conta AWS para executar chamadas na API do Amazon Bedrock

# %%
import boto3
import numpy as np

boto_session = boto3.Session()
credentials = boto_session.get_credentials()

# %%
bedrock_models = boto3.client('bedrock')
bedrock_models.list_foundation_models()

# %% [markdown]
# Criando uma runtime para executar chamadas para os modelos fundacionais

# %%
bedrock = boto3.client('bedrock-runtime')

# %%
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

bedrock_embeddings = BedrockEmbeddings(client=bedrock, model_id="amazon.titan-embed-text-v1")
llm = Bedrock(model_id="anthropic.claude-v2", client=bedrock, model_kwargs={'max_tokens_to_sample':300})

# %% [markdown]
# ### 2 - Lendo um arquivo PDF com a LGPD e entendendo os embeddings

# O PDF com o texto foi gerado com base [neste link](https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/L13709compilado.htm).

# %%
import glob

data_path = './data/'
data_path_files = data_path + '*.pdf'

pdf_files =  glob.glob(data_path_files)
pdf_files

# %% [markdown]
# Quebrando o arquivo PDF em menores blocos de texto

# %%
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader

loader = PyPDFDirectoryLoader(data_path)

documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    #chunk_size = 1000,
    chunk_size = 500,
    chunk_overlap  = 100,
)

docs = text_splitter.split_documents(documents)

# %%
avg_doc_length = lambda documents: sum([len(doc.page_content) for doc in documents])//len(documents)
avg_char_count_pre = avg_doc_length(documents)
avg_char_count_post = avg_doc_length(docs)
print(f'Average length among {len(documents)} documents loaded is {avg_char_count_pre} characters.')
print(f'After the split we have {len(docs)} documents more than the original {len(documents)}.')
print(f'Average length among {len(docs)} documents (after split) is {avg_char_count_post} characters.')

# %% [markdown]
# Chamando a API pura do Bedrock para demonstração, gerando um embedding e visualizando o resultado.

# %%
import json


def create_embedding_bedrock(text, bedrock_client):
    payload = {"inputText": f"{text}"}
    body = json.dumps(payload)
    modelId = "amazon.titan-embed-g1-text-02"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_client.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    embedding = response_body.get("embedding")
    return embedding

# %%
sample_embedding = create_embedding_bedrock(docs[1].page_content, bedrock)
print(f"The embedding vector has {len(sample_embedding)} values\n{sample_embedding[0:3]+['...']+sample_embedding[-3:]}")

# %%
docs[1]

# %% [markdown]
# ### 3 - Lendo o arquivo e gerando uma base de vetores (em memória)

# %%
from langchain.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

vectorstore_faiss = FAISS.from_documents(
    docs,
    bedrock_embeddings,
)

wrapper_store_faiss = VectorStoreIndexWrapper(vectorstore=vectorstore_faiss)

# %% [markdown]
# ### 4 - Executando perguntas e obtendo respostas do LLM

# %%
question = "Quem é o titular de um dado? Mostre-me a referência no contexto"

# %%
answer = wrapper_store_faiss.query(question=question, llm=llm)
print(answer)

# %%
question = "Quem é o titular de um dado? Você consegue explicar para um leigo que não conhece de leis?"

# %%
answer = wrapper_store_faiss.query(question=question, llm=llm)
print(answer)

# %% [markdown]
# ### 5 - Outros exemplos

# %%
question = "O número de telefone e o endereço de IP de acesso à Internet são considerados dados pessoais?"
answer = wrapper_store_faiss.query(question=question, llm=llm)
print(answer)

# %%
question = "Quais são os dados sensíveis de uma pessoa?"
answer = wrapper_store_faiss.query(question=question, llm=llm)
print(answer)

# %% [markdown]
# ---

# %%
question = "Qual o papel do controlador?"
answer = wrapper_store_faiss.query(question=question, llm=llm)
print(answer)

# %%
question =  """ Estou colocando sistema de comandas no meu bar. Para permitir que as pessoas consumam nas
                comandas preciso fazer um cadastro prévio que irá conter os seguintes dados: nome completo, data de
                nascimento, CPF, identidade, nome dos pais, endereço, e-mail e estado civil.
                Desta forma, estarei violando a LGPD? Explique de forma bem simples """

answer = wrapper_store_faiss.query(question=question, llm=llm)
print(answer)

# %%
question = "Quem pode realizar o tratamento dos dados?"
answer = wrapper_store_faiss.query(question=question, llm=llm)
print(answer)
