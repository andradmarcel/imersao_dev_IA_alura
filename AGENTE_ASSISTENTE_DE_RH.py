# Bibliotecas padrão do Python
import os  # Para interagir com o sistema operacional, como acessar variáveis de ambiente.
from pathlib import Path  # Para manipular caminhos de arquivos de forma orientada a objetos.
from typing import Literal, List, Dict, TypedDict, Optional  # Para anotações de tipo mais específicas.
from IPython.display import display, Image

# Bibliotecas de terceiros (instaladas via pip)
from dotenv import load_dotenv  # Para carregar variáveis de ambiente de um arquivo .env.
from pydantic import BaseModel, Field  # Para definir a estrutura de saída e validar dados.

# Bibliotecas do LangChain
from langchain.chains.combine_documents import create_stuff_documents_chain  # Para criar uma cadeia que "recheia" o prompt com documentos.
from langchain_community.document_loaders import PyMuPDFLoader  # Para carregar e extrair texto de arquivos PDF.
from langchain_community.vectorstores import FAISS  # Para criar um banco de vetores em memória.
from langchain_core.messages import SystemMessage, HumanMessage  # Para estruturar as mensagens da conversa com o modelo.
from langchain_core.prompts import ChatPromptTemplate  # Para criar templates de prompts para o chat.
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings  # Para interagir com os modelos do Google (Gemini).
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Para dividir textos longos em pedaços menores (chunks).
from langgraph.graph import StateGraph, START, END

# Funções auxiliares de formatação
from formatadores import formatar_citacoes


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

TRIAGEM_PROMPT = (
    "Você é um triador de Service Desk para políticas internas da empresa Carraro Desenvolvimento. "
    "Dada a mensagem do usuário, retorne SOMENTE um JSON com:\n"
    "{\n"
    '  "decisao": "AUTO_RESOLVER" | "PEDIR_INFO" | "ABRIR_CHAMADO",\n'
    '  "urgencia": "BAIXA" | "MEDIA" | "ALTA",\n'
    '  "campos_faltantes": ["..."]\n'
    "}\n" 
    "Regras:\n"
    '- **AUTO_RESOLVER**: Perguntas claras sobre regras ou procedimentos descritos nas políticas '
    '(Ex: "Posso reembolsar a internet do meu home office?", "Como funciona a política de alimentação em viagens?").\n'
    '- **PEDIR_INFO**: Mensagens vagas ou que faltam informações para identificar o tema ou contexto'
    '(Ex: "Preciso de ajuda com uma política", "Tenho uma dúvida geral").\n'
    '- **ABRIR_CHAMADO**: Pedidos de exceção, liberação, aprovação ou acesso especial,'
    'ou quando o usuário explicitamente pede para abrir um chamado'
    '(Ex: "Quero exceção para trabalhar 5 dias remoto.", "Solicito liberação para anexos externos.",'
    '"Por favor, abra um chamado para o RH.").'
    "Analise a mensagem e decida a ação mais apropriada."
)

class TriagemOut(BaseModel):
    decisao: Literal["AUTO_RESOLVER", "PEDIR_INFO", "ABRIR_CHAMADO"]
    urgencia: Literal["BAIXA", "MEDIA", "ALTA"]
    campos_faltantes: List[str] = Field(default_factory=list)

llm_triagem = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    api_key=GOOGLE_API_KEY
)

triagem_chain = llm_triagem.with_structured_output(TriagemOut)

def triagem(mensagem: str) -> Dict:
    saida: TriagemOut = triagem_chain.invoke([SystemMessage(content=TRIAGEM_PROMPT), HumanMessage(content=mensagem)])
    return saida.model_dump() 


    

docs = []
pasta_politicas = Path("RAG_Politicas")

for documento in pasta_politicas.glob("*.pdf"):
    try:
        loader = PyMuPDFLoader(str(documento)) 
        docs.extend(loader.load())
        print(f"Carregado com sucesso o arquivo: {documento.name}") 
    except Exception as e:  
        print(f"Erro ao carregar arquivo: {documento.name}: {e}")
        
# print (f"Total de documentos: {len(docs)} ")        

splitter = RecursiveCharacterTextSplitter(chunk_size= 300, chunk_overlap= 30)

chunks = splitter.split_documents(docs)     
   
# for chunk in chunks:       #NAO NECESSARIO, APENAS PARA VER CADA UM DOS CHUNKS 
#     print(chunk)
#     print(100*"-")

embeddings = GoogleGenerativeAIEmbeddings(
    model= "models/gemini-embedding-001",
    google_api_key= GOOGLE_API_KEY
)

vectorstore = FAISS.from_documents(chunks, embeddings)

retriever = vectorstore.as_retriever(search_type= "similarity_score_threshold", 
                                     search_kwargs={"score_threshold":0.3, "k": 4})

prompt_rag = ChatPromptTemplate.from_messages([
    ("system",
     "Você é um Assistente de Políticas Internas (RH/IT) da empresa Carraro Desenvolvimento. "
     "Responda SOMENTE com base no contexto fornecido. "
     "Se não houver base suficiente, responda que não esta claro nas politicas da empresa \n"
     "e que precisa de mais informacoes."),

    ("human", "Pergunta: {input}\n\nContexto:\n{context}")
    
])

document_chain = create_stuff_documents_chain(llm_triagem, prompt_rag)


def perguntar_politica_rag(pergunta: str) -> dict:
    docs_relacionados = retriever.invoke(pergunta)
    
    if not docs_relacionados:
        return {"answer": "Não esta claro nas politicas da empresa.",
                "citacoes": [],
                "contexto_encontrado": False
                }

    answer = document_chain.invoke({"input": pergunta,
                                    "context": docs_relacionados})
    
    txt = (answer or "").strip()
    
    if txt.rstrip(".!?") == "Não esta claro nas politicas da empresa.":
        return {"answer": "Não esta claro nas politicas da empresa.",
                "citacoes": [],
                "contexto_encontrado": False
                }
        
    return {    "answer": txt,
                "citacoes": formatar_citacoes(docs_relacionados, pergunta),
                "contexto_encontrado": True
                }    
    

# perguntas_teste = [
#           "Posso reembolsar a internet?",
#           "Quero mais 5 dias de trabalho remoto. Como faço?",
#           "Posso reembolsar cursos ou treinamentos da Alura?",
#           "Quantas capivaras tem no Rio Pinheiros?",
#           "To com uma duvida sobre o vale alimentacao!",
#           "Preciso de mais alguns dias remoto e tambem quero reembolsar a internet, como eu faço?"
#           ]

# for msg_teste in perguntas_teste:
#     print(f"Pergunta: {msg_teste}\nResposta: {triagem(msg_teste)}\n")


# for msg_teste in perguntas_teste:
#     resposta = perguntar_politica_rag(msg_teste)
#     print(f"Pergunta: {msg_teste}")
#     print(f"Resposta: {resposta['answer']}")
#     print(100 * "-")
#     if resposta["contexto_encontrado"]:
#         print(f"Citações:")
#         for citacao in resposta["citacoes"]:
#             print(f"  - Documento: {citacao['documento']} (página {citacao['pagina']})")
#             print(f"    Trecho: \"...{citacao['trecho']}...\"")
#     print(100 * "-")
        

class AgentState(TypedDict, total = False):
    pergunta: str
    triagem: dict
    resposta: Optional[str]
    citacoes: List[dict]
    rag_sucesso: bool
    acao_final: str
    
def node_triagem(state: AgentState) -> AgentState:
    print("Executando nó triagem...") 
    return {"triagem": triagem(state["pergunta"])}  

def node_auto_resolver(state: AgentState) -> AgentState:
    print("Executando nó auto_resolver...") 
    resposta_rag = perguntar_politica_rag(state["pergunta"])
    
    update: AgentState = {
        "resposta": resposta_rag["answer"],
        "citacoes": resposta_rag.get("citacoes", []),
        "rag_sucesso": resposta_rag["contexto_encontrado"],
    }
    
    if resposta_rag["contexto_encontrado"]:
        update["acao_final"] = "AUTO_RESOLVER"
    return update 

def node_pedir_info(state: AgentState) -> AgentState:
    print("Executando nó pedir info...")    
    faltantes = state["triagem"].get("campos_faltantes", [])     
    detalhe = ",".join(faltantes) if faltantes else "Tema e contexto especifico"
    return {
        "resposta": f"Para avançar, preciso de detalhes sobre: {detalhe}",
        "citacoes": [],
        "acao_final": "PEDIR_INFO"
    }
def node_abrir_chamado(state: AgentState) -> AgentState:
    print("Executando nó de abrir chamado...") 
    triagem = state["triagem"]
    return{
        "resposta": f"Abrindo chamado com urgencia {triagem["urgencia"]}. Descrição: {state["pergunta"][:150]}",
        "citacoes": [],
        "acao_final": "ABRIR_CHAMADO"
    }
    

KEYWORDS_ABRIR_CHAMADO = ["aprovação", "exceção", "liberação", "abrir ticket", "abrir chamado", "acesso especial"]

def decidir_pos_triagem(state: AgentState) -> str:
    print("Decidindo após a triagem...")
    decisao = state["triagem"]["decisao"]
    
    if decisao == "AUTO_RESOLVER": return "auto"
    if decisao == "PEDIR_INFO": return "info"
    if decisao == "ABRIR_CHAMADO": return "chamado"
    

def decidir_pos_auto_resolver(state: AgentState) -> str:
    print("Decidindo após a busca (RAG)...")
    if state.get("rag_sucesso"): 
        print("--> RAG teve sucesso. Finalizando.")
        return "ok"

    state_da_pergunta = (state["pergunta"] or "").lower()
    
    if any(k in state_da_pergunta for k in KEYWORDS_ABRIR_CHAMADO): 
        print("Rag falhou, mas foram encontradas keywords de abertura de ticket. Abrindo...")
        return "chamado"
    
    print("Rag falhou, sem keywords, vou pedir mais informações...")
    return "info"
    
    
workflow = StateGraph(AgentState)

workflow.add_node("triagem", node_triagem)
workflow.add_node("auto_resolver", node_auto_resolver)
workflow.add_node("pedir_info", node_pedir_info)
workflow.add_node("abrir_chamado", node_abrir_chamado)

workflow.add_edge(START, "triagem")
workflow.add_conditional_edges("triagem", decidir_pos_triagem, {
    "auto": "auto_resolver",
    "info": "pedir_info",
    "chamado": "abrir_chamado"
})
   
workflow.add_conditional_edges("auto_resolver", decidir_pos_auto_resolver, {
    "info": "pedir_info",
    "chamado": "abrir_chamado",
    "ok": END
})

workflow.add_edge("pedir_info", END)
workflow.add_edge("abrir_chamado", END)

grafo = workflow.compile()


try:
    print("Gerando a imagem do grafo...")
    # A primeira linha continua igual, ela gera os dados da imagem.
    graph_bytes = grafo.get_graph().draw_mermaid_png()

    # Define o nome do arquivo onde a imagem será salva.
    nome_arquivo_grafo = "fluxo_do_agente.png"

    # Abre (ou cria) um arquivo em modo de "escrita binária" (wb) e salva os dados da imagem nele.
    with open(nome_arquivo_grafo, "wb") as f:
        f.write(graph_bytes)

    # Imprime uma mensagem de sucesso no terminal.
    print(f"Visualização do grafo salva com sucesso em: '{nome_arquivo_grafo}'")

except Exception as e:
    # Se ocorrer um erro (ex: bibliotecas faltando), uma mensagem será exibida.
    print(f"Não foi possível gerar a imagem do grafo. Erro: {e}")
    
perguntas_teste = [
          "Posso reembolsar a internet?",
        #   "Quero mais 5 dias de trabalho remoto. Como faço?",
        #   "Posso reembolsar cursos ou treinamentos da Alura?",
        #   "É possível reembolsar certificações do Google Cloud?",
        #   "Posso obter o Google Gemini de graça?",
        #   "Qual é a palavra-chave da aula de hoje?",
        #   "Preciso de uma aprovação para comprar um software.",
        #   "Quantas capivaras tem no Rio Pinheiros?"
          ]

for msg_teste in perguntas_teste:
    resposta_final = grafo.invoke({"pergunta": msg_teste})
    triag = resposta_final.get("triagem", {})

    print(f"Pergunta: {msg_teste}")
    print(f"Decisao: {triag.get("decisao")}, Urgencia: {triag.get("urgencia")}, Acao final: {resposta_final.get("acao_final")}")
    print(f"Resposta: {resposta_final.get("resposta")}")
    
    if resposta_final.get("citacoes"):
        print("CITAÇÕES:")
        for citacao in resposta_final.get("citacoes"):
            print(f" - Documento: {citacao['documento']}, Página: {citacao['pagina']}")
            print(f"   Trecho: {citacao['trecho']}")

    print("------------------------------------")