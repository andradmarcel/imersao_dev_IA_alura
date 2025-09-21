# --- 1. IMPORTAÇÃO DAS BIBLIOTECAS ---
# Pense nesta seção como a nossa "caixa de ferramentas". Antes de começar a construir nosso agente,
# nós pegamos todas as ferramentas (bibliotecas) que vamos precisar.

# --- Ferramentas que já vêm com o Python ---
import os  # A ferramenta 'os' nos deixa conversar com o computador (sistema operacional), como por exemplo, para pegar segredos guardados.
from pathlib import Path  # A ferramenta 'Path' é um jeito inteligente de lidar com nomes de pastas e arquivos, para não nos perdermos.
from typing import Literal, List, Dict, TypedDict, Optional  # A ferramenta 'typing' nos ajuda a deixar "etiquetas" no nosso código,
                                                            # dizendo que tipo de coisa cada variável deve guardar (um texto, uma lista, etc.).
                                                            # Isso não muda como o código funciona, mas o deixa mais fácil de ler.
from IPython.display import display, Image # Ferramentas para mostrar imagens bonitas na tela, muito usadas em cadernos de código.

# --- Ferramentas que instalamos de fora (com 'pip') ---
from dotenv import load_dotenv  # Esta ferramenta lê um arquivo secreto chamado '.env' onde guardamos nossa chave de API.
from pydantic import BaseModel, Field  # Pydantic é como um "fiscal de qualidade". Usamos o 'BaseModel' para criar um molde
                                       # de como a resposta da IA deve ser, garantindo que ela sempre venha no formato certo.


# --- Ferramentas do LangChain (o cérebro da nossa operação) ---
from langchain.chains.combine_documents import create_stuff_documents_chain  # Uma função que cria uma "corrente" que pega vários documentos e os "enfia" (stuff) dentro de uma instrução para a IA.
from langchain_community.document_loaders import PyMuPDFLoader  # Um "carregador" especializado em abrir arquivos PDF e ler o texto dentro deles.
from langchain_community.vectorstores import FAISS  # Uma "estante de livros mágica" (Vector Store) que organiza nossos textos de um jeito super rápido para fazer buscas.
from langchain_core.messages import SystemMessage, HumanMessage  # "Balões de fala" para conversar com a IA. 'SystemMessage' é o que nós (o sistema) dizemos para a IA, e 'HumanMessage' é o que o usuário diz.
from langchain_core.prompts import ChatPromptTemplate  # Um "molde de carta" para escrever nossas instruções para a IA de forma organizada.
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings  # As ferramentas que nos conectam com os cérebros do Google (Gemini). Uma para conversar e outra para transformar texto em "códigos de significado" (embeddings).
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Uma "tesoura inteligente" que corta textos muito grandes em pedaços menores.
from langgraph.graph import StateGraph, START, END # A ferramenta mais importante! Com o 'StateGraph', nós desenhamos o "mapa de trabalho" do nosso agente, com começo ('START'), fim ('END') e todas as etapas no meio.


# --- 2. CONFIGURAÇÃO INICIAL E PREPARAÇÃO DO AMBIENTE ---
# A função `load_dotenv()` é chamada. Ela vai procurar o arquivo `.env` e carregar os segredos que estão lá para o nosso programa poder usar.
load_dotenv()
# Aqui, criamos uma "caixinha" (variável) chamada `GOOGLE_API_KEY`.
# O valor que colocamos dentro dela é o que a função `os.getenv("GOOGLE_API_KEY")` nos devolve.
# Essa função busca no ambiente do computador pelo segredo com o nome "GOOGLE_API_KEY".
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- 3. FERRAMENTAS DO AGENTE: TRIAGEM E RAG ---

# --- 3.1. FERRAMENTA DE TRIAGEM ---
# Criamos uma "caixinha" constante chamada `TRIAGEM_PROMPT`.
# Dentro dela, guardamos um texto longo que são as instruções para a IA fazer a triagem.
# O prompt de sistema para a tarefa de triagem. Define o papel e as regras para o LLM.
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

# Aqui, estamos desenhando um "molde" com Pydantic. É como um manual de instruções para a IA.
# O nome do nosso molde é `TriagemOut`.
class TriagemOut(BaseModel):
    # O molde diz que a resposta da IA DEVE ter uma "decisao", que só pode ser um desses três textos.
    decisao: Literal["AUTO_RESOLVER", "PEDIR_INFO", "ABRIR_CHAMADO"]
    # Também deve ter uma "urgencia", que só pode ser um desses três textos.
    urgencia: Literal["BAIXA", "MEDIA", "ALTA"]
    # E deve ter "campos_faltantes", que será uma lista de textos. Se a IA não encontrar nenhum,
    # `default_factory=list` garante que teremos uma lista vazia, e não um erro.
    campos_faltantes: List[str] = Field(default_factory=list)

# Agora, construímos o nosso robô de triagem. Criamos uma "caixinha" chamada `llm_triagem`.
# Dentro dela, colocamos um novo robô `ChatGoogleGenerativeAI` que acabamos de criar.
# Nós configuramos o robô para usar o modelo "gemini-1.5-flash" e damos a ele nossa chave secreta (`GOOGLE_API_KEY`) para funcionar.
llm_triagem = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0, # Temperatura 0 diz para o robô ser o mais lógico e menos criativo possível.
    api_key=GOOGLE_API_KEY
)

# Criamos uma "corrente" (chain) chamada `triagem_chain`.
# Essa corrente é o nosso robô `llm_triagem` usando um superpoder: `with_structured_output`.
# Esse superpoder força o robô a sempre dar respostas que se encaixam perfeitamente no nosso molde `TriagemOut`.
triagem_chain = llm_triagem.with_structured_output(TriagemOut)

# Função que encapsula a lógica de triagem.
# Aqui, definimos uma nova "receita" chamada `triagem`. Ela recebe uma `mensagem` (texto) como ingrediente.
def triagem(mensagem: str) -> Dict:
    # Dentro da receita, usamos nossa `triagem_chain` para trabalhar.
    # A função `invoke` manda para a IA as instruções (`TRIAGEM_PROMPT`) e a mensagem do usuário.
    # O resultado, que já vem no formato do nosso molde, é guardado na caixinha `saida`.
    # Invoca a cadeia com o prompt do sistema e a mensagem do usuário.
    saida: TriagemOut = triagem_chain.invoke([SystemMessage(content=TRIAGEM_PROMPT), HumanMessage(content=mensagem)])
    # Por fim, a receita devolve o conteúdo da caixinha `saida` de uma forma simples (um dicionário), usando `.model_dump()`.
    return saida.model_dump() 


    

docs = []
# Criamos uma caixinha `pasta_politicas` que guarda um "mapa" para a pasta chamada "politicas".
pasta_politicas = Path("RAG_Politicas")

# Usamos um laço `for` para olhar cada item que o mapa encontrar.
# `pasta_politicas.glob("*.pdf")` é como dizer: "No local que esse mapa aponta, me dê todos os arquivos que terminam com .pdf".
for documento in pasta_politicas.glob("*.pdf"):
    # O bloco `try...except` é uma rede de segurança. Tentamos fazer o que está no `try`.
    # Se der algum erro, em vez de o programa quebrar, ele pula para o `except` e nos avisa.
    try:
        loader = PyMuPDFLoader(str(documento)) 
        docs.extend(loader.load())
        print(f"Carregado com sucesso o arquivo: {documento.name}") 
    except Exception as e:  
        print(f"Erro ao carregar arquivo: {documento.name}: {e}")
        
# Divisão em Chunks para caber no limite de contexto do LLM.
# Criamos uma "tesoura inteligente" chamada `splitter`.
splitter = RecursiveCharacterTextSplitter(chunk_size= 300, chunk_overlap= 30)
# Usamos a tesoura para cortar todos os documentos da caixa `docs` e guardamos os pedacinhos na caixa `chunks`.
chunks = splitter.split_documents(docs)     
   
# Criação dos Embeddings (Vetorização), que representam o significado do texto.
# Criamos um robô `embeddings` que transforma palavras em "códigos de significado" (vetores de números).
embeddings = GoogleGenerativeAIEmbeddings(
    model= "models/gemini-embedding-001",
    google_api_key= GOOGLE_API_KEY
)

# Criação do Vector Store, que armazena os chunks e seus vetores.
# FAISS é uma biblioteca do Facebook AI para busca de similaridade eficiente.
# Criamos nossa "estante mágica" `vectorstore`.
# A função `FAISS.from_documents` pega os pedaços de texto (`chunks`) e usa o robô `embeddings`
# para criar um código para cada um, organizando tudo na estante para buscas rápidas.
vectorstore = FAISS.from_documents(chunks, embeddings)

# Criação do Retriever, a interface de busca no Vector Store.
# Criamos o `retriever`, que é o nosso "bibliotecário". É ele quem vai saber como procurar na estante mágica.
retriever = vectorstore.as_retriever(search_type= "similarity_score_threshold", 
                                     search_kwargs={"score_threshold":0.3, "k": 4})
# Preparação da Cadeia de Resposta (RAG Chain)
prompt_rag = ChatPromptTemplate.from_messages([
    ("system",
     "Você é um Assistente de Políticas Internas (RH/IT) da empresa Carraro Desenvolvimento. "
     "Responda SOMENTE com base no contexto fornecido. "
     "Se não houver base suficiente, responda que não esta claro nas politicas da empresa \n"
     "e que precisa de mais informacoes."),
    # `{input}` e `{context}` são espaços em branco que serão preenchidos depois.
    ("human", "Pergunta: {input}\n\nContexto:\n{context}")
])
# Criamos a `document_chain`, que junta o robô de triagem (`llm_triagem`) com o nosso molde de carta (`prompt_rag`).
document_chain = create_stuff_documents_chain(llm_triagem, prompt_rag)

# Formatadores
import re, pathlib

# Remove espaços em branco extras de uma string.
# `re.sub(...)` é uma função que procura um padrão em um texto e o substitui por outra coisa.
def _clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

# Extrai um trecho de texto em torno da primeira palavra-chave encontrada da pergunta.
def extrair_trecho(texto: str, query: str, janela: int = 240) -> str:
    txt = _clean_text(texto)
    # `[... for ... in ...]` é um jeito rápido de criar uma lista.
    # Aqui, criamos uma lista de palavras importantes da pergunta do usuário.
    termos = [t.lower() for t in re.findall(r"\w+", query or "") if len(t) >= 4]
    pos = -1
    for t in termos:
        pos = txt.lower().find(t)
        # `break` para o laço assim que a primeira palavra for encontrada.
        if pos != -1: break
    if pos == -1: pos = 0
    # `max()` e `min()` garantem que não vamos tentar pegar um pedaço de texto que está fora do livro.
    ini, fim = max(0, pos - janela//2), min(len(txt), pos + janela//2)
    # `txt[ini:fim]` é como recortar uma fatia da string, do início (`ini`) ao fim (`fim`).
    return txt[ini:fim]

# Formata a lista de documentos retornados pelo retriever em um formato de citação mais amigável.
def formatar_citacoes(docs_rel: List, query: str) -> List[Dict]:
    # `seen = set()` cria um conjunto, que é como uma lista que não aceita itens repetidos.
    # Usamos isso para não mostrar a mesma citação da mesma página duas vezes.
    cites, seen = [], set()
    for d in docs_rel:
        src = pathlib.Path(d.metadata.get("source","")).name
        page = int(d.metadata.get("page", 0)) + 1
        key = (src, page)
        if key in seen:
            # `continue` pula para a próxima iteração do laço.
            continue
        seen.add(key)
        # `cites.append(...)` adiciona um novo item (um dicionário com os detalhes da citação) na nossa lista `cites`.
        cites.append({"documento": src, "pagina": page, "trecho": extrair_trecho(d.page_content, query)})
    # `cites[:3]` pega apenas os 3 primeiros itens da lista de citações.
    return cites[:3]


# Função que encapsula o fluxo de RAG.
def perguntar_politica_rag(pergunta: str) -> dict:
    # 1. Busca: O bibliotecário (`retriever`) é invocado com a `pergunta`. Ele busca na estante
    # e devolve os documentos mais parecidos, que guardamos na caixa `docs_relacionados`.
    docs_relacionados = retriever.invoke(pergunta)
    
    # Se nenhum documento for encontrado, retorna uma resposta padrão.
    if not docs_relacionados:
        return {"answer": "Não esta claro nas politicas da empresa.",
                "citacoes": [],
                "contexto_encontrado": False
                }

    # 2. Geração: Invocamos a `document_chain`. Ela preenche o molde de carta com a `pergunta`
    # e os `docs_relacionados` e envia para a IA, que gera uma resposta. Guardamos na caixa `answer`.
    answer = document_chain.invoke({"input": pergunta,
                                    "context": docs_relacionados})
    
    txt = (answer or "").strip()
    
    # `txt.rstrip(".!?")` remove os pontos do final da frase para a comparação ficar mais precisa.
    # Verifica se o LLM não conseguiu encontrar a resposta no contexto fornecido.
    if txt.rstrip(".!?") == "Não esta claro nas politicas da empresa.":
        return {"answer": "Não esta claro nas politicas da empresa.",
                "citacoes": [],
                "contexto_encontrado": False
                }
        
    return {    "answer": txt,
                "citacoes": formatar_citacoes(docs_relacionados, pergunta),
                "contexto_encontrado": True
                }    
# --- 4. CONSTRUÇÃO DO AGENTE COM LANGGRAPH ---
# Usamos o LangGraph para criar um fluxograma (grafo) que orquestra nossas ferramentas.

# --- 4.1. DEFINIÇÃO DO ESTADO DO AGENTE ---
# O 'AgentState' é o "diário" do nosso agente. É um tipo de dicionário especial que guarda
# tudo o que acontece em cada etapa da jornada. `total=False` significa que nem todas as
# chaves precisam estar preenchidas o tempo todo.
class AgentState(TypedDict, total = False):
    pergunta: str
    triagem: dict
    resposta: Optional[str]
    citacoes: List[dict]
    rag_sucesso: bool
    acao_final: str
    
# --- 4.2. DEFINIÇÃO DOS NÓS DO GRAFO ---
# Cada "nó" (node) é uma "parada" no nosso mapa de trabalho. É uma função que faz uma tarefa.
# Ela sempre recebe o diário (`state`) e devolve um dicionário com as atualizações para o diário.

# Nó 1: Triagem da pergunta inicial do usuário.
def node_triagem(state: AgentState) -> AgentState:
    print("Executando nó triagem...") 
    return {"triagem": triagem(state["pergunta"])}  

# Nó 2: Tenta resolver a pergunta usando a busca em documentos (RAG).
def node_auto_resolver(state: AgentState) -> AgentState:
    print("Executando nó auto_resolver...") 
    # Acessamos a pergunta que está guardada no diário (`state`) usando `state["pergunta"]`.
    # O resultado da busca é guardado na caixa `resposta_rag`.
    resposta_rag = perguntar_politica_rag(state["pergunta"])
    
    update: AgentState = {
        "resposta": resposta_rag["answer"],
        "citacoes": resposta_rag.get("citacoes", []),
        "rag_sucesso": resposta_rag["contexto_encontrado"],
    }
    
    if resposta_rag["contexto_encontrado"]:
        update["acao_final"] = "AUTO_RESOLVER"
    return update 

# Nó 3: Pede mais informações ao usuário quando a pergunta é vaga.
def node_pedir_info(state: AgentState) -> AgentState:
    print("Executando nó pedir info...")    
    # O método `.get(chave, [])` é um jeito seguro de pegar algo de um dicionário.
    # Se a chave "campos_faltantes" não existir, ele nos dá uma lista vazia em vez de um erro.
    faltantes = state["triagem"].get("campos_faltantes", [])     
    # `",".join(faltantes)` junta os itens da lista `faltantes` em um único texto, separados por vírgula.
    detalhe = ",".join(faltantes) if faltantes else "Tema e contexto especifico"
    return {
        "resposta": f"Para avançar, preciso de detalhes sobre: {detalhe}",
        "citacoes": [],
        "acao_final": "PEDIR_INFO"
    }
# Nó 4: Abre um chamado para a equipe de RH/TI quando necessário.
def node_abrir_chamado(state: AgentState) -> AgentState:
    print("Executando nó de abrir chamado...") 
    triagem = state["triagem"]
    return{
        "resposta": f"Abrindo chamado com urgencia {triagem["urgencia"]}. Descrição: {state["pergunta"][:150]}",
        "citacoes": [],
        "acao_final": "ABRIR_CHAMADO"
    }
    
# --- 4.3. DEFINIÇÃO DAS ARESTAS CONDICIONAIS (A INTELIGÊNCIA DO FLUXO) ---
# Funções que olham para o estado atual e decidem para qual nó o agente deve ir em seguida.
# São como as "setas" do nosso mapa, que podem apontar para lugares diferentes.

KEYWORDS_ABRIR_CHAMADO = ["aprovação", "exceção", "liberação", "abrir ticket", "abrir chamado", "acesso especial"]

# Decide o próximo passo com base no resultado da triagem.
def decidir_pos_triagem(state: AgentState) -> str:
    print("Decidindo após a triagem...")
    # A função de decisão devolve apenas um texto. Esse texto é a "chave" que o LangGraph usa para saber para qual nó ir.
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
    
    # `any(...)` verifica se QUALQUER uma das palavras-chave da nossa lista está na pergunta do usuário.
    # É um jeito rápido de checar várias coisas de uma vez.
    if any(k in state_da_pergunta for k in KEYWORDS_ABRIR_CHAMADO): 
        # Se a busca falhou, mas a pergunta contém palavras-chave para abrir chamado, vai para esse nó.
        print("--> RAG falhou, mas encontrei keywords. Roteando para abrir chamado.")
        return "chamado"
    print("--> RAG falhou. Roteando para pedir mais informações.")
    return "info"
    
# --- 4.4. MONTAGEM DO GRAFO (O FLUXOGRAMA) ---
# Agora, juntamos todas as peças que criamos para montar o mapa de trabalho do agente.
# 1. Criamos um novo mapa (`StateGraph`) e dizemos que ele vai usar nosso diário (`AgentState`).
workflow = StateGraph(AgentState)

# 2. Adicionamos as "paradas" (nós) ao nosso mapa, dando um nome para cada uma.
workflow.add_node("triagem", node_triagem)
workflow.add_node("auto_resolver", node_auto_resolver)
workflow.add_node("pedir_info", node_pedir_info)
workflow.add_node("abrir_chamado", node_abrir_chamado)

# 3. Marcamos o ponto de partida (`START`) do mapa, que sempre será o nó "triagem".
workflow.add_edge(START, "triagem")

# 4. Adicionamos as "setas com condição".
# Depois do nó "triagem", a função `decidir_pos_triagem` é chamada.
# O dicionário `{...}` mapeia o resultado da função para o próximo nó.
# Ex: Se a função devolver "auto", o fluxo vai para "auto_resolver".
workflow.add_conditional_edges("triagem", decidir_pos_triagem, {
    "auto": "auto_resolver",
    "info": "pedir_info",
    "chamado": "abrir_chamado"
})
   
workflow.add_conditional_edges("auto_resolver", decidir_pos_auto_resolver, {
    "info": "pedir_info",
    "chamado": "abrir_chamado",
    # `END` é a parada final, que significa que o trabalho acabou.
    "ok": END
})

# 5. Adicionamos as setas que sempre levam para o fim do mapa.
workflow.add_edge("pedir_info", END)
workflow.add_edge("abrir_chamado", END)

# 6. `compile()` transforma nosso mapa de desenho em um objeto que o computador consegue executar.
grafo = workflow.compile()

# --- 5. VISUALIZAÇÃO E EXECUÇÃO DO AGENTE ---
# Gera uma representação visual do nosso fluxograma.
# try:
#     graph_bytes = grafo.get_graph().draw_mermaid_png()
#     display(Image(graph_bytes))
# except Exception as e:
#     print(f"Não foi possível gerar a imagem do grafo. Erro: {e}")

# Lista de perguntas para testar todas as rotas possíveis do nosso agente.
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

# Loop para executar o agente para cada pergunta de teste.
for msg_teste in perguntas_teste:
    # O método `invoke` dá o "play" no nosso mapa. Ele começa no `START` e segue as setas até chegar no `END`.
    # O resultado final de tudo o que aconteceu no diário é guardado na caixa `resposta_final`.
    resposta_final = grafo.invoke({"pergunta": msg_teste})
    # Extrai o resultado da triagem para exibição.
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