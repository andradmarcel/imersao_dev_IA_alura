# 🤖 Agente de Atendimento de RH com RAG e LangGraph
* 
**Triagem Automatizada:** O agente interpreta a pergunta do usuário e a direciona para a ferramenta correta.
* 
**Respostas Baseadas em Documentos (RAG):** Utiliza a técnica de *Retrieval-Augmented Generation* para buscar informações em arquivos PDF de políticas internas, garantindo respostas fiéis à fonte.
* 
**Busca Semântica Inteligente:** Emprega o **FAISS** e a API de embeddings do **Google Gemini** para encontrar os trechos mais relevantes dos documentos, mesmo que a pergunta não use as mesmas palavras-chave.
* 
**Orquestração com LangGraph:** O fluxo de execução do agente é gerenciado como um grafo de estados, permitindo um controle robusto e modular sobre o processo de decisão.
## 🛠️ Arquitetura e Funcionamento
O fluxo de trabalho do agente foi estruturado da seguinte forma:
1.  **Ingestão e Vetorização:** Documentos de políticas internas (`.pdf`) são carregados, divididos em blocos de texto (`chunks`) e transformados em vetores numéricos (embeddings) pela API do Gemini.
2.  **Armazenamento em Banco Vetorial:** Esses vetores são armazenados e indexados no **FAISS**, criando uma base de conhecimento otimizada para busca semântica.
3.  **Orquestração do Fluxo (LangGraph):** Ao receber uma pergunta, o **LangGraph** inicia o grafo. O primeiro nó (agente) analisa a intenção do usuário e decide qual ferramenta usar (neste caso, a ferramenta de busca RAG).
4.  **Recuperação de Informação (Retrieval):** A pergunta do usuário é convertida em um vetor, e o FAISS realiza uma busca de similaridade para encontrar os `chunks` mais relevantes na base de conhecimento.
5.  **Geração da Resposta (Generation):** Os `chunks` recuperados são injetados em um prompt, juntamente com a pergunta original. O **Google Gemini** utiliza esse contexto para gerar uma resposta final, coesa e precisa.
## 🚀 Como Executar o Projeto
1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/andradmarcel/imersao_dev_IA_alura.git
    cd imersao_dev_IA_alura
    ```
2.  **Crie um ambiente virtual e instale as dependências:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows: venv\Scripts\activate
    pip install langchain langgraph faiss-cpu python-dotenv google-generativeai
    ```
3.  **Configure suas variáveis de ambiente:**
    * Crie um arquivo `.env` na raiz do projeto.
    * Adicione sua chave da API do Google Gemini:
        ```env
        GOOGLE_API_KEY="SUA_API_KEY_AQUI"
        ```
4.  **Adicione os documentos:**
    * Coloque os arquivos `.pdf` com as políticas da empresa em um diretório específico (ex: `/documentos`).
5.  **Execute o agente:**
    ```bash
    python main.py
    ```
## 💡 Habilidades e Competências Demonstradas
* **Orquestração de Agentes de IA:** Uso do LangGraph para criar fluxos de decisão complexos e modulares.
* **Retrieval-Augmented Generation (RAG):** Implementação de sistemas que baseiam respostas de LLMs em fontes de conhecimento externas.
* **Busca Semântica:** Aplicação de embeddings e bancos de dados vetoriais (FAISS) para busca de informação por significado.
* **Processamento de Linguagem Natural (PLN):** Manipulação e interpretação de texto para extração de informações.
* **Consumo de APIs:** Integração com a API do Google Gemini para embeddings e geração de texto.
* **Engenharia de Prompts:** Elaboração de prompts eficazes para guiar o modelo a gerar respostas precisas e contextuais.
