# 🤖 Agente de Atendimento de RH com RAG e LangGraph

Este projeto apresenta um agente de IA autônomo projetado para otimizar o atendimento de RH, respondendo a dúvidas sobre políticas internas da empresa de forma automatizada e precisa. O agente utiliza um sistema de Geração Aumentada por Recuperação (RAG) para consultar documentos e fornecer respostas contextuais.

Este repositório foi desenvolvido durante a **Imersão Dev Agentes de IA**, uma parceria da **Alura** com o **Google Gemini**.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0086D1?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB2ZXJzaW9uPSIxLjIiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgdmlld0JveD0iMCAwIDE5MTUgMTk1MyIgd2lkdGg9IjE5MTUiIGhlaWdodD0iMTk1MyI+PHN0eWxlPi5he2ZpbGw6I2ZmZn08L3N0eWxlPjxwYXRoIGZpbGwtcnVsZT0iZXZlbm9kZCIgY2xhc3M9ImEiIGQ9Im0xNTA0LjkgMTk1Mi45bC0zMzUtMTkzLjloLTY2My4zbC0yMTMuMyAxMjEuNmgtMjkzLjNsMzM1LjEtMTkzLjloNjYzLjNsMjEzLjMtMTIxLjZoMjkzLjN6bS0xMDcwLjUtNjI0LjdsLTMzNS4xLTE5My45aDI5My4zbDIxMy4zIDEyMS42aDY2My4zbDMzNS0xOTMuOWgtMjkzLjNsLTIxMy40IDEyMS42aC02NjMuM3ptMCA2MjQuOGwtMzM1LjEtMTkzLjloMjkzLjNsMjEzLjMgMTIxLjZoNjYzLjNsMzM1LTE5My45aC0yOTMuM2wtMjEzLjQgMTIxLjZoLTY2My4zeiIvPjxwYXRoIGNsYXNzPSJhIiBkPSJtNDM0LjQgMTk1Mi45bDMzNS4xLTE5My45di0xMTc3LjVsLTMzNS4xIDE5My45em0xMDcwLjUtMTU2Ni4zbDMzNS0xOTMuOWgtMjkzLjNsLTIxMy40IDEyMS42aC02NjMuM2wtMzM1LjEtMTkzLjloMjkzLjNsMjEzLjMgMTIxLjZoNjYzLjN6bS0xMDcwLjYtMTkzLjlsLTMzNS4xLTE5My45aDI5My4zbDIxMy4zIDEyMS42aDY2My4zbDMzNS0xOTMuOWgtMjkzLjNsLTIxMy40IDEyMS42aC02NjMuM3ptMCA2MjQuOGwtMzM1LjEtMTkzLjloMjkzLjNsMjEzLjMgMTIxLjZoNjYzLjNsMzM1LTE5My45aC0yOTMuM2wtMjEzLjQgMTIxLjZoLTY2My4zeiIvPjwvc3ZnPg==)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-8E75B4?style=for-the-badge&logo=google&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-4A90E2?style=for-the-badge&logo=facebook&logoColor=white)

---

## 🎯 Objetivo do Projeto

O principal desafio abordado por este projeto é a alta demanda e a repetitividade das perguntas direcionadas às equipes de RH e TI sobre as políticas internas da empresa. Isso consome um tempo valioso que poderia ser usado em tarefas mais estratégicas.

A solução foi desenvolver um **agente autônomo** que centraliza o conhecimento da empresa em uma base vetorial e utiliza modelos de linguagem avançados para automatizar a triagem e a resposta a essas dúvidas, oferecendo informações precisas e instantâneas aos colaboradores.

## ✨ Funcionalidades Principais

* **Triagem Automatizada:** O agente interpreta a pergunta do usuário e a direciona para a ferramenta correta.
* **Respostas Baseadas em Documentos (RAG):** Utiliza a técnica de *Retrieval-Augmented Generation* para buscar informações em arquivos PDF de políticas internas, garantindo respostas fiéis à fonte.
* **Busca Semântica Inteligente:** Emprega o **FAISS** e a API de embeddings do **Google Gemini** para encontrar os trechos mais relevantes dos documentos, mesmo que a pergunta não use as mesmas palavras-chave.
* **Orquestração com LangGraph:** O fluxo de execução do agente é gerenciado como um grafo de estados, permitindo um controle robusto e modular sobre o processo de decisão.

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
    git clone [https://github.com/seu-usuario/seu-repositorio.git](https://github.com/seu-usuario/seu-repositorio.git)
    cd seu-repositorio
    ```

2.  **Crie um ambiente virtual e instale as dependências:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Configure suas variáveis de ambiente:**
    * Crie um arquivo `.env` na raiz do projeto.
    * Adicione sua chave da API do Google Gemini:
        ```
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
