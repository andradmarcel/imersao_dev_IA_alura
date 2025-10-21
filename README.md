# Agente de Atendimento de RH com RAG e LangGraph

**Tecnologias:** Python | LangChain | Google Gemini | FAISS | LangGraph

---

## 📋 Descrição do Projeto

### 🎯 Problema Solucionado
Desenvolvi um agente autônomo para automatizar a triagem e resposta de dúvidas sobre políticas internas da empresa, reduzindo a carga sobre a equipe de RH/TI.

### ⚙️ Implementação Técnica
O fluxo foi orquestrado com **LangGraph**, criando um grafo de estados que direciona a solicitação do usuário. Implementei um sistema de **RAG (Retrieval-Augmented Generation)** que vetoriza documentos de políticas internas (.pdf) usando **FAISS** e a API de embeddings do Gemini, permitindo que o agente encontre e sintetize respostas precisas.

### 💡 Habilidades Demonstradas
- Orquestração de Agentes (LangGraph)
- RAG (Retrieval-Augmented Generation)
- Busca Semântica
- Processamento de Linguagem Natural (PLN)
- Consumo de APIs (Google Gemini)
- Engenharia de Prompts

---

## 🎯 Visão Geral
Este repositório reúne todos os códigos, anotações e experiências desenvolvidos durante a Imersão Dev Agentes de IA, promovida pela Alura em parceria com o Google Gemini. O objetivo da imersão foi capacitar desenvolvedores a criar agentes inteligentes, trabalhar com processamento de linguagem natural, engenharia de prompts, RAG (Retrieval-Augmented Generation), embeddings e orquestração de fluxos de IA.

## 🧠 Objetivos do Projeto
- Desenvolver agentes inteligentes capazes de compreender comandos, buscar informações em documentos e solucionar problemas práticos.
- Praticar integração com a API Gemini para soluções de NLP (Natural Language Processing).
- Implementar técnicas modernas como orquestração de agentes com LangGraph e criação de bases de conhecimento com RAG.
- Aplicar engenharia de prompts e integração de documentos via embeddings.
