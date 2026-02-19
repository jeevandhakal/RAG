**ASSIGNMENT - 2**

**Document Q&A with RAG**

Build a Simple RAG System using LangChain & ChromaDB

## Objective

Build a RAG (Retrieval Augmented Generation) system that loads a PDF document, creates embeddings, stores them in a ChromaDB vector database, and allows users to ask questions that are answered based on the document content.

## Workflow

Your RAG system should follow this process:

**Document Loading** - Load a PDF file from the data/ directory

**Text Splitting** - Split document into smaller chunks

**Embedding & Storage** - Create embeddings and store in ChromaDB

**User Query** - Accept a question from the user

**Answer Generation** - Retrieve relevant chunks and generate answer using LLM

## Requirements

**Document Loader**

- Load documents from a data/ directory

**Text Splitter**

- Use RecursiveCharacterTextSplitter
- Configure chunk_size and chunk_overlap (suggested: 1000 and 200)

**Vector Store**

- Use ChromaDB as the vector database (free, open-source)
- Use Jina AI embeddings (free alternative) <https://jina.ai/embeddings>

**RAG Chain**

- Create a retriever from the vector store
- Use RetrievalQA chain to connect retriever with LLM
- Use OpenAI GPT model (or any compatible LLM)

## Output Format

- Accept user questions via command line input
- Print answers based on the document content
- Handle cases where information is not found in the document

## Test Queries

Run your RAG system with these 3 queries and save results to output/results.txt:

QUERIES = \[

"What is Crosswalk guards?",

"What to do if moving through an intersection with a green signal?",

"What to do when approached by an emergency vehicle?"

\]

**Submission:** Create a GitHub repository with all your code and files. Share the repository link as your submission.

## ⭐ Bonus Points

- **Source Citations** - Show which page/chunk the answer came from, extract from document
- **Multiple Documents** - Support loading multiple PDF files

## Documents to Use

- Use given PDF: **DH-Chapter2.pdf**

## Resources

- <https://docs.langchain.com/oss/python/langchain/rag>
- <https://docs.langchain.com/oss/python/integrations/vectorstores/index#chroma>
- <https://jina.ai/>
- <https://jina.ai/embeddings>
