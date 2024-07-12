# InsightBot AI 🤖

InsightBot AI is an adaptive recommendation chatbot that uses Retrieval-Augmented Generation (RAG) and a vector database. The project allows users to upload PDF and JSON files, process the text, and ask questions based on the content of the files.

## Features

- **File Upload**: Supports PDF and JSON file uploads.
- **Text Processing**: Extracts and processes text from uploaded files.
- **Text Chunking**: Splits text into manageable chunks.
- **Text Embedding**: Uses Google Generative AI Embeddings to convert text chunks into vector representations.
- **Vector Store**: Stores embeddings in a FAISS vector database for efficient similarity search.
- **Question Answering**: Provides answers to user queries based on the content of the uploaded files.
- **Interactive Interface**: Built with Streamlit for easy interaction and user experience.

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/InsightBot-AI.git
    cd InsightBot-AI
    ```

2. **Set up a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up environment variables**:
    - Create a `.env` file in the root directory of the project.
    - Add your Google API key to the `.env` file:
        ```
        GOOGLE_API_KEY=your_google_api_key
        ```

## Usage

1. **Run the Streamlit app**:
    ```bash
    streamlit run app.py
    ```

2. **Upload Files**:
    - Use the file uploader in the Streamlit interface to upload PDF or JSON files.

3. **Process Files**:
    - Click the "Submit & Process" button to extract and process text from the uploaded files.

4. **Ask Questions**:
    - Enter your question in the text input field and press Enter.
    - The bot will provide answers based on the content of the uploaded files.

## Project Structure

- `app.py`: The main Streamlit app script.
- `requirements.txt`: List of required Python packages.
- `.env`: Environment file for storing API keys (not included in the repository).

## Approach Taken

The approach taken for this project involved the following steps:
1. **Loading Environment Variables**: Using the `dotenv` library to load the Google API key securely.
2. **Reading Files**: Handling PDF files using `PyPDF2` and JSON files using the `json` library.
3. **Text Chunking**: Splitting the text into manageable chunks using `langchain`'s `RecursiveCharacterTextSplitter`.
4. **Embedding Text**: Using Google Generative AI Embeddings to embed the text chunks into vector representations.
5. **Creating Vector Store**: Storing the embeddings in a FAISS vector database for efficient similarity search.
6. **Conversational Chain**: Setting up a conversational chain using `langchain` to handle question-answering based on the embedded text.
7. **Streamlit Interface**: Building an interactive user interface with Streamlit for file uploads and question input.

## Challenges Faced

The following challenges were encountered during the project:
1. **Rate Limiting**: The Google Generative AI API has strict rate limits, which caused quota exceeded errors during the embedding process.
2. **Handling Large Files**: Processing large files efficiently without exceeding API rate limits or running into memory issues.
3. **Ensuring Accuracy**: Maintaining the accuracy of the question-answering system while handling diverse file formats and content types.

## Overcoming Challenges

The challenges were overcome using the following strategies:
1. **Batch Processing with Delays**: To handle rate limiting, text chunks were processed in smaller batches with delays between each batch.
2. **Retry Logic**: Implemented retry logic to handle rate limit errors by waiting and retrying the requests.
3. **Monitoring and Adjustments**: Regularly monitored the processing time and adjusted batch sizes and delays to optimize performance.
4. **Enhanced Error Handling**: Improved error handling to gracefully manage different types of errors and provide meaningful feedback to users.

## Conclusion

The InsightBot AI project successfully developed a recommendation chatbot using RAG and a vector database. By implementing effective strategies to overcome challenges, the project provided a robust solution for processing and querying text from PDF and JSON files. The approach taken and the lessons learned from this project can serve as a valuable reference for future projects involving text processing and conversational AI.
