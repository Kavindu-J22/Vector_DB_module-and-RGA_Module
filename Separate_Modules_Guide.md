# Running Vector DB and RAG Modules Separately in Google Colab

This guide explains how to run the Vector Database Module and RAG Module as separate, independent processes in Google Colab.

## 🎯 Why Run Separately?

### Benefits:
- **🔧 Modular Development**: Test and debug each component independently
- **💾 Resource Management**: Use different Colab sessions for different tasks
- **🔄 Flexibility**: Run Vector DB once, use RAG multiple times
- **👥 Team Collaboration**: Different team members can work on different modules
- **⚡ Performance**: Dedicated resources for each module

## 📋 Prerequisites

1. **Google Account** with Colab access
2. **Pinecone Account** with API key and environment
3. **Project Files** uploaded to Google Drive or GitHub
4. **(Optional)** OpenAI API key for enhanced responses

## 🗄️ Part 1: Vector Database Module

### Purpose:
- Process legal documents (Acts, Cases, Regulations)
- Generate embeddings using Legal-BERT/Sentence-BERT
- Store in Pinecone vector database
- Create searchable legal document index

### Steps:
1. **Open `Vector_DB_Colab.ipynb`** in Google Colab
2. **Upload your project zip** to Google Drive
3. **Configure Pinecone credentials** in the notebook
4. **Run all cells** to process documents
5. **Verify database status** - note the vector count
6. **Save results** to Google Drive

### Expected Output:
```
✅ Vector Database Status:
📊 Index Name: sri-lankan-legal-docs
📈 Total Vectors: 15,847
📏 Dimension: 768
🎉 Vector Database is ready for RAG Module!
```

### Time Required: 
- **Small dataset** (< 100 docs): 10-15 minutes
- **Medium dataset** (100-1000 docs): 30-60 minutes  
- **Large dataset** (> 1000 docs): 1-3 hours

## 🤖 Part 2: RAG Module

### Purpose:
- Connect to existing vector database
- Provide web interface for legal questions
- Generate multiple response options
- Handle conversation history and user interactions

### Steps:
1. **Open `RAG_Module_Colab.ipynb`** in Google Colab
2. **Use SAME Pinecone credentials** as Vector DB module
3. **Test database connection** - should show existing vectors
4. **Launch web interface** - get public ngrok URL
5. **Share URL** with users for legal question answering

### Expected Output:
```
🎉 Sri Lankan Legal AI Assistant is READY!
🌐 Access your Legal AI Assistant at:
   https://abc123.ngrok.io
```

### Time Required:
- **Setup**: 5-10 minutes
- **Runtime**: Continuous (as long as Colab session is active)

## 🔄 Workflow Sequence

### Recommended Order:

1. **First Time Setup:**
   ```
   Vector DB Module → RAG Module
   ```

2. **Development/Testing:**
   ```
   Vector DB Module (when documents change)
   ↓
   RAG Module (for each testing session)
   ```

3. **Production Use:**
   ```
   Vector DB Module (run once or when updating documents)
   ↓
   RAG Module (run continuously for users)
   ```

## 📊 Resource Management

### Vector DB Module:
- **RAM Usage**: High during processing
- **GPU Usage**: Medium (for embeddings)
- **Storage**: Temporary (results go to Pinecone)
- **Session Duration**: Can be closed after completion

### RAG Module:
- **RAM Usage**: Medium (for inference)
- **GPU Usage**: Low to Medium
- **Storage**: Minimal (logs and cache)
- **Session Duration**: Keep active for continuous service

## 🔧 Configuration Management

### Shared Configuration:
Both modules need the **SAME** Pinecone settings:
```python
PINECONE_API_KEY = 'your-key-here'
PINECONE_ENVIRONMENT = 'your-env-here'
INDEX_NAME = 'sri-lankan-legal-docs'
```

### Module-Specific Settings:

**Vector DB Module:**
```python
# Document processing settings
CHUNK_SIZE = 512
OVERLAP = 50
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
```

**RAG Module:**
```python
# Response generation settings
OPENAI_API_KEY = 'your-openai-key'  # Optional
MAX_RESPONSES = 3
RESPONSE_STYLES = ['professional', 'detailed', 'concise']
```

## 🚀 Quick Start Commands

### For Vector DB Module:
```python
# In Vector_DB_Colab.ipynb
# 1. Set ZIP_PATH to your project location
ZIP_PATH = '/content/drive/MyDrive/your-project.zip'

# 2. Set Pinecone credentials
PINECONE_API_KEY = 'your-key'
PINECONE_ENVIRONMENT = 'your-env'

# 3. Run all cells
```

### For RAG Module:
```python
# In RAG_Module_Colab.ipynb
# 1. Use SAME credentials as Vector DB
PINECONE_API_KEY = 'same-key-as-vector-db'
PINECONE_ENVIRONMENT = 'same-env-as-vector-db'
INDEX_NAME = 'sri-lankan-legal-docs'

# 2. Run all cells to get public URL
```

## 🔍 Troubleshooting

### Common Issues:

1. **"Index not found" in RAG Module**
   - Make sure Vector DB Module completed successfully
   - Check that both modules use identical Pinecone credentials
   - Verify index name matches exactly

2. **"No vectors found" in RAG Module**
   - Vector DB Module may have failed during processing
   - Check Vector DB logs for errors
   - Re-run Vector DB Module with smaller document batch

3. **RAG Module web interface not loading**
   - Wait longer for Streamlit to start (up to 30 seconds)
   - Try restarting the cell
   - Use CLI mode as alternative

4. **Memory errors**
   - Use GPU runtime in Colab
   - Process documents in smaller batches
   - Restart runtime and clear outputs

### Quick Fixes:
```python
# Restart processes
!pkill -f streamlit
!pkill -f python

# Check Pinecone connection
import pinecone
pinecone.init(api_key='your-key', environment='your-env')
print(pinecone.list_indexes())

# Verify index stats
index = pinecone.Index('sri-lankan-legal-docs')
print(index.describe_index_stats())
```

## 📈 Scaling Considerations

### For Large Document Collections:
1. **Batch Processing**: Process documents in chunks
2. **Multiple Sessions**: Use separate Colab sessions for different document types
3. **Incremental Updates**: Add new documents without reprocessing everything
4. **Resource Monitoring**: Watch RAM and GPU usage

### For High User Traffic:
1. **Multiple RAG Instances**: Run RAG Module in multiple Colab sessions
2. **Load Balancing**: Distribute users across different ngrok URLs
3. **Caching**: Implement response caching for common queries
4. **Monitoring**: Track usage and performance metrics

## 🎯 Best Practices

1. **Always run Vector DB Module first** before RAG Module
2. **Use consistent naming** for indexes and configurations
3. **Save results regularly** to Google Drive
4. **Monitor resource usage** to avoid session timeouts
5. **Test with small datasets** before processing large collections
6. **Keep backup copies** of configuration files
7. **Document your API keys** securely (not in notebooks)

## 🔗 Integration Points

The modules connect through:
- **Pinecone Vector Database**: Shared storage layer
- **Index Name**: Must match exactly
- **Embedding Model**: Should be consistent
- **Document Metadata**: Shared schema

This separation allows for flexible development and deployment while maintaining seamless integration between the document processing and question-answering components.

---

**🎉 With this setup, you can run a complete Sri Lankan Legal AI system using separate, manageable Google Colab notebooks!**
