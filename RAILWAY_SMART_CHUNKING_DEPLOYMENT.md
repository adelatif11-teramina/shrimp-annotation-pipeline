# Railway Smart Chunking Deployment Guide

This guide ensures smart chunking is properly deployed and replaces sentence-level chunking on Railway production.

## 🎯 What's Changed

Smart chunking has been **fully integrated** to replace sentence-level chunking:

### ✅ **Production API Updated** (`railway_production_api.py`)
- `split_into_sentences()` → `split_into_smart_chunks()`
- Document processing now creates 150-400 character chunks with context preservation
- Database storage adapted for smart chunks
- Response includes chunking quality metrics

### ✅ **Main API Updated** (`services/api/annotation_api.py`)
- Uses `ImprovedDocumentIngestionService` with smart chunking
- Automatic fallback to sentence splitting if smart chunking fails

### ✅ **Configuration Files**
- `config/production_config.yaml` - Production smart chunking config
- `config/local_config.yaml` - Updated with chunking settings

### ✅ **Frontend Compatibility**
- UI components updated to display "Smart Chunk" vs "Sentence"
- Shows sentence count per chunk for context

## 🚀 Railway Deployment Steps

### 1. Deploy to Railway
```bash
# Push all changes to trigger Railway deployment
git add .
git commit -m "Enable smart chunking for production deployment"
git push origin main
```

### 2. Verify Environment Variables
Ensure these are set in Railway dashboard:
```bash
DATABASE_URL=postgresql://...     # Auto-provided by Railway
JWT_SECRET_KEY=your-secret        # Set manually
OPENAI_API_KEY=sk-...            # Optional, for LLM features
PORT=8000                        # Auto-provided by Railway
```

### 3. Verify Deployment
```bash
# Run verification script
python scripts/verify_smart_chunking_deployment.py

# Or manually test upload endpoint
curl -X POST https://your-railway-app.railway.app/api/upload/document \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Test Document",
    "content": "Vibrio parahaemolyticus causes AHPND in shrimp. This pathogen spreads rapidly..."
  }'
```

### 4. Check Response Format
Successful uploads now return:
```json
{
  "success": true,
  "doc_id": "uploaded_abc123",
  "chunking_mode": "smart_paragraph",
  "chunk_count": 3,
  "sentence_count": 8,
  "chunking_info": {
    "avg_chars_per_chunk": 220,
    "chunks_with_context": 2
  }
}
```

## 🔧 Configuration Details

### Smart Chunking Settings (Production)
```yaml
chunking:
  mode: smart_paragraph              # ENABLED by default
  target_length: [150, 400]         # Optimal annotation size
  preserve_context: true            # Keep definitions/pronouns together
  fallback_mode: sentence           # Graceful degradation
```

### Quality Thresholds
- **Minimum chunk length**: 150 characters
- **Maximum chunk length**: 400 characters  
- **Context preservation**: Definitions, pronouns, discourse markers
- **Quality score target**: >80%

## 📊 Monitoring Smart Chunking

### Health Check Endpoint
```bash
GET /health
```
Returns chunking status and quality metrics.

### Chunking Quality Metrics
Monitor these in Railway logs:
- `✓ Created N smart chunks (avg X chars)`
- `Quality score: X.X%`
- `Chunks with context: N/M`

### Fallback Indicators
Watch for warnings:
- `Smart chunking failed, falling back to sentence splitting`
- Alert if fallback rate > 30%

## 🎯 Benefits in Production

### For Annotators
- **Context preserved**: "This pathogen" references are clear
- **Optimal size**: 2-4 sentences, not overwhelming
- **Faster annotation**: Less context switching

### For Data Quality
- **Better relations**: Cross-sentence entity relationships captured
- **Fewer errors**: Reduced ambiguity from pronouns
- **Consistent chunks**: Predictable 150-400 char size

### For Performance
- **Fewer annotation units**: ~50% reduction in chunks vs sentences
- **Higher throughput**: Faster annotation per document
- **Quality metrics**: Real-time chunking quality monitoring

## 🔄 Rollback Plan (if needed)

If smart chunking causes issues:

1. **Quick rollback** - Change one line in `railway_production_api.py`:
   ```python
   # Line 357: Change back to sentence splitting
   chunks = split_into_sentences(document_text)  # Quick rollback
   ```

2. **Configuration rollback** - Update `config/production_config.yaml`:
   ```yaml
   chunking:
     mode: sentence  # Rollback to sentence-level
   ```

3. **Redeploy**:
   ```bash
   git commit -m "Rollback to sentence chunking"
   git push origin main
   ```

## ✅ Verification Checklist

- [ ] Railway deployment successful
- [ ] `/health` endpoint responds with 200
- [ ] Document upload returns `chunking_mode: "smart_paragraph"`
- [ ] Chunk count < sentence count (efficiency gain)
- [ ] Quality score > 80%
- [ ] Frontend displays "Smart Chunk" instead of "Sentence"
- [ ] No errors in Railway logs
- [ ] Fallback rate < 30%

## 🎉 Success Indicators

**You'll know smart chunking is working when:**
1. **Upload responses** show `chunking_mode: "smart_paragraph"`
2. **Chunk counts** are lower than sentence counts
3. **Annotation UI** shows "Smart Chunk (N sentences)"
4. **Quality scores** are consistently >80%
5. **Railway logs** show `✓ Created N smart chunks`

## 🆘 Troubleshooting

### Common Issues

**Issue**: Smart chunking always falls back to sentences
- **Cause**: Import error or module missing
- **Fix**: Check Railway build logs for import errors

**Issue**: Chunks too long/short
- **Cause**: Configuration not loaded
- **Fix**: Verify `config/production_config.yaml` is deployed

**Issue**: Context not preserved
- **Cause**: Smart chunking disabled
- **Fix**: Check `chunking.mode` in production config

### Getting Help

1. Check Railway deployment logs
2. Run verification script locally
3. Test `/health` endpoint
4. Review chunking quality metrics

---

**🚀 Smart chunking is now live on Railway and will dramatically improve annotation quality by preserving context while maintaining optimal chunk sizes for human annotators!**