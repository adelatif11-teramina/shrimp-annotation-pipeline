# Railway OpenAI Setup Guide

## Quick Setup for Triplet Generation

To enable real AI-powered triplet generation, you need to set your OpenAI API key in Railway's environment variables.

## Railway Environment Variable Setup

1. **Go to your Railway project dashboard**
2. **Navigate to Variables tab**
3. **Add new environment variable:**
   - **Name**: `OPENAI_API_KEY`
   - **Value**: `[Use the OpenAI API key from your local environment: $OPENAI_API_KEY]`

4. **Redeploy your Railway service**

## After Setup

Once the environment variable is set and deployed:

- ✅ **Triplet tab will show real AI-generated triplets**
- ✅ **Entity extraction with shrimp aquaculture ontology**
- ✅ **Knowledge graph relations with audit workflow**
- ✅ **Rule-based cross-checking and validation**

## Current Behavior

**Without API key**: Shows mock triplets with message to set OPENAI_API_KEY
**With API key**: Full LLM-powered triplet generation using GPT-4o-mini

## Security Note

- Never commit API keys to version control
- Use Railway's environment variables for secure storage
- The API key is safely configured in Railway's secure environment

## Verification

After setting up, check the Railway logs for:
```
✅ OpenAI API key found: sk-proj--l...
✅ Successfully imported full annotation API
```

If you see this, triplet generation is working!