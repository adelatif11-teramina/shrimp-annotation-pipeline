# Railway Deployment Checklist ✅

## PostgreSQL Integration Complete

✅ **Auto-save for annotations** - Saves every 30 seconds to localStorage + server  
✅ **Network failure recovery** - Progressive retry with offline queue  
✅ **Session management** - Draft restoration and unload protection  
✅ **PostgreSQL support** - Connection pooling for 3 concurrent users  
✅ **Railway deployment ready** - Automatic fallback strategy  

## Deployment Steps

### 1. Railway Project Setup
- [ ] Create new Railway project
- [ ] Add PostgreSQL service to project
- [ ] Deploy from GitHub repository

### 2. Environment Variables
Set these in Railway dashboard:
```
DATABASE_URL=<automatically provided by Railway PostgreSQL service>
RAILWAY_ENVIRONMENT=production
PORT=8000
JWT_SECRET_KEY=<generate secure key>
```

Optional:
```
OPENAI_API_KEY=<if using OpenAI for candidates>
```

### 3. Deployment Configuration
Railway will automatically:
- Use `railway.toml` for deployment config
- Run `railway_start.sh` as startup command
- Detect `DATABASE_URL` and use PostgreSQL production API
- Fall back to in-memory API if PostgreSQL fails
- Set up database tables on first run

## Features Working in Production

### For 3-Person Annotation Team

**Error Recovery & Data Protection:**
- ✅ Auto-save every 30 seconds prevents work loss
- ✅ Network failure detection with retry queue
- ✅ Browser crash protection with draft restoration
- ✅ Unload warning if unsaved changes exist

**PostgreSQL Concurrent Support:**
- ✅ Connection pooling (10 connections + 20 overflow)
- ✅ Proper transaction handling
- ✅ Database migrations on startup
- ✅ Performance indexes for fast queries

**Production API Features:**
- ✅ Health check endpoint (`/api/health`)
- ✅ Draft save/restore endpoints
- ✅ Complete annotation workflow
- ✅ Statistics and monitoring
- ✅ Triage queue management

## Architecture

```
Railway Deployment
├── PostgreSQL Service (managed by Railway)
├── Web Service (your app)
│   ├── Database Setup (automatic)
│   ├── Production API (concurrent users)
│   ├── React Frontend (built)
│   └── Fallback API (if PostgreSQL fails)
```

## Startup Flow

1. Railway detects `DATABASE_URL` environment variable
2. Runs `scripts/setup_railway_database.py` to initialize tables
3. Starts `services/api/production_api.py` with PostgreSQL
4. Serves React frontend from `/ui/build`
5. If PostgreSQL fails, falls back to `railway_api.py` (in-memory)

## API Endpoints

**Health & Status:**
- `GET /api/health` - Database connectivity check
- `GET /api/statistics/overview` - System stats

**Annotation Workflow:**
- `GET /api/triage/queue` - Get annotation queue
- `GET /api/triage/next` - Get next item to annotate
- `POST /api/annotations/decide` - Submit annotation decision

**Error Recovery:**
- `POST /api/annotations/draft` - Save annotation draft
- `GET /api/annotations/draft/{item_id}` - Load annotation draft
- `DELETE /api/annotations/draft` - Clear draft after submission

## Monitoring

**Built-in Health Checks:**
- Database connection status
- API response times
- Active draft count
- Failed operation queue size

**User-Visible Indicators:**
- Connection status indicator in UI
- Auto-save status in annotation workspace
- Network recovery progress
- Draft restoration dialogs

## What's Different from Standard Setup

**Production Enhancements:**
1. PostgreSQL instead of SQLite (handles concurrent users)
2. Auto-save with localStorage + server backup
3. Network failure recovery with progressive retry
4. Draft management for crash protection
5. Connection pooling for performance
6. Automatic database setup on deploy

**Fallback Strategy:**
- If PostgreSQL fails: Falls back to in-memory storage
- If full API fails: Falls back to minimal Railway API
- Always serves React frontend when available

## Testing in Railway

After deployment, test these scenarios:

**Error Recovery:**
1. Start annotating an item
2. Disconnect network for 30 seconds  
3. Continue annotating (should queue operations)
4. Reconnect network (should retry automatically)

**Auto-save:**
1. Start annotating an item
2. Make changes and wait 30 seconds
3. Refresh browser
4. Should see draft restoration dialog

**Concurrent Users:**
1. Have 3 people log in simultaneously
2. All should be able to annotate different items
3. No database locking or connection issues

## Next Steps After Deployment

1. **Monitor logs** for any database connection issues
2. **Test with 3 concurrent users** to verify PostgreSQL performance  
3. **Verify auto-save** works in production environment
4. **Check error recovery** handles Railway network conditions
5. **Review health check** endpoint for monitoring

## Support

If issues arise:
- Check Railway logs for database connection errors
- Use health check endpoint to diagnose problems
- Fallback API should keep system functional
- All error recovery features work offline