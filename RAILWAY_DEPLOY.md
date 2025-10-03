# Railway Deployment Guide

Deploy your shrimp annotation pipeline to Railway in just a few clicks!

## 🚀 Quick Deploy (Recommended)

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/QXzqBG)

## 📋 Manual Deployment Steps

### Step 1: Create Railway Account
1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub (easiest option)
3. Connect your GitHub account

### Step 2: Deploy from GitHub

1. **Create New Project**:
   - Click "New Project" on Railway dashboard
   - Select "Deploy from GitHub repo"
   - Choose `adelatif11-teramina/shrimp-annotation-pipeline`

2. **Railway will automatically**:
   - Detect the project as Python
   - Use the Dockerfile or nixpacks.toml configuration
   - Install dependencies
   - Build the frontend
   - Deploy the application

### Step 3: Configure Environment Variables

In your Railway project dashboard, go to **Variables** and add:

**Required:**
```bash
JWT_SECRET_KEY=your-secure-jwt-secret-here
```

**Optional (but recommended):**
```bash
OPENAI_API_KEY=your-openai-api-key
ENVIRONMENT=production
LOG_LEVEL=INFO
RATE_LIMIT_ENABLED=true
```

**Generate a secure JWT secret:**
```bash
# Run this in your terminal to generate a secure secret
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### Step 4: Deploy and Access

1. **Deploy**: Railway automatically deploys after adding environment variables
2. **Get URL**: Your app will be available at a Railway-provided URL like:
   `https://your-app-name.up.railway.app`
3. **Custom Domain** (optional): Add your own domain in Railway settings

## 📊 What Gets Deployed

Your Railway deployment includes:

- ✅ **Full API Backend** - FastAPI server with all endpoints
- ✅ **React Frontend** - Built and served statically
- ✅ **SQLite Database** - File-based database (perfect for single user)
- ✅ **Authentication** - JWT-based security
- ✅ **Rate Limiting** - API protection
- ✅ **Health Checks** - Automatic monitoring
- ✅ **Auto-scaling** - Handles traffic spikes

## 🔧 Railway Configuration Files

The following files configure your Railway deployment:

- `railway.toml` - Railway-specific settings
- `nixpacks.toml` - Build configuration  
- `Dockerfile` - Container configuration
- `requirements.txt` - Python dependencies

## 💰 Cost Estimate

**Railway Pricing:**
- **Hobby Plan**: $5/month - Perfect for single user
- **Pro Plan**: $20/month - For production use
- **Pay-as-you-go**: ~$0.02/hour when running

**For single user**: Expect $5-10/month total cost.

## 🔍 Monitoring Your Deployment

### Check Deployment Status
1. Go to your Railway project dashboard
2. Click on your service
3. View **Deployments** tab for build logs
4. View **Metrics** tab for performance

### Health Check
Your app includes a health endpoint:
```bash
curl https://your-app.up.railway.app/health
```

### View Logs
In Railway dashboard:
1. Click your service
2. Go to **Logs** tab
3. See real-time application logs

## ⚙️ Railway Features You Get

- **Auto-Deploy**: Pushes to GitHub automatically deploy
- **SSL Certificate**: HTTPS enabled by default
- **Environment Variables**: Secure configuration management
- **Metrics**: CPU, memory, and request monitoring
- **Custom Domains**: Add your own domain
- **Database Backups**: Automatic SQLite backups
- **Rolling Deployments**: Zero-downtime updates

## 🚨 Troubleshooting

### Common Issues

**Build Fails:**
- Check build logs in Railway dashboard
- Ensure all dependencies in `requirements.txt`
- Verify Node.js version compatibility

**App Won't Start:**
- Check you've set `JWT_SECRET_KEY` environment variable
- Review application logs in Railway dashboard
- Verify PORT environment variable is used correctly

**Database Issues:**
- SQLite is used by default (no external database needed)
- Database file persists in Railway's volume storage
- For PostgreSQL, add Railway PostgreSQL service

### Health Check Failed
```bash
# Test health endpoint
curl https://your-app.up.railway.app/health

# Should return:
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "version": "1.0.0"
}
```

### Performance Optimization

**For better performance:**
1. Upgrade to Railway Pro plan
2. Add PostgreSQL database service
3. Enable Redis for caching
4. Use CDN for frontend assets

## 🔄 Updates and Maintenance

### Auto-Deploy from GitHub
1. Any push to `main` branch automatically deploys
2. Railway rebuilds and redeploys your app
3. Zero-downtime rolling deployment

### Manual Deploy
1. Go to Railway dashboard
2. Click your service
3. Click **Deploy** → **Redeploy**

### Environment Updates
1. Update variables in Railway dashboard
2. Click **Redeploy** to apply changes

## 🌐 Custom Domain Setup

1. **Buy Domain**: Get a domain from any registrar
2. **Railway Settings**: 
   - Go to project Settings
   - Add custom domain
   - Copy Railway's DNS settings
3. **DNS Configuration**:
   - Add CNAME record pointing to Railway
   - Wait for SSL certificate generation

## 📈 Scaling

Railway automatically scales based on:
- **CPU Usage**: Scales up during high processing
- **Memory Usage**: Allocates more memory as needed
- **Request Volume**: Handles traffic spikes

For heavy usage:
- Upgrade to Pro plan
- Add PostgreSQL for better database performance
- Consider Redis for session storage

## ✅ Success!

Once deployed, your annotation pipeline will be available at:
`https://your-app.up.railway.app`

Features available:
- Full annotation interface
- API endpoints for automation
- Health monitoring
- Secure authentication
- Professional deployment

Ready to start annotating! 🦐

## 🆘 Support

- **Railway Docs**: [docs.railway.app](https://docs.railway.app)
- **Railway Discord**: [discord.gg/railway](https://discord.gg/railway)
- **Project Issues**: [GitHub Issues](https://github.com/adelatif11-teramina/shrimp-annotation-pipeline/issues)