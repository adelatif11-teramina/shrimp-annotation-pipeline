# GitHub Push Instructions

## Step 1: Create Repository on GitHub

1. Open your browser and go to: https://github.com/new
2. Fill in these details:
   - **Repository name**: `shrimp-annotation-pipeline`
   - **Description**: `Production-ready human-in-the-loop annotation pipeline for shrimp aquaculture domain`
   - **Public/Private**: Choose Public (recommended for portfolio)
   - **DO NOT** check any initialization options (no README, no .gitignore, no license)
3. Click "Create repository"

## Step 2: Push Your Code

After creating the repository, GitHub will show you the repository URL. It should look like:
`https://github.com/letive/shrimp-annotation-pipeline.git`

Run these commands in your terminal:

```bash
# Add GitHub as remote origin
git remote add origin https://github.com/letive/shrimp-annotation-pipeline.git

# Push your code to GitHub
git push -u origin main
```

## Step 3: Authentication

When you run `git push`, you'll be asked for:
- **Username**: letive
- **Password**: Use a Personal Access Token (NOT your GitHub password)

### To create a Personal Access Token:
1. Go to: https://github.com/settings/tokens/new
2. Give it a name like "shrimp-pipeline-push"
3. Select expiration (30 days is fine)
4. Check these scopes:
   - ✅ repo (all of them)
   - ✅ workflow (optional, for GitHub Actions)
5. Click "Generate token"
6. **COPY THE TOKEN IMMEDIATELY** (you won't see it again)
7. Use this token as your password when git asks

## Step 4: Verify Success

After pushing, your repository will be available at:
https://github.com/letive/shrimp-annotation-pipeline

## Step 5: Final GitHub Setup

Once pushed, go to your repository on GitHub and:

1. **Add Topics** (on the right side of the repo page):
   - shrimp
   - aquaculture
   - annotation
   - nlp
   - machine-learning
   - python
   - fastapi
   - react

2. **Update Settings**:
   - Go to Settings → Options
   - Add website URL if you deploy it
   - Enable Issues
   - Enable Discussions (optional)

3. **Set up GitHub Pages** (optional, for documentation):
   - Settings → Pages
   - Source: Deploy from a branch
   - Branch: main, folder: /docs

## Troubleshooting

If you get an error about authentication:
1. Make sure you're using a Personal Access Token, not your password
2. Check that the token has 'repo' scope
3. Try using SSH instead of HTTPS (requires SSH key setup)

If you get an error about the remote already existing:
```bash
git remote remove origin
git remote add origin https://github.com/letive/shrimp-annotation-pipeline.git
```

## Success! 🎉

Once pushed, your production-ready annotation pipeline will be on GitHub, ready to share or deploy!