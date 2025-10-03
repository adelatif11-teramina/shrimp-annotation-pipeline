#!/bin/bash
# Script to push the shrimp-annotation-pipeline to GitHub

echo "🚀 GitHub Repository Setup for Shrimp Annotation Pipeline"
echo "=========================================================="
echo ""
echo "Step 1: Create a new repository on GitHub"
echo "1. Go to: https://github.com/new"
echo "2. Repository name: shrimp-annotation-pipeline"
echo "3. Description: Production-ready human-in-the-loop annotation pipeline for shrimp aquaculture domain"
echo "4. Choose: Public (or Private if you prefer)"
echo "5. DON'T initialize with README, .gitignore, or license"
echo "6. Click 'Create repository'"
echo ""
echo "Step 2: After creating the repository, GitHub will show you commands."
echo "Copy the repository URL (it should look like: https://github.com/letive/shrimp-annotation-pipeline.git)"
echo ""
read -p "Enter your GitHub repository URL: " REPO_URL

if [ -z "$REPO_URL" ]; then
    echo "❌ No repository URL provided. Exiting."
    exit 1
fi

echo ""
echo "🔗 Adding GitHub remote..."
git remote add origin "$REPO_URL"

echo "📤 Pushing to GitHub..."
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Successfully pushed to GitHub!"
    echo "🌐 Your repository is now available at: $REPO_URL"
    echo ""
    echo "Next steps:"
    echo "1. Add a LICENSE file (MIT or Apache 2.0 recommended)"
    echo "2. Enable GitHub Actions for CI/CD"
    echo "3. Add topics: shrimp, aquaculture, annotation, nlp, machine-learning"
    echo "4. Consider adding GitHub Pages for documentation"
else
    echo ""
    echo "❌ Push failed. You may need to:"
    echo "1. Check your GitHub authentication"
    echo "2. Generate a Personal Access Token at: https://github.com/settings/tokens"
    echo "3. Use the token as your password when prompted"
fi