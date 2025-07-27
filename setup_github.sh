#!/bin/bash

# GitHub Repository Setup Script for MLB Betting RL System

echo "🚀 Setting up GitHub repository connection..."
echo ""

# Remove the placeholder remote
git remote remove origin 2>/dev/null

echo "📋 Please provide your GitHub repository details:"
echo ""

# Get GitHub username
read -p "Enter your GitHub username: " GITHUB_USERNAME

# Get repository name
read -p "Enter your repository name: " REPO_NAME

# Construct the repository URL
REPO_URL="https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"

echo ""
echo "🔗 Adding remote origin: $REPO_URL"
git remote add origin "$REPO_URL"

echo ""
echo "📤 Pushing to GitHub..."
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Successfully pushed to GitHub!"
    echo "🌐 Your repository is now available at: https://github.com/$GITHUB_USERNAME/$REPO_NAME"
    echo ""
    echo "📋 Next steps:"
    echo "1. Set up GitHub Actions (optional)"
    echo "2. Add collaborators (if needed)"
    echo "3. Set up branch protection rules"
    echo "4. Configure deployment secrets"
else
    echo ""
    echo "❌ Failed to push to GitHub. Please check:"
    echo "1. Repository exists on GitHub"
    echo "2. You have write access"
    echo "3. GitHub credentials are configured"
    echo ""
    echo "🔧 To configure GitHub credentials:"
    echo "   git config --global user.name 'Your Name'"
    echo "   git config --global user.email 'your.email@example.com'"
fi 