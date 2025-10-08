# Draft Management Guide

The annotation system includes an auto-save feature that creates drafts to prevent data loss. This guide explains how to manage the "Draft Found" dialog and related settings.

## Understanding the Draft System

### What are Drafts?
- **Drafts** are automatically saved copies of your annotation work in progress
- They save every 30 seconds while you're working
- They include entities, relations, topics, notes, and confidence levels
- They're stored locally in your browser and optionally on the server

### When do Drafts Appear?
You'll see the "Draft Found" dialog when:
- You return to an annotation item that has unsaved work
- You refresh the page while working on an annotation
- You close and reopen your browser with pending work

## Managing the Draft Dialog

### Option 1: User Settings (Recommended)

Navigate to **Settings → General → Draft Management** to configure:

#### Draft Behavior Options:
- **Ask what to do** (Default): Shows the dialog asking if you want to restore or discard
- **Always restore drafts**: Automatically loads your previous work without asking
- **Always discard drafts**: Automatically discards drafts and starts fresh

#### Additional Settings:
- **Draft Retention**: How many days to keep drafts (default: 7 days)
- **Show Draft Dialog**: Toggle to disable the dialog entirely

#### How to Set Up:
1. Go to **Settings** page
2. Click **General** tab
3. Scroll to **Draft Management** section
4. Choose your preferred **Draft Behavior**
5. Click **Save Settings**

### Option 2: Manual Draft Cleanup

If you want to clear all existing drafts:

1. Go to **Settings → General → Draft Management**
2. Click **Clear All Drafts** button
3. Confirm the action
4. All saved drafts will be permanently deleted

### Option 3: Browser Storage Cleanup

For advanced users, you can manually clear drafts from browser storage:

1. Open browser Developer Tools (F12)
2. Go to Application/Storage tab
3. Find Local Storage → your domain
4. Delete keys starting with `annotation_draft_`

## Recommendations by Use Case

### For Regular Annotators:
- **Setting**: "Always restore drafts" 
- **Benefit**: Seamlessly continue where you left off
- **Best for**: Users who work on annotations across multiple sessions

### For Quality Control/Reviewers:
- **Setting**: "Always discard drafts"
- **Benefit**: Always start with fresh, original data
- **Best for**: Reviewers who need to see original candidates without user modifications

### For Testing/Development:
- **Setting**: "Ask what to do"
- **Benefit**: Full control over each decision
- **Best for**: Testing different scenarios or troubleshooting

### For Shared Computers:
- **Setting**: "Always discard drafts" + Clear drafts regularly
- **Benefit**: Prevents mixing work from different users
- **Best for**: Lab computers or shared workstations

## Technical Details

### Storage Locations:
- **Local Storage**: Browser localStorage with keys like `annotation_draft_[item_id]`
- **Server Storage**: Optional backup via API endpoint `/api/annotations/draft`
- **Settings Storage**: User preferences in `annotation_settings` localStorage key

### Auto-cleanup:
- Drafts older than the retention period are automatically removed
- Expired drafts are cleaned up when the page loads
- Completed annotations automatically clear their drafts

### Data Included in Drafts:
```json
{
  "itemId": "annotation_item_id",
  "timestamp": "2025-01-26T10:30:00Z",
  "data": {
    "entities": [...],
    "relations": [...], 
    "topics": [...],
    "notes": "user notes",
    "confidence": "high"
  }
}
```

## Troubleshooting

### Draft Dialog Keeps Appearing:
1. Check if auto-save is working (look for save status indicators)
2. Verify your draft behavior setting in Settings
3. Clear all drafts if the issue persists

### Drafts Not Being Saved:
1. Check browser localStorage quota (may be full)
2. Verify auto-save is enabled in Settings
3. Check browser console for errors

### Lost Work Despite Auto-save:
1. Check if drafts exist: Settings → Clear All Drafts (don't click, just check)
2. Look for expired drafts (check retention period)
3. Check if the item ID changed

### Performance Issues:
1. Clear old drafts regularly
2. Reduce draft retention period
3. Disable server backup if using local-only mode

## API Endpoints

For developers integrating with the draft system:

- `POST /api/annotations/draft` - Save draft to server
- `GET /api/annotations/draft/{item_id}` - Retrieve draft
- `DELETE /api/annotations/draft/{item_id}` - Delete draft

## Best Practices

1. **Regular Cleanup**: Clear drafts weekly to maintain performance
2. **Consistent Settings**: Use the same draft behavior across team members
3. **Backup Strategy**: Enable server drafts for important work
4. **Team Guidelines**: Establish team policies for draft management
5. **Testing**: Test your preferred settings with sample data first

## Conclusion

The draft system is designed to protect your work while being flexible enough to adapt to different workflows. Choose the settings that best match your annotation style and update them as your needs change.

For additional help, refer to the main annotation guidelines or contact your system administrator.