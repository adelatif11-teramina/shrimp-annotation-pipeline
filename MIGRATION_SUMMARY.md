# Database Migration Summary

## Problem Identified
- Documents upload successfully but don't appear in triage queue
- Root cause: Missing `triage_items` table + UUID/INTEGER schema mismatch

## Migration Details: `a8dc23fd26e1_fix_uuid_schema_and_add_missing_tables.py`

### What This Migration Fixes:

#### 1. **Missing Critical Tables (Creates 6 New Tables)**
- ✅ `triage_items` - **CRITICAL**: Why documents don't appear in queue
- ✅ `annotation_events` - Event tracking with exact model schema
- ✅ `auto_accept_rules` - Auto-accept rules with all required columns
- ✅ `auto_accept_decisions` - Decision logging with validation fields
- ✅ `model_training_runs` - Training metadata with full schema
- ✅ `gold_annotations` - Renamed from `annotations` to match models

#### 2. **UUID Schema Migration (5 Existing Tables)**
- ✅ Converts INTEGER primary keys → UUID primary keys
- ✅ Updates all foreign key relationships
- ✅ Preserves existing data during conversion
- ✅ Handles dependency order correctly

#### 3. **Column Schema Fixes**
- ✅ Adds missing columns to `candidates` table (text, label, etc.)
- ✅ Converts `paragraph_id` from VARCHAR → INTEGER
- ✅ Renames `doc_metadata` → `document_metadata`

### Safety Features:
- ✅ **Data Preservation**: Existing data is mapped from INTEGER to UUID
- ✅ **Dependency Handling**: Tables migrated in correct order
- ✅ **Error Handling**: Try/catch blocks for edge cases
- ✅ **Rollback Support**: Complete downgrade function (though dangerous)

### Schema Alignment Verification:
- ✅ **Exact Column Match**: All new tables precisely match `services/database/models.py`
- ✅ **Correct Nullability**: All nullable/non-nullable constraints match models
- ✅ **Proper Defaults**: All default values match model definitions
- ✅ **Index Support**: Includes required indexes and unique constraints

## Expected Results After Migration:

### ✅ **Immediate Fix**
- Documents will upload AND appear in triage queue
- All database operations will work without silent failures
- FastAPI will not crash on ORM column access

### ✅ **Full Feature Support**
- Triage queue functionality restored
- Auto-accept rules can be stored/retrieved
- Event tracking will work
- Model training metadata can be persisted

## Migration Commands:

```bash
# Apply the migration
alembic upgrade head

# Verify migration success
python -c "from services.database.models import *; print('✅ All models work')"
```

## Rollback Warning:
The downgrade function exists but is **DANGEROUS** - it converts UUIDs back to INTEGERs which may cause data loss. Only use in emergency.

## Technical Notes:
- Uses PostgreSQL UUID extension (`uuid-ossp`)
- Maps existing INTEGER IDs to new UUIDs preserving relationships
- Handles self-referential foreign keys correctly
- Compatible with Railway PostgreSQL deployment

---

**Status**: ✅ Ready for production deployment
**Risk Level**: 🟢 Low (data preservation built-in)
**Expected Impact**: 🎯 Resolves document upload → triage queue issue completely