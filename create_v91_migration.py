#!/usr/bin/env python3
"""
Database Migration Script for Trading Bot v9.1
==============================================

This script creates the Alembic migration for v9.1 database schema changes.
Run this script to generate the migration file automatically.

Usage:
    python create_v91_migration.py

Requirements:
    - Alembic installed
    - Database models updated in database.py
    - alembic.ini configured
"""

import os
import sys
import subprocess
from datetime import datetime


def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(
            command, shell=True, check=True, capture_output=True, text=True
        )
        print(f"✅ {description} - SUCCESS")
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        print(f"   Error: {e.stderr.strip()}")
        return False


def check_alembic_setup():
    """Check if Alembic is properly set up."""
    print("🔍 Checking Alembic setup...")

    # Check if alembic.ini exists
    if not os.path.exists("alembic.ini"):
        print("❌ alembic.ini not found!")
        return False

    # Check if alembic directory exists
    if not os.path.exists("alembic"):
        print("❌ alembic directory not found!")
        return False

    # Check if versions directory exists
    if not os.path.exists("alembic/versions"):
        print("❌ alembic/versions directory not found!")
        return False

    print("✅ Alembic setup verified")
    return True


def create_migration():
    """Create the v9.1 migration."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    migration_message = f"v91_enhanced_trading_features_{timestamp}"

    print(f"🚀 Creating migration: {migration_message}")

    # Generate migration
    command = f'alembic revision --autogenerate -m "{migration_message}"'
    if not run_command(command, "Generating migration"):
        return False

    print("✅ Migration created successfully!")
    print("\n📋 Next steps:")
    print("1. Review the generated migration file in alembic/versions/")
    print("2. Run: alembic upgrade head")
    print("3. Verify database schema changes")

    return True


def main():
    """Main migration creation process."""
    print("=" * 60)
    print("🗄️  TRADING BOT v9.1 - DATABASE MIGRATION CREATOR")
    print("=" * 60)

    # Check if we're in the right directory
    if not os.path.exists("database.py"):
        print("❌ database.py not found! Run this script from the bot directory.")
        sys.exit(1)

    # Check Alembic setup
    if not check_alembic_setup():
        print("\n❌ Alembic not properly set up!")
        print("Run: alembic init alembic")
        sys.exit(1)

    # Create migration
    if create_migration():
        print("\n🎉 Migration creation completed!")
    else:
        print("\n❌ Migration creation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
