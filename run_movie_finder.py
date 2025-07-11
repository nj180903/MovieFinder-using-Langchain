#!/usr/bin/env python3
"""
Movie Finder System Runner
Demonstrates the complete two-phase workflow
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"🎬 {title}")
    print("="*60)

def print_step(step_num, description):
    """Print a formatted step"""
    print(f"\n📋 Step {step_num}: {description}")
    print("-" * 40)

def check_dependencies():
    """Check if required dependencies are installed"""
    print_step(1, "Checking Dependencies")
    
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'langchain-community', 
        'langchain-huggingface', 'chromadb'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("💡 Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("\n✅ All dependencies are available!")
    return True

def run_phase_1():
    """Run Phase 1: Vector Store Creation"""
    print_step(2, "Phase 1 - Vector Store Creation")
    
    # Check if vector store already exists
    vector_store_path = './movie_store_vetcor/'
    if os.path.exists(vector_store_path):
        print(f"⚠️ Vector store already exists at: {vector_store_path}")
        choice = input("Do you want to rebuild it? (y/n): ").lower().strip()
        if choice != 'y':
            print("✅ Using existing vector store")
            return True
    
    print("🚀 Running setup.py...")
    
    try:
        # Run setup.py
        result = subprocess.run([sys.executable, 'setup.py'], 
                              capture_output=True, text=True, cwd='.')
        
        if result.returncode == 0:
            print("✅ Phase 1 completed successfully!")
            print(result.stdout)
            return True
        else:
            print("❌ Phase 1 failed!")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running setup.py: {str(e)}")
        return False

def run_phase_2():
    """Run Phase 2: Launch Streamlit App"""
    print_step(3, "Phase 2 - Launching Movie Finder App")
    
    print("🎬 Starting Streamlit application...")
    print("📱 The app will open in your browser automatically")
    print("🔄 Press Ctrl+C to stop the application")
    
    try:
        # Run streamlit app
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'app.py'], 
                      cwd='.')
        
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running Streamlit app: {str(e)}")

def run_test_workflow():
    """Run test workflow with mock data"""
    print_step("TEST", "Running Test Workflow")
    
    print("🧪 Running test with mock data...")
    
    try:
        result = subprocess.run([sys.executable, 'test_vector_store_only.py'], 
                              capture_output=True, text=True, cwd='.')
        
        if result.returncode == 0:
            print("✅ Test workflow completed successfully!")
            print(result.stdout)
            return True
        else:
            print("❌ Test workflow failed!")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running test: {str(e)}")
        return False

def main():
    """Main function"""
    print_header("Movie Finder System")
    
    print("""
🎯 This script will guide you through the complete Movie Finder workflow:

📋 Workflow Overview:
1. Check dependencies
2. Phase 1: Create vector store (setup.py)
3. Phase 2: Launch movie finder app (app.py)

⚙️ Options:
1. Run complete workflow (recommended)
2. Run Phase 1 only (vector store creation)
3. Run Phase 2 only (launch app)
4. Run test workflow (with mock data)
5. Exit
""")
    
    while True:
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            # Complete workflow
            print_header("Complete Workflow")
            
            if not check_dependencies():
                print("\n❌ Please install missing dependencies first!")
                continue
            
            if run_phase_1():
                print("\n🎉 Phase 1 completed! Starting Phase 2...")
                time.sleep(2)
                run_phase_2()
            else:
                print("\n❌ Phase 1 failed! Cannot proceed to Phase 2.")
            
        elif choice == '2':
            # Phase 1 only
            print_header("Phase 1 Only")
            
            if not check_dependencies():
                print("\n❌ Please install missing dependencies first!")
                continue
            
            if run_phase_1():
                print("\n🎉 Phase 1 completed successfully!")
                print("💡 You can now run Phase 2 or use option 3 to launch the app")
            
        elif choice == '3':
            # Phase 2 only
            print_header("Phase 2 Only")
            
            vector_store_path = './movie_store_vetcor/'
            if not os.path.exists(vector_store_path):
                print("❌ Vector store not found!")
                print("🚨 Please run Phase 1 first (option 1 or 2)")
                continue
            
            run_phase_2()
            
        elif choice == '4':
            # Test workflow
            print_header("Test Workflow")
            
            if run_test_workflow():
                print("\n🎉 Test completed successfully!")
            else:
                print("\n❌ Test failed!")
            
        elif choice == '5':
            # Exit
            print("\n👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1-5.")

if __name__ == "__main__":
    main() 