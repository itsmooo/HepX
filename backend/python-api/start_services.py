#!/usr/bin/env python3
"""
HepX Services Startup Script
============================

This script helps you start both the Python API server and Node.js server
for the hepatitis prediction application.

Usage:
    python start_services.py [--python-port PORT] [--node-port PORT]

Examples:
    python start_services.py                    # Use default ports
    python start_services.py --python-port 8001 # Custom Python port
    python start_services.py --node-port 5001   # Custom Node port
"""

import subprocess
import sys
import time
import argparse
import signal
import os
from pathlib import Path

class ServiceManager:
    def __init__(self, python_port=8000, node_port=5000):
        self.python_port = python_port
        self.node_port = node_port
        self.processes = []
        
    def check_dependencies(self):
        """Check if required dependencies are installed"""
        print("🔍 Checking dependencies...")
        
        # Check Python dependencies
        try:
            import fastapi
            import uvicorn
            import tensorflow
            import pandas
            import numpy
            import sklearn
            print("✅ Python dependencies found")
        except ImportError as e:
            print(f"❌ Missing Python dependency: {e}")
            print("💡 Install with: pip install -r requirements.txt")
            return False
            
        # Check Node.js
        try:
            result = subprocess.run(['node', '--version'], 
                                  capture_output=True, text=True, check=True)
            print(f"✅ Node.js found: {result.stdout.strip()}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Node.js not found")
            print("💡 Install Node.js from: https://nodejs.org/")
            return False
            
        # Check if model files exist
        possible_model_paths = [
            Path("./enhanced_hepatitis_outputs/models/best_enhanced_model.keras"),
            Path("./hepatitis_thesis_outputs/models/best_model.keras"),
            Path("./hepatitis_thesis_outputs/models/hepatitis_model.keras")
        ]
        
        possible_artifacts_paths = [
            Path("./enhanced_hepatitis_outputs/models/preprocessing_artifacts.pkl"),
            Path("./hepatitis_thesis_outputs/models/preprocessing_artifacts.pkl")
        ]
        
        # Find existing model and artifacts
        model_path = None
        artifacts_path = None
        
        for path in possible_model_paths:
            if path.exists():
                model_path = path
                break
                
        for path in possible_artifacts_paths:
            if path.exists():
                artifacts_path = path
                break
        
        if not model_path or not artifacts_path:
            print("❌ Model files not found")
            print("💡 Available models:", [str(p) for p in possible_model_paths if p.exists()])
            print("💡 Available artifacts:", [str(p) for p in possible_artifacts_paths if p.exists()])
            print("💡 Run improved_training.py or hepatitis_thesis_training.py first to generate the model")
            return False
            
        print("✅ Model files found")
        return True
    
    def start_python_service(self):
        """Start the Python FastAPI service"""
        print(f"🐍 Starting Python API server on port {self.python_port}...")
        
        try:
            process = subprocess.Popen([
                sys.executable, 'predict.py', '--server', '--port', str(self.python_port)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            self.processes.append(('Python API', process))
            
            # Wait a moment and check if it started successfully
            time.sleep(2)
            if process.poll() is None:
                print(f"✅ Python API server started successfully on http://localhost:{self.python_port}")
                print(f"📖 API Documentation: http://localhost:{self.python_port}/docs")
                return True
            else:
                stdout, stderr = process.communicate()
                print(f"❌ Python API server failed to start")
                print(f"Error: {stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Failed to start Python service: {e}")
            return False
    
    def start_node_service(self):
        """Start the Node.js Express service"""
        print(f"🟢 Starting Node.js server on port {self.node_port}...")
        
        try:
            # Set environment variables
            env = os.environ.copy()
            env['PORT'] = str(self.node_port)
            env['PYTHON_API_URL'] = f'http://localhost:{self.python_port}'
            
            process = subprocess.Popen([
                'node', 'server.js'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
            
            self.processes.append(('Node.js Server', process))
            
            # Wait a moment and check if it started successfully
            time.sleep(2)
            if process.poll() is None:
                print(f"✅ Node.js server started successfully on http://localhost:{self.node_port}")
                print(f"🔗 API Status: http://localhost:{self.node_port}/api/status")
                print(f"🔗 Python Service Status: http://localhost:{self.node_port}/api/python-status")
                return True
            else:
                stdout, stderr = process.communicate()
                print(f"❌ Node.js server failed to start")
                print(f"Error: {stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Failed to start Node.js service: {e}")
            return False
    
    def monitor_services(self):
        """Monitor running services"""
        print("\n🔄 Services are running. Press Ctrl+C to stop all services.\n")
        print("📊 Service Status:")
        print(f"   🐍 Python API: http://localhost:{self.python_port}")
        print(f"   🟢 Node.js API: http://localhost:{self.node_port}")
        print(f"   📖 API Docs: http://localhost:{self.python_port}/docs")
        print("\n" + "="*50)
        
        try:
            while True:
                # Check if all processes are still running
                running_count = 0
                for name, process in self.processes:
                    if process.poll() is None:
                        running_count += 1
                    else:
                        print(f"⚠️  {name} has stopped unexpectedly")
                
                if running_count == 0:
                    print("❌ All services have stopped")
                    break
                    
                time.sleep(5)
                
        except KeyboardInterrupt:
            print("\n🛑 Shutdown signal received...")
            self.stop_services()
    
    def stop_services(self):
        """Stop all running services"""
        print("🔄 Stopping services...")
        
        for name, process in self.processes:
            if process.poll() is None:
                print(f"🛑 Stopping {name}...")
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=5)
                    print(f"✅ {name} stopped gracefully")
                except subprocess.TimeoutExpired:
                    print(f"⚠️  Force killing {name}...")
                    process.kill()
                    process.wait()
                    print(f"✅ {name} force stopped")
        
        print("✅ All services stopped")
    
    def run(self):
        """Main execution flow"""
        print("🚀 HepX Services Manager")
        print("=" * 30)
        
        # Check dependencies
        if not self.check_dependencies():
            print("\n❌ Dependency check failed. Please fix the issues above.")
            return False
        
        print("\n🔄 Starting services...")
        
        # Start Python service first
        if not self.start_python_service():
            print("\n❌ Failed to start Python service")
            return False
        
        # Wait a bit for Python service to fully initialize
        time.sleep(3)
        
        # Start Node.js service
        if not self.start_node_service():
            print("\n❌ Failed to start Node.js service")
            self.stop_services()
            return False
        
        # Monitor services
        self.monitor_services()
        return True

def main():
    parser = argparse.ArgumentParser(description='Start HepX services')
    parser.add_argument('--python-port', type=int, default=8000,
                       help='Port for Python API service (default: 8000)')
    parser.add_argument('--node-port', type=int, default=5000,
                       help='Port for Node.js service (default: 5000)')
    
    args = parser.parse_args()
    
    # Setup signal handler
    manager = ServiceManager(args.python_port, args.node_port)
    
    def signal_handler(sig, frame):
        print("\n🛑 Received interrupt signal")
        manager.stop_services()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the service manager
    success = manager.run()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()