# HepX Services Architecture

This document explains the new decoupled architecture for the HepX hepatitis prediction application.

## Architecture Overview

The application now consists of two independent services:

1. **Python API Service** (FastAPI) - Handles ML predictions
2. **Node.js API Service** (Express) - Handles web requests, authentication, and data management

```
Frontend (Next.js) 
       ↓
Node.js API Server (Express) 
       ↓ HTTP calls
Python API Server (FastAPI)
       ↓
ML Model (TensorFlow/Keras)
```

## Services

### 1. Python API Service (Port 8000)

**File**: `predict.py`

**Features**:
- FastAPI-based REST API
- Pre-loaded ML model for fast predictions
- Automatic model validation
- Interactive API documentation
- CORS enabled for cross-origin requests

**Endpoints**:
- `GET /` - Health check
- `GET /health` - Detailed health status
- `POST /predict` - Make hepatitis predictions
- `GET /docs` - Interactive API documentation

### 2. Node.js API Service (Port 5000)

**File**: `server.js`

**Features**:
- Express.js web server
- User authentication
- File uploads
- Database connectivity
- Communication with Python API via HTTP

**Key Changes**:
- Replaced direct Python process spawning with HTTP calls
- Added Python service health checking
- Enhanced error handling for service communication

## Installation & Setup

### Prerequisites

1. **Python 3.8+** with pip
2. **Node.js 16+** with npm
3. **Trained ML model files** (run `improved_training.py` if not available)

### Python Dependencies

```bash
cd backend
pip install -r requirements.txt
```

Required packages:
- fastapi==0.104.1
- uvicorn[standard]==0.24.0
- pandas==2.1.3
- numpy==1.25.2
- tensorflow==2.15.0
- scikit-learn==1.3.2
- pydantic==2.5.0

### Node.js Dependencies

```bash
cd backend
npm install
```

## Running the Services

### Option 1: Automated Startup (Recommended)

Use the provided startup script to run both services:

```bash
cd backend
python start_services.py
```

With custom ports:
```bash
python start_services.py --python-port 8001 --node-port 5001
```

The script will:
- Check all dependencies
- Verify model files exist
- Start Python API service
- Start Node.js service with proper environment variables
- Monitor both services
- Handle graceful shutdown

### Option 2: Manual Startup

#### Start Python API Service

```bash
cd backend
python predict.py --server --port 8000
```

#### Start Node.js Service

```bash
cd backend
export PYTHON_API_URL=http://localhost:8000
export PORT=5000
node server.js
```

## Environment Variables

### Node.js Service

- `PORT`: Server port (default: 5000)
- `PYTHON_API_URL`: Python API endpoint (default: http://localhost:8000)
- `MONGODB_URI`: MongoDB connection string
- `JWT_SECRET`: JWT secret for authentication

### Python Service

- No additional environment variables needed
- Port is specified via command line argument

## API Usage

### Python API Direct Usage

```bash
# Health check
curl http://localhost:8000/health

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": "31-45",
    "gender": "female",
    "symptoms": {
      "jaundice": true,
      "dark_urine": true,
      "fatigue": 7,
      "nausea": true
    },
    "riskFactors": ["recentTravel"]
  }'
```

### Node.js API Usage

```bash
# Check Python service status
curl http://localhost:5000/api/python-status

# Make prediction via Node.js
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": "31-45",
    "gender": "female",
    "symptoms": {
      "jaundice": true,
      "dark_urine": true,
      "fatigue": 7,
      "nausea": true
    },
    "riskFactors": ["recentTravel"]
  }'
```

## Service Communication

The Node.js service communicates with the Python service via HTTP:

1. **Request Flow**: Frontend → Node.js → Python API → ML Model
2. **Response Flow**: ML Model → Python API → Node.js → Frontend
3. **Error Handling**: Each service handles its own errors and provides meaningful messages
4. **Health Monitoring**: Node.js can check Python service health

## Development

### Running in Development Mode

#### Python API Development

```bash
cd backend
uvicorn predict:app --reload --host 0.0.0.0 --port 8000
```

#### Node.js Development

```bash
cd backend
npm run dev  # or nodemon server.js
```

### Testing

#### Test Python API

```bash
# Start the Python service
python predict.py --server

# Run tests (if available)
pytest tests/
```

#### Test Node.js API

```bash
# Start both services
python start_services.py

# Test endpoints
npm test  # or your preferred testing framework
```

## Monitoring & Logs

### Python Service Logs

- Console output shows request/response details
- Uvicorn provides automatic request logging
- Model loading status and errors

### Node.js Service Logs

- Express request logging
- Python API communication status
- Authentication and database operations

### Health Checking

```bash
# Check Python service health
curl http://localhost:8000/health

# Check Python service via Node.js
curl http://localhost:5000/api/python-status
```

## Troubleshooting

### Common Issues

1. **Python service won't start**
   - Check if model files exist: `ls improved_hepatitis_outputs/models/`
   - Verify Python dependencies: `pip list`
   - Check port availability: `lsof -i :8000`

2. **Node.js can't connect to Python service**
   - Verify Python service is running: `curl http://localhost:8000/health`
   - Check `PYTHON_API_URL` environment variable
   - Verify network connectivity

3. **Model loading errors**
   - Ensure TensorFlow version compatibility
   - Check model file integrity
   - Run `improved_training.py` to regenerate models

### Error Messages

- **"Model not loaded"**: Python service model initialization failed
- **"Python service is not available"**: Node.js can't reach Python service
- **"Model files not found"**: Run training script first

## Production Deployment

### Docker Deployment

Create separate containers for each service:

```dockerfile
# Python API Dockerfile
FROM python:3.9
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "predict.py", "--server", "--port", "8000"]
```

```dockerfile
# Node.js API Dockerfile
FROM node:16
COPY package*.json ./
RUN npm install
COPY . .
EXPOSE 5000
CMD ["node", "server.js"]
```

### Process Management

Use PM2 for Node.js service:

```bash
pm2 start server.js --name "hepx-node"
```

Use systemd for Python service:

```ini
[Unit]
Description=HepX Python API
After=network.target

[Service]
Type=simple
User=hepx
WorkingDirectory=/path/to/backend
ExecStart=/usr/bin/python predict.py --server --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

## Benefits of New Architecture

1. **Scalability**: Services can be scaled independently
2. **Maintainability**: Clear separation of concerns
3. **Development**: Teams can work on services independently
4. **Deployment**: Services can be deployed on different servers
5. **Monitoring**: Each service can be monitored separately
6. **Performance**: Pre-loaded models reduce prediction latency
7. **Reliability**: Service failures don't affect each other