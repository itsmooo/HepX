# HepaPredict Backend

This is the backend server for the HepaPredict application with MongoDB authentication and hepatitis prediction functionality.

## Features

- **User Authentication**: Register, login, logout with JWT tokens
- **User Management**: Profile updates, password changes
- **Hepatitis Prediction**: AI-powered prediction using machine learning
- **MongoDB Integration**: User data storage and management
- **Security**: Password hashing, JWT authentication, protected routes

## Setup Instructions

### 1. Install Dependencies

```bash
npm install
```

### 2. Environment Variables

Create a `.env` file in the backend directory with the following variables:

```env
NODE_ENV=development
PORT=5000
MONGODB_URI=mongodb://localhost:27017/hepapredict
JWT_SECRET=your-super-secret-jwt-key-change-this-in-production
```

### 3. MongoDB Setup

Make sure MongoDB is running on your system. You can use:
- Local MongoDB installation
- MongoDB Atlas (cloud)
- Docker MongoDB container

### 4. Start the Server

```bash
npm start
```

The server will run on `http://localhost:5000`

## API Endpoints

### Authentication Routes

- `POST /api/auth/register` - Register a new user
- `POST /api/auth/login` - Login user
- `GET /api/auth/me` - Get current user (protected)
- `PUT /api/auth/profile` - Update user profile (protected)
- `PUT /api/auth/change-password` - Change password (protected)
- `POST /api/auth/logout` - Logout user (protected)

### Prediction Routes

- `POST /api/predict` - Make hepatitis prediction
- `POST /api/train-model` - Train the ML model
- `GET /api/model-results` - Get model results
- `GET /api/visualizations` - Get visualization images

## User Model

```javascript
{
  firstName: String (required),
  lastName: String (required),
  email: String (required, unique),
  password: String (required, hashed),
  role: String (enum: ['user', 'admin']),
  isActive: Boolean,
  lastLogin: Date,
  predictions: [ObjectId],
  timestamps: true
}
```

## Security Features

- Password hashing with bcryptjs
- JWT token authentication
- Protected routes with middleware
- Input validation
- Error handling

## Database Schema

The application uses MongoDB with the following collections:
- `users` - User accounts and profiles
- `predictions` - User prediction history (future feature)

## Frontend Integration

The frontend can connect to these endpoints using the API URL:
`http://localhost:5000/api`

Make sure to include the JWT token in the Authorization header for protected routes:
```
Authorization: Bearer <token>
``` 