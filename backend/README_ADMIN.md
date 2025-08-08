# Admin API Documentation

This document describes the admin-only API endpoints for managing users and predictions.

## Authentication

All admin endpoints require:
1. Valid JWT token in Authorization header: `Bearer <token>`
2. User must have `admin` role

## Admin Endpoints

### Dashboard Statistics

**GET** `/api/admin/dashboard`

Get overall system statistics including user counts, prediction counts, and recent activity.

**Response:**
```json
{
  "success": true,
  "data": {
    "totalUsers": 150,
    "totalPredictions": 1250,
    "activeUsers": 145,
    "adminUsers": 3,
    "predictionsByClass": [
      { "_id": "Hepatitis A", "count": 800 },
      { "_id": "Hepatitis C", "count": 450 }
    ],
    "recentPredictions": 25,
    "recentUsers": 5
  }
}
```

### User Management

#### Get All Users

**GET** `/api/admin/users?page=1&limit=10&search=john&role=user&isActive=true`

**Query Parameters:**
- `page` (optional): Page number (default: 1)
- `limit` (optional): Items per page (default: 10)
- `search` (optional): Search by name or email
- `role` (optional): Filter by role (user/admin)
- `isActive` (optional): Filter by active status (true/false)

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "_id": "user_id",
      "firstName": "John",
      "lastName": "Doe",
      "email": "john@example.com",
      "role": "user",
      "isActive": true,
      "lastLogin": "2024-01-15T10:30:00.000Z",
      "predictions": [
        {
          "_id": "prediction_id",
          "prediction": {
            "predicted_class": "Hepatitis A",
            "confidence": 0.85
          },
          "createdAt": "2024-01-15T10:30:00.000Z"
        }
      ],
      "createdAt": "2024-01-01T00:00:00.000Z"
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 10,
    "total": 150,
    "pages": 15
  }
}
```

#### Get User by ID

**GET** `/api/admin/users/:id`

**Response:**
```json
{
  "success": true,
  "data": {
    "_id": "user_id",
    "firstName": "John",
    "lastName": "Doe",
    "email": "john@example.com",
    "role": "user",
    "isActive": true,
    "lastLogin": "2024-01-15T10:30:00.000Z",
    "predictions": [...],
    "createdAt": "2024-01-01T00:00:00.000Z"
  }
}
```

#### Create User

**POST** `/api/admin/users`

**Body:**
```json
{
  "firstName": "Jane",
  "lastName": "Smith",
  "email": "jane@example.com",
  "password": "password123",
  "role": "user"
}
```

**Response:**
```json
{
  "success": true,
  "message": "User created successfully",
  "data": {
    "_id": "new_user_id",
    "firstName": "Jane",
    "lastName": "Smith",
    "email": "jane@example.com",
    "role": "user",
    "isActive": true
  }
}
```

#### Update User

**PUT** `/api/admin/users/:id`

**Body:**
```json
{
  "firstName": "Jane",
  "lastName": "Johnson",
  "email": "jane.johnson@example.com",
  "role": "admin",
  "isActive": true
}
```

**Response:**
```json
{
  "success": true,
  "message": "User updated successfully",
  "data": {
    "_id": "user_id",
    "firstName": "Jane",
    "lastName": "Johnson",
    "email": "jane.johnson@example.com",
    "role": "admin",
    "isActive": true
  }
}
```

#### Delete User

**DELETE** `/api/admin/users/:id`

**Response:**
```json
{
  "success": true,
  "message": "User deleted successfully"
}
```

### Prediction Management

#### Get All Predictions

**GET** `/api/admin/predictions?page=1&limit=10&userId=user_id&predictedClass=Hepatitis A&status=completed`

**Query Parameters:**
- `page` (optional): Page number (default: 1)
- `limit` (optional): Items per page (default: 10)
- `userId` (optional): Filter by user ID
- `predictedClass` (optional): Filter by prediction class
- `status` (optional): Filter by status (pending/completed/failed)

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "_id": "prediction_id",
      "user": {
        "_id": "user_id",
        "firstName": "John",
        "lastName": "Doe",
        "email": "john@example.com"
      },
      "age": "31-45",
      "gender": "male",
      "symptoms": {
        "jaundice": true,
        "dark_urine": false,
        "pain": 5,
        "fatigue": 3,
        "nausea": true,
        "vomiting": false,
        "fever": true,
        "loss_of_appetite": false,
        "joint_pain": false
      },
      "riskFactors": ["recent_travel"],
      "prediction": {
        "predicted_class": "Hepatitis A",
        "confidence": 0.85,
        "probability_Hepatitis_A": 0.85,
        "probability_Hepatitis_C": 0.15
      },
      "status": "completed",
      "notes": "High confidence prediction",
      "createdAt": "2024-01-15T10:30:00.000Z"
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 10,
    "total": 1250,
    "pages": 125
  }
}
```

#### Get Prediction by ID

**GET** `/api/admin/predictions/:id`

**Response:**
```json
{
  "success": true,
  "data": {
    "_id": "prediction_id",
    "user": {
      "_id": "user_id",
      "firstName": "John",
      "lastName": "Doe",
      "email": "john@example.com"
    },
    "age": "31-45",
    "gender": "male",
    "symptoms": {...},
    "riskFactors": ["recent_travel"],
    "prediction": {
      "predicted_class": "Hepatitis A",
      "confidence": 0.85,
      "probability_Hepatitis_A": 0.85,
      "probability_Hepatitis_C": 0.15
    },
    "status": "completed",
    "notes": "High confidence prediction",
    "createdAt": "2024-01-15T10:30:00.000Z"
  }
}
```

#### Update Prediction

**PUT** `/api/admin/predictions/:id`

**Body:**
```json
{
  "status": "completed",
  "notes": "Updated notes for this prediction"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Prediction updated successfully",
  "data": {
    "_id": "prediction_id",
    "status": "completed",
    "notes": "Updated notes for this prediction"
  }
}
```

#### Delete Prediction

**DELETE** `/api/admin/predictions/:id`

**Response:**
```json
{
  "success": true,
  "message": "Prediction deleted successfully"
}
```

## Error Responses

All endpoints return consistent error responses:

```json
{
  "success": false,
  "message": "Error description",
  "error": "Detailed error message (in development)"
}
```

Common HTTP status codes:
- `200`: Success
- `201`: Created
- `400`: Bad Request
- `401`: Unauthorized
- `403`: Forbidden
- `404`: Not Found
- `500`: Internal Server Error

## Usage Examples

### Using curl

```bash
# Get all users
curl -H "Authorization: Bearer YOUR_JWT_TOKEN" \
     http://localhost:5000/api/admin/users

# Create a new user
curl -X POST \
     -H "Authorization: Bearer YOUR_JWT_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"firstName":"John","lastName":"Doe","email":"john@example.com","password":"password123","role":"user"}' \
     http://localhost:5000/api/admin/users

# Get dashboard stats
curl -H "Authorization: Bearer YOUR_JWT_TOKEN" \
     http://localhost:5000/api/admin/dashboard
```

### Using JavaScript/Fetch

```javascript
// Get all users
const response = await fetch('http://localhost:5000/api/admin/users', {
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  }
});

const data = await response.json();
console.log(data);
```
