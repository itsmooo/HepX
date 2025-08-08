import fetch from 'node-fetch'

const BASE_URL = 'http://localhost:5000/api'
let adminToken = ''
let testUserId = ''
let testPredictionId = ''

// Test data
const testUser = {
  firstName: 'Test',
  lastName: 'Admin',
  email: 'admin@test.com',
  password: 'password123',
  role: 'admin'
}

const testRegularUser = {
  firstName: 'Test',
  lastName: 'User',
  email: 'user@test.com',
  password: 'password123',
  role: 'user'
}

// Helper function to make authenticated requests
async function makeRequest(endpoint, options = {}) {
  const url = `${BASE_URL}${endpoint}`
  const headers = {
    'Content-Type': 'application/json',
    ...(adminToken && { 'Authorization': `Bearer ${adminToken}` }),
    ...options.headers
  }

  const response = await fetch(url, {
    ...options,
    headers
  })

  const data = await response.json()
  return { response, data }
}

// Test functions
async function testAuth() {
  console.log('\n=== Testing Authentication ===')
  
  // Register admin user
  console.log('1. Registering admin user...')
  const { data: registerData } = await makeRequest('/auth/register', {
    method: 'POST',
    body: JSON.stringify(testUser)
  })
  
  if (registerData.success) {
    console.log('✅ Admin user registered successfully')
    adminToken = registerData.data.token
  } else {
    console.log('❌ Failed to register admin user:', registerData.message)
    return false
  }

  // Login admin user
  console.log('2. Logging in admin user...')
  const { data: loginData } = await makeRequest('/auth/login', {
    method: 'POST',
    body: JSON.stringify({
      email: testUser.email,
      password: testUser.password
    })
  })

  if (loginData.success) {
    console.log('✅ Admin user logged in successfully')
    adminToken = loginData.data.token
  } else {
    console.log('❌ Failed to login admin user:', loginData.message)
    return false
  }

  return true
}

async function testDashboard() {
  console.log('\n=== Testing Dashboard ===')
  
  const { data } = await makeRequest('/admin/dashboard')
  
  if (data.success) {
    console.log('✅ Dashboard stats retrieved successfully')
    console.log('📊 Dashboard data:', data.data)
  } else {
    console.log('❌ Failed to get dashboard stats:', data.message)
  }
}

async function testUserManagement() {
  console.log('\n=== Testing User Management ===')
  
  // Create a test user
  console.log('1. Creating test user...')
  const { data: createData } = await makeRequest('/admin/users', {
    method: 'POST',
    body: JSON.stringify(testRegularUser)
  })
  
  if (createData.success) {
    console.log('✅ Test user created successfully')
    testUserId = createData.data._id
  } else {
    console.log('❌ Failed to create test user:', createData.message)
    return
  }

  // Get all users
  console.log('2. Getting all users...')
  const { data: usersData } = await makeRequest('/admin/users')
  
  if (usersData.success) {
    console.log('✅ Users retrieved successfully')
    console.log(`📋 Found ${usersData.data.length} users`)
  } else {
    console.log('❌ Failed to get users:', usersData.message)
  }

  // Get specific user
  console.log('3. Getting specific user...')
  const { data: userData } = await makeRequest(`/admin/users/${testUserId}`)
  
  if (userData.success) {
    console.log('✅ User retrieved successfully')
    console.log('👤 User data:', userData.data)
  } else {
    console.log('❌ Failed to get user:', userData.message)
  }

  // Update user
  console.log('4. Updating user...')
  const { data: updateData } = await makeRequest(`/admin/users/${testUserId}`, {
    method: 'PUT',
    body: JSON.stringify({
      firstName: 'Updated',
      lastName: 'User'
    })
  })
  
  if (updateData.success) {
    console.log('✅ User updated successfully')
  } else {
    console.log('❌ Failed to update user:', updateData.message)
  }
}

async function testPredictionManagement() {
  console.log('\n=== Testing Prediction Management ===')
  
  // Get all predictions
  console.log('1. Getting all predictions...')
  const { data: predictionsData } = await makeRequest('/admin/predictions')
  
  if (predictionsData.success) {
    console.log('✅ Predictions retrieved successfully')
    console.log(`📊 Found ${predictionsData.data.length} predictions`)
    
    if (predictionsData.data.length > 0) {
      testPredictionId = predictionsData.data[0]._id
    }
  } else {
    console.log('❌ Failed to get predictions:', predictionsData.message)
  }

  // Get specific prediction if available
  if (testPredictionId) {
    console.log('2. Getting specific prediction...')
    const { data: predictionData } = await makeRequest(`/admin/predictions/${testPredictionId}`)
    
    if (predictionData.success) {
      console.log('✅ Prediction retrieved successfully')
      console.log('🔮 Prediction data:', predictionData.data)
    } else {
      console.log('❌ Failed to get prediction:', predictionData.message)
    }

    // Update prediction
    console.log('3. Updating prediction...')
    const { data: updateData } = await makeRequest(`/admin/predictions/${testPredictionId}`, {
      method: 'PUT',
      body: JSON.stringify({
        notes: 'Test note from admin'
      })
    })
    
    if (updateData.success) {
      console.log('✅ Prediction updated successfully')
    } else {
      console.log('❌ Failed to update prediction:', updateData.message)
    }
  }
}

async function cleanup() {
  console.log('\n=== Cleaning Up ===')
  
  // Delete test user
  if (testUserId) {
    console.log('1. Deleting test user...')
    const { data: deleteUserData } = await makeRequest(`/admin/users/${testUserId}`, {
      method: 'DELETE'
    })
    
    if (deleteUserData.success) {
      console.log('✅ Test user deleted successfully')
    } else {
      console.log('❌ Failed to delete test user:', deleteUserData.message)
    }
  }

  // Delete test prediction
  if (testPredictionId) {
    console.log('2. Deleting test prediction...')
    const { data: deletePredictionData } = await makeRequest(`/admin/predictions/${testPredictionId}`, {
      method: 'DELETE'
    })
    
    if (deletePredictionData.success) {
      console.log('✅ Test prediction deleted successfully')
    } else {
      console.log('❌ Failed to delete test prediction:', deletePredictionData.message)
    }
  }
}

// Main test function
async function runTests() {
  console.log('🚀 Starting Admin API Tests...')
  
  try {
    // Test authentication
    const authSuccess = await testAuth()
    if (!authSuccess) {
      console.log('❌ Authentication failed, stopping tests')
      return
    }

    // Test dashboard
    await testDashboard()

    // Test user management
    await testUserManagement()

    // Test prediction management
    await testPredictionManagement()

    // Cleanup
    await cleanup()

    console.log('\n🎉 All tests completed!')
  } catch (error) {
    console.error('❌ Test failed with error:', error.message)
  }
}

// Run tests if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runTests()
}

export { runTests }
