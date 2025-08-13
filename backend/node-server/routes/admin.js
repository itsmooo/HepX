import express from 'express'
import User from '../models/User.js'
import Prediction from '../models/Prediction.js'
import { protect, authorize } from '../middleware/auth.js'

const router = express.Router()

// @desc    Get all users (with pagination and filtering)
// @route   GET /api/admin/users
// @access  Admin only
export const getAllUsers = async (req, res) => {
  try {
    const page = parseInt(req.query.page) || 1
    const limit = parseInt(req.query.limit) || 10
    const search = req.query.search || ''
    const role = req.query.role || ''
    const isActive = req.query.isActive

    const skip = (page - 1) * limit

    // Build query
    let query = {}
    
    if (search) {
      query.$or = [
        { firstName: { $regex: search, $options: 'i' } },
        { lastName: { $regex: search, $options: 'i' } },
        { email: { $regex: search, $options: 'i' } }
      ]
    }

    if (role) {
      query.role = role
    }

    if (isActive !== undefined) {
      query.isActive = isActive === 'true'
    }

    const users = await User.find(query)
      .select('-password')
      .populate('predictions', 'prediction.predicted_class prediction.confidence createdAt')
      .sort({ createdAt: -1 })
      .skip(skip)
      .limit(limit)

    const total = await User.countDocuments(query)

    res.json({
      success: true,
      data: users,
      pagination: {
        page,
        limit,
        total,
        pages: Math.ceil(total / limit)
      }
    })
  } catch (error) {
    console.error('Get all users error:', error)
    res.status(500).json({
      success: false,
      message: 'Server error',
      error: error.message
    })
  }
}

// @desc    Get single user by ID
// @route   GET /api/admin/users/:id
// @access  Admin only
export const getUserById = async (req, res) => {
  try {
    const user = await User.findById(req.params.id)
      .select('-password')
      .populate('predictions')

    if (!user) {
      return res.status(404).json({
        success: false,
        message: 'User not found'
      })
    }

    res.json({
      success: true,
      data: user
    })
  } catch (error) {
    console.error('Get user by ID error:', error)
    res.status(500).json({
      success: false,
      message: 'Server error',
      error: error.message
    })
  }
}

// @desc    Create new user
// @route   POST /api/admin/users
// @access  Admin only
export const createUser = async (req, res) => {
  try {
    const { firstName, lastName, email, password, role } = req.body

    // Check if user exists
    const userExists = await User.findOne({ email })
    if (userExists) {
      return res.status(400).json({
        success: false,
        message: 'User already exists with this email'
      })
    }

    // Create user
    const user = await User.create({
      firstName,
      lastName,
      email,
      password,
      role: role || 'user'
    })

    res.status(201).json({
      success: true,
      message: 'User created successfully',
      data: user
    })
  } catch (error) {
    console.error('Create user error:', error)
    res.status(500).json({
      success: false,
      message: 'Server error',
      error: error.message
    })
  }
}

// @desc    Update user
// @route   PUT /api/admin/users/:id
// @access  Admin only
export const updateUser = async (req, res) => {
  try {
    const { firstName, lastName, email, role, isActive } = req.body

    const user = await User.findById(req.params.id)
    if (!user) {
      return res.status(404).json({
        success: false,
        message: 'User not found'
      })
    }

    // Check if email is being changed and if it already exists
    if (email && email !== user.email) {
      const emailExists = await User.findOne({ email })
      if (emailExists) {
        return res.status(400).json({
          success: false,
          message: 'Email already exists'
        })
      }
    }

    // Update fields
    user.firstName = firstName || user.firstName
    user.lastName = lastName || user.lastName
    user.email = email || user.email
    user.role = role || user.role
    user.isActive = isActive !== undefined ? isActive : user.isActive

    const updatedUser = await user.save()

    res.json({
      success: true,
      message: 'User updated successfully',
      data: updatedUser
    })
  } catch (error) {
    console.error('Update user error:', error)
    res.status(500).json({
      success: false,
      message: 'Server error',
      error: error.message
    })
  }
}

// @desc    Delete user
// @route   DELETE /api/admin/users/:id
// @access  Admin only
export const deleteUser = async (req, res) => {
  try {
    const user = await User.findById(req.params.id)
    if (!user) {
      return res.status(404).json({
        success: false,
        message: 'User not found'
      })
    }

    // Check if user is trying to delete themselves
    if (user._id.toString() === req.user._id.toString()) {
      return res.status(400).json({
        success: false,
        message: 'Cannot delete your own account'
      })
    }

    // Delete associated predictions
    await Prediction.deleteMany({ user: user._id })

    // Delete user
    await User.findByIdAndDelete(req.params.id)

    res.json({
      success: true,
      message: 'User deleted successfully'
    })
  } catch (error) {
    console.error('Delete user error:', error)
    res.status(500).json({
      success: false,
      message: 'Server error',
      error: error.message
    })
  }
}

// @desc    Get all predictions (with pagination and filtering)
// @route   GET /api/admin/predictions
// @access  Admin only
export const getAllPredictions = async (req, res) => {
  try {
    const page = parseInt(req.query.page) || 1
    const limit = parseInt(req.query.limit) || 10
    const userId = req.query.userId || ''
    const predictedClass = req.query.predictedClass || ''
    const status = req.query.status || ''

    const skip = (page - 1) * limit

    // Build query
    let query = {}
    
    if (userId) {
      query.user = userId
    }

    if (predictedClass) {
      query['prediction.predicted_class'] = predictedClass
    }

    if (status) {
      query.status = status
    }

    const predictions = await Prediction.find(query)
      .populate('user', 'firstName lastName email')
      .sort({ createdAt: -1 })
      .skip(skip)
      .limit(limit)

    const total = await Prediction.countDocuments(query)

    res.json({
      success: true,
      data: predictions,
      pagination: {
        page,
        limit,
        total,
        pages: Math.ceil(total / limit)
      }
    })
  } catch (error) {
    console.error('Get all predictions error:', error)
    res.status(500).json({
      success: false,
      message: 'Server error',
      error: error.message
    })
  }
}

// @desc    Get single prediction by ID
// @route   GET /api/admin/predictions/:id
// @access  Admin only
export const getPredictionById = async (req, res) => {
  try {
    const prediction = await Prediction.findById(req.params.id)
      .populate('user', 'firstName lastName email')

    if (!prediction) {
      return res.status(404).json({
        success: false,
        message: 'Prediction not found'
      })
    }

    res.json({
      success: true,
      data: prediction
    })
  } catch (error) {
    console.error('Get prediction by ID error:', error)
    res.status(500).json({
      success: false,
      message: 'Server error',
      error: error.message
    })
  }
}

// @desc    Update prediction
// @route   PUT /api/admin/predictions/:id
// @access  Admin only
export const updatePrediction = async (req, res) => {
  try {
    const { status, notes } = req.body

    const prediction = await Prediction.findById(req.params.id)
    if (!prediction) {
      return res.status(404).json({
        success: false,
        message: 'Prediction not found'
      })
    }

    // Update fields
    if (status) prediction.status = status
    if (notes !== undefined) prediction.notes = notes

    const updatedPrediction = await prediction.save()

    res.json({
      success: true,
      message: 'Prediction updated successfully',
      data: updatedPrediction
    })
  } catch (error) {
    console.error('Update prediction error:', error)
    res.status(500).json({
      success: false,
      message: 'Server error',
      error: error.message
    })
  }
}

// @desc    Delete prediction
// @route   DELETE /api/admin/predictions/:id
// @access  Admin only
export const deletePrediction = async (req, res) => {
  try {
    const prediction = await Prediction.findById(req.params.id)
    if (!prediction) {
      return res.status(404).json({
        success: false,
        message: 'Prediction not found'
      })
    }

    await Prediction.findByIdAndDelete(req.params.id)

    res.json({
      success: true,
      message: 'Prediction deleted successfully'
    })
  } catch (error) {
    console.error('Delete prediction error:', error)
    res.status(500).json({
      success: false,
      message: 'Server error',
      error: error.message
    })
  }
}

// @desc    Get dashboard statistics
// @route   GET /api/admin/dashboard
// @access  Admin only
export const getDashboardStats = async (req, res) => {
  try {
    const totalUsers = await User.countDocuments()
    const totalPredictions = await Prediction.countDocuments()
    const activeUsers = await User.countDocuments({ isActive: true })
    const adminUsers = await User.countDocuments({ role: 'admin' })

    // Get predictions by class
    const predictionsByClass = await Prediction.aggregate([
      {
        $group: {
          _id: '$prediction.predicted_class',
          count: { $sum: 1 }
        }
      }
    ])

    // Get recent predictions (last 7 days)
    const sevenDaysAgo = new Date()
    sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7)
    
    const recentPredictions = await Prediction.countDocuments({
      createdAt: { $gte: sevenDaysAgo }
    })

    // Get recent users (last 7 days)
    const recentUsers = await User.countDocuments({
      createdAt: { $gte: sevenDaysAgo }
    })

    res.json({
      success: true,
      data: {
        totalUsers,
        totalPredictions,
        activeUsers,
        adminUsers,
        predictionsByClass,
        recentPredictions,
        recentUsers
      }
    })
  } catch (error) {
    console.error('Get dashboard stats error:', error)
    res.status(500).json({
      success: false,
      message: 'Server error',
      error: error.message
    })
  }
}

export default router
