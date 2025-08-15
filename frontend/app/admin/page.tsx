"use client";

import type React from "react";
import { useState, useEffect } from "react";
import toast, { Toaster } from "react-hot-toast";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { motion, AnimatePresence } from "framer-motion";
import {
  Users,
  FileText,
  BarChart3,
  Activity,
  Search,
  Filter,
  Plus,
  Edit,
  Trash2,
  Eye,
  MoreHorizontal,
  TrendingUp,
  TrendingDown,
  UserCheck,
  UserX,
  Clock,
  CheckCircle,
  AlertTriangle,
  XCircle,
  Download,
  RefreshCw,
  Settings,
  Shield,
  Database,
  Zap,
  Loader2,
} from "lucide-react";

interface DashboardStats {
  totalUsers: number;
  totalPredictions: number;
  activeUsers: number;
  adminUsers: number;
  predictionsByClass: Array<{ _id: string; count: number }>;
  recentPredictions: number;
  recentUsers: number;
}

interface User {
  _id: string;
  firstName: string;
  lastName: string;
  email: string;
  role: string;
  isActive: boolean;
  lastLogin?: string;
  predictions: Array<{
    _id: string;
    prediction: {
      predicted_class: string;
      confidence: number;
    };
    createdAt: string;
  }>;
  createdAt: string;
}

interface Prediction {
  _id: string;
  user: {
    firstName: string;
    lastName: string;
    email: string;
  } | null;
  age: string;
  gender: string;
  symptoms: any;
  prediction: {
    predicted_class: string;
    confidence: number;
    probability_Hepatitis_A: number;
    probability_Hepatitis_C: number;
  };
  status: string;
  createdAt: string;
}

export default function AdminDashboard() {
  const [activeTab, setActiveTab] = useState("overview");
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [users, setUsers] = useState<User[]>([]);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [recentActivities, setRecentActivities] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [filterRole, setFilterRole] = useState("all");
  const [filterStatus, setFilterStatus] = useState("all");
  const [refreshing, setRefreshing] = useState(false);
  const [showCreateUser, setShowCreateUser] = useState(false);
  const [showEditUser, setShowEditUser] = useState(false);
  const [showViewUser, setShowViewUser] = useState(false);
  const [selectedUser, setSelectedUser] = useState<User | null>(null);
  const [createError, setCreateError] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [updating, setUpdating] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [deleteItem, setDeleteItem] = useState<{type: 'user' | 'prediction', id: string, name: string} | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [showViewPrediction, setShowViewPrediction] = useState(false);
  const [showEditPrediction, setShowEditPrediction] = useState(false);
  const [selectedPrediction, setSelectedPrediction] = useState<Prediction | null>(null);
  const [updatingPrediction, setUpdatingPrediction] = useState(false);
  const [editPredictionForm, setEditPredictionForm] = useState({
    status: "completed",
    notes: ""
  });
  const [newUserForm, setNewUserForm] = useState({
    firstName: "",
    lastName: "",
    email: "",
    password: "",
    role: "user"
  });
  const [editUserForm, setEditUserForm] = useState({
    firstName: "",
    lastName: "",
    email: "",
    role: "user",
    isActive: true
  });

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const generateRecentActivities = (users: User[], predictions: Prediction[]) => {
    const activities: any[] = [];
    
    // Get recent users (last 24 hours)
    const oneDayAgo = new Date();
    oneDayAgo.setDate(oneDayAgo.getDate() - 1);
    
    users.forEach(user => {
      const createdDate = new Date(user.createdAt);
      if (createdDate > oneDayAgo) {
        activities.push({
          action: `${user.firstName} ${user.lastName} registered`,
          time: getTimeAgo(createdDate),
          type: "user"
        });
      }
      
      // Add login activity if lastLogin is recent
      if (user.lastLogin) {
        const loginDate = new Date(user.lastLogin);
        if (loginDate > oneDayAgo) {
          activities.push({
            action: `${user.firstName} ${user.lastName} logged in`,
            time: getTimeAgo(loginDate),
            type: "login"
          });
        }
      }
    });
    
    // Get recent predictions (last 24 hours)
    predictions.forEach(prediction => {
      const createdDate = new Date(prediction.createdAt);
      if (createdDate > oneDayAgo) {
        const userName = prediction.user ? 
          `${prediction.user.firstName} ${prediction.user.lastName}` : 
          'Guest user';
        activities.push({
          action: `${userName} received ${prediction.prediction.predicted_class} prediction`,
          time: getTimeAgo(createdDate),
          type: "prediction"
        });
      }
    });
    
    // Sort by most recent first and limit to 5
    return activities
      .sort((a, b) => new Date(b.time).getTime() - new Date(a.time).getTime())
      .slice(0, 5);
  };

  const getTimeAgo = (date: Date) => {
    const now = new Date();
    const diffInMinutes = Math.floor((now.getTime() - date.getTime()) / (1000 * 60));
    
    if (diffInMinutes < 1) return "Just now";
    if (diffInMinutes < 60) return `${diffInMinutes} minute${diffInMinutes > 1 ? 's' : ''} ago`;
    
    const diffInHours = Math.floor(diffInMinutes / 60);
    if (diffInHours < 24) return `${diffInHours} hour${diffInHours > 1 ? 's' : ''} ago`;
    
    const diffInDays = Math.floor(diffInHours / 24);
    return `${diffInDays} day${diffInDays > 1 ? 's' : ''} ago`;
  };

  const fetchDashboardData = async () => {
    try {
      setError(null);
      const token = localStorage.getItem("token");
      if (!token) {
        // Redirect to login if no token
        window.location.href = "/";
        return;
      }

      const headers = {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      };

      // Fetch dashboard stats
      const statsResponse = await fetch("/api/admin/dashboard", { headers });
      if (statsResponse.ok) {
        const statsData = await statsResponse.json();
        setStats(statsData.data);
      } else if (statsResponse.status === 401) {
        // Unauthorized - redirect to login
        localStorage.removeItem("token");
        localStorage.removeItem("user");
        window.location.href = "/";
        return;
      } else {
        console.error("Failed to fetch dashboard stats:", statsResponse.status);
        setError("Failed to load dashboard statistics");
      }

      // Fetch users
      const usersResponse = await fetch("/api/admin/users", { headers });
      let usersData = [];
      if (usersResponse.ok) {
        const response = await usersResponse.json();
        usersData = response.data;
        setUsers(usersData);
      } else if (usersResponse.status === 401) {
        // Unauthorized - redirect to login
        localStorage.removeItem("token");
        localStorage.removeItem("user");
        window.location.href = "/";
        return;
      } else {
        console.error("Failed to fetch users:", usersResponse.status);
        setError("Failed to load users");
      }

      // Fetch predictions
      const predictionsResponse = await fetch("/api/admin/predictions", { headers });
      let predictionsData = [];
      if (predictionsResponse.ok) {
        const response = await predictionsResponse.json();
        predictionsData = response.data;
        setPredictions(predictionsData);
      } else if (predictionsResponse.status === 401) {
        // Unauthorized - redirect to login
        localStorage.removeItem("token");
        localStorage.removeItem("user");
        window.location.href = "/";
        return;
      } else {
        console.error("Failed to fetch predictions:", predictionsResponse.status);
        setError("Failed to load predictions");
      }

      // Generate real activities from actual data
      const activities = generateRecentActivities(usersData, predictionsData);
      setRecentActivities(activities);

    } catch (error) {
      console.error("Error fetching dashboard data:", error);
      setError("An error occurred while loading dashboard data");
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const handleRefresh = () => {
    setRefreshing(true);
    fetchDashboardData();
  };

  const openCreateUser = () => {
    setCreateError(null);
    setShowCreateUser(true);
  };

  const closeCreateUser = () => {
    setShowCreateUser(false);
    setNewUserForm({ firstName: "", lastName: "", email: "", password: "", role: "user" });
    setCreating(false);
    setCreateError(null);
  };

  const openViewUser = (user: User) => {
    setSelectedUser(user);
    setShowViewUser(true);
  };

  const closeViewUser = () => {
    setShowViewUser(false);
    setSelectedUser(null);
  };

  const openEditUser = (user: User) => {
    setSelectedUser(user);
    setEditUserForm({
      firstName: user.firstName,
      lastName: user.lastName,
      email: user.email,
      role: user.role,
      isActive: user.isActive
    });
    setShowEditUser(true);
  };

  const closeEditUser = () => {
    setShowEditUser(false);
    setSelectedUser(null);
    setEditUserForm({ firstName: "", lastName: "", email: "", role: "user", isActive: true });
    setUpdating(false);
  };

  const openDeleteConfirm = (type: 'user' | 'prediction', id: string, name: string) => {
    setDeleteItem({ type, id, name });
    setShowDeleteConfirm(true);
  };

  const closeDeleteConfirm = () => {
    setShowDeleteConfirm(false);
    setDeleteItem(null);
    setDeleting(false);
  };

  const confirmDelete = async () => {
    if (!deleteItem) return;
    
    setDeleting(true);
    
    try {
      if (deleteItem.type === 'user') {
        await handleDeleteUserConfirmed(deleteItem.id);
      } else {
        await handleDeletePredictionConfirmed(deleteItem.id);
      }
      closeDeleteConfirm();
    } catch (error) {
      setDeleting(false);
    }
  };

  const openViewPrediction = async (predictionId: string) => {
    try {
      const token = localStorage.getItem("token");
      if (!token) {
        toast.error("Authentication required");
        window.location.href = "/";
        return;
      }
      
      const response = await fetch(`/api/admin/predictions/${predictionId}`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        toast.error("Failed to load prediction details");
        return;
      }
      
      setSelectedPrediction(data.data);
      setShowViewPrediction(true);
      
    } catch (e: any) {
      toast.error(e.message || "Failed to load prediction details");
    }
  };

  const closeViewPrediction = () => {
    setShowViewPrediction(false);
    setSelectedPrediction(null);
  };

  const openEditPrediction = (prediction: Prediction) => {
    setSelectedPrediction(prediction);
    setEditPredictionForm({
      status: prediction.status,
      notes: ""
    });
    setShowEditPrediction(true);
  };

  const closeEditPrediction = () => {
    setShowEditPrediction(false);
    setSelectedPrediction(null);
    setEditPredictionForm({ status: "completed", notes: "" });
    setUpdatingPrediction(false);
  };

  const handleEditPredictionSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedPrediction) return;
    
    setUpdatingPrediction(true);
    
    try {
      const token = localStorage.getItem("token");
      if (!token) {
        toast.error("Authentication required");
        window.location.href = "/";
        return;
      }
      
      const response = await fetch(`/api/admin/predictions/${selectedPrediction._id}`, {
        method: "PUT",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(editPredictionForm),
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        toast.error(data?.message || "Failed to update prediction");
        return;
      }
      
      toast.success("Prediction updated successfully!");
      await fetchDashboardData();
      closeEditPrediction();
      
    } catch (err: any) {
      toast.error(err.message || "Failed to update prediction");
    } finally {
      setUpdatingPrediction(false);
    }
  };

  const handleEditInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value, type, checked } = e.target;
    setEditUserForm(prev => ({ 
      ...prev, 
      [name]: type === 'checkbox' ? checked : value 
    }));
  };

  const handleEditRoleChange = (value: string) => {
    setEditUserForm(prev => ({ ...prev, role: value }));
  };

  const handleEditUserSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedUser) return;
    
    setUpdating(true);
    
    try {
      const token = localStorage.getItem("token");
      if (!token) {
        toast.error("Authentication required");
        window.location.href = "/";
        return;
      }
      
      const response = await fetch(`/api/admin/users/${selectedUser._id}`, {
        method: "PUT",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(editUserForm),
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        toast.error(data?.message || "Failed to update user");
        return;
      }
      
      toast.success("User updated successfully!");
      await fetchDashboardData();
      closeEditUser();
      
    } catch (err: any) {
      toast.error(err.message || "Failed to update user");
    } finally {
      setUpdating(false);
    }
  };

  const handleCreateInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setNewUserForm(prev => ({ ...prev, [name]: value }));
  };

  const handleCreateRoleChange = (value: string) => {
    setNewUserForm(prev => ({ ...prev, role: value }));
  };

  const handleCreateUserSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    // Validate password length
    if (newUserForm.password.length < 6) {
      toast.error("Password must be at least 6 characters long");
      return;
    }
    
    setCreating(true);
    setCreateError(null);
    
    try {
      const token = localStorage.getItem("token");
      if (!token) {
        toast.error("Authentication required");
        window.location.href = "/";
        return;
      }
      
      const response = await fetch("/api/admin/users", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(newUserForm),
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        if (response.status === 401) {
          localStorage.removeItem("token");
          localStorage.removeItem("user");
          window.location.href = "/";
          return;
        }
        
        // Handle specific password validation error
        if (data?.message?.includes("Password must be at least")) {
          toast.error("Password must be at least 6 characters long");
        } else {
          toast.error(data?.message || "Failed to create user");
        }
        return;
      }
      
      // Success
      toast.success("User created successfully!");
      await fetchDashboardData();
      setActiveTab("users");
      closeCreateUser();
      
    } catch (err: any) {
      toast.error(err.message || "Failed to create user");
    } finally {
      setCreating(false);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "completed":
        return "bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400";
      case "pending":
        return "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400";
      case "failed":
        return "bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400";
      default:
        return "bg-gray-100 text-gray-800 dark:bg-gray-900/20 dark:text-gray-400";
    }
  };

  const getRoleColor = (role: string) => {
    return role === "admin" 
      ? "bg-purple-100 text-purple-800 dark:bg-purple-900/20 dark:text-purple-400"
      : "bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400";
  };

  const filteredUsers = users.filter(user => {
    const matchesSearch = user.firstName.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         user.lastName.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         user.email.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesRole = filterRole === "all" || user.role === filterRole;
    return matchesSearch && matchesRole;
  });

  const handleEditUser = async (userId: string, updates: Partial<User>) => {
    try {
      const token = localStorage.getItem("token");
      if (!token) {
        toast.error("Authentication required");
        window.location.href = "/";
        return;
      }
      
      const response = await fetch(`/api/admin/users/${userId}`, {
        method: "PUT",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(updates),
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        toast.error(data?.message || "Failed to update user");
        return;
      }
      
      toast.success("User updated successfully!");
      await fetchDashboardData();
    } catch (e: any) {
      toast.error(e.message || "Failed to update user");
    }
  };

  const handleDeleteUser = async (userId: string, userName: string) => {
    openDeleteConfirm('user', userId, userName);
  };

  const handleDeleteUserConfirmed = async (userId: string) => {
    try {
      const token = localStorage.getItem("token");
      if (!token) {
        toast.error("Authentication required");
        window.location.href = "/";
        return;
      }
      
      const response = await fetch(`/api/admin/users/${userId}`, {
        method: "DELETE",
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        toast.error(data?.message || "Failed to delete user");
        return;
      }
      
      toast.success("User deleted successfully!");
      await fetchDashboardData();
    } catch (e: any) {
      toast.error(e.message || "Failed to delete user");
    }
  };



  const handleUpdatePrediction = async (predictionId: string, updates: { status?: string; notes?: string }) => {
    try {
      const token = localStorage.getItem("token");
      if (!token) {
        toast.error("Authentication required");
        window.location.href = "/";
        return;
      }
      
      const response = await fetch(`/api/admin/predictions/${predictionId}`, {
        method: "PUT",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(updates),
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        toast.error(data?.message || "Failed to update prediction");
        return;
      }
      
      toast.success("Prediction updated successfully!");
      await fetchDashboardData();
    } catch (e: any) {
      toast.error(e.message || "Failed to update prediction");
    }
  };

  const handleDeletePrediction = async (predictionId: string, predictionInfo: string) => {
    openDeleteConfirm('prediction', predictionId, predictionInfo);
  };

  const handleDeletePredictionConfirmed = async (predictionId: string) => {
    try {
      const token = localStorage.getItem("token");
      if (!token) {
        toast.error("Authentication required");
        window.location.href = "/";
        return;
      }
      
      const response = await fetch(`/api/admin/predictions/${predictionId}`, {
        method: "DELETE",
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        toast.error(data?.message || "Failed to delete prediction");
        return;
      }
      
      toast.success("Prediction deleted successfully!");
      await fetchDashboardData();
    } catch (e: any) {
      toast.error(e.message || "Failed to delete prediction");
    }
  };

  const filteredPredictions = predictions.filter(prediction => {
    const matchesStatus = filterStatus === "all" || prediction.status === filterStatus;
    return matchesStatus;
  });

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
        <div className="container mx-auto px-4 py-8">
          <div className="flex flex-col items-center justify-center h-96">
            <Loader2 className="h-12 w-12 animate-spin text-blue-600 mb-4" />
            <h2 className="text-xl font-semibold text-slate-900 dark:text-white mb-2">
              Loading Admin Dashboard
            </h2>
            <p className="text-slate-500 dark:text-slate-400">
              Please wait while we fetch your data...
            </p>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
        <div className="container mx-auto px-4 py-8">
          <div className="flex flex-col items-center justify-center h-96">
            <AlertTriangle className="h-12 w-12 text-red-500 mb-4" />
            <h2 className="text-xl font-semibold text-slate-900 dark:text-white mb-2">
              Error Loading Dashboard
            </h2>
            <p className="text-slate-500 dark:text-slate-400 mb-4">
              {error}
            </p>
            <Button onClick={handleRefresh} className="bg-blue-600 hover:bg-blue-700">
              <RefreshCw className="h-4 w-4 mr-2" />
              Try Again
            </Button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
      <Toaster 
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: '#363636',
            color: '#fff',
          },
        }}
      />
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8 gap-4"
        >
          <div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              Admin Dashboard
            </h1>
            <p className="text-slate-500 dark:text-slate-400 mt-2">
              Manage users, monitor predictions, and oversee system analytics
            </p>
          </div>
          <div className="flex gap-3">
            <Button 
              variant="outline" 
              className="flex items-center gap-2"
              onClick={handleRefresh}
              disabled={refreshing}
            >
              {refreshing ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <RefreshCw className="h-4 w-4" />
              )}
              Refresh
            </Button>
            <Button className="bg-blue-600 hover:bg-blue-700 flex items-center gap-2">
              <Download className="h-4 w-4" />
              Export Data
            </Button>
          </div>
        </motion.div>

        {/* Stats Cards */}
        {stats && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8"
          >
            {[
              {
                title: "Total Users",
                value: stats.totalUsers,
                icon: <Users className="h-6 w-6" />,
                color: "bg-blue-500",
                trend: "+12%",
                trendUp: true,
              },
              {
                title: "Total Predictions",
                value: stats.totalPredictions,
                icon: <FileText className="h-6 w-6" />,
                color: "bg-green-500",
                trend: "+8%",
                trendUp: true,
              },
              {
                title: "Active Users",
                value: stats.activeUsers,
                icon: <UserCheck className="h-6 w-6" />,
                color: "bg-purple-500",
                trend: "+5%",
                trendUp: true,
              },
              {
                title: "Admin Users",
                value: stats.adminUsers,
                icon: <Shield className="h-6 w-6" />,
                color: "bg-orange-500",
                trend: "0%",
                trendUp: false,
              },
            ].map((stat, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
              >
                <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl transition-all duration-300 group">
                  <div className={`absolute top-0 right-0 w-32 h-32 ${stat.color} opacity-10 rounded-full -translate-y-16 translate-x-16 group-hover:scale-110 transition-transform duration-300`}></div>
                  <CardContent className="p-6 relative z-10">
                    <div className="flex justify-between items-start">
                      <div>
                        <p className="text-sm font-medium text-slate-500 dark:text-slate-400">
                          {stat.title}
                        </p>
                        <h3 className="text-3xl font-bold mt-2 text-slate-900 dark:text-white">
                          {stat.value.toLocaleString()}
                        </h3>
                        <div className="flex items-center gap-1 mt-2">
                          {stat.trendUp ? (
                            <TrendingUp className="h-4 w-4 text-green-500" />
                          ) : (
                            <TrendingDown className="h-4 w-4 text-red-500" />
                          )}
                          <span className={`text-sm font-medium ${stat.trendUp ? 'text-green-500' : 'text-red-500'}`}>
                            {stat.trend}
                          </span>
                        </div>
                      </div>
                      <div className={`p-3 rounded-full ${stat.color} text-white group-hover:scale-110 transition-transform duration-300`}>
                        {stat.icon}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </motion.div>
        )}

        {/* Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid grid-cols-3 mb-8 w-full max-w-md bg-white dark:bg-slate-800 shadow-lg">
            <TabsTrigger
              value="overview"
              className="data-[state=active]:bg-gradient-to-r data-[state=active]:from-blue-600 data-[state=active]:to-purple-600 data-[state=active]:text-white"
            >
              Overview
            </TabsTrigger>
            <TabsTrigger
              value="users"
              className="data-[state=active]:bg-gradient-to-r data-[state=active]:from-blue-600 data-[state=active]:to-purple-600 data-[state=active]:text-white"
            >
              Users
            </TabsTrigger>
            <TabsTrigger
              value="predictions"
              className="data-[state=active]:bg-gradient-to-r data-[state=active]:from-blue-600 data-[state=active]:to-purple-600 data-[state=active]:text-white"
            >
              Predictions
            </TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Recent Activity */}
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.2 }}
              >
                <Card className="border-0 shadow-lg">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Activity className="h-5 w-5 text-blue-600" />
                      Recent Activity
                    </CardTitle>
                    <CardDescription>
                      Latest system activities and user interactions
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {recentActivities.length > 0 ? recentActivities.map((activity, index) => (
                        <motion.div
                          key={index}
                          initial={{ opacity: 0, x: -20 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: 0.1 * index }}
                          className="flex items-center gap-3 p-3 rounded-lg bg-slate-50 dark:bg-slate-800 hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors"
                        >
                          <div className={`p-2 rounded-full ${
                            activity.type === "user" ? "bg-blue-100 text-blue-600" :
                            activity.type === "prediction" ? "bg-green-100 text-green-600" :
                            activity.type === "login" ? "bg-purple-100 text-purple-600" :
                            "bg-orange-100 text-orange-600"
                          }`}>
                            {activity.type === "user" && <Users className="h-4 w-4" />}
                            {activity.type === "prediction" && <FileText className="h-4 w-4" />}
                            {activity.type === "login" && <UserCheck className="h-4 w-4" />}
                            {activity.type === "system" && <Settings className="h-4 w-4" />}
                          </div>
                          <div className="flex-1">
                            <p className="font-medium text-slate-900 dark:text-white">{activity.action}</p>
                            <p className="text-sm text-slate-500 dark:text-slate-400">{activity.time}</p>
                          </div>
                        </motion.div>
                      )) : (
                        <div className="text-center py-8">
                          <Activity className="h-12 w-12 text-slate-300 dark:text-slate-600 mx-auto mb-3" />
                          <p className="text-slate-500 dark:text-slate-400 text-sm">
                            No recent activities in the last 24 hours
                          </p>
                          <p className="text-slate-400 dark:text-slate-500 text-xs mt-1">
                            Activities will appear here when users register, login, or make predictions
                          </p>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </motion.div>

              {/* Predictions by Class */}
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.3 }}
              >
                <Card className="border-0 shadow-lg">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <BarChart3 className="h-5 w-5 text-green-600" />
                      Predictions by Class
                    </CardTitle>
                    <CardDescription>
                      Distribution of hepatitis predictions
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {stats?.predictionsByClass.map((item, index) => (
                        <motion.div
                          key={index}
                          initial={{ opacity: 0, x: 20 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: 0.1 * index }}
                          className="space-y-2"
                        >
                          <div className="flex justify-between items-center">
                            <span className="font-medium text-slate-900 dark:text-white">
                              {item._id}
                            </span>
                            <span className="text-sm text-slate-500 dark:text-slate-400">
                              {item.count} predictions
                            </span>
                          </div>
                          <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2">
                            <div
                              className="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full transition-all duration-500"
                              style={{ width: `${(item.count / (stats?.totalPredictions || 1)) * 100}%` }}
                            ></div>
                          </div>
                        </motion.div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            </div>
          </TabsContent>

          {/* Users Tab */}
          <TabsContent value="users">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
            >
              <Card className="border-0 shadow-lg">
                <CardHeader>
                  <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                    <div>
                      <CardTitle className="flex items-center gap-2">
                        <Users className="h-5 w-5 text-blue-600" />
                        User Management
                      </CardTitle>
                      <CardDescription>
                        Manage system users and their permissions
                      </CardDescription>
                    </div>
                    <Button onClick={openCreateUser} className="bg-blue-600 hover:bg-blue-700 flex items-center gap-2">
                      <Plus className="h-4 w-4" />
                      Add User
                    </Button>
                  </div>
                </CardHeader>
                <CardContent>
                  {/* Filters */}
                  <div className="flex flex-col md:flex-row gap-4 mb-6">
                    <div className="flex-1 relative">
                      <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-slate-400" />
                      <Input
                        placeholder="Search users..."
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        className="max-w-sm pl-10"
                      />
                    </div>
                    <Select value={filterRole} onValueChange={setFilterRole}>
                      <SelectTrigger className="w-full md:w-48">
                        <SelectValue placeholder="Filter by role" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="all">All Roles</SelectItem>
                        <SelectItem value="user">User</SelectItem>
                        <SelectItem value="admin">Admin</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  {/* Users Table */}
                  <div className="rounded-lg border">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Name</TableHead>
                          <TableHead>Email</TableHead>
                          <TableHead>Role</TableHead>
                          <TableHead>Status</TableHead>
                          <TableHead>Predictions</TableHead>
                          <TableHead>Last Login</TableHead>
                          <TableHead>Actions</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {filteredUsers.map((user) => (
                          <TableRow key={user._id} className="hover:bg-slate-50 dark:hover:bg-slate-800">
                            <TableCell>
                              <div>
                                <p className="font-medium">{user.firstName} {user.lastName}</p>
                              </div>
                            </TableCell>
                            <TableCell>{user.email}</TableCell>
                            <TableCell>
                              <Badge className={getRoleColor(user.role)}>
                                {user.role}
                              </Badge>
                            </TableCell>
                            <TableCell>
                              <Badge className={user.isActive ? "bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400" : "bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400"}>
                                {user.isActive ? "Active" : "Inactive"}
                              </Badge>
                            </TableCell>
                            <TableCell>{user.predictions.length}</TableCell>
                            <TableCell>
                              {user.lastLogin ? new Date(user.lastLogin).toLocaleDateString() : "Never"}
                            </TableCell>
                            <TableCell>
                              <div className="flex gap-2">
                                <Button 
                                  variant="outline" 
                                  size="sm"
                                  onClick={() => openViewUser(user)}
                                  title="View user details"
                                >
                                  <Eye className="h-4 w-4" />
                                </Button>
                                <Button 
                                  variant="outline" 
                                  size="sm" 
                                  onClick={() => openEditUser(user)}
                                  title="Edit user"
                                >
                                  <Edit className="h-4 w-4" />
                                </Button>
                                <Button 
                                  variant="outline" 
                                  size="sm" 
                                  onClick={() => handleDeleteUser(user._id, `${user.firstName} ${user.lastName}`)}
                                  title="Delete user"
                                >
                                  <Trash2 className="h-4 w-4" />
                                </Button>
                              </div>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          </TabsContent>

          {/* Predictions Tab */}
          <TabsContent value="predictions">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
            >
              <Card className="border-0 shadow-lg">
                <CardHeader>
                  <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                    <div>
                      <CardTitle className="flex items-center gap-2">
                        <FileText className="h-5 w-5 text-green-600" />
                        Prediction Management
                      </CardTitle>
                      <CardDescription>
                        Monitor and manage all system predictions
                      </CardDescription>
                    </div>
                    <Button className="bg-green-600 hover:bg-green-700 flex items-center gap-2">
                      <Download className="h-4 w-4" />
                      Export
                    </Button>
                  </div>
                </CardHeader>
                <CardContent>
                  {/* Filters */}
                  <div className="flex flex-col md:flex-row gap-4 mb-6">
                    <Select value={filterStatus} onValueChange={setFilterStatus}>
                      <SelectTrigger className="w-full md:w-48">
                        <SelectValue placeholder="Filter by status" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="all">All Status</SelectItem>
                        <SelectItem value="completed">Completed</SelectItem>
                        <SelectItem value="pending">Pending</SelectItem>
                        <SelectItem value="failed">Failed</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  {/* Predictions Table */}
                  <div className="rounded-lg border">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>User</TableHead>
                          <TableHead>Prediction</TableHead>
                          <TableHead>Confidence</TableHead>
                          <TableHead>Status</TableHead>
                          <TableHead>Date</TableHead>
                          <TableHead>Actions</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {filteredPredictions.map((prediction) => (
                          <TableRow key={prediction._id} className="hover:bg-slate-50 dark:hover:bg-slate-800">
                            <TableCell>
                              <div>
                                <p className="font-medium">
                                  {prediction.user ? 
                                    `${prediction.user.firstName} ${prediction.user.lastName}` : 
                                    'Guest User'
                                  }
                                </p>
                                <p className="text-sm text-slate-500">
                                  {prediction.user?.email || 'No email (guest)'}
                                </p>
                              </div>
                            </TableCell>
                            <TableCell>
                              <Badge className="bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400">
                                {prediction.prediction.predicted_class}
                              </Badge>
                            </TableCell>
                            <TableCell>
                              <div className="flex items-center gap-2">
                                <div className="w-16 bg-slate-200 dark:bg-slate-700 rounded-full h-2">
                                  <div
                                    className="bg-gradient-to-r from-green-500 to-blue-500 h-2 rounded-full"
                                    style={{ width: `${prediction.prediction.confidence * 100}%` }}
                                  ></div>
                                </div>
                                <span className="text-sm font-medium">
                                  {(prediction.prediction.confidence * 100).toFixed(1)}%
                                </span>
                              </div>
                            </TableCell>
                            <TableCell>
                              <Badge className={getStatusColor(prediction.status)}>
                                {prediction.status}
                              </Badge>
                            </TableCell>
                            <TableCell>
                              {new Date(prediction.createdAt).toLocaleDateString()}
                            </TableCell>
                            <TableCell>
                              <div className="flex gap-2">
                                <Button 
                                  variant="outline" 
                                  size="sm"
                                  onClick={() => openViewPrediction(prediction._id)}
                                  title="View prediction details"
                                >
                                  <Eye className="h-4 w-4" />
                                </Button>
                                <Button 
                                  variant="outline" 
                                  size="sm"
                                  onClick={() => openEditPrediction(prediction)}
                                  title="Edit prediction"
                                >
                                  <Edit className="h-4 w-4" />
                                </Button>
                                <Button 
                                  variant="outline" 
                                  size="sm"
                                  onClick={() => {
                                    const predictionInfo = prediction.user ? 
                                      `${prediction.prediction.predicted_class} for ${prediction.user.firstName} ${prediction.user.lastName}` :
                                      `${prediction.prediction.predicted_class} for Guest User`;
                                    handleDeletePrediction(prediction._id, predictionInfo);
                                  }}
                                  title="Delete prediction"
                                >
                                  <Trash2 className="h-4 w-4" />
                                </Button>
                              </div>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          </TabsContent>
        </Tabs>

        {/* View User Modal */}
        {showViewUser && selectedUser && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
            <div className="w-full max-w-lg rounded-xl bg-white dark:bg-slate-900 p-6 shadow-2xl">
              <div className="mb-4">
                <h3 className="text-xl font-semibold text-slate-900 dark:text-white">User Details</h3>
                <p className="text-sm text-slate-500 dark:text-slate-400">View user information</p>
              </div>
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="text-sm font-medium text-slate-700 dark:text-slate-300">First Name</label>
                    <p className="text-slate-900 dark:text-white">{selectedUser.firstName}</p>
                  </div>
                  <div>
                    <label className="text-sm font-medium text-slate-700 dark:text-slate-300">Last Name</label>
                    <p className="text-slate-900 dark:text-white">{selectedUser.lastName}</p>
                  </div>
                </div>
                <div>
                  <label className="text-sm font-medium text-slate-700 dark:text-slate-300">Email</label>
                  <p className="text-slate-900 dark:text-white">{selectedUser.email}</p>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="text-sm font-medium text-slate-700 dark:text-slate-300">Role</label>
                    <Badge className={getRoleColor(selectedUser.role)}>
                      {selectedUser.role}
                    </Badge>
                  </div>
                  <div>
                    <label className="text-sm font-medium text-slate-700 dark:text-slate-300">Status</label>
                    <Badge className={selectedUser.isActive ? "bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400" : "bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400"}>
                      {selectedUser.isActive ? "Active" : "Inactive"}
                    </Badge>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="text-sm font-medium text-slate-700 dark:text-slate-300">Predictions</label>
                    <p className="text-slate-900 dark:text-white">{selectedUser.predictions.length}</p>
                  </div>
                  <div>
                    <label className="text-sm font-medium text-slate-700 dark:text-slate-300">Last Login</label>
                    <p className="text-slate-900 dark:text-white">
                      {selectedUser.lastLogin ? new Date(selectedUser.lastLogin).toLocaleDateString() : "Never"}
                    </p>
                  </div>
                </div>
                <div>
                  <label className="text-sm font-medium text-slate-700 dark:text-slate-300">Member Since</label>
                  <p className="text-slate-900 dark:text-white">{new Date(selectedUser.createdAt).toLocaleDateString()}</p>
                </div>
              </div>
              <div className="flex justify-end gap-2 pt-6">
                <Button type="button" variant="outline" onClick={closeViewUser}>
                  Close
                </Button>
                <Button type="button" onClick={() => { closeViewUser(); openEditUser(selectedUser); }} className="bg-blue-600 hover:bg-blue-700">
                  Edit User
                </Button>
              </div>
            </div>
          </div>
        )}

        {/* Edit User Modal */}
        {showEditUser && selectedUser && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
            <div className="w-full max-w-lg rounded-xl bg-white dark:bg-slate-900 p-6 shadow-2xl">
              <div className="mb-4">
                <h3 className="text-xl font-semibold text-slate-900 dark:text-white">Edit User</h3>
                <p className="text-sm text-slate-500 dark:text-slate-400">Update user information</p>
              </div>
              <form onSubmit={handleEditUserSubmit} className="space-y-4">
                <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                  <Input
                    name="firstName"
                    placeholder="First name"
                    value={editUserForm.firstName}
                    onChange={handleEditInputChange}
                    required
                  />
                  <Input
                    name="lastName"
                    placeholder="Last name"
                    value={editUserForm.lastName}
                    onChange={handleEditInputChange}
                    required
                  />
                </div>
                <Input
                  type="email"
                  name="email"
                  placeholder="Email"
                  value={editUserForm.email}
                  onChange={handleEditInputChange}
                  required
                />
                <div className="space-y-2">
                  <Select value={editUserForm.role} onValueChange={handleEditRoleChange}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select role" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="user">User</SelectItem>
                      <SelectItem value="admin">Admin</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="isActive"
                    name="isActive"
                    checked={editUserForm.isActive}
                    onChange={handleEditInputChange}
                    className="rounded"
                  />
                  <label htmlFor="isActive" className="text-sm font-medium text-slate-700 dark:text-slate-300">
                    Active User
                  </label>
                </div>
                <div className="flex justify-end gap-2 pt-2">
                  <Button type="button" variant="outline" onClick={closeEditUser}>
                    Cancel
                  </Button>
                  <Button type="submit" disabled={updating} className="bg-blue-600 hover:bg-blue-700">
                    {updating ? "Updating..." : "Update User"}
                  </Button>
                </div>
              </form>
            </div>
          </div>
        )}

        {/* Create User Modal */}
        {showCreateUser && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
            <div className="w-full max-w-lg rounded-xl bg-white dark:bg-slate-900 p-6 shadow-2xl">
              <div className="mb-4">
                <h3 className="text-xl font-semibold text-slate-900 dark:text-white">Create User</h3>
                <p className="text-sm text-slate-500 dark:text-slate-400">Add a new user account</p>
              </div>
              {createError && (
                <div className="mb-3 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-800 dark:bg-red-900/30 dark:text-red-300">
                  {createError}
                </div>
              )}
              <form onSubmit={handleCreateUserSubmit} className="space-y-4">
                <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                  <Input
                    name="firstName"
                    placeholder="First name"
                    value={newUserForm.firstName}
                    onChange={handleCreateInputChange}
                    required
                  />
                  <Input
                    name="lastName"
                    placeholder="Last name"
                    value={newUserForm.lastName}
                    onChange={handleCreateInputChange}
                    required
                  />
                </div>
                <Input
                  type="email"
                  name="email"
                  placeholder="Email"
                  value={newUserForm.email}
                  onChange={handleCreateInputChange}
                  required
                />
                <Input
                  type="password"
                  name="password"
                  placeholder="Temporary password"
                  value={newUserForm.password}
                  onChange={handleCreateInputChange}
                  required
                />
                <div className="space-y-2">
                  <Select value={newUserForm.role} onValueChange={handleCreateRoleChange}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select role" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="user">User</SelectItem>
                      <SelectItem value="admin">Admin</SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-slate-500 dark:text-slate-400">Only admins can access this page. Creating an admin user here is allowed.</p>
                </div>
                <div className="flex justify-end gap-2 pt-2">
                  <Button type="button" variant="outline" onClick={closeCreateUser}>
                    Cancel
                  </Button>
                  <Button type="submit" disabled={creating} className="bg-blue-600 hover:bg-blue-700">
                    {creating ? "Creating..." : "Create"}
                  </Button>
                </div>
              </form>
            </div>
          </div>
        )}

        {/* Prediction View Modal */}
        {showViewPrediction && selectedPrediction && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 20 }}
              className="w-full max-w-2xl rounded-2xl bg-white dark:bg-slate-900 p-6 shadow-2xl border border-slate-200 dark:border-slate-700 max-h-[80vh] overflow-y-auto"
            >
              {/* Header */}
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                    <FileText className="h-6 w-6 text-white" />
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-slate-900 dark:text-white">
                      Prediction Details
                    </h3>
                    <p className="text-sm text-slate-500 dark:text-slate-400">
                      View complete prediction information
                    </p>
                  </div>
                </div>
                <Button variant="outline" size="sm" onClick={closeViewPrediction}>
                  <XCircle className="h-4 w-4" />
                </Button>
              </div>

              {/* Content */}
              <div className="space-y-6">
                {/* User Information */}
                <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-lg p-4 border border-blue-200 dark:border-blue-800">
                  <h4 className="font-semibold text-blue-900 dark:text-blue-300 mb-3 flex items-center gap-2">
                    <Users className="h-4 w-4" />
                    Patient Information
                  </h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <label className="text-sm font-medium text-slate-700 dark:text-slate-300">Name</label>
                      <p className="text-slate-900 dark:text-white">
                        {selectedPrediction.user ? 
                          `${selectedPrediction.user.firstName} ${selectedPrediction.user.lastName}` : 
                          'Guest User'
                        }
                      </p>
                    </div>
                    <div>
                      <label className="text-sm font-medium text-slate-700 dark:text-slate-300">Email</label>
                      <p className="text-slate-900 dark:text-white">
                        {selectedPrediction.user?.email || 'No email (guest)'}
                      </p>
                    </div>
                    <div>
                      <label className="text-sm font-medium text-slate-700 dark:text-slate-300">Age Group</label>
                      <p className="text-slate-900 dark:text-white">{selectedPrediction.age}</p>
                    </div>
                    <div>
                      <label className="text-sm font-medium text-slate-700 dark:text-slate-300">Gender</label>
                      <p className="text-slate-900 dark:text-white capitalize">{selectedPrediction.gender}</p>
                    </div>
                  </div>
                </div>

                {/* Prediction Results */}
                <div className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg p-4 border border-green-200 dark:border-green-800">
                  <h4 className="font-semibold text-green-900 dark:text-green-300 mb-3 flex items-center gap-2">
                    <BarChart3 className="h-4 w-4" />
                    Prediction Results
                  </h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <label className="text-sm font-medium text-slate-700 dark:text-slate-300">Predicted Class</label>
                      <Badge className="bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400 mt-1">
                        {selectedPrediction.prediction.predicted_class}
                      </Badge>
                    </div>
                    <div>
                      <label className="text-sm font-medium text-slate-700 dark:text-slate-300">Confidence</label>
                      <div className="flex items-center gap-2 mt-1">
                        <div className="w-20 bg-slate-200 dark:bg-slate-700 rounded-full h-2">
                          <div
                            className="bg-gradient-to-r from-green-500 to-blue-500 h-2 rounded-full"
                            style={{ width: `${selectedPrediction.prediction.confidence * 100}%` }}
                          ></div>
                        </div>
                        <span className="text-sm font-medium text-slate-900 dark:text-white">
                          {(selectedPrediction.prediction.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                    <div>
                      <label className="text-sm font-medium text-slate-700 dark:text-slate-300">Hepatitis A Probability</label>
                      <p className="text-slate-900 dark:text-white">
                        {(selectedPrediction.prediction.probability_Hepatitis_A * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div>
                      <label className="text-sm font-medium text-slate-700 dark:text-slate-300">Hepatitis C Probability</label>
                      <p className="text-slate-900 dark:text-white">
                        {(selectedPrediction.prediction.probability_Hepatitis_C * 100).toFixed(1)}%
                      </p>
                    </div>
                  </div>
                </div>

                {/* Status and Date */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="text-sm font-medium text-slate-700 dark:text-slate-300">Status</label>
                    <Badge className={getStatusColor(selectedPrediction.status)} style={{ marginTop: '4px' }}>
                      {selectedPrediction.status}
                    </Badge>
                  </div>
                  <div>
                    <label className="text-sm font-medium text-slate-700 dark:text-slate-300">Created Date</label>
                    <p className="text-slate-900 dark:text-white">
                      {new Date(selectedPrediction.createdAt).toLocaleDateString()}
                    </p>
                  </div>
                </div>
              </div>

              {/* Footer */}
              <div className="flex justify-end gap-3 mt-6 pt-4 border-t border-slate-200 dark:border-slate-700">
                <Button variant="outline" onClick={closeViewPrediction}>
                  Close
                </Button>
                <Button 
                  onClick={() => { closeViewPrediction(); openEditPrediction(selectedPrediction); }}
                  className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white"
                >
                  <Edit className="h-4 w-4 mr-2" />
                  Edit Prediction
                </Button>
              </div>
            </motion.div>
          </div>
        )}

        {/* Prediction Edit Modal */}
        {showEditPrediction && selectedPrediction && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 20 }}
              className="w-full max-w-lg rounded-2xl bg-white dark:bg-slate-900 p-6 shadow-2xl border border-slate-200 dark:border-slate-700"
            >
              {/* Header */}
              <div className="flex items-center gap-3 mb-6">
                <div className="w-12 h-12 rounded-full bg-gradient-to-br from-purple-500 to-blue-600 flex items-center justify-center">
                  <Edit className="h-6 w-6 text-white" />
                </div>
                <div>
                  <h3 className="text-xl font-bold text-slate-900 dark:text-white">
                    Edit Prediction
                  </h3>
                  <p className="text-sm text-slate-500 dark:text-slate-400">
                    Update prediction status and notes
                  </p>
                </div>
              </div>

              {/* Form */}
              <form onSubmit={handleEditPredictionSubmit} className="space-y-4">
                <div>
                  <label className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-2 block">
                    Status
                  </label>
                  <Select 
                    value={editPredictionForm.status} 
                    onValueChange={(value) => setEditPredictionForm(prev => ({ ...prev, status: value }))}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select status" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="pending">Pending</SelectItem>
                      <SelectItem value="completed">Completed</SelectItem>
                      <SelectItem value="failed">Failed</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <label className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-2 block">
                    Notes (Optional)
                  </label>
                  <textarea
                    className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-lg bg-white dark:bg-slate-800 text-slate-900 dark:text-white resize-none"
                    rows={3}
                    placeholder="Add any notes about this prediction..."
                    value={editPredictionForm.notes}
                    onChange={(e) => setEditPredictionForm(prev => ({ ...prev, notes: e.target.value }))}
                  />
                </div>

                {/* Action Buttons */}
                <div className="flex gap-3 pt-4">
                  <Button
                    type="button"
                    variant="outline"
                    onClick={closeEditPrediction}
                    disabled={updatingPrediction}
                    className="flex-1"
                  >
                    Cancel
                  </Button>
                  <Button
                    type="submit"
                    disabled={updatingPrediction}
                    className="flex-1 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white"
                  >
                    {updatingPrediction ? (
                      <>
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                        Updating...
                      </>
                    ) : (
                      <>
                        <CheckCircle className="h-4 w-4 mr-2" />
                        Update Prediction
                      </>
                    )}
                  </Button>
                </div>
              </form>
            </motion.div>
          </div>
        )}

        {/* Beautiful Delete Confirmation Modal */}
        {showDeleteConfirm && deleteItem && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 20 }}
              className="w-full max-w-md rounded-2xl bg-white dark:bg-slate-900 p-6 shadow-2xl border border-slate-200 dark:border-slate-700"
            >
              {/* Header with Icon */}
              <div className="flex items-center justify-center mb-4">
                <div className="w-16 h-16 rounded-full bg-gradient-to-br from-red-500 to-red-600 flex items-center justify-center shadow-lg">
                  <AlertTriangle className="h-8 w-8 text-white" />
                </div>
              </div>

              {/* Title */}
              <div className="text-center mb-4">
                <h3 className="text-xl font-bold text-slate-900 dark:text-white mb-2">
                  Confirm Deletion
                </h3>
                <p className="text-slate-600 dark:text-slate-400">
                  This action cannot be undone. Are you absolutely sure?
                </p>
              </div>

              {/* Content */}
              <div className="bg-gradient-to-r from-red-50 to-orange-50 dark:from-red-900/20 dark:to-orange-900/20 rounded-lg p-4 mb-6 border border-red-200 dark:border-red-800">
                <div className="flex items-start gap-3">
                  <div className="w-5 h-5 rounded-full bg-red-500 flex items-center justify-center flex-shrink-0 mt-0.5">
                    <XCircle className="h-3 w-3 text-white" />
                  </div>
                  <div>
                    <p className="font-semibold text-red-900 dark:text-red-300 mb-1">
                      You're about to delete this {deleteItem.type}:
                    </p>
                    <p className="text-red-800 dark:text-red-400 font-medium">
                      {deleteItem.name}
                    </p>
                    {deleteItem.type === 'user' && (
                      <p className="text-sm text-red-700 dark:text-red-500 mt-2">
                        This will also permanently delete all predictions associated with this user.
                      </p>
                    )}
                  </div>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex gap-3">
                <Button
                  type="button"
                  variant="outline"
                  onClick={closeDeleteConfirm}
                  disabled={deleting}
                  className="flex-1 border-slate-300 hover:bg-slate-50 dark:border-slate-600 dark:hover:bg-slate-800"
                >
                  <XCircle className="h-4 w-4 mr-2" />
                  Cancel
                </Button>
                <Button
                  type="button"
                  onClick={confirmDelete}
                  disabled={deleting}
                  className="flex-1 bg-gradient-to-r from-red-600 to-red-700 hover:from-red-700 hover:to-red-800 text-white border-0 shadow-lg"
                >
                  {deleting ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Deleting...
                    </>
                  ) : (
                    <>
                      <Trash2 className="h-4 w-4 mr-2" />
                      Delete {deleteItem.type}
                    </>
                  )}
                </Button>
              </div>

              {/* Warning Footer */}
              <div className="mt-4 text-center">
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  This action is permanent and cannot be reversed
                </p>
              </div>
            </motion.div>
          </div>
        )}
      </div>
    </div>
  );
}

// Create User Modal
/* Rendered inside component return above via portal-like overlay */
