"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import Header from "../common/header";
import Footer from "../common/footer";
import { 
  User, 
  Settings, 
  Activity, 
  Heart, 
  TrendingUp, 
  Calendar,
  Shield,
  Bell,
  Edit,
  Save,
  X,
  Camera,
  MapPin,
  Phone,
  Mail,
  Globe,
  Award,
  Target,
  BarChart3,
  Clock,
  CheckCircle,
  AlertCircle
} from "lucide-react";

interface User {
  _id: string;
  firstName: string;
  lastName: string;
  email: string;
  role: string;
  token: string;
}

const ProfilePage = () => {
  const [user, setUser] = useState<User | null>(null);
  const [isEditing, setIsEditing] = useState(false);
  const [editForm, setEditForm] = useState({
    firstName: "",
    lastName: "",
    email: "",
    phone: "",
    location: "",
    bio: ""
  });

  useEffect(() => {
    const storedUser = localStorage.getItem("user");
    if (storedUser) {
      const userData = JSON.parse(storedUser);
      setUser(userData);
      setEditForm({
        firstName: userData.firstName,
        lastName: userData.lastName,
        email: userData.email,
        phone: "",
        location: "",
        bio: ""
      });
    }
  }, []);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setEditForm({
      ...editForm,
      [e.target.name]: e.target.value,
    });
  };

  const handleSave = () => {
    // Here you would typically update the user data via API
    console.log("Saving profile:", editForm);
    setIsEditing(false);
  };

  const handleCancel = () => {
    setEditForm({
      firstName: user?.firstName || "",
      lastName: user?.lastName || "",
      email: user?.email || "",
      phone: "",
      location: "",
      bio: ""
    });
    setIsEditing(false);
  };

  if (!user) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900 flex items-center justify-center">
        <Card className="w-full max-w-md">
          <CardContent className="p-6 text-center">
            <User className="h-12 w-12 text-slate-400 mx-auto mb-4" />
            <h2 className="text-xl font-semibold text-slate-900 dark:text-white mb-2">
              Not Logged In
            </h2>
            <p className="text-slate-600 dark:text-slate-400 mb-4">
              Please log in to view your profile.
            </p>
            <Button className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700">
              Go to Login
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900">
      <Header />
      <div className="container mx-auto px-4 py-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          {/* Header */}
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-slate-900 dark:text-white mb-2">
              My Profile
            </h1>
            <p className="text-slate-600 dark:text-slate-400">
              Manage your account and view your health prediction history
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Profile Card */}
            <div className="lg:col-span-1">
              <Card className="border-0 shadow-xl bg-white/80 dark:bg-slate-800/80 backdrop-blur-xl">
                <CardContent className="p-6">
                  <div className="text-center mb-6">
                    <div className="relative inline-block">
                      <div className="w-24 h-24 rounded-full bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center text-2xl font-bold text-white shadow-lg mb-4">
                        {user.firstName.charAt(0)}{user.lastName.charAt(0)}
                      </div>
                      <Button
                        size="icon"
                        className="absolute -bottom-2 -right-2 w-8 h-8 rounded-full bg-blue-600 hover:bg-blue-700"
                      >
                        <Camera className="h-4 w-4" />
                      </Button>
                    </div>
                    <h2 className="text-xl font-semibold text-slate-900 dark:text-white">
                      {user.firstName} {user.lastName}
                    </h2>
                    <p className="text-slate-600 dark:text-slate-400">{user.email}</p>
                    <div className="flex items-center justify-center space-x-2 mt-2">
                      <div className="w-2 h-2 rounded-full bg-emerald-500"></div>
                      <span className="text-xs text-emerald-600 dark:text-emerald-400 font-medium">
                        Online
                      </span>
                    </div>
                  </div>

                  {/* Quick Stats */}
                  <div className="grid grid-cols-3 gap-4 mb-6">
                    <div className="text-center p-3 rounded-lg bg-blue-50 dark:bg-blue-900/20">
                      <Activity className="h-6 w-6 text-blue-600 dark:text-blue-400 mx-auto mb-2" />
                      <p className="text-xs text-slate-600 dark:text-slate-400">Predictions</p>
                      <p className="text-lg font-bold text-slate-900 dark:text-white">12</p>
                    </div>
                    <div className="text-center p-3 rounded-lg bg-emerald-50 dark:bg-emerald-900/20">
                      <Heart className="h-6 w-6 text-emerald-600 dark:text-emerald-400 mx-auto mb-2" />
                      <p className="text-xs text-slate-600 dark:text-slate-400">Health Score</p>
                      <p className="text-lg font-bold text-slate-900 dark:text-white">85%</p>
                    </div>
                    <div className="text-center p-3 rounded-lg bg-purple-50 dark:bg-purple-900/20">
                      <TrendingUp className="h-6 w-6 text-purple-600 dark:text-purple-400 mx-auto mb-2" />
                      <p className="text-xs text-slate-600 dark:text-slate-400">Accuracy</p>
                      <p className="text-lg font-bold text-slate-900 dark:text-white">92%</p>
                    </div>
                  </div>

                  {/* Contact Info */}
                  <div className="space-y-3">
                    <div className="flex items-center space-x-3 text-sm">
                      <Mail className="h-4 w-4 text-slate-400" />
                      <span className="text-slate-600 dark:text-slate-400">{user.email}</span>
                    </div>
                    <div className="flex items-center space-x-3 text-sm">
                      <Phone className="h-4 w-4 text-slate-400" />
                      <span className="text-slate-600 dark:text-slate-400">+1 (555) 123-4567</span>
                    </div>
                    <div className="flex items-center space-x-3 text-sm">
                      <MapPin className="h-4 w-4 text-slate-400" />
                      <span className="text-slate-600 dark:text-slate-400">New York, NY</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Main Content */}
            <div className="lg:col-span-2">
              <Tabs defaultValue="overview" className="w-full">
                <TabsList className="grid w-full grid-cols-4 bg-slate-100 dark:bg-slate-800">
                  <TabsTrigger value="overview" className="data-[state=active]:bg-white dark:data-[state=active]:bg-slate-700">
                    Overview
                  </TabsTrigger>
                  <TabsTrigger value="predictions" className="data-[state=active]:bg-white dark:data-[state=active]:bg-slate-700">
                    Predictions
                  </TabsTrigger>
                  <TabsTrigger value="settings" className="data-[state=active]:bg-white dark:data-[state=active]:bg-slate-700">
                    Settings
                  </TabsTrigger>
                  <TabsTrigger value="security" className="data-[state=active]:bg-white dark:data-[state=active]:bg-slate-700">
                    Security
                  </TabsTrigger>
                </TabsList>

                {/* Overview Tab */}
                <TabsContent value="overview" className="mt-6">
                  <Card className="border-0 shadow-xl bg-white/80 dark:bg-slate-800/80 backdrop-blur-xl">
                    <CardHeader>
                      <CardTitle className="flex items-center space-x-2">
                        <User className="h-5 w-5" />
                        <span>Profile Information</span>
                        {!isEditing && (
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => setIsEditing(true)}
                            className="ml-auto"
                          >
                            <Edit className="h-4 w-4 mr-2" />
                            Edit
                          </Button>
                        )}
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      {isEditing ? (
                        <div className="space-y-4">
                          <div className="grid grid-cols-2 gap-4">
                            <div>
                              <Label htmlFor="firstName">First Name</Label>
                              <Input
                                id="firstName"
                                name="firstName"
                                value={editForm.firstName}
                                onChange={handleInputChange}
                                className="mt-1"
                              />
                            </div>
                            <div>
                              <Label htmlFor="lastName">Last Name</Label>
                              <Input
                                id="lastName"
                                name="lastName"
                                value={editForm.lastName}
                                onChange={handleInputChange}
                                className="mt-1"
                              />
                            </div>
                          </div>
                          <div>
                            <Label htmlFor="email">Email</Label>
                            <Input
                              id="email"
                              name="email"
                              type="email"
                              value={editForm.email}
                              onChange={handleInputChange}
                              className="mt-1"
                            />
                          </div>
                          <div>
                            <Label htmlFor="phone">Phone</Label>
                            <Input
                              id="phone"
                              name="phone"
                              value={editForm.phone}
                              onChange={handleInputChange}
                              className="mt-1"
                            />
                          </div>
                          <div>
                            <Label htmlFor="location">Location</Label>
                            <Input
                              id="location"
                              name="location"
                              value={editForm.location}
                              onChange={handleInputChange}
                              className="mt-1"
                            />
                          </div>
                          <div>
                            <Label htmlFor="bio">Bio</Label>
                            <Input
                              id="bio"
                              name="bio"
                              value={editForm.bio}
                              onChange={handleInputChange}
                              className="mt-1"
                            />
                          </div>
                          <div className="flex space-x-2">
                            <Button onClick={handleSave} className="bg-blue-600 hover:bg-blue-700">
                              <Save className="h-4 w-4 mr-2" />
                              Save
                            </Button>
                            <Button variant="outline" onClick={handleCancel}>
                              <X className="h-4 w-4 mr-2" />
                              Cancel
                            </Button>
                          </div>
                        </div>
                      ) : (
                        <div className="space-y-4">
                          <div className="grid grid-cols-2 gap-4">
                            <div>
                              <Label className="text-sm text-slate-500">First Name</Label>
                              <p className="text-slate-900 dark:text-white font-medium">{user.firstName}</p>
                            </div>
                            <div>
                              <Label className="text-sm text-slate-500">Last Name</Label>
                              <p className="text-slate-900 dark:text-white font-medium">{user.lastName}</p>
                            </div>
                          </div>
                          <div>
                            <Label className="text-sm text-slate-500">Email</Label>
                            <p className="text-slate-900 dark:text-white font-medium">{user.email}</p>
                          </div>
                          <div>
                            <Label className="text-sm text-slate-500">Phone</Label>
                            <p className="text-slate-900 dark:text-white font-medium">+1 (555) 123-4567</p>
                          </div>
                          <div>
                            <Label className="text-sm text-slate-500">Location</Label>
                            <p className="text-slate-900 dark:text-white font-medium">New York, NY</p>
                          </div>
                          <div>
                            <Label className="text-sm text-slate-500">Bio</Label>
                            <p className="text-slate-900 dark:text-white font-medium">
                              Health-conscious individual passionate about preventive care and AI-driven health insights.
                            </p>
                          </div>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                </TabsContent>

                {/* Predictions Tab */}
                <TabsContent value="predictions" className="mt-6">
                  <Card className="border-0 shadow-xl bg-white/80 dark:bg-slate-800/80 backdrop-blur-xl">
                    <CardHeader>
                      <CardTitle className="flex items-center space-x-2">
                        <Activity className="h-5 w-5" />
                        <span>Prediction History</span>
                      </CardTitle>
                      <CardDescription>
                        View your past health predictions and results
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        {/* Recent Predictions */}
                        <div className="grid gap-4">
                          {[1, 2, 3].map((item) => (
                            <div key={item} className="flex items-center space-x-4 p-4 rounded-lg border border-slate-200 dark:border-slate-700">
                              <div className="w-12 h-12 rounded-full bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center">
                                <Activity className="h-6 w-6 text-white" />
                              </div>
                              <div className="flex-1">
                                <h3 className="font-semibold text-slate-900 dark:text-white">
                                  Hepatitis Prediction #{item}
                                </h3>
                                <p className="text-sm text-slate-600 dark:text-slate-400">
                                  Risk Level: Low • Confidence: 92%
                                </p>
                                <p className="text-xs text-slate-500 dark:text-slate-500">
                                  {new Date().toLocaleDateString()}
                                </p>
                              </div>
                              <div className="flex items-center space-x-2">
                                <CheckCircle className="h-5 w-5 text-emerald-500" />
                                <span className="text-sm text-emerald-600 dark:text-emerald-400 font-medium">
                                  Completed
                                </span>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>

                {/* Settings Tab */}
                <TabsContent value="settings" className="mt-6">
                  <Card className="border-0 shadow-xl bg-white/80 dark:bg-slate-800/80 backdrop-blur-xl">
                    <CardHeader>
                      <CardTitle className="flex items-center space-x-2">
                        <Settings className="h-5 w-5" />
                        <span>Account Settings</span>
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-6">
                        <div className="flex items-center justify-between p-4 rounded-lg border border-slate-200 dark:border-slate-700">
                          <div className="flex items-center space-x-3">
                            <Bell className="h-5 w-5 text-slate-400" />
                            <div>
                              <h3 className="font-semibold text-slate-900 dark:text-white">Email Notifications</h3>
                              <p className="text-sm text-slate-600 dark:text-slate-400">Receive updates about your predictions</p>
                            </div>
                          </div>
                          <Button variant="outline" size="sm">Configure</Button>
                        </div>
                        
                        <div className="flex items-center justify-between p-4 rounded-lg border border-slate-200 dark:border-slate-700">
                          <div className="flex items-center space-x-3">
                            <Globe className="h-5 w-5 text-slate-400" />
                            <div>
                              <h3 className="font-semibold text-slate-900 dark:text-white">Language</h3>
                              <p className="text-sm text-slate-600 dark:text-slate-400">English (US)</p>
                            </div>
                          </div>
                          <Button variant="outline" size="sm">Change</Button>
                        </div>
                        
                        <div className="flex items-center justify-between p-4 rounded-lg border border-slate-200 dark:border-slate-700">
                          <div className="flex items-center space-x-3">
                            <Award className="h-5 w-5 text-slate-400" />
                            <div>
                              <h3 className="font-semibold text-slate-900 dark:text-white">Privacy</h3>
                              <p className="text-sm text-slate-600 dark:text-slate-400">Manage your data and privacy settings</p>
                            </div>
                          </div>
                          <Button variant="outline" size="sm">Manage</Button>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>

                {/* Security Tab */}
                <TabsContent value="security" className="mt-6">
                  <Card className="border-0 shadow-xl bg-white/80 dark:bg-slate-800/80 backdrop-blur-xl">
                    <CardHeader>
                      <CardTitle className="flex items-center space-x-2">
                        <Shield className="h-5 w-5" />
                        <span>Security Settings</span>
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-6">
                        <div className="flex items-center justify-between p-4 rounded-lg border border-slate-200 dark:border-slate-700">
                          <div className="flex items-center space-x-3">
                            <Target className="h-5 w-5 text-slate-400" />
                            <div>
                              <h3 className="font-semibold text-slate-900 dark:text-white">Change Password</h3>
                              <p className="text-sm text-slate-600 dark:text-slate-400">Update your account password</p>
                            </div>
                          </div>
                          <Button variant="outline" size="sm">Update</Button>
                        </div>
                        
                        <div className="flex items-center justify-between p-4 rounded-lg border border-slate-200 dark:border-slate-700">
                          <div className="flex items-center space-x-3">
                            <BarChart3 className="h-5 w-5 text-slate-400" />
                            <div>
                              <h3 className="font-semibold text-slate-900 dark:text-white">Two-Factor Authentication</h3>
                              <p className="text-sm text-slate-600 dark:text-slate-400">Add an extra layer of security</p>
                            </div>
                          </div>
                          <Button variant="outline" size="sm">Enable</Button>
                        </div>
                        
                        <div className="flex items-center justify-between p-4 rounded-lg border border-slate-200 dark:border-slate-700">
                          <div className="flex items-center space-x-3">
                            <Clock className="h-5 w-5 text-slate-400" />
                            <div>
                              <h3 className="font-semibold text-slate-900 dark:text-white">Login History</h3>
                              <p className="text-sm text-slate-600 dark:text-slate-400">View recent login activity</p>
                            </div>
                          </div>
                          <Button variant="outline" size="sm">View</Button>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>
              </Tabs>
            </div>
          </div>
        </motion.div>
      </div>
      <Footer />
    </div>
  );
};

export default ProfilePage; 