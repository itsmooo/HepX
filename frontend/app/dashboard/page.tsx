"use client";

import { useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { motion } from "framer-motion";
import {
  Activity,
  BarChart3,
  FileText,
  Users,
  BookOpen,
  ArrowRight,
  Bell,
  Calendar,
  CheckCircle2,
  AlertTriangle,
  Heart,
} from "lucide-react";
import Link from "next/link";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState("overview");

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8 gap-4">
        <div>
          <h1 className="text-3xl font-bold text-slate-900 dark:text-white">
            Dashboard
          </h1>
          <p className="text-slate-500 dark:text-slate-400 mt-1">
            Welcome to HepaPredict. Monitor your assessments and learn about
            hepatitis.
          </p>
        </div>
        <div className="flex gap-3">
          <Button variant="outline" className="flex items-center gap-2">
            <Bell className="h-4 w-4" />
            <span className="hidden sm:inline">Notifications</span>
            <span className="inline-flex items-center justify-center w-5 h-5 text-xs font-bold text-white bg-blue-600 rounded-full">
              3
            </span>
          </Button>
          <Button className="bg-blue-600 hover:bg-blue-700">
            <FileText className="h-4 w-4 mr-2" />
            New Assessment
          </Button>
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid grid-cols-3 mb-8 w-full max-w-md">
          <TabsTrigger
            value="overview"
            className="data-[state=active]:bg-blue-600 data-[state=active]:text-white"
          >
            Overview
          </TabsTrigger>
          <TabsTrigger
            value="history"
            className="data-[state=active]:bg-blue-600 data-[state=active]:text-white"
          >
            History
          </TabsTrigger>
          <TabsTrigger
            value="resources"
            className="data-[state=active]:bg-blue-600 data-[state=active]:text-white"
          >
            Resources
          </TabsTrigger>
        </TabsList>

        <TabsContent value="overview">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            {[
              {
                title: "Risk Level",
                value: "Low",
                icon: <Activity className="h-5 w-5 text-green-500" />,
                color: "bg-green-100 dark:bg-green-900/20",
                textColor: "text-green-700 dark:text-green-400",
              },
              {
                title: "Assessments",
                value: "3",
                icon: <FileText className="h-5 w-5 text-blue-500" />,
                color: "bg-blue-100 dark:bg-blue-900/20",
                textColor: "text-blue-700 dark:text-blue-400",
              },
              {
                title: "Next Checkup",
                value: "15 days",
                icon: <Calendar className="h-5 w-5 text-purple-500" />,
                color: "bg-purple-100 dark:bg-purple-900/20",
                textColor: "text-purple-700 dark:text-purple-400",
              },
            ].map((stat, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
              >
                <Card>
                  <CardContent className="p-6">
                    <div className="flex justify-between items-center">
                      <div>
                        <p className="text-sm font-medium text-slate-500 dark:text-slate-400">
                          {stat.title}
                        </p>
                        <h3
                          className={`text-2xl font-bold mt-1 ${stat.textColor}`}
                        >
                          {stat.value}
                        </h3>
                      </div>
                      <div className={`p-3 rounded-full ${stat.color}`}>
                        {stat.icon}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <motion.div
              className="lg:col-span-2"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
            >
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-xl font-semibold flex items-center">
                    <BarChart3 className="h-5 w-5 mr-2 text-blue-600" />
                    Risk Assessment Trends
                  </CardTitle>
                  <CardDescription>
                    Your hepatitis risk assessment over time
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="h-[250px] flex items-center justify-center">
                    <div className="w-full space-y-6">
                      {[
                        { date: "March 2023", risk: 35, label: "Low Risk" },
                        { date: "June 2023", risk: 28, label: "Low Risk" },
                        {
                          date: "September 2023",
                          risk: 15,
                          label: "Very Low Risk",
                        },
                      ].map((assessment, i) => (
                        <div key={i} className="space-y-2">
                          <div className="flex justify-between text-sm">
                            <span className="font-medium">
                              {assessment.date}
                            </span>
                            <span
                              className={
                                assessment.risk > 50
                                  ? "text-red-600"
                                  : "text-green-600"
                              }
                            >
                              {assessment.label}
                            </span>
                          </div>
                          <div className="w-full bg-slate-100 dark:bg-slate-700 rounded-full h-3">
                            <div
                              className={`h-3 rounded-full ${
                                assessment.risk > 70
                                  ? "bg-red-600"
                                  : assessment.risk > 50
                                  ? "bg-orange-500"
                                  : assessment.risk > 30
                                  ? "bg-yellow-500"
                                  : "bg-green-500"
                              }`}
                              style={{ width: `${assessment.risk}%` }}
                            ></div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </CardContent>
                <CardFooter className="pt-0">
                  <Button variant="outline" className="w-full">
                    View Detailed History
                  </Button>
                </CardFooter>
              </Card>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
            >
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-xl font-semibold flex items-center">
                    <Bell className="h-5 w-5 mr-2 text-blue-600" />
                    Health Reminders
                  </CardTitle>
                  <CardDescription>
                    Upcoming actions and reminders
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {[
                      {
                        title: "Schedule checkup",
                        date: "In 15 days",
                        icon: <Calendar className="h-4 w-4 text-blue-600" />,
                        priority: "medium",
                      },
                      {
                        title: "Complete vaccination",
                        date: "Overdue by 5 days",
                        icon: (
                          <AlertTriangle className="h-4 w-4 text-amber-600" />
                        ),
                        priority: "high",
                      },
                      {
                        title: "Review latest results",
                        date: "New results available",
                        icon: <FileText className="h-4 w-4 text-green-600" />,
                        priority: "low",
                      },
                    ].map((reminder, i) => (
                      <div
                        key={i}
                        className={`p-3 rounded-lg border ${
                          reminder.priority === "high"
                            ? "border-amber-200 bg-amber-50 dark:border-amber-900 dark:bg-amber-900/20"
                            : "border-slate-200 dark:border-slate-700"
                        }`}
                      >
                        <div className="flex justify-between items-center">
                          <div className="flex items-center gap-3">
                            <div className="flex-shrink-0">{reminder.icon}</div>
                            <div>
                              <p className="font-medium text-slate-900 dark:text-white">
                                {reminder.title}
                              </p>
                              <p className="text-sm text-slate-500 dark:text-slate-400">
                                {reminder.date}
                              </p>
                            </div>
                          </div>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-8 w-8 p-0"
                          >
                            <CheckCircle2 className="h-4 w-4" />
                          </Button>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
                <CardFooter className="pt-0">
                  <Button variant="outline" className="w-full">
                    Manage Reminders
                  </Button>
                </CardFooter>
              </Card>
            </motion.div>
          </div>

          <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-6">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
            >
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-xl font-semibold flex items-center">
                    <BookOpen className="h-5 w-5 mr-2 text-blue-600" />
                    Educational Resources
                  </CardTitle>
                  <CardDescription>Learn more about hepatitis</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {[
                      {
                        title: "Understanding Hepatitis Types",
                        category: "Guide",
                      },
                      { title: "Prevention Strategies", category: "Article" },
                      {
                        title: "Symptoms & Warning Signs",
                        category: "Checklist",
                      },
                    ].map((resource, i) => (
                      <div
                        key={i}
                        className="flex justify-between items-center p-3 rounded-lg border border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors"
                      >
                        <div>
                          <p className="font-medium text-slate-900 dark:text-white">
                            {resource.title}
                          </p>
                          <p className="text-sm text-slate-500 dark:text-slate-400">
                            {resource.category}
                          </p>
                        </div>
                        <Button variant="ghost" size="sm" className="h-8">
                          <ArrowRight className="h-4 w-4" />
                        </Button>
                      </div>
                    ))}
                  </div>
                </CardContent>
                <CardFooter className="pt-0">
                  <Link href="/resources" className="w-full">
                    <Button variant="outline" className="w-full">
                      View All Resources
                    </Button>
                  </Link>
                </CardFooter>
              </Card>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 }}
            >
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-xl font-semibold flex items-center">
                    <Users className="h-5 w-5 mr-2 text-blue-600" />
                    Community Support
                  </CardTitle>
                  <CardDescription>
                    Connect with others for support
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {[
                      {
                        title: "Hepatitis Support Group",
                        members: "1,245 members",
                      },
                      {
                        title: "Treatment Experiences",
                        members: "876 members",
                      },
                      { title: "Caregivers Network", members: "532 members" },
                    ].map((group, i) => (
                      <div
                        key={i}
                        className="flex justify-between items-center p-3 rounded-lg border border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors"
                      >
                        <div>
                          <p className="font-medium text-slate-900 dark:text-white">
                            {group.title}
                          </p>
                          <p className="text-sm text-slate-500 dark:text-slate-400">
                            {group.members}
                          </p>
                        </div>
                        <Button variant="ghost" size="sm" className="h-8">
                          <ArrowRight className="h-4 w-4" />
                        </Button>
                      </div>
                    ))}
                  </div>
                </CardContent>
                <CardFooter className="pt-0">
                  <Button variant="outline" className="w-full">
                    Join Community
                  </Button>
                </CardFooter>
              </Card>
            </motion.div>
          </div>
        </TabsContent>

        <TabsContent value="history">
          <Card>
            <CardHeader>
              <CardTitle>Assessment History</CardTitle>
              <CardDescription>
                View your previous hepatitis risk assessments
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                {[
                  {
                    date: "September 15, 2023",
                    result: "Low Risk",
                    type: "Hepatitis C",
                    score: 15,
                    status: "completed",
                  },
                  {
                    date: "June 22, 2023",
                    result: "Low Risk",
                    type: "Hepatitis B",
                    score: 28,
                    status: "completed",
                  },
                  {
                    date: "March 10, 2023",
                    result: "Low Risk",
                    type: "Hepatitis A",
                    score: 35,
                    status: "completed",
                  },
                ].map((assessment, i) => (
                  <div
                    key={i}
                    className="border border-slate-200 dark:border-slate-700 rounded-lg overflow-hidden"
                  >
                    <div className="p-4 bg-slate-50 dark:bg-slate-800 border-b border-slate-200 dark:border-slate-700 flex justify-between items-center">
                      <div>
                        <p className="font-medium text-slate-900 dark:text-white">
                          {assessment.date}
                        </p>
                        <p className="text-sm text-slate-500 dark:text-slate-400">
                          Assessment #{3 - i}
                        </p>
                      </div>
                      <Badge
                        className={
                          assessment.score > 70
                            ? "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300"
                            : assessment.score > 50
                            ? "bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-300"
                            : assessment.score > 30
                            ? "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300"
                            : "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300"
                        }
                      >
                        {assessment.result}
                      </Badge>
                    </div>
                    <div className="p-4">
                      <div className="flex justify-between items-center mb-2">
                        <p className="text-sm font-medium text-slate-900 dark:text-white">
                          Risk Score
                        </p>
                        <p className="text-sm text-slate-500 dark:text-slate-400">
                          {assessment.score}%
                        </p>
                      </div>
                      <Progress value={assessment.score} className="h-2" />

                      <div className="mt-4 flex justify-between items-center">
                        <div>
                          <p className="text-sm text-slate-500 dark:text-slate-400">
                            Type
                          </p>
                          <p className="font-medium text-slate-900 dark:text-white">
                            {assessment.type}
                          </p>
                        </div>
                        <div>
                          <p className="text-sm text-slate-500 dark:text-slate-400">
                            Status
                          </p>
                          <p className="font-medium text-green-600 dark:text-green-400 capitalize">
                            {assessment.status}
                          </p>
                        </div>
                        <Button variant="outline" size="sm">
                          View Details
                        </Button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="resources">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[
              {
                title: "Understanding Hepatitis",
                description:
                  "Learn about the different types of hepatitis, their causes, and how they affect the liver.",
                icon: <BookOpen className="h-10 w-10 text-blue-600" />,
                category: "Educational",
              },
              {
                title: "Prevention Strategies",
                description:
                  "Discover effective ways to prevent hepatitis infection through vaccination and lifestyle changes.",
                icon: <Activity className="h-10 w-10 text-green-600" />,
                category: "Guide",
              },
              {
                title: "Treatment Options",
                description:
                  "Explore current treatment approaches for different types of hepatitis.",
                icon: <FileText className="h-10 w-10 text-purple-600" />,
                category: "Medical",
              },
              {
                title: "Living with Hepatitis",
                description:
                  "Tips and advice for managing daily life with a hepatitis diagnosis.",
                icon: <Users className="h-10 w-10 text-amber-600" />,
                category: "Lifestyle",
              },
              {
                title: "Support Resources",
                description:
                  "Find support groups, counseling services, and other resources for hepatitis patients.",
                icon: <Heart className="h-10 w-10 text-red-600" />,
                category: "Support",
              },
              {
                title: "Research & News",
                description:
                  "Stay updated with the latest research findings and news about hepatitis treatments.",
                icon: <BarChart3 className="h-10 w-10 text-indigo-600" />,
                category: "News",
              },
            ].map((resource, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.1 }}
              >
                <Card className="h-full flex flex-col">
                  <CardHeader>
                    <div className="p-3 bg-slate-100 dark:bg-slate-800 rounded-full w-16 h-16 flex items-center justify-center mb-4">
                      {resource.icon}
                    </div>
                    <CardTitle>{resource.title}</CardTitle>
                    <Badge className="w-fit">{resource.category}</Badge>
                  </CardHeader>
                  <CardContent className="flex-grow">
                    <p className="text-slate-500 dark:text-slate-400">
                      {resource.description}
                    </p>
                  </CardContent>
                  <CardFooter className="pt-0">
                    <Button className="w-full">Read More</Button>
                  </CardFooter>
                </Card>
              </motion.div>
            ))}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
