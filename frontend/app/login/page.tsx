"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import toast from "react-hot-toast";
import LoginForm from "@/app/common/login-form";

export default function LoginPage() {
  const router = useRouter();
  const [open, setOpen] = useState(true);

  const handleClose = () => {
    setOpen(false);
    router.push("/");
  };

  return (
    <div className="min-h-[70vh] flex items-center justify-center">
      {open && (
        <LoginForm
          onSwitchToRegister={() => router.push("/")}
          onClose={() => {
            toast.success("Logged in successfully");
            handleClose();
            router.refresh();
          }}
        />
      )}
    </div>
  );
}

