"use client"

import * as React from "react"
import Link from "next/link"
import { Brain, Shield } from "lucide-react";

import {
  NavigationMenu,
  NavigationMenuItem,
  NavigationMenuLink,
  NavigationMenuList,
} from "@/components/ui/navigation-menu"

export default function Navbar() {
  return (
      <header className="border-b bg-background/80 backdrop-blur-sm">
        <div className="container mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Brain className="w-8 h-8 text-primary" />
            <div>
              <h1 className="text-2xl font-bold">TBNet</h1>
              <p className="text-muted-foreground text-sm">
                Advanced Tuberculosis Detection System
              </p>
            </div>
          </div>
        </div>
      </header>
  )
}
