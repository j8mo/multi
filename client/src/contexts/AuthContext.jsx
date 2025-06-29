import React, { createContext, useContext, useEffect, useState } from 'react'
import { supabase } from '../lib/supabase'
import toast from 'react-hot-toast'

const AuthContext = createContext({})

export const useAuth = () => {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Get initial session
    supabase.auth.getSession().then(({ data: { session } }) => {
      console.log('Initial session check:', session?.user?.email || 'No session')
      setUser(session?.user ?? null)
      setLoading(false)
    })

    // Listen for auth changes
    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      async (event, session) => {
        console.log('Auth state changed:', event, session?.user?.email)
        setUser(session?.user ?? null)
        setLoading(false)
        
        // Handle successful signup
        if (event === 'SIGNED_UP' && session?.user) {
          console.log('User signed up successfully:', session.user.email)
        }
        
        // Handle successful signin
        if (event === 'SIGNED_IN' && session?.user) {
          console.log('User signed in successfully:', session.user.email)
          // Force redirect to dashboard after successful OAuth signin
          if (window.location.hash.includes('access_token')) {
            console.log('OAuth callback detected, redirecting to dashboard')
            window.location.href = '/dashboard'
          }
        }
        
        // Handle token refresh
        if (event === 'TOKEN_REFRESHED' && session?.user) {
          console.log('Token refreshed for user:', session.user.email)
        }
      }
    )

    return () => subscription.unsubscribe()
  }, [])

  const signInWithGoogle = async () => {
    try {
      const { error } = await supabase.auth.signInWithOAuth({
        provider: 'google',
        options: {
          redirectTo: `${window.location.origin}/dashboard`,
          queryParams: {
            access_type: 'offline',
            prompt: 'consent',
          }
        }
      })
      if (error) {
        if (error.message.includes('provider is not enabled')) {
          toast.error('Google authentication is not configured. Please use email authentication instead.')
          return
        }
        if (error.message.includes('redirect_uri_mismatch')) {
          toast.error('Google OAuth is not properly configured. Please use email authentication instead.')
          return
        }
        throw error
      }
    } catch (error) {
      toast.error('Failed to sign in with Google. Please use email authentication instead.')
      console.error('Error:', error)
    }
  }

  const signInWithEmail = async (email, password) => {
    try {
      const { data, error } = await supabase.auth.signInWithPassword({
        email,
        password
      })
      if (error) throw error
      
      console.log('Sign in successful:', data.user?.email)
      toast.success('Signed in successfully')
    } catch (error) {
      console.error('Sign in error:', error)
      toast.error(error.message || 'Failed to sign in')
      throw error
    }
  }

  const signUpWithEmail = async (email, password) => {
    try {
      const { data, error } = await supabase.auth.signUp({
        email,
        password,
        options: {
          emailRedirectTo: `${window.location.origin}/dashboard`
        }
      })
      
      if (error) throw error
      
      console.log('Sign up successful:', data.user?.email)
      
      if (data.user && !data.session) {
        toast.success('Please check your email to verify your account before signing in.')
      } else if (data.user && data.session) {
        toast.success('Account created successfully! Welcome to MultiTalk.')
      }
    } catch (error) {
      console.error('Sign up error:', error)
      toast.error(error.message || 'Failed to create account')
      throw error
    }
  }

  const signOut = async () => {
    try {
      const { error } = await supabase.auth.signOut()
      if (error) throw error
      toast.success('Signed out successfully')
    } catch (error) {
      toast.error('Failed to sign out')
      console.error('Error:', error)
    }
  }

  const value = {
    user,
    loading,
    signInWithGoogle,
    signInWithEmail,
    signUpWithEmail,
    signOut
  }

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  )
}