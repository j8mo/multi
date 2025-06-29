import React, { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { supabase } from '../lib/supabase'

const AuthCallback = () => {
  const navigate = useNavigate()

  useEffect(() => {
    const handleAuthCallback = async () => {
      try {
        console.log('Processing OAuth callback...')
        console.log('Current URL:', window.location.href)
        
        // Get the session after OAuth callback
        const { data: { session }, error } = await supabase.auth.getSession()
        
        if (error) {
          console.error('Error getting session:', error)
          navigate('/login?error=auth_callback_failed')
          return
        }
        
        if (session && session.user) {
          console.log('OAuth callback successful, user:', session.user.email)
          navigate('/dashboard')
        } else {
          console.log('No session found, redirecting to login')
          navigate('/login')
        }
      } catch (error) {
        console.error('OAuth callback error:', error)
        navigate('/login?error=callback_error')
      }
    }

    // Handle the callback after component mounts
    handleAuthCallback()
  }, [navigate])

  return (
    <div className="min-h-screen flex items-center justify-center">
      <div className="text-center">
        <div className="loading-spinner mb-4"></div>
        <p className="text-gray-600">Completing login...</p>
      </div>
    </div>
  )
}

export default AuthCallback
