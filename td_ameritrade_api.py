"""
TD Ameritrade API Integration for Options Chain Data
Includes authentication, real-time data streaming, and advanced options analytics
"""

import requests
import json
import time
import websocket
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import os
from urllib.parse import urlencode
import base64
import threading
from queue import Queue


class TDAmeritradeAuth:
    """
    Handle TD Ameritrade OAuth 2.0 authentication
    """
    
    def __init__(self, client_id: str, redirect_uri: str = 'http://localhost:8080'):
        """
        Initialize TD Ameritrade authentication
        
        Args:
            client_id: Your TD Ameritrade app client ID
            redirect_uri: Redirect URI registered with your app
        """
        self.client_id = client_id
        self.redirect_uri = redirect_uri
        self.token_path = 'td_token.json'
        self.access_token = None
        self.refresh_token = None
        self.token_expiry = None
        
        # API endpoints
        self.auth_url = 'https://auth.tdameritrade.com/auth'
        self.token_url = 'https://api.tdameritrade.com/v1/oauth2/token'
        
        # Load existing token if available
        self.load_token()
    
    def get_auth_url(self) -> str:
        """Generate the authorization URL for initial authentication"""
        params = {
            'response_type': 'code',
            'redirect_uri': self.redirect_uri,
            'client_id': f'{self.client_id}@AMER.OAUTHAP'
        }
        return f"{self.auth_url}?{urlencode(params)}"
    
    def authenticate_with_code(self, auth_code: str):
        """
        Exchange authorization code for access and refresh tokens
        
        Args:
            auth_code: Authorization code from TD Ameritrade
        """
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        data = {
            'grant_type': 'authorization_code',
            'refresh_token': '',
            'access_type': 'offline',
            'code': auth_code,
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri
        }
        
        response = requests.post(self.token_url, headers=headers, data=data)
        
        if response.status_code == 200:
            token_data = response.json()
            self.access_token = token_data['access_token']
            self.refresh_token = token_data['refresh_token']
            self.token_expiry = time.time() + token_data['expires_in']
            self.save_token()
            print("Authentication successful!")
        else:
            raise Exception(f"Authentication failed: {response.text}")
    
    def refresh_access_token(self):
        """Refresh the access token using the refresh token"""
        if not self.refresh_token:
            raise Exception("No refresh token available. Please authenticate first.")
        
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        data = {
            'grant_type': 'refresh_token',
            'refresh_token': self.refresh_token,
            'client_id': self.client_id
        }
        
        response = requests.post(self.token_url, headers=headers, data=data)
        
        if response.status_code == 200:
            token_data = response.json()
            self.access_token = token_data['access_token']
            self.token_expiry = time.time() + token_data['expires_in']
            self.save_token()
            print("Token refreshed successfully!")
        else:
            raise Exception(f"Token refresh failed: {response.text}")
    
    def get_headers(self) -> Dict:
        """Get headers with valid access token"""
        if not self.access_token or time.time() >= self.token_expiry:
            self.refresh