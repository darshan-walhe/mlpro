import os
import getpass
import requests
from requests.auth import HTTPBasicAuth

def upload_to_nextcloud():
    print("🚂 Nextcloud Project Uploader")
    print("-" * 30)
    
    # Nextcloud details
    base_url = "https://ivo.lv.tab.digital"
    
    # ---------------------------------------------------------
    # ZERO-CREDENTIALS / "DUMMY ACCOUNT" METHOD
    # Instead of a real account, we use a Public Share Token.
    # ---------------------------------------------------------
    share_token = os.environ.get("NEXTCLOUD_SHARE_TOKEN")
    if not share_token:
        print("To avoid logging in, create a public 'File drop (upload only)' share link in Nextcloud.")
        print("Example link: https://ivo.lv.tab.digital/s/AbCdEfGhIjKlMn")
        print("The token is the last part: AbCdEfGhIjKlMn\n")
        share_token = input("Enter your Nextcloud Public Share Token: ").strip()
    
    # For public shares, the WebDAV URL is different
    webdav_url = f"{base_url}/public.php/webdav"
    
    # Nextcloud public WebDAV uses the share token as the username
    # and a blank password (unless you set a password on the link).
    auth = HTTPBasicAuth(share_token, "")
    
    # Create a session to reuse the SSL connection
    session = requests.Session()
    session.auth = auth
    
    # Verify connection to the shared folder
    print("\nConnecting to Nextcloud...")
    try:
        response = session.request("PROPFIND", webdav_url)
        
        if response.status_code in [200, 207]: # 207 Multi-Status is standard for WebDAV PROPFIND
            print("✅ Connection successful. Target folder ready.")
        else:
            print(f"❌ Failed to connect. Status Code: {response.status_code}")
            print("Please check your share token and try again.")
            return
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return

    # Files to upload (skipping venv and git/cache folders)
    project_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("\nUploading files...")
    for root, dirs, files in os.walk(project_dir):
        # Skip certain directories
        dirs[:] = [d for d in dirs if d not in ['venv', '.git', '__pycache__', 'node_modules', '.pytest_cache']]
        
        for file in files:
            if file == "upload_to_nextcloud.py":
                continue
                
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, project_dir)
            remote_path = f"{webdav_url}/{relative_path.replace(os.sep, '/')}"
            
            # Create subdirectories if needed (handles deep nesting)
            rel_dir = os.path.dirname(relative_path).replace(os.sep, '/')
            if rel_dir:
                parts = rel_dir.split('/')
                current_dir = webdav_url
                for part in parts:
                    current_dir = f"{current_dir}/{part}"
                    session.request("MKCOL", current_dir)
            
            print(f"Uploading {relative_path}...")
            try:
                with open(local_path, 'rb') as f:
                    file_data = f.read() # Read into memory to avoid chunked encoding issues
                r = session.put(remote_path, data=file_data)
                if r.status_code in [201, 204]:
                    print(f"  -> Success")
                else:
                    print(f"  -> Failed ({r.status_code})")
            except Exception as e:
                print(f"  -> Exception occurred: {e}")

    print("\n🎉 Upload complete! Check your Nextcloud dashboard.")

if __name__ == "__main__":
    upload_to_nextcloud()
