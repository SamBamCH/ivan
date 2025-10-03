"""
Run this script ONCE to create and configure your Selenium Chrome profile.
After running this, you can manually log into DraftKings, then close the browser.
"""

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import os
import time

def setup_chrome_profile():
    # Define your profile directory
    user_data_dir = r"C:\Users\ivan-\ivan\chrome-selenium\dk-profile"
    profile_directory = "Default"
    
    # Create the directory if it doesn't exist
    os.makedirs(user_data_dir, exist_ok=True)
    
    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument(f"--user-data-dir={user_data_dir}")
    chrome_options.add_argument(f"--profile-directory={profile_directory}")
    
    # Optional: Disable some automation flags that websites might detect
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    print(f"Creating Chrome profile at: {user_data_dir}")
    print("A Chrome window will open. Please:")
    print("1. Navigate to DraftKings.com")
    print("2. Log in with your credentials")
    print("3. Complete any 2FA/verification if needed")
    print("4. Once logged in, CLOSE THE BROWSER WINDOW")
    print("\nStarting browser in 3 seconds...")
    time.sleep(3)
    
    # Launch Chrome
    driver = webdriver.Chrome(options=chrome_options)
    
    # Navigate to DraftKings
    driver.get("https://www.draftkings.com/")
    
    # Wait for user to log in and close browser
    print("\nWaiting for you to log in and close the browser...")
    print("(The script will exit when you close the browser window)")
    
    try:
        # Keep the browser open until user closes it
        while True:
            time.sleep(1)
    except Exception:
        pass
    finally:
        try:
            driver.quit()
        except:
            pass
    
    print("\n✓ Profile setup complete!")
    print(f"Your profile is saved at: {user_data_dir}")
    print("You can now run your main script (get_contest_data.py)")

if __name__ == "__main__":
    setup_chrome_profile()