import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta

# Constants for matching existing columns
COLUMNS = [
    'PNR Number', 'Train Number', 'Date of Journey', 'Class of Travel',
    'Quota', 'Source Station', 'Destination Station', 'Booking Date',
    'Current Status', 'Number of Passengers', 'Age of Passengers',
    'Booking Channel', 'Travel Distance', 'Number of Stations',
    'Travel Time', 'Train Type', 'Seat Availability',
    'Special Considerations', 'Holiday or Peak Season', 'Waitlist Position',
    'Confirmation Status'
]

CLASS_OPTIONS = ["Sleeper", "1AC", "2AC", "3AC"]
QUOTA_OPTIONS = ["General", "Tatkal", "Premium Tatkal", "Ladies"]
AGE_GROUP_OPTIONS = ["Adult", "Child", "Senior Citizen"]
BOOKING_CHANNEL_OPTIONS = ["IRCTC Website", "Mobile App", "Counter"]
TRAIN_TYPE_OPTIONS = ["Express", "Superfast", "Rajdhani", "Shatabdi"]
PEAK_SEASON_OPTIONS = ["Yes", "No"]

def generate_synthetic_data(num_samples=30000):
    np.random.seed(42)
    random.seed(42)

    data = []
    base_date = datetime(2025, 1, 1)

    for i in range(num_samples):
        # Base features
        pnr = f"{random.randint(1000000000, 9999999999)}"
        train_num = f"{random.randint(10000, 99999)}"
        
        # Journey & Booking dates (Days Before Journey)
        days_before = np.random.randint(0, 121)  # Fixed: match UI slider range (0–120)
        journey_date = base_date + timedelta(days=np.random.randint(0, 365))
        booking_date = journey_date - timedelta(days=days_before)

        class_travel = np.random.choice(CLASS_OPTIONS, p=[0.4, 0.1, 0.2, 0.3])
        quota = np.random.choice(QUOTA_OPTIONS)
        age = np.random.choice(AGE_GROUP_OPTIONS)
        channel = np.random.choice(BOOKING_CHANNEL_OPTIONS)
        train_type = np.random.choice(TRAIN_TYPE_OPTIONS)
        peak_season = np.random.choice(PEAK_SEASON_OPTIONS)
        
        passengers = np.random.randint(1, 6)
        distance = np.random.randint(100, 2000)
        stations = np.random.randint(2, 30)
        time = round(distance / 60.0, 1)
        
        special_considerations = "Yes" if np.random.random() < 0.05 else np.nan

        # Is the ticket initially waitlisted or confirmed?
        # Let's say 30% are initially confirmed, 70% are waitlisted.
        is_initially_waitlisted = np.random.random() < 0.7
        
        score = 0.0
        
        if not is_initially_waitlisted:
            waitlist_pos = np.nan
            seat_avail = np.random.randint(1, 200)
            current_status = "Confirmed"
            conf_status = "Confirmed"
        else:
            seat_avail = 0
            # Waitlist number between 1 and 200
            wl_number = np.random.randint(1, 200)
            waitlist_pos = f"WL{wl_number:03d}"
            
            # Predict realistic confirmation chance based on waitlist and days
            # Higher days before = good. Lower WL = good.
            # Convert to a logistic probability curve
            # Base probability
            base_score = 2.0
            
            # Penalty for long waitlist
            base_score -= (wl_number / 20.0)
            
            # Bonus for booking early
            base_score += (days_before / 30.0)
            
            # Penalty for more passengers
            base_score -= (passengers * 0.4)
            
            # Class penalties
            if class_travel == "1AC": base_score -= 1.5
            elif class_travel == "2AC": base_score -= 1.0
            
            if peak_season == "Yes": base_score -= 1.0
            if special_considerations == "Yes": base_score += 1.0
            
            # Convert score to probability
            prob = 1 / (1 + np.exp(-base_score))
            
            if np.random.random() < prob:
                conf_status = "Confirmed"
                current_status = "Confirmed"
            else:
                conf_status = "Not Confirmed"
                current_status = "Waitlisted"
        
        row = {
            'PNR Number': pnr,
            'Train Number': train_num,
            'Date of Journey': journey_date.strftime("%Y-%m-%d"),
            'Class of Travel': class_travel,
            'Quota': quota,
            'Source Station': f"Station {np.random.randint(1, 100)}",
            'Destination Station': f"Station {np.random.randint(101, 200)}",
            'Booking Date': booking_date.strftime("%Y-%m-%d"),
            'Current Status': current_status,
            'Number of Passengers': passengers,
            'Age of Passengers': age,
            'Booking Channel': channel,
            'Travel Distance': distance,
            'Number of Stations': stations,
            'Travel Time': time,
            'Train Type': train_type,
            'Seat Availability': seat_avail,
            'Special Considerations': special_considerations,
            'Holiday or Peak Season': peak_season,
            'Waitlist Position': waitlist_pos,
            'Confirmation Status': conf_status
        }
        data.append(row)

    df = pd.DataFrame(data, columns=COLUMNS)
    return df

if __name__ == "__main__":
    filepath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "Railway Ticket Confirmation.csv")
    print(f"Generating realistic synthetic dataset and saving to {filepath}")
    df = generate_synthetic_data(30000)
    df.to_csv(filepath, index=False)
    print("Dataset generated successfully!")
    print(df['Confirmation Status'].value_counts())
    
    # Check confirmation rate for waitlisted tickets
    waitlisted = df[df['Waitlist Position'].notna()]
    print("\nWaitlisted tickets ONLY - Confirmation Status:")
    print(waitlisted['Confirmation Status'].value_counts())
