import pandas as pd
import matplotlib.pyplot as plt

def analyze_memory_leak(file_path, memory_column):
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Ensure the correct column exists
        if memory_column not in df.columns:
            print(f"Column '{memory_column}' not found in the file.")
            print(f"Available columns: {df.columns}")
            return

        # Convert timestamps if available
        if 'Time' in df.columns:
            df['Time'] = pd.to_datetime(df['Time'])

        # Plot memory usage over time
        plt.figure(figsize=(10, 5))
        plt.plot(df['Time'], df[memory_column], label="Memory Usage", color="red")
        plt.xlabel("Time")
        plt.ylabel("Memory (MB)")
        plt.title("Memory Usage Over Time")
        plt.legend()
        plt.grid()
        plt.show()

        # Check for increasing trend (basic memory leak detection)
        if df[memory_column].is_monotonic_increasing:
            print("⚠️ Potential memory leak detected: Memory usage is continuously increasing.")
        else:
            print("✅ No consistent memory leak detected.")

    except FileNotFoundError:
        print("Error: File not found. Please check the filename and try again.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Get file path from user
file_path = input("Enter the path of the Performance Monitor CSV file: ")
memory_column = input("Enter the memory column to analyze (e.g., 'Working Set'): ")

analyze_memory_leak(file_path, memory_column)

